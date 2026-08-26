"""Fixed-reference deviation experiment helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import torch

from aura.AURA import AURA
from aura.optimization import optimize_controls
from methods.plan import ControlEdge, duration_seconds_to_steps
from methods.MPPI import MPPIController, MPPIParameters
from simulation.pushing_model import get_pushing_model
from train_model import load_opt_model_2
from utils.experiment_io import write_csv
from utils.utils import arrayDistance, is_state_array_valid

@dataclass(frozen=True)
class ReferenceTrajectory:
    """One immutable nominal rollout shared by every tracking method."""

    system: str
    environment: str
    initial_state: np.ndarray
    states: np.ndarray
    controls: np.ndarray
    control_duration: float
    generation_attempts: int = 1

    def __post_init__(self) -> None:
        states = np.asarray(self.states, dtype=float)
        controls = np.asarray(self.controls, dtype=float)
        initial = np.asarray(self.initial_state, dtype=float).reshape(-1)
        if states.ndim != 2 or controls.ndim != 2:
            raise ValueError("reference states and controls must be rank-two arrays")
        if len(states) != len(controls) + 1:
            raise ValueError("a reference must contain one more state than control")
        if not np.array_equal(states[0], initial):
            raise ValueError("reference state zero must equal the initial state")
        states = states.copy()
        controls = controls.copy()
        initial = initial.copy()
        states.setflags(write=False)
        controls.setflags(write=False)
        initial.setflags(write=False)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "initial_state", initial)

    @property
    def num_controls(self) -> int:
        return int(len(self.controls))

    def as_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "environment": self.environment,
            "initial_state": self.initial_state.tolist(),
            "x_ref": self.states.tolist(),
            "u_ref": self.controls.tolist(),
            "control_duration": float(self.control_duration),
            "generation_attempts": int(self.generation_attempts),
        }


def sample_push_control(
    system, rng: np.random.Generator, *, environment: str
) -> np.ndarray:
    if environment == "mujoco":
        return np.asarray(
            [0.0, rng.uniform(-0.32, 0.32), rng.uniform(0.045, 0.085)],
            dtype=float,
        )
    bounds = np.asarray(system.control_bounds, dtype=float)
    faces = np.asarray([0.0, 0.25, 0.5, 0.75], dtype=float)
    faces = faces[(faces >= bounds[0, 0]) & (faces <= bounds[0, 1])]
    if not len(faces):
        raise ValueError("pushing control bounds contain no valid face")
    side_intervals = [
        (bounds[1, 0], min(bounds[1, 1], -0.05)),
        (max(bounds[1, 0], 0.05), bounds[1, 1]),
    ]
    side_intervals = [(low, high) for low, high in side_intervals if low < high]
    if side_intervals:
        widths = np.asarray([high - low for low, high in side_intervals])
        interval = side_intervals[
            int(rng.choice(len(side_intervals), p=widths / widths.sum()))
        ]
        side = rng.uniform(*interval)
    else:
        side = rng.uniform(bounds[1, 0], bounds[1, 1])
    return np.asarray(
        [
            rng.choice(faces),
            side,
            rng.uniform(bounds[2, 0], bounds[2, 1]),
        ],
        dtype=float,
    )


def sample_random_controls(
    system,
    num_controls: int,
    rng: np.random.Generator,
    *,
    environment: str = "gaussian",
    start_state: np.ndarray | None = None,
    duration: float | None = None,
) -> list[np.ndarray]:
    controls: list[np.ndarray] = []
    current = (
        np.asarray(start_state, dtype=float).reshape(-1).copy()
        if start_state is not None
        else None
    )

    def ordinary_control() -> np.ndarray:
        if system.name == "dubins_airplane":
            return np.asarray(
                [
                    rng.uniform(-0.08, 0.08),
                    rng.uniform(-0.4, 0.4),
                    rng.uniform(-0.2, 0.2),
                ],
                dtype=float,
            )
        return np.asarray(
            [rng.uniform(low, high) for low, high in system.control_bounds],
            dtype=float,
        )

    for _ in range(int(num_controls)):
        if system.name == "pushing_object":
            control = sample_push_control(system, rng, environment=environment)
        elif system.name == "dubins_airplane" and current is not None and duration:
            candidates = []
            for _ in range(256):
                proposed = ordinary_control()
                next_state = np.asarray(
                    system.propagate(current, proposed, float(duration)), dtype=float
                )
                if not np.isfinite(next_state).all():
                    continue
                if any(
                    next_state[index] < low or next_state[index] > high
                    for index, (low, high) in enumerate(system.state_bounds)
                ):
                    continue
                clearance = min(
                    min(next_state[index] - low, high - next_state[index])
                    / max(high - low, 1e-12)
                    for index, (low, high) in enumerate(system.state_bounds[:3])
                )
                candidates.append((clearance, proposed, next_state))
            if not candidates:
                raise RuntimeError("could not sample a valid airplane control")
            candidates.sort(key=lambda item: item[0], reverse=True)
            _, control, current = candidates[
                int(rng.integers(0, min(16, len(candidates))))
            ]
        else:
            control = ordinary_control()
        controls.append(np.asarray(control, dtype=float))
    return controls


def sample_mujoco_push_sequence(
    system,
    start_state: np.ndarray,
    num_controls: int,
    rng: np.random.Generator,
    duration: float,
) -> list[np.ndarray]:
    controls: list[np.ndarray] = []
    current = np.asarray(start_state, dtype=float).reshape(-1).copy()
    previous_face: float | None = None
    for _ in range(int(num_controls)):
        candidates = []
        for _ in range(160):
            face = float(rng.choice([0.0, 0.25, 0.5, 0.75]))
            control = np.asarray(
                [face, rng.uniform(-0.32, 0.32), rng.uniform(0.035, 0.075)]
            )
            next_state = np.asarray(
                system.propagate(current, control, duration), dtype=float
            )
            if not (-0.12 <= next_state[0] <= 0.76 and -0.82 <= next_state[1] <= -0.34):
                continue
            score = float(np.linalg.norm(next_state[:2] - current[:2]))
            score -= 0.22 * abs(float(next_state[1] + 0.58))
            if previous_face is not None and face == previous_face:
                score -= 0.025
            score += float(rng.normal(0.0, 0.006))
            candidates.append((score, control, next_state))
        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            _, control, next_state = candidates[
                int(rng.integers(0, min(8, len(candidates))))
            ]
        else:
            control = np.asarray([0.0, 0.0, 0.045])
            next_state = np.asarray(
                system.propagate(current, control, duration), dtype=float
            )
        controls.append(control)
        current = next_state.copy()
        current[2] = (current[2] + np.pi) % (2.0 * np.pi) - np.pi
        previous_face = float(control[0])
    return controls


def propagate_reference_states(
    system, start_state: np.ndarray, controls, duration: float
) -> list[np.ndarray]:
    states = [np.asarray(start_state, dtype=float).copy()]
    for control in controls:
        states.append(
            np.asarray(system.propagate(states[-1], control, duration), dtype=float)
        )
    return states


def _sample_interior_initial_state(
    system,
    rng: np.random.Generator,
    *,
    boundary_margin_fraction: float,
    state_bounds: list[list[float]] | list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Sample inside the system's native bounds.

    The repository has local Gaussian state samplers but no uniform valid
    initial-state sampler.  This small adapter therefore samples the existing
    state bounds directly.  A subsequent nominal-rollout validity check is the
    authoritative rejection criterion.
    """

    margin = float(boundary_margin_fraction)
    if not 0.0 <= margin < 0.5:
        raise ValueError("boundary_margin_fraction must lie in [0, 0.5)")
    bounds = np.asarray(state_bounds or system.state_bounds, dtype=float)
    lows = bounds[:, 0]
    highs = bounds[:, 1]
    widths = highs - lows
    inner_low = lows + margin * widths
    inner_high = highs - margin * widths
    bounded = rng.uniform(inner_low, inner_high)
    if system.name in ("kinematic_car", "pushing_object"):
        return np.asarray([bounded[0], bounded[1], rng.uniform(-np.pi, np.pi)])
    return np.asarray(bounded, dtype=float)


def _state_is_valid(system, state: np.ndarray, config: dict) -> bool:
    state = np.asarray(state, dtype=float).reshape(-1)
    if not np.isfinite(state).all():
        return False
    bounds = np.asarray(system.state_bounds, dtype=float)
    if state.size < len(bounds):
        return False
    if np.any(state[: len(bounds)] < bounds[:, 0]) or np.any(
        state[: len(bounds)] > bounds[:, 1]
    ):
        return False
    return bool(
        is_state_array_valid(
            state,
            system=system.name,
            config={"state_bounds": system.state_bounds},
            obstacle_config=config.get("obstacles"),
        )
    )


def generate_nominal_trajectory(
    system,
    initial_state: np.ndarray,
    num_controls: int,
    rng: np.random.Generator,
    config: dict,
) -> ReferenceTrajectory:
    """Sample controls once and propagate them with the nominal model Gamma."""

    duration = float(config["control_duration"])
    if system.name == "pushing_object" and config["environment"] == "mujoco":
        # Reuse the submitted tracking driver's table-aware sequence sampler.
        # Independent random face-0 pushes can make a ten-step valid sequence
        # vanishingly unlikely and are not the protocol used by that driver.
        controls = sample_mujoco_push_sequence(
            system,
            np.asarray(initial_state, dtype=float),
            int(num_controls),
            rng,
            duration,
        )
    else:
        controls = sample_random_controls(
            system,
            int(num_controls),
            rng,
            environment=str(config["environment"]),
        )
    bounds = np.asarray(system.control_bounds, dtype=float)
    controls_array = np.asarray(controls, dtype=float)
    controls_array = np.clip(controls_array, bounds[:, 0], bounds[:, 1])
    states = propagate_reference_states(system, initial_state, controls_array, duration)
    return ReferenceTrajectory(
        system=system.name,
        environment=str(config["environment"]),
        initial_state=states[0],
        states=np.asarray(states, dtype=float),
        controls=controls_array,
        control_duration=duration,
    )


def generate_valid_reference(
    system,
    num_controls: int,
    rng: np.random.Generator,
    config: dict,
    *,
    max_attempts: int = 10_000,
    boundary_margin_fraction: float = 0.20,
) -> ReferenceTrajectory:
    """Rejection-sample only on nominal validity, never method performance."""

    for attempt in range(1, int(max_attempts) + 1):
        initial = _sample_interior_initial_state(
            system,
            rng,
            boundary_margin_fraction=boundary_margin_fraction,
            state_bounds=config.get("initial_state_bounds"),
        )
        try:
            reference = generate_nominal_trajectory(
                system, initial, num_controls, rng, config
            )
        except (FloatingPointError, RuntimeError, ValueError):
            continue
        if all(_state_is_valid(system, state, config) for state in reference.states):
            return ReferenceTrajectory(
                system=reference.system,
                environment=reference.environment,
                initial_state=reference.initial_state,
                states=reference.states,
                controls=reference.controls,
                control_duration=reference.control_duration,
                generation_attempts=attempt,
            )
    raise RuntimeError(
        f"failed to sample a valid {system.name} nominal trajectory after "
        f"{max_attempts} attempts"
    )

METHOD_ORDER = ("aura", "mppi", "open_loop")


def canonicalize_state(system_name: str, state) -> np.ndarray:
    values = np.asarray(state, dtype=float).reshape(-1)
    dimensions = {
        "kinematic_car": 3,
        "pushing_object": 3,
        "double_integrator": 6,
        "dubins_airplane": 6,
    }
    return values[: dimensions.get(system_name, len(values))]


def load_pushing_optimizer(system, learning_rate: float, iterations: int):
    model = get_pushing_model(
        system.object_shape,
        model_name=getattr(system, "model_name", "cracker_box_flipped"),
        model_path=getattr(system, "model_path", None),
    )
    return load_opt_model_2(
        model, lr=float(learning_rate), epochs=int(iterations)
    )


class PickerSimulator:
    def __init__(self, config: dict | None = None):
        self.config = dict(config or {})


class PickerPlanner:
    obstacle_config = None
    optimizer_child_match_tolerance = 1e-5

    def __init__(self, validation_step: float, propagation_step: float):
        self.motion_validation_step_size = float(validation_step)
        self.propagation_step_size = float(propagation_step)

    def duration_seconds_to_steps(self, duration: float) -> int:
        return duration_seconds_to_steps(
            duration,
            self.propagation_step_size,
            min_steps=1,
            max_steps=1,
        )


def create_aura_picker(system, duration: float, simulator_config: dict) -> AURA:
    picker = AURA.__new__(AURA)
    picker.system = system
    picker.simulator = PickerSimulator(simulator_config)
    picker.planner = PickerPlanner(
        max(0.02, min(0.05, float(duration) / 20.0)), float(duration)
    )
    picker.propagation_step_size = float(duration)
    picker.opt_model = None
    picker.last_control_decision = {}
    return picker


@dataclass
class MethodResult:
    method: str
    initial_state: np.ndarray
    states: list[np.ndarray] = field(default_factory=list)
    controls: list[np.ndarray] = field(default_factory=list)
    timings: list[dict[str, float]] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    success: bool = True
    failure_reason: str = ""

    @property
    def completed_steps(self) -> int:
        return len(self.controls)


def _synchronize(device: torch.device | str | None) -> None:
    if device is None:
        return
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)


def _reset_plant(plant, reference: ReferenceTrajectory) -> np.ndarray:
    plant.reset()
    plant.set_state(reference.initial_state.tolist())
    return canonicalize_state(reference.system, plant.get_state())


class AURALocalTrackingAdapter:
    """Precompute Algorithm 3 batches for immutable nominal edges."""

    global_replanning_enabled = False

    def __init__(self, system, config: dict):
        self.system = system
        self.config = config
        self.duration = float(config["control_duration"])
        self.picker: AURA = create_aura_picker(
            system, self.duration, config["simulator_config"]
        )
        self.opt_model = None
        if system.name == "pushing_object":
            self.opt_model = load_pushing_optimizer(
                system,
                float(config["aura"]["learning_rate"]),
                int(config["aura"]["gradient_iterations"]),
            )
        self.picker.opt_model = self.opt_model
        self.edges: list[ControlEdge] = []
        self.results: list[dict[str, Any] | None] = []
        self.precompute_times: list[float] = []

    def _seed(self, seed: int) -> None:
        np.random.seed(int(seed) % (2**32 - 1))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    def precompute(self, reference: ReferenceTrajectory, seed: int) -> None:
        """Run all local optimizations without reading any executed plant state."""

        self.edges = []
        self.results = []
        self.precompute_times = []
        aura = self.config["aura"]
        requested_device = self.config.get("resolved_device")
        for index, nominal_control in enumerate(reference.controls):
            edge = ControlEdge(
                source_state=reference.states[index],
                target_state=reference.states[index + 1],
                control=nominal_control,
                duration_steps=1,
                duration_seconds=self.duration,
                edge_id=f"reference:{index}",
            )
            self.edges.append(edge)
            self._seed(int(seed) + 104_729 * (index + 1))
            _synchronize(requested_device)
            started = time.perf_counter()
            result = optimize_controls(
                system=self.system,
                # This is the nominal current reference state, not an
                # observation from the plant.
                next_state=reference.states[index],
                child_edges=[edge],
                integration_step_size=self.duration,
                model=self.opt_model,
                num_states=int(aura["batch_size"]),
                position_std=float(aura["position_std"]),
                rotation_std=float(aura["rotation_std"]),
                velocity_std=float(aura["velocity_std"]),
                num_steps=int(aura["gradient_iterations"]),
                learning_rate=float(aura["learning_rate"]),
                requested_device=requested_device,
            )
            _synchronize(requested_device)
            self.precompute_times.append(time.perf_counter() - started)
            self.results.append(result)

    def select(self, index: int, current_state: np.ndarray) -> tuple[np.ndarray, dict]:
        edge = self.edges[index]
        selection = self.picker.pick_next_control(
            system=self.system,
            optimization_result=self.results[index],
            current_state=np.asarray(current_state, dtype=float),
            next_state=edge.target_state,
            children_edges=[edge],
            fallback_edge=edge,
        )
        diagnostic = dict(self.picker.last_control_decision or {})
        diagnostic["selection_source"] = selection.source
        diagnostic["global_replanning_enabled"] = False
        return np.asarray(selection.control, dtype=float).reshape(-1), diagnostic


def mppi_cost_weights(system_name: str) -> dict[str, list[float] | float]:
    """Expose the state/control weights already encoded by MPPIController."""

    if system_name == "double_integrator":
        return {
            "Q": [6.0, 6.0, 6.0, 3.0, 3.0, 3.0],
            "Q_terminal": [40.0] * 6,
            "R": 1.0,
        }
    if system_name == "kinematic_car":
        return {"Q": [4.0, 4.0, 0.3], "Q_terminal": [30.0, 30.0, 4.0], "R": 1.0}
    if system_name == "pushing_object":
        return {"Q": [50.0, 50.0, 5.0], "Q_terminal": [100.0, 100.0, 50.0], "R": 1.0}
    raise ValueError(system_name)


class ReferenceTrackingMPPI(MPPIController):
    """Configure the existing MPPI sampler/rollout/update for fixed references."""

    def __init__(self, *args, **kwargs):
        tracking_cost_weights = kwargs.pop("tracking_cost_weights", None)
        tracking_iterations = kwargs.pop("tracking_iterations", 1)
        nominal_deadband = kwargs.pop("nominal_deadband", -1.0)
        super().__init__(*args, **kwargs)
        weights = dict(mppi_cost_weights(self.system_name))
        if tracking_cost_weights:
            weights.update(dict(tracking_cost_weights))
        self.q = torch.as_tensor(weights["Q"], dtype=self.dtype, device=self.device)
        self.q_terminal = torch.as_tensor(
            weights["Q_terminal"], dtype=self.dtype, device=self.device
        )
        self.r_scale = float(weights["R"])
        self.tracking_iterations = int(tracking_iterations)
        self.nominal_deadband = float(nominal_deadband)
        if self.q.numel() != self.state_dim or self.q_terminal.numel() != self.state_dim:
            raise ValueError("MPPI tracking Q/Q_terminal dimensions must match state")
        if not np.isfinite(self.r_scale) or self.r_scale < 0.0:
            raise ValueError("MPPI tracking R must be finite and nonnegative")
        if self.tracking_iterations < 1:
            raise ValueError("MPPI tracking_iterations must be positive")
        if not np.isfinite(self.nominal_deadband):
            raise ValueError("MPPI nominal_deadband must be finite")

    def initialize_nominal(self, controls: np.ndarray) -> None:
        controls = np.array(controls, dtype=float, copy=True)
        if controls.ndim != 2 or len(controls) < 1:
            raise ValueError("MPPI nominal initialization requires controls")
        internal = torch.as_tensor(controls, dtype=self.dtype, device=self.device).clone()
        if self.is_pushing:
            internal[:, 0] = torch.round(internal[:, 0] * 4.0)
        internal = torch.maximum(
            torch.minimum(internal, self.internal_control_high),
            self.internal_control_low,
        )
        count = min(len(internal), self.parameters.horizon_steps)
        self.control_sequence[:count] = internal[:count]
        if count < self.parameters.horizon_steps:
            self.control_sequence[count:] = internal[count - 1]

    def _internal_reference_controls(self, controls: torch.Tensor) -> torch.Tensor:
        internal = controls.clone()
        if self.is_pushing:
            internal[:, 0] = torch.round(internal[:, 0] * 4.0)
        return torch.maximum(
            torch.minimum(internal, self.internal_control_high),
            self.internal_control_low,
        )

    @torch.no_grad()
    def advance_with_nominal_reference(self, controls: np.ndarray) -> np.ndarray:
        """Execute the reference action and keep the receding sequence aligned."""

        reference = torch.as_tensor(
            np.array(controls, dtype=float, copy=True),
            dtype=self.dtype,
            device=self.device,
        )
        horizon = int(len(reference))
        internal = self._internal_reference_controls(reference)
        self.control_sequence[:horizon] = internal
        command = self.physical_controls(internal[0]).detach().cpu().numpy()
        self.control_sequence[:-1] = self.control_sequence[1:].clone()
        if self.parameters.horizon_steps > 1:
            self.control_sequence[-1] = self.control_sequence[-2].clone()
        return command

    def _reference_state_cost(
        self, state: torch.Tensor, reference: torch.Tensor, *, terminal: bool
    ) -> torch.Tensor:
        residual = state - reference.reshape(1, -1)
        if self.system_name in ("kinematic_car", "pushing_object"):
            residual = residual.clone()
            residual[:, 2] = torch.remainder(
                residual[:, 2] + torch.pi, 2.0 * torch.pi
            ) - torch.pi
        weights = self.q_terminal if terminal else self.q
        cost = (residual.square() * weights).sum(dim=1)
        return cost + self.bounds_cost(state) + self.obstacle_cost(state)

    @torch.no_grad()
    def command_reference(
        self,
        state: np.ndarray,
        state_references: np.ndarray,
        control_references: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Perform one standard MPPI update over a shortened reference horizon."""

        x_ref = torch.as_tensor(
            np.array(state_references, dtype=float, copy=True),
            dtype=self.dtype,
            device=self.device,
        )
        u_ref = torch.as_tensor(
            np.array(control_references, dtype=float, copy=True),
            dtype=self.dtype,
            device=self.device,
        )
        horizon = int(len(u_ref))
        if x_ref.ndim != 2 or u_ref.ndim != 2 or len(x_ref) != horizon:
            raise ValueError("MPPI state/control reference windows must align")
        if not 1 <= horizon <= self.parameters.horizon_steps:
            raise ValueError("invalid active MPPI reference horizon")

        # MPPI's proposal and likelihood correction are expressed in the
        # controller's internal coordinates (including pushing's categorical
        # face index).  Reference tracking therefore has to center both the
        # explicit action cost and the path-integral correction on u_ref.  The
        # original goal-reaching controller centers these terms on zero; using
        # that absolute-control prior here systematically suppresses nonzero
        # nominal controls.
        u_ref_internal = self._internal_reference_controls(u_ref)
        # After the previous command shifts U, initialize the newly exposed
        # horizon tail with the corresponding nominal reference control.
        self.control_sequence[horizon - 1] = u_ref_internal[horizon - 1]

        state_tensor = torch.as_tensor(
            np.array(state, dtype=float, copy=True),
            dtype=self.dtype,
            device=self.device,
        ).reshape(1, -1)
        effective_sample_sizes = []
        maximum_weights = []
        update_norms = []
        minimum = torch.tensor(float("nan"), dtype=self.dtype, device=self.device)
        for _ in range(self.tracking_iterations):
            sampled_internal, perturbation = self.sample_controls()
            sampled_physical = self.physical_controls(sampled_internal)
            rollout_state = state_tensor.expand(
                self.parameters.num_samples, -1
            ).clone()
            costs = torch.zeros(
                self.parameters.num_samples, dtype=self.dtype, device=self.device
            )
            for step in range(horizon):
                action = sampled_physical[:, step]
                rollout_state = self.propagate(rollout_state, action)
                costs += self._reference_state_cost(
                    rollout_state, x_ref[step], terminal=False
                )
                control_residual = sampled_internal[:, step] - u_ref_internal[step]
                costs += self.r_scale * control_residual.square().sum(dim=1)
                proposal_residual = (
                    self.control_sequence[step] - u_ref_internal[step]
                )
                cross = (
                    proposal_residual
                    * self.inverse_noise_variance
                    * perturbation[:, step]
                )
                costs += (
                    self.parameters.control_cost_scale
                    * self.parameters.temperature
                    * cross.sum(dim=1)
                )
            costs += self._reference_state_cost(
                rollout_state, x_ref[horizon - 1], terminal=True
            )
            if not torch.isfinite(costs).all():
                raise FloatingPointError("MPPI produced non-finite tracking costs")

            minimum = costs.amin()
            weights = torch.softmax(
                -(costs - minimum) / self.parameters.temperature, dim=0
            )
            update = torch.einsum(
                "k,ktm->tm", weights, perturbation[:, :horizon]
            )
            self.control_sequence[:horizon] = torch.maximum(
                torch.minimum(
                    self.control_sequence[:horizon] + update,
                    self.internal_control_high,
                ),
                self.internal_control_low,
            )
            effective_sample_sizes.append(
                float(torch.reciprocal(weights.square().sum()).detach().cpu())
            )
            maximum_weights.append(float(weights.amax().detach().cpu()))
            update_norms.append(float(torch.linalg.vector_norm(update).detach().cpu()))
        command = self.physical_controls(self.control_sequence[0]).detach().cpu().numpy()
        self.control_sequence[:-1] = self.control_sequence[1:].clone()
        if self.parameters.horizon_steps > 1:
            self.control_sequence[-1] = self.control_sequence[-2].clone()
        return command, {
            "active_horizon": float(horizon),
            "optimization_iterations": float(self.tracking_iterations),
            "minimum_rollout_cost": float(minimum.detach().cpu()),
            "maximum_weight": float(maximum_weights[-1]),
            "mean_maximum_weight": float(np.mean(maximum_weights)),
            "effective_sample_size": float(effective_sample_sizes[-1]),
            "mean_effective_sample_size": float(np.mean(effective_sample_sizes)),
            "mean_update_norm": float(np.mean(update_norms)),
        }


def run_open_loop(
    reference: ReferenceTrajectory,
    plant,
    config: dict,
) -> MethodResult:
    initial = _reset_plant(plant, reference)
    result = MethodResult("open_loop", initial, states=[initial.copy()])
    try:
        for nominal in reference.controls:
            started = time.perf_counter()
            control = np.asarray(nominal, dtype=float).copy()
            selection_time = time.perf_counter() - started
            propagation_started = time.perf_counter()
            next_state = canonicalize_state(
                reference.system,
                plant.execute_segment(control, reference.control_duration),
            )
            propagation_time = time.perf_counter() - propagation_started
            result.controls.append(control)
            result.states.append(next_state)
            result.timings.append(
                {
                    "computation_time": selection_time,
                    "open_loop_selection_time": selection_time,
                    "plant_propagation_time": propagation_time,
                }
            )
            result.diagnostics.append({})
    except Exception as exc:
        result.success = False
        result.failure_reason = repr(exc)
    return result


def run_aura_tracking(
    reference: ReferenceTrajectory,
    aura_optimizer: AURALocalTrackingAdapter,
    plant,
    config: dict,
    *,
    seed: int,
    force_nominal: bool = False,
) -> MethodResult:
    if force_nominal:
        precompute_times = [0.0] * reference.num_controls
    else:
        # This entire call consumes only the immutable nominal reference.
        aura_optimizer.precompute(reference, seed)
        precompute_times = aura_optimizer.precompute_times
    initial = _reset_plant(plant, reference)
    result = MethodResult("aura", initial, states=[initial.copy()])
    try:
        for index, nominal in enumerate(reference.controls):
            started = time.perf_counter()
            if force_nominal:
                control = np.asarray(nominal, dtype=float).copy()
                diagnostic = {
                    "selection_source": "forced_nominal",
                    "global_replanning_enabled": False,
                }
            else:
                control, diagnostic = aura_optimizer.select(index, result.states[-1])
            selection_time = time.perf_counter() - started
            propagation_started = time.perf_counter()
            next_state = canonicalize_state(
                reference.system,
                plant.execute_segment(control, reference.control_duration),
            )
            propagation_time = time.perf_counter() - propagation_started
            result.controls.append(control)
            result.states.append(next_state)
            result.timings.append(
                {
                    "computation_time": float(precompute_times[index]) + selection_time,
                    "aura_precompute_time": float(precompute_times[index]),
                    "aura_selection_time": selection_time,
                    "plant_propagation_time": propagation_time,
                }
            )
            result.diagnostics.append(diagnostic)
    except Exception as exc:
        result.success = False
        result.failure_reason = repr(exc)
    return result


def run_mppi_tracking(
    reference: ReferenceTrajectory,
    mppi_controller: ReferenceTrackingMPPI,
    plant,
    config: dict,
    *,
    force_nominal: bool = False,
) -> MethodResult:
    initial = _reset_plant(plant, reference)
    result = MethodResult("mppi", initial, states=[initial.copy()])
    if not force_nominal:
        mppi_controller.initialize_nominal(
            reference.controls[: mppi_controller.parameters.horizon_steps]
        )
    try:
        for index, nominal in enumerate(reference.controls):
            started = time.perf_counter()
            if force_nominal:
                control = np.asarray(nominal, dtype=float).copy()
                diagnostic = {"selection_source": "forced_nominal"}
            else:
                horizon = min(
                    mppi_controller.parameters.horizon_steps,
                    reference.num_controls - index,
                )
                current_reference_error = float(
                    arrayDistance(
                        result.states[-1],
                        reference.states[index],
                        system=reference.system,
                    )
                )
                nominal_deadband = float(
                    getattr(mppi_controller, "nominal_deadband", -1.0)
                )
                if nominal_deadband >= 0.0 and current_reference_error <= nominal_deadband:
                    control = mppi_controller.advance_with_nominal_reference(
                        reference.controls[index : index + horizon]
                    )
                    diagnostic = {
                        "selection_source": "nominal_deadband",
                        "active_horizon": float(horizon),
                        "current_reference_error": current_reference_error,
                        "nominal_deadband": nominal_deadband,
                    }
                else:
                    control, diagnostic = mppi_controller.command_reference(
                        result.states[-1],
                        reference.states[index + 1 : index + 1 + horizon],
                        reference.controls[index : index + horizon],
                    )
                    diagnostic["selection_source"] = "mppi"
                    diagnostic["current_reference_error"] = current_reference_error
                    diagnostic["nominal_deadband"] = nominal_deadband
            _synchronize(mppi_controller.device)
            optimization_time = time.perf_counter() - started
            propagation_started = time.perf_counter()
            next_state = canonicalize_state(
                reference.system,
                plant.execute_segment(control, reference.control_duration),
            )
            propagation_time = time.perf_counter() - propagation_started
            result.controls.append(np.asarray(control, dtype=float))
            result.states.append(next_state)
            result.timings.append(
                {
                    "computation_time": optimization_time,
                    "mppi_optimization_time": optimization_time,
                    "plant_propagation_time": propagation_time,
                }
            )
            result.diagnostics.append(diagnostic)
    except Exception as exc:
        result.success = False
        result.failure_reason = repr(exc)
    return result


def make_mppi_controller(
    system,
    reference: ReferenceTrajectory,
    parameters: MPPIParameters,
    config: dict,
    *,
    seed: int,
) -> ReferenceTrackingMPPI:
    return ReferenceTrackingMPPI(
        system=system,
        goal_state=reference.states[-1].copy(),
        propagation_step_size=reference.control_duration,
        parameters=parameters,
        obstacle_config=config.get("obstacles"),
        model_name=config.get("model_name"),
        model_path=config.get("model_path"),
        seed=seed,
        device=config["resolved_device"],
        tracking_cost_weights=config["mppi"].get("cost_weights"),
        tracking_iterations=int(config["mppi"].get("optimization_iterations", 1)),
        nominal_deadband=float(config["mppi"].get("nominal_deadband", -1.0)),
    )

METHOD_LABELS = {"aura": "AURA", "mppi": "MPPI", "open_loop": "Open Loop"}


def _json_vector(value: np.ndarray | list[float] | None) -> str:
    if value is None:
        return "null"
    return json.dumps(np.asarray(value, dtype=float).reshape(-1).tolist())


def method_result_to_step_rows(
    *,
    reference: ReferenceTrajectory,
    result: MethodResult,
    trial: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Emit one row for every requested step, including failure placeholders."""

    rows: list[dict[str, Any]] = []
    cumulative = 0.0
    for index in range(reference.num_controls):
        completed = index < result.completed_steps
        if completed:
            error = float(
                arrayDistance(
                    result.states[index + 1],
                    reference.states[index + 1],
                    system=reference.system,
                )
            )
            cumulative += error
            timing = result.timings[index]
            diagnostics = result.diagnostics[index]
            executed = result.states[index + 1]
            executed_control = result.controls[index]
            failure_reason = ""
        else:
            error = math.nan
            timing = {}
            diagnostics = {}
            executed = None
            executed_control = None
            failure_reason = result.failure_reason or "method_did_not_complete_step"
        rows.append(
            {
                "system": reference.system,
                "environment": reference.environment,
                "trial": int(trial),
                "step": index + 1,
                "method": result.method,
                "seed": int(seed),
                "x_reference": _json_vector(reference.states[index + 1]),
                "x_executed": _json_vector(executed),
                "u_reference": _json_vector(reference.controls[index]),
                "u_executed": _json_vector(executed_control),
                "instantaneous_tracking_error": error,
                "cumulative_tracking_error": cumulative if completed else math.nan,
                "control_duration": float(reference.control_duration),
                "computation_time": float(timing.get("computation_time", math.nan)),
                "aura_precompute_time": float(
                    timing.get("aura_precompute_time", math.nan)
                ),
                "aura_selection_time": float(
                    timing.get("aura_selection_time", math.nan)
                ),
                "mppi_optimization_time": float(
                    timing.get("mppi_optimization_time", math.nan)
                ),
                "open_loop_selection_time": float(
                    timing.get("open_loop_selection_time", math.nan)
                ),
                "plant_propagation_time": float(
                    timing.get("plant_propagation_time", math.nan)
                ),
                "success": bool(completed),
                "failure_reason": failure_reason,
                "controller_diagnostics": json.dumps(diagnostics, sort_keys=True),
            }
        )
    return rows


def method_result_to_trial_row(
    *,
    reference: ReferenceTrajectory,
    result: MethodResult,
    trial: int,
    seed: int,
) -> dict[str, Any]:
    errors = [
        float(
            arrayDistance(
                result.states[index + 1],
                reference.states[index + 1],
                system=reference.system,
            )
        )
        for index in range(result.completed_steps)
    ]
    complete = result.success and result.completed_steps == reference.num_controls
    precompute = [t.get("aura_precompute_time", math.nan) for t in result.timings]
    selection = [t.get("aura_selection_time", math.nan) for t in result.timings]
    online = [
        t.get("mppi_optimization_time", t.get("open_loop_selection_time", math.nan))
        for t in result.timings
    ]
    return {
        "system": reference.system,
        "environment": reference.environment,
        "trial": int(trial),
        "method": result.method,
        "seed": int(seed),
        "success": bool(complete),
        "failure_reason": result.failure_reason,
        "completed_steps": int(result.completed_steps),
        "requested_steps": int(reference.num_controls),
        "mean_tracking_error": float(np.mean(errors)) if complete else math.nan,
        "partial_mean_tracking_error": float(np.mean(errors)) if errors else math.nan,
        "terminal_tracking_error": float(errors[-1]) if errors else math.nan,
        "total_computation_time": float(
            np.nansum([t.get("computation_time", math.nan) for t in result.timings])
        ),
        "aura_precompute_time": float(np.nansum(precompute)),
        "aura_selection_time": float(np.nansum(selection)),
        "online_optimization_time": float(np.nansum(online)),
        "initial_state": _json_vector(result.initial_state),
        "x_ref": json.dumps(reference.states.tolist()),
        "u_ref": json.dumps(reference.controls.tolist()),
        "generation_attempts": int(reference.generation_attempts),
    }


def _float(row: dict, key: str) -> float:
    try:
        return float(row.get(key, math.nan))
    except (TypeError, ValueError):
        return math.nan


def _truth(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _sample_stats(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = int(len(array))
    if n == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "std_dev": math.nan,
            "median": math.nan,
            "standard_error": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
        }
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "std_dev": std,
        "median": float(np.median(array)),
        "standard_error": se,
        "ci_low": mean - 1.96 * se,
        "ci_high": mean + 1.96 * se,
    }


def _bootstrap_mean_ci(
    values: np.ndarray, *, num_resamples: int, seed: int
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(int(seed))
    # Chunking avoids a large temporary allocation for unusually large runs.
    samples: list[np.ndarray] = []
    remaining = int(num_resamples)
    while remaining:
        count = min(remaining, 2_000)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        samples.append(values[indices].mean(axis=1))
        remaining -= count
    boot = np.concatenate(samples)
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high)


def _condition_key(row: dict) -> tuple[str, str]:
    return str(row["system"]), str(row["environment"])


def compute_tracking_statistics(
    step_metrics: list[dict[str, Any]],
    trial_metrics: list[dict[str, Any]],
    *,
    num_controls: int,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 91_337,
) -> dict[str, list[dict[str, Any]]]:
    """Compute all headline values from whole, jointly completed trials."""

    if int(bootstrap_resamples) < 10_000:
        raise ValueError("at least 10,000 bootstrap resamples are required")
    conditions = sorted({_condition_key(row) for row in trial_metrics})
    joint: dict[tuple[str, str], set[int]] = {}
    success_rows: list[dict[str, Any]] = []
    for system, environment in conditions:
        per_method: dict[str, set[int]] = {}
        condition_trials = [
            row for row in trial_metrics if _condition_key(row) == (system, environment)
        ]
        all_trials = {int(row["trial"]) for row in condition_trials}
        for method in METHOD_ORDER:
            successful = {
                int(row["trial"])
                for row in condition_trials
                if row["method"] == method and _truth(row["success"])
            }
            per_method[method] = successful
            success_rows.append(
                {
                    "system": system,
                    "environment": environment,
                    "method": method,
                    "successful_trials": len(successful),
                    "attempted_trials": len(all_trials),
                    "success_rate": (
                        len(successful) / len(all_trials) if all_trials else math.nan
                    ),
                }
            )
        joint[(system, environment)] = set.intersection(
            *(per_method[method] for method in METHOD_ORDER)
        )

    step_statistics: list[dict[str, Any]] = []
    for system, environment in conditions:
        for method in METHOD_ORDER:
            for step in range(1, int(num_controls) + 1):
                matching = [
                    row
                    for row in step_metrics
                    if _condition_key(row) == (system, environment)
                    and row["method"] == method
                    and int(row["step"]) == step
                    and int(row["trial"]) in joint[(system, environment)]
                ]
                instantaneous = _sample_stats(
                    _float(row, "instantaneous_tracking_error") for row in matching
                )
                cumulative = _sample_stats(
                    _float(row, "cumulative_tracking_error") for row in matching
                )
                step_statistics.append(
                    {
                        "system": system,
                        "environment": environment,
                        "method": method,
                        "step": step,
                        **instantaneous,
                        "cumulative_mean": cumulative["mean"],
                        "cumulative_std_dev": cumulative["std_dev"],
                        "cumulative_median": cumulative["median"],
                        "cumulative_standard_error": cumulative["standard_error"],
                        "cumulative_ci_low": cumulative["ci_low"],
                        "cumulative_ci_high": cumulative["ci_high"],
                        "scope": "jointly_completed_trials",
                    }
                )

    success_lookup = {
        (row["system"], row["environment"], row["method"]): row
        for row in success_rows
    }
    summary: list[dict[str, Any]] = []
    trial_means: dict[tuple[str, str, str], dict[int, float]] = {}
    for system, environment in conditions:
        for method_index, method in enumerate(METHOD_ORDER):
            values_by_trial = {
                int(row["trial"]): _float(row, "mean_tracking_error")
                for row in trial_metrics
                if _condition_key(row) == (system, environment)
                and row["method"] == method
                and int(row["trial"]) in joint[(system, environment)]
            }
            trial_means[(system, environment, method)] = values_by_trial
            values = np.asarray(list(values_by_trial.values()), dtype=float)
            stats = _sample_stats(values)
            low, high = _bootstrap_mean_ci(
                values,
                num_resamples=bootstrap_resamples,
                seed=bootstrap_seed + method_index,
            )
            success = success_lookup[(system, environment, method)]
            summary.append(
                {
                    "system": system,
                    "environment": environment,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "mean_tracking_error": stats["mean"],
                    "std_dev": stats["std_dev"],
                    "ci_low": low,
                    "ci_high": high,
                    "num_joint_trials": stats["n"],
                    "successful_trials": success["successful_trials"],
                    "attempted_trials": success["attempted_trials"],
                    "success_rate": success["success_rate"],
                    "bootstrap_resamples": int(bootstrap_resamples),
                    "bootstrap_seed": int(bootstrap_seed),
                    "scope": "jointly_completed_trials",
                }
            )

    paired: list[dict[str, Any]] = []
    comparisons = (
        ("aura_minus_open_loop", "aura", "open_loop"),
        ("aura_minus_mppi", "aura", "mppi"),
        ("mppi_minus_open_loop", "mppi", "open_loop"),
    )
    for system, environment in conditions:
        for comparison, left, right in comparisons:
            for step in range(1, int(num_controls) + 1):
                keyed: dict[str, dict[int, float]] = {}
                for method in (left, right):
                    keyed[method] = {
                        int(row["trial"]): _float(
                            row, "instantaneous_tracking_error"
                        )
                        for row in step_metrics
                        if _condition_key(row) == (system, environment)
                        and row["method"] == method
                        and int(row["step"]) == step
                        and int(row["trial"]) in joint[(system, environment)]
                    }
                ids = sorted(set(keyed[left]) & set(keyed[right]))
                stats = _sample_stats(
                    keyed[left][trial] - keyed[right][trial] for trial in ids
                )
                paired.append(
                    {
                        "system": system,
                        "environment": environment,
                        "comparison": comparison,
                        "step": step,
                        **stats,
                        "scope": "jointly_completed_trials",
                    }
                )
            left_values = trial_means[(system, environment, left)]
            right_values = trial_means[(system, environment, right)]
            ids = sorted(set(left_values) & set(right_values))
            differences = np.asarray(
                [left_values[trial] - right_values[trial] for trial in ids],
                dtype=float,
            )
            stats = _sample_stats(differences)
            low, high = _bootstrap_mean_ci(
                differences,
                num_resamples=bootstrap_resamples,
                seed=bootstrap_seed + 100 + len(paired),
            )
            paired.append(
                {
                    "system": system,
                    "environment": environment,
                    "comparison": comparison,
                    "step": "overall_trial_mean",
                    **stats,
                    "ci_low": low,
                    "ci_high": high,
                    "scope": "jointly_completed_trials",
                }
            )
    return {
        "summary": summary,
        "step_statistics": step_statistics,
        "paired_comparisons": paired,
        "success_rates": success_rows,
    }


def _format_interval(row: dict[str, Any]) -> str:
    return (
        f"{float(row['mean_tracking_error']):.6f} "
        f"[{float(row['ci_low']):.6f}, {float(row['ci_high']):.6f}]"
    )


def write_summary_tables(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(output_dir / "tracking_summary.csv", rows)
    conditions = sorted({_condition_key(row) for row in rows})
    lookup = {
        (row["system"], row["environment"], row["method"]): row for row in rows
    }
    if len(conditions) == 1:
        system, environment = conditions[0]
        markdown = [
            f"# Tracking summary: {system} / {environment}",
            "",
            "Results use jointly completed trials; confidence intervals bootstrap whole trials.",
            "",
            "| Method | Mean Tracking Error | Std. Dev. | 95% CI | Success |",
            "|---|---:|---:|---:|---:|",
        ]
        latex = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Fixed-reference local tracking performance. Confidence intervals bootstrap whole trials.}",
            r"\label{tab:closed_loop_tracking}",
            r"\begin{tabular}{lccc}",
            r"\hline",
            r"Method & Mean Tracking Error & Std. Dev. & 95\% CI \\",
            r"\hline",
        ]
        for method in METHOD_ORDER:
            row = lookup[(system, environment, method)]
            markdown.append(
                f"| {row['method_label']} | {float(row['mean_tracking_error']):.6f} | "
                f"{float(row['std_dev']):.6f} | "
                f"[{float(row['ci_low']):.6f}, {float(row['ci_high']):.6f}] | "
                f"{int(row['successful_trials'])}/{int(row['attempted_trials'])} |"
            )
            latex.append(
                f"{row['method_label']} & {float(row['mean_tracking_error']):.6f} & "
                f"{float(row['std_dev']):.6f} & "
                f"[{float(row['ci_low']):.6f}, {float(row['ci_high']):.6f}] \\\\"
            )
    else:
        markdown = [
            "# Fixed-reference tracking summary",
            "",
            "Values are mean tracking error with whole-trial 95% bootstrap CI.",
            "",
            "| System / Environment | AURA | MPPI | Open Loop |",
            "|---|---:|---:|---:|",
        ]
        latex = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Fixed-reference local tracking performance (mean and whole-trial 95\% bootstrap CI).}",
            r"\label{tab:closed_loop_tracking}",
            r"\begin{tabular}{lccc}",
            r"\hline",
            r"System / Environment & AURA & MPPI & Open Loop \\",
            r"\hline",
        ]
        for system, environment in conditions:
            values = [
                _format_interval(lookup[(system, environment, method)])
                for method in METHOD_ORDER
            ]
            title = f"{system} / {environment}"
            markdown.append(f"| {title} | " + " | ".join(values) + " |")
            latex.append(
                title.replace("_", r"\_")
                + " & "
                + " & ".join(values)
                + r" \\"
            )
    latex.extend([r"\hline", r"\end{tabular}"])
    latex.append(r"\end{table}" if len(conditions) == 1 else r"\end{table*}")
    (output_dir / "tracking_summary.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    (output_dir / "tracking_summary.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )
