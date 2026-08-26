"""Vectorized Model Predictive Path Integral control for AURA systems.

The controller implements the standard path-integral update for continuous
controls and a categorical relaxation for the pushing face selector.  It uses
the same nominal dynamics, control bounds, learned pushing model, and state
conventions as the planners in this repository.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import torch

from propagators import (
    double_integrator,
    dubins_airplane,
    kinematic_car,
    pushing_object,
)
from propagators.propagator import wrap_angle_torch
from simulation.pushing_model import get_pushing_model


SUPPORTED_SYSTEMS = {
    "kinematic_car",
    "double_integrator",
    "pushing_object",
    "dubins_airplane",
}


def validate_system_name(name: str) -> str:
    """Return a supported canonical system name."""

    system_name = str(name).strip().lower()
    if system_name not in SUPPORTED_SYSTEMS:
        raise ValueError(f"MPPI is not implemented for system {name!r}")
    return system_name


@dataclass(frozen=True)
class MPPIParameters:
    """Numerical parameters for one MPPI controller."""

    horizon_steps: int
    num_samples: int
    temperature: float
    control_noise_std: tuple[float, ...]
    action_duration_steps: int = 1
    control_cost_scale: float = 1.0
    goal_cost_mode: str = "direct_goal_region"
    goal_cost_scale: float = 1.0
    smoothness_cost_scale: float = 0.0
    zero_anchor_pushing_model: bool = False
    short_push_cost_scale: float = 0.0
    minimum_preferred_push_distance: float = 0.0

    def validate(self, control_dim: int) -> None:
        if int(self.horizon_steps) < 1:
            raise ValueError("MPPI horizon_steps must be positive")
        if int(self.num_samples) < 2:
            raise ValueError("MPPI num_samples must be at least two")
        if not math.isfinite(float(self.temperature)) or self.temperature <= 0.0:
            raise ValueError("MPPI temperature must be finite and positive")
        if len(self.control_noise_std) != int(control_dim):
            raise ValueError(
                "MPPI control_noise_std dimension does not match the control space"
            )
        if any(
            not math.isfinite(float(v)) or float(v) <= 0.0
            for v in self.control_noise_std
        ):
            raise ValueError(
                "all MPPI control noise standard deviations must be positive"
            )
        if int(self.action_duration_steps) != 1:
            raise ValueError(
                "Figure 7 MPPI uses one propagation tick per receding-horizon action"
            )
        if self.goal_cost_mode not in {
            "direct_goal_region",
            "bearing_funnel",
            "shared_state_metric",
        }:
            raise ValueError(
                "MPPI goal_cost_mode must be 'direct_goal_region', "
                "'bearing_funnel', or 'shared_state_metric'"
            )
        if (
            not math.isfinite(float(self.goal_cost_scale))
            or float(self.goal_cost_scale) <= 0.0
        ):
            raise ValueError("MPPI goal_cost_scale must be finite and positive")
        if (
            not math.isfinite(float(self.smoothness_cost_scale))
            or float(self.smoothness_cost_scale) < 0.0
        ):
            raise ValueError(
                "MPPI smoothness_cost_scale must be finite and nonnegative"
            )
        if not isinstance(self.zero_anchor_pushing_model, bool):
            raise ValueError("MPPI zero_anchor_pushing_model must be boolean")
        if (
            not math.isfinite(float(self.short_push_cost_scale))
            or float(self.short_push_cost_scale) < 0.0
        ):
            raise ValueError("MPPI short_push_cost_scale must be finite and nonnegative")
        if (
            not math.isfinite(float(self.minimum_preferred_push_distance))
            or float(self.minimum_preferred_push_distance) < 0.0
        ):
            raise ValueError(
                "MPPI minimum_preferred_push_distance must be finite and nonnegative"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_parameters(system_name: str) -> MPPIParameters:
    """Return the controller sizes used by the paper's legacy baseline.

    Noise scales are expressed in the current repository's control encoding.
    In particular, the pushing face coordinate is an internal categorical
    index in ``[0, 3]`` and is converted to ``{0, .25, .5, .75}`` only when a
    control is propagated.
    """

    key = validate_system_name(system_name)
    if key == "kinematic_car":
        return MPPIParameters(
            # A 15-second horizon covers a useful nonholonomic car maneuver;
            # the same controller budget and objective are used in Gaussian
            # and MuJoCo environments.
            horizon_steps=15,
            num_samples=512,
            temperature=2.0,
            control_noise_std=(0.23, 0.12),
            goal_cost_mode="shared_state_metric",
        )
    if key == "double_integrator":
        return MPPIParameters(
            horizon_steps=5,
            num_samples=512,
            temperature=2.0,
            control_noise_std=(0.18, 0.18, 0.18),
        )
    if key == "pushing_object":
        return MPPIParameters(
            horizon_steps=2,
            num_samples=2048,
            temperature=0.5,
            control_noise_std=(1.0, 0.20, 0.05),
        )
    if key == "dubins_airplane":
        return MPPIParameters(
            horizon_steps=5,
            num_samples=512,
            temperature=2.0,
            control_noise_std=(0.15, 0.4, 0.4),
        )
    raise ValueError(f"MPPI is not implemented for system {system_name!r}")


def parameters_from_config(system_name: str, config: dict) -> MPPIParameters:
    """Apply an optional ``mppi`` mapping on top of the audited defaults."""

    defaults = default_parameters(system_name)
    override = dict(config.get("mppi") or {})
    values = asdict(defaults)
    if (
        validate_system_name(system_name) == "dubins_airplane"
        and str(config.get("panel_id", "")).lower() == "dubins_airplane_gaussian"
    ):
        values["horizon_steps"] = 5
        values["num_samples"] = 512
        values["temperature"] = 1.4
        values["control_noise_std"] = (0.15, 0.10, 0.10)
    for key in values:
        if key in override:
            values[key] = override[key]
    values["control_noise_std"] = tuple(float(v) for v in values["control_noise_std"])
    return MPPIParameters(**values)


class MPPIController:
    """Batched MPPI controller sharing AURA's nominal system models."""

    def __init__(
        self,
        system,
        goal_state,
        *,
        propagation_step_size: float,
        goal_threshold: float = 0.0,
        parameters: MPPIParameters | None = None,
        obstacle_config: dict | None = None,
        model_name: str | None = None,
        model_path: str | None = None,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ):
        self.system = system
        self.system_name = validate_system_name(system.name)
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.propagation_step_size = float(propagation_step_size)
        if (
            not math.isfinite(self.propagation_step_size)
            or self.propagation_step_size <= 0.0
        ):
            raise ValueError("propagation_step_size must be finite and positive")
        self.parameters = parameters or default_parameters(self.system_name)
        self.parameters.validate(len(system.control_bounds))
        self.goal_threshold = float(goal_threshold)
        if not math.isfinite(self.goal_threshold) or self.goal_threshold < 0.0:
            raise ValueError("MPPI goal_threshold must be finite and nonnegative")
        self.goal_state = torch.as_tensor(
            np.asarray(goal_state, dtype=float), dtype=self.dtype, device=self.device
        ).reshape(-1)
        self.state_dim = int(self.goal_state.numel())
        self.control_dim = len(system.control_bounds)
        self.is_pushing = self.system_name == "pushing_object"
        self.obstacle_config = dict(obstacle_config or {})
        self.configure_controls(seed)
        self.configure_state_bounds()
        self.configure_pushing_model(model_name, model_path)

    def configure_controls(self, seed: int) -> None:
        """Build physical/internal control bounds and the nominal sequence."""

        physical_bounds = np.asarray(self.system.control_bounds, dtype=float)
        if physical_bounds.shape != (self.control_dim, 2):
            raise ValueError("malformed system control bounds")
        self.physical_low = torch.as_tensor(
            physical_bounds[:, 0], dtype=self.dtype, device=self.device
        )
        self.physical_high = torch.as_tensor(
            physical_bounds[:, 1], dtype=self.dtype, device=self.device
        )
        self.maximum_push_distance = (
            float(physical_bounds[2, 1]) if self.is_pushing else 0.0
        )
        if (
            self.is_pushing
            and self.parameters.minimum_preferred_push_distance
            > self.maximum_push_distance
        ):
            raise ValueError(
                "MPPI minimum_preferred_push_distance exceeds the pushing distance bound"
            )

        internal_bounds = physical_bounds.copy()
        if self.is_pushing:
            internal_bounds[0] = (0.0, 3.0)
        self.internal_control_low = torch.as_tensor(
            internal_bounds[:, 0], dtype=self.dtype, device=self.device
        )
        self.internal_control_high = torch.as_tensor(
            internal_bounds[:, 1], dtype=self.dtype, device=self.device
        )
        self.control_noise_std = torch.as_tensor(
            self.parameters.control_noise_std, dtype=self.dtype, device=self.device
        )
        self.inverse_noise_variance = torch.reciprocal(self.control_noise_std.square())
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))

        self.control_sequence = torch.zeros(
            (self.parameters.horizon_steps, self.control_dim),
            dtype=self.dtype,
            device=self.device,
        )
        if self.is_pushing:
            self.control_sequence[:, 0] = 1.5
            self.control_sequence[:, 1] = 0.0
            self.control_sequence[:, 2] = 0.5 * (
                self.internal_control_low[2] + self.internal_control_high[2]
            )

    def configure_state_bounds(self) -> None:
        """Create tensors used to penalize rollout bound violations."""

        state_bounds = np.asarray(self.system.state_bounds, dtype=float)
        self.state_bound_dim = min(len(state_bounds), self.state_dim)
        self.state_low = torch.as_tensor(
            state_bounds[: self.state_bound_dim, 0],
            dtype=self.dtype,
            device=self.device,
        )
        self.state_high = torch.as_tensor(
            state_bounds[: self.state_bound_dim, 1],
            dtype=self.dtype,
            device=self.device,
        )

    def configure_pushing_model(
        self,
        model_name: str | None,
        model_path: str | None,
    ) -> None:
        """Load the learned model only for the pushing system."""

        self.pushing_model = None
        if self.is_pushing:
            self.pushing_model = get_pushing_model(
                self.system.object_shape,
                model_name=str(
                    model_name
                    or getattr(self.system, "model_name", "cracker_box_flipped")
                ),
                model_path=model_path or getattr(self.system, "model_path", None),
            ).to(self.device)
            self.pushing_model.eval()

    def reset(self) -> None:
        self.control_sequence.zero_()
        if self.is_pushing:
            self.control_sequence[:, 0] = 1.5
            self.control_sequence[:, 2] = 0.5 * (
                self.internal_control_low[2] + self.internal_control_high[2]
            )

    def physical_controls(self, internal: torch.Tensor) -> torch.Tensor:
        controls = internal.clone()
        if self.is_pushing:
            face_index = torch.clamp(torch.round(controls[..., 0]), 0.0, 3.0)
            controls[..., 0] = 0.25 * face_index
        return torch.maximum(
            torch.minimum(controls, self.physical_high), self.physical_low
        )

    def propagate(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        duration = self.propagation_step_size * int(
            self.parameters.action_duration_steps
        )
        if self.system_name == "kinematic_car":
            return kinematic_car.propagate_torch(state, control, duration)
        if self.system_name == "double_integrator":
            return double_integrator.propagate_torch(state, control, duration)
        if self.system_name == "pushing_object":
            relative = self.pushing_model(control)[:, :3]
            if self.parameters.zero_anchor_pushing_model:
                zero_distance_control = control.clone()
                zero_distance_control[:, 2] = 0.0
                zero_distance_relative = self.pushing_model(zero_distance_control)[:, :3]
                relative = torch.stack(
                    [
                        relative[:, 0] - zero_distance_relative[:, 0],
                        relative[:, 1] - zero_distance_relative[:, 1],
                        wrap_angle_torch(
                            relative[:, 2] - zero_distance_relative[:, 2]
                        ),
                    ],
                    dim=1,
                )
            return pushing_object.compose_torch(state, relative)
        if self.system_name == "dubins_airplane":
            return dubins_airplane.propagate_torch(state, control, duration)
        raise AssertionError(self.system_name)

    def bounds_cost(self, state: torch.Tensor) -> torch.Tensor:
        bounded = state[:, : self.state_bound_dim]
        low_violation = torch.relu(self.state_low - bounded)
        high_violation = torch.relu(bounded - self.state_high)
        return 250.0 * (low_violation.square() + high_violation.square()).sum(dim=1)

    def obstacle_cost(self, state: torch.Tensor) -> torch.Tensor:
        config = self.obstacle_config
        if not config.get("enabled", False) or state.shape[1] < 2:
            return torch.zeros(state.shape[0], dtype=self.dtype, device=self.device)
        p = state[:, :2]
        distances: list[torch.Tensor] = []
        safety = float(config.get("safety_radius", 0.10))
        for cx, cy, radius in config.get("circles", []):
            center = torch.tensor([cx, cy], dtype=self.dtype, device=self.device)
            distances.append(
                torch.linalg.vector_norm(p - center, dim=1) - float(radius) - safety
            )
        for xmin, ymin, xmax, ymax in config.get("aabbs", []):
            center = torch.tensor(
                [0.5 * (xmin + xmax), 0.5 * (ymin + ymax)],
                dtype=self.dtype,
                device=self.device,
            )
            half = torch.tensor(
                [0.5 * abs(xmax - xmin), 0.5 * abs(ymax - ymin)],
                dtype=self.dtype,
                device=self.device,
            )
            delta = torch.abs(p - center) - half
            outside = torch.linalg.vector_norm(torch.relu(delta), dim=1)
            inside = torch.minimum(
                torch.maximum(delta[:, 0], delta[:, 1]), torch.zeros_like(outside)
            )
            distances.append(outside + inside - safety)
        if not distances:
            return torch.zeros(state.shape[0], dtype=self.dtype, device=self.device)
        distance = torch.stack(distances, dim=1).amin(dim=1)
        return 60.0 * torch.nn.functional.softplus(-distance / 0.08) * 0.08

    def state_cost(self, state: torch.Tensor, *, terminal: bool) -> torch.Tensor:
        if self.system_name == "double_integrator":
            cost = self.double_integrator_cost(state, terminal)
        elif self.system_name == "dubins_airplane":
            cost = self.dubins_airplane_cost(state, terminal)
        else:
            cost = self.planar_cost(state, terminal)
        return cost + self.bounds_cost(state) + self.obstacle_cost(state)

    def preferred_push_distance(self, state: torch.Tensor) -> torch.Tensor:
        """Return an adaptive soft distance target for pushing controls."""

        if not self.is_pushing or self.parameters.short_push_cost_scale <= 0.0:
            return torch.zeros(state.shape[0], dtype=self.dtype, device=self.device)
        position_error = torch.linalg.vector_norm(
            state[:, :2] - self.goal_state[:2], dim=1
        )
        yaw_error = torch.abs(wrap_angle_torch(state[:, 2] - self.goal_state[2]))
        # OMPL's SE(2) metric uses translation plus half the angular distance.
        goal_error = position_error + 0.5 * yaw_error
        remaining = torch.relu(goal_error - self.goal_threshold)
        preferred = remaining / float(self.parameters.horizon_steps)
        preferred = torch.clamp(
            preferred,
            min=float(self.parameters.minimum_preferred_push_distance),
            max=self.maximum_push_distance,
        )
        return torch.where(remaining > 0.0, preferred, torch.zeros_like(preferred))

    def short_push_cost(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
    ) -> torch.Tensor:
        """Softly discourage contact-only pushes while outside the goal region."""

        preferred = self.preferred_push_distance(state)
        shortfall = torch.relu(preferred - control[:, 2])
        return float(self.parameters.short_push_cost_scale) * shortfall.square()

    def double_integrator_cost(
        self,
        state: torch.Tensor,
        terminal: bool,
    ) -> torch.Tensor:
        position_error = state[:, :3] - self.goal_state[:3]
        velocity_error = state[:, 3:6] - self.goal_state[3:6]
        if terminal:
            return 40.0 * (
                position_error.square().sum(dim=1) + velocity_error.square().sum(dim=1)
            )
        return 6.0 * position_error.square().sum(
            dim=1
        ) + 3.0 * velocity_error.square().sum(dim=1)

    def dubins_airplane_cost(
        self,
        state: torch.Tensor,
        terminal: bool,
    ) -> torch.Tensor:
        position_error = state[:, :3] - self.goal_state[:3]
        yaw_error = wrap_angle_torch(state[:, 3] - self.goal_state[3])
        pitch_error = state[:, 4] - self.goal_state[4]
        speed_error = state[:, 5] - self.goal_state[5]
        if terminal:
            return (
                60.0 * position_error.square().sum(dim=1)
                + 10.0 * yaw_error.square()
                + 10.0 * pitch_error.square()
                + 20.0 * speed_error.square()
            )
        return (
            15.0 * position_error.square().sum(dim=1)
            + 3.0 * yaw_error.square()
            + 3.0 * pitch_error.square()
            + 5.0 * speed_error.square()
        )

    def planar_cost(self, state: torch.Tensor, terminal: bool) -> torch.Tensor:
        delta = self.goal_state[:2] - state[:, :2]
        distance = torch.linalg.vector_norm(delta, dim=1)
        yaw_error = wrap_angle_torch(state[:, 2] - self.goal_state[2])
        if (
            not self.is_pushing
            and self.parameters.goal_cost_mode == "shared_state_metric"
        ):
            weight = 30.0 if terminal else 4.0
            return (
                self.parameters.goal_cost_scale
                * weight
                * (distance.square() + yaw_error.square())
            )
        if terminal:
            position_weight = 100.0 if self.is_pushing else 30.0
            yaw_weight = 50.0 if self.is_pushing else 4.0
            return position_weight * distance.square() + yaw_weight * yaw_error.square()
        if self.parameters.goal_cost_mode == "bearing_funnel":
            bearing = torch.atan2(delta[:, 1], delta[:, 0])
            funnel = 0.5 if self.is_pushing else 0.8
            alpha = torch.clamp(distance / funnel, 0.0, 1.0)
            desired = wrap_angle_torch(
                bearing + (1.0 - alpha) * wrap_angle_torch(self.goal_state[2] - bearing)
            )
            yaw_error = wrap_angle_torch(state[:, 2] - desired)
        position_weight = 50.0 if self.is_pushing else 4.0
        yaw_weight = 5.0 if self.is_pushing else 0.3
        return position_weight * distance.square() + yaw_weight * yaw_error.square()

    def sample_controls(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample rollout controls and return their MPPI update perturbations.

        For continuous controls the path-integral likelihood correction and
        update use the Gaussian perturbation that was actually drawn.  The
        rollout control itself is projected into the admissible control box.
        Using ``projected_control - U`` as the Gaussian perturbation changes
        the proposal distribution at a bound and is not the standard MPPI
        update.  Pushing's face selector is categorical rather than
        continuous, so that coordinate necessarily uses its snapped delta.
        """
        noise = (
            torch.randn(
                (
                    self.parameters.num_samples,
                    self.parameters.horizon_steps,
                    self.control_dim,
                ),
                dtype=self.dtype,
                device=self.device,
                generator=self.generator,
            )
            * self.control_noise_std
        )
        sampled = self.control_sequence.unsqueeze(0) + noise
        sampled = torch.maximum(
            torch.minimum(sampled, self.internal_control_high),
            self.internal_control_low,
        )
        if self.is_pushing:
            sampled[..., 0] = torch.round(sampled[..., 0])
        update_perturbation = noise
        if self.is_pushing:
            update_perturbation = update_perturbation.clone()
            update_perturbation[..., 0] = (
                sampled[..., 0] - self.control_sequence.unsqueeze(0)[..., 0]
            )
        return sampled, update_perturbation

    def rollout_costs(
        self,
        initial_state: torch.Tensor,
        controls: torch.Tensor,
        perturbations: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate the sampled horizons and accumulate their MPPI costs."""

        state = initial_state.expand(self.parameters.num_samples, -1).clone()
        costs = torch.zeros(
            self.parameters.num_samples,
            dtype=self.dtype,
            device=self.device,
        )
        previous_control = None
        for step in range(self.parameters.horizon_steps):
            control = controls[:, step]
            if self.is_pushing:
                costs += self.short_push_cost(state, control)
            state = self.propagate(state, control)
            costs += self.state_cost(state, terminal=False)
            if (
                previous_control is not None
                and self.parameters.smoothness_cost_scale > 0.0
            ):
                change = (control - previous_control).square().sum(dim=1)
                costs += self.parameters.smoothness_cost_scale * change
            previous_control = control
            correction = (
                self.control_sequence[step]
                * self.inverse_noise_variance
                * perturbations[:, step]
            )
            costs += (
                self.parameters.control_cost_scale
                * self.parameters.temperature
                * correction.sum(dim=1)
            )
        return costs + self.state_cost(state, terminal=True)

    def update_control_sequence(
        self,
        costs: torch.Tensor,
        perturbations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the path-integral weighted control update."""

        minimum = costs.amin()
        weights = torch.softmax(
            -(costs - minimum) / self.parameters.temperature,
            dim=0,
        )
        update = torch.einsum("k,ktm->tm", weights, perturbations)
        self.control_sequence = torch.maximum(
            torch.minimum(
                self.control_sequence + update,
                self.internal_control_high,
            ),
            self.internal_control_low,
        )
        return minimum, weights

    def advance_control_sequence(self) -> np.ndarray:
        """Return the first physical action and shift the receding horizon."""

        command = (
            self.physical_controls(self.control_sequence[0].clone())
            .detach()
            .cpu()
            .numpy()
        )
        self.control_sequence[:-1] = self.control_sequence[1:].clone()
        if self.parameters.horizon_steps > 1:
            self.control_sequence[-1] = self.control_sequence[-2].clone()
        return command

    @torch.no_grad()
    def command(self, state) -> tuple[np.ndarray, dict[str, float]]:
        state_tensor = torch.as_tensor(
            np.asarray(state, dtype=float), dtype=self.dtype, device=self.device
        ).reshape(1, -1)
        if state_tensor.shape[1] != self.state_dim:
            raise ValueError(
                f"MPPI expected a {self.state_dim}D state, got {state_tensor.shape[1]}"
            )
        preferred_push_distance = self.preferred_push_distance(state_tensor)[0]
        sampled_internal, perturbation = self.sample_controls()
        sampled_physical = self.physical_controls(sampled_internal)
        costs = self.rollout_costs(state_tensor, sampled_physical, perturbation)
        if not torch.isfinite(costs).all():
            raise FloatingPointError("MPPI produced non-finite rollout costs")
        minimum, weights = self.update_control_sequence(costs, perturbation)
        command = self.advance_control_sequence()
        return command, {
            "minimum_rollout_cost": float(minimum.detach().cpu()),
            "maximum_weight": float(weights.amax().detach().cpu()),
            "effective_sample_size": float(
                torch.reciprocal(weights.square().sum()).detach().cpu()
            ),
            "preferred_push_distance": float(
                preferred_push_distance.detach().cpu()
            ),
        }

    @torch.no_grad()
    def predict_next(self, state, control) -> np.ndarray:
        state_tensor = torch.as_tensor(
            np.asarray(state, dtype=float), dtype=self.dtype, device=self.device
        ).reshape(1, -1)
        control_tensor = torch.as_tensor(
            np.asarray(control, dtype=float), dtype=self.dtype, device=self.device
        ).reshape(1, -1)
        result = self.propagate(state_tensor, control_tensor)
        return result[0].detach().cpu().numpy().astype(float)
