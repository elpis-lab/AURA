"""Particle-based RandUp-RRT for OMPL control problems.

This module implements the practical sampled reachable-set approximation from
Robust-RRT.  It is intentionally a finite-particle baseline: safety is checked
for the sampled particles (plus optional geometric padding), not proved for the
continuous disturbance support.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
import time
from typing import Any

import numpy as np
from ompl import base as ob

from propagators.propagator import wrap_angle_numpy
from utils.utils import (
    are_state_arrays_valid,
    is_state_array_valid,
    normalize_obstacle_config,
)


NO_UNCERTAINTY_MODES = {"none", "deterministic", "zero"}
GAUSSIAN_UNCERTAINTY_MODES = {
    "gaussian_process",
    "matched_gaussian",
    "aura_optimizer_gaussian",
    "gaussian_process_oracle",
    "oracle_gaussian_process",
}


@dataclass(frozen=True)
class RandUpRRTConfig:
    """Numerical and uncertainty parameters for one RandUP-RRT instance."""

    num_particles: int = 50
    padding_epsilon: float = 0.0
    max_iterations: int = 2_147_483_647
    planning_time: float = 10.0
    random_seed: int = 0
    control_duration_min: int = 1
    control_duration_max: int = 5
    optimize_after_first_solution: bool = False
    goal_requires_all_particles: bool = True
    uncertainty_mode: str = "gaussian_process"
    position_std: float = 0.0
    rotation_std: float = 0.0
    velocity_std: float = 0.0
    initial_position_std: float = 0.0
    initial_rotation_std: float = 0.0
    goal_bias: float = 0.05

    def validate(self) -> None:
        if int(self.num_particles) < 1:
            raise ValueError("randup_num_particles must be at least 1")
        if (
            not math.isfinite(float(self.padding_epsilon))
            or float(self.padding_epsilon) < 0.0
        ):
            raise ValueError("randup_padding_epsilon must be finite and nonnegative")
        if int(self.max_iterations) < 1:
            raise ValueError("randup_max_iterations must be at least 1")
        if (
            not math.isfinite(float(self.planning_time))
            or float(self.planning_time) <= 0.0
        ):
            raise ValueError("randup_planning_time must be finite and positive")
        if int(self.control_duration_min) < 1:
            raise ValueError("randup_control_duration_min must be at least 1")
        if int(self.control_duration_max) < int(self.control_duration_min):
            raise ValueError(
                "randup_control_duration_max must be >= randup_control_duration_min"
            )
        if not 0.0 <= float(self.goal_bias) <= 1.0:
            raise ValueError("goal_bias must lie in [0, 1]")
        for name in (
            "position_std",
            "rotation_std",
            "velocity_std",
            "initial_position_std",
            "initial_rotation_std",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        mode = str(self.uncertainty_mode).strip().lower()
        if mode not in NO_UNCERTAINTY_MODES | GAUSSIAN_UNCERTAINTY_MODES:
            raise ValueError(
                "randup_uncertainty_mode must be one of none, deterministic, "
                "zero, gaussian_process, matched_gaussian, "
                "aura_optimizer_gaussian, "
                "gaussian_process_oracle, or oracle_gaussian_process"
            )

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "RandUpRRTConfig":
        """Build from the public ``randup_*`` experiment configuration keys."""

        return cls(
            num_particles=int(values.get("randup_num_particles", 50)),
            padding_epsilon=float(values.get("randup_padding_epsilon", 0.0)),
            max_iterations=int(values.get("randup_max_iterations", 2_147_483_647)),
            planning_time=float(
                values.get(
                    "randup_planning_time",
                    values.get("planning_time", 10.0),
                )
            ),
            random_seed=int(values.get("randup_random_seed", values.get("seed", 0))),
            control_duration_min=int(
                values.get(
                    "randup_control_duration_min",
                    values.get("min_control_duration", 1),
                )
            ),
            control_duration_max=int(
                values.get(
                    "randup_control_duration_max",
                    values.get("max_control_duration", 5),
                )
            ),
            optimize_after_first_solution=bool(
                values.get("randup_optimize_after_first_solution", False)
            ),
            goal_requires_all_particles=bool(
                values.get("randup_goal_requires_all_particles", True)
            ),
            uncertainty_mode=str(
                values.get("randup_uncertainty_mode", "gaussian_process")
            ),
            position_std=float(
                values.get(
                    "randup_position_std",
                    values.get("sampling_position_std", 0.0),
                )
            ),
            rotation_std=float(
                values.get(
                    "randup_rotation_std",
                    values.get("sampling_rotation_std", 0.0),
                )
            ),
            velocity_std=float(
                values.get(
                    "randup_velocity_std",
                    values.get(
                        "sampling_velocity_std",
                        values.get("sampling_position_std", 0.0),
                    ),
                )
            ),
            initial_position_std=float(values.get("randup_initial_position_std", 0.0)),
            initial_rotation_std=float(values.get("randup_initial_rotation_std", 0.0)),
            goal_bias=float(values.get("goal_bias", 0.05)),
        )

    @property
    def is_oracle(self) -> bool:
        return str(self.uncertainty_mode).strip().lower() in {
            "gaussian_process_oracle",
            "oracle_gaussian_process",
        }


@dataclass
class RandUpNode:
    """One nominal exploration state and its sampled reachable-set endpoint."""

    nominal_state: Any
    nominal_values: np.ndarray
    particles: np.ndarray
    parent: int | None
    control: Any | None
    control_values: np.ndarray | None
    duration_steps: int
    duration_seconds: float
    persistent_parameters: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=float)
    )


@dataclass(frozen=True)
class RandUpControlPath:
    """Ownership-safe control path returned to AURA's execution pipeline."""

    states: tuple[tuple[float, ...], ...]
    controls: tuple[tuple[float, ...], ...]
    duration_steps: tuple[int, ...]
    duration_seconds: tuple[float, ...]

    def getStateCount(self) -> int:
        return len(self.states)

    def getControlCount(self) -> int:
        return len(self.controls)


class RandUpRRT:
    """OMPL control-planning wrapper using sampled uncertain reachable sets.

    OMPL's Python binding does not expose its nearest-neighbor container
    templates.  Nodes therefore use a linear nearest query over nominal OMPL
    states while retaining OMPL's state-space distance and state/control
    samplers. The local nanobind build also cannot safely retain Python
    subclasses of ``ob.Planner`` that allocate states inside ``solve``; this
    class therefore wraps the OMPL planner interface and returns the same
    control-sequence dictionary consumed by the rest of AURA. Constructing a
    ``PathControl`` from Python is also unsafe in this build because nanobind
    and ``PathControl`` both claim ownership of appended raw pointers.
    """

    def __init__(
        self,
        space_information,
        *,
        system,
        config: RandUpRRTConfig,
        start_state: np.ndarray,
        goal_state: np.ndarray,
        goal_threshold: float,
        obstacle_config: dict | None = None,
    ):
        config.validate()
        self._space_information = space_information
        self._problem_definition = None
        self._name = "RandUP-RRT"
        self.system = system
        self.config = config
        self.obstacle_config = normalize_obstacle_config(
            deepcopy(obstacle_config) if obstacle_config is not None else None
        )
        self._validity_config = {
            "state_bounds": [list(bounds) for bounds in system.state_bounds],
            "obstacles": self.obstacle_config,
        }
        self._base_safety_radius = float(
            (self.obstacle_config or {}).get("safety_radius", 0.0)
        )
        self._rng = np.random.default_rng(int(config.random_seed))
        self._start_state_values = np.asarray(start_state, dtype=float).reshape(-1)
        self._goal_state_values = np.asarray(goal_state, dtype=float).reshape(-1)
        self._goal_threshold = float(goal_threshold)
        if not math.isfinite(self._goal_threshold) or self._goal_threshold < 0.0:
            raise ValueError("goal_threshold must be finite and nonnegative")
        self._state_sampler = None
        self._control_sampler = None
        self.nodes: list[RandUpNode] = []
        self.goal_node_index: int | None = None
        self.last_solution_path = None
        self.last_solution_particle_sets: list[np.ndarray] = []
        self.last_solution_persistent_parameters: list[np.ndarray] = []
        self.last_expansion_rejection = ""
        self.failure_reason = ""
        self.stats = self.new_stats()

    def getName(self) -> str:
        return self._name

    def getSpaceInformation(self):
        return self._space_information

    def setProblemDefinition(self, problem_definition) -> None:
        self._problem_definition = problem_definition

    def getProblemDefinition(self):
        if self._problem_definition is None:
            raise RuntimeError("RandUP-RRT requires an OMPL ProblemDefinition")
        return self._problem_definition

    def checkValidity(self) -> None:
        pdef = self.getProblemDefinition()
        if int(pdef.getStartStateCount()) < 1:
            raise ValueError("RandUP-RRT requires at least one start state")

    def new_stats(self) -> dict[str, Any]:
        return {
            "planning_success": False,
            "failure_reason": "",
            "planning_time": 0.0,
            "iterations": 0,
            "tree_expansions": 0,
            "accepted_edges": 0,
            "tree_size": 0,
            "dynamics_propagations": 0,
            "particle_propagations": 0,
            "particle_collision_rejections": 0,
            "nominal_collision_rejections": 0,
            "numerical_propagation_failures": 0,
            "last_expansion_rejection": "",
            "num_particles": int(self.config.num_particles),
            "padding_epsilon": float(self.config.padding_epsilon),
            "uncertainty_mode": str(self.config.uncertainty_mode),
            "goal_requires_all_particles": bool(
                self.config.goal_requires_all_particles
            ),
            "optimize_after_first_solution": bool(
                self.config.optimize_after_first_solution
            ),
            "solution_candidates": 0,
            "solution_improvements": 0,
            "first_solution_time": None,
            "best_solution_duration_steps": None,
            "best_solution_duration_seconds": None,
            "best_nominal_goal_distance": float("inf"),
            "best_worst_particle_goal_distance": float("inf"),
        }

    def ensure_samplers(self) -> None:
        si = self.getSpaceInformation()
        if self._state_sampler is None:
            self._state_sampler = si.allocStateSampler()
        if self._control_sampler is None:
            self._control_sampler = si.allocControlSampler()

    def clear(self):
        # The local nanobind wrappers returned by alloc/clone own their C++
        # objects. Calling freeState/freeControl manually double-releases them.
        self.nodes.clear()
        self.goal_node_index = None
        self.last_solution_path = None
        self.last_solution_particle_sets = []
        self.last_solution_persistent_parameters = []
        self.last_expansion_rejection = ""
        self.failure_reason = ""
        self.stats = self.new_stats()
        self._rng = np.random.default_rng(int(self.config.random_seed))
        self._state_sampler = None
        self._control_sampler = None

    def state_to_numpy(self, state) -> np.ndarray:
        return self.system.ompl_state_to_numpy(state)

    def set_state(self, state, values: np.ndarray) -> None:
        self.system.set_ompl_state(state, values)

    def control_to_numpy(self, control) -> np.ndarray:
        dimension = int(self.getSpaceInformation().getControlSpace().getDimension())
        return np.array([control[index] for index in range(dimension)], dtype=float)

    def sample_control(self, sampled_control) -> tuple[np.ndarray, int]:
        """Draw one unbiased OMPL control and one uniform control duration."""

        self._control_sampler.sample(sampled_control)
        control = self.control_to_numpy(sampled_control).copy()
        steps = int(
            self._rng.integers(
                int(self.config.control_duration_min),
                int(self.config.control_duration_max) + 1,
            )
        )
        return control, steps

    def state_valid(self, state: np.ndarray) -> bool:
        return bool(
            is_state_array_valid(
                state,
                system=self.system.name,
                config=self._validity_config,
                obstacle_config=self.obstacle_config,
                safety_radius_override=(
                    self._base_safety_radius + float(self.config.padding_epsilon)
                ),
            )
        )

    def sample_initial_particles(self, start: np.ndarray) -> np.ndarray:
        particles = np.repeat(
            np.asarray(start, dtype=float).reshape(1, -1),
            int(self.config.num_particles),
            axis=0,
        )
        if float(self.config.initial_position_std) > 0.0:
            position_dimension = min(2, particles.shape[1])
            particles[:, :position_dimension] += self._rng.normal(
                0.0,
                float(self.config.initial_position_std),
                size=(len(particles), position_dimension),
            )
        if (
            self.system.name in ("kinematic_car", "pushing_object")
            and float(self.config.initial_rotation_std) > 0.0
        ):
            particles[:, 2] = wrap_angle_numpy(
                particles[:, 2]
                + self._rng.normal(
                    0.0,
                    float(self.config.initial_rotation_std),
                    size=len(particles),
                )
            )
        return particles

    def initialize_root(self) -> bool:
        pdef = self.getProblemDefinition()
        if int(pdef.getStartStateCount()) < 1:
            self.failure_reason = "invalid_root_particle_set:no_start_state"
            return False
        start_array = self._start_state_values.copy()
        particles = self.sample_initial_particles(start_array)
        if not np.all(np.isfinite(particles)) or any(
            not self.state_valid(particle) for particle in particles
        ):
            self.failure_reason = "invalid_root_particle_set"
            return False
        persistent = np.empty((int(self.config.num_particles), 0), dtype=float)
        self.nodes.append(
            RandUpNode(
                nominal_state=start_array.copy(),
                nominal_values=start_array.copy(),
                particles=particles,
                parent=None,
                control=None,
                control_values=None,
                duration_steps=0,
                duration_seconds=0.0,
                persistent_parameters=persistent,
            )
        )
        self.stats["tree_size"] = 1
        self.update_goal_statistics(self.nodes[0])
        return True

    def update_goal_statistics(self, node: RandUpNode) -> None:
        nominal_distance = self.state_distance(
            node.nominal_values, self._goal_state_values
        )
        worst_particle_distance = max(
            self.state_distance(particle, self._goal_state_values)
            for particle in np.asarray(node.particles, dtype=float)
        )
        self.stats["best_nominal_goal_distance"] = min(
            float(self.stats["best_nominal_goal_distance"]), nominal_distance
        )
        self.stats["best_worst_particle_goal_distance"] = min(
            float(self.stats["best_worst_particle_goal_distance"]),
            worst_particle_distance,
        )

    def nearest_index(self, sampled_state) -> int:
        sampled_values = self.state_to_numpy(sampled_state)
        values = np.asarray([node.nominal_values for node in self.nodes], dtype=float)
        if self.system.name in ("kinematic_car", "pushing_object"):
            position = np.linalg.norm(values[:, :2] - sampled_values[:2], axis=1)
            angle = np.abs(wrap_angle_numpy(values[:, 2] - sampled_values[2]))
            distances = position + 0.5 * angle
        elif self.system.name == "dubins_airplane":
            distances = np.asarray(
                [self.system.state_distance(value, sampled_values) for value in values],
                dtype=float,
            )
        else:
            distances = np.linalg.norm(values - sampled_values, axis=1)
        return int(np.argmin(distances))

    def state_distance(self, first: np.ndarray, second: np.ndarray) -> float:
        """Match the repository's OMPL state-space goal/exploration metric."""

        first = np.asarray(first, dtype=float).reshape(-1)
        second = np.asarray(second, dtype=float).reshape(-1)
        if self.system.name in ("kinematic_car", "pushing_object"):
            position = float(np.linalg.norm(first[:2] - second[:2]))
            angle = float(abs(wrap_angle_numpy(first[2] - second[2])))
            # OMPL's default SE2 weights are 1 for R^2 and 0.5 for SO(2).
            return position + 0.5 * angle
        if self.system.name == "dubins_airplane":
            return self.system.state_distance(first, second)
        return float(np.linalg.norm(first - second))

    def apply_time_varying_uncertainty(self, particles: np.ndarray) -> np.ndarray:
        mode = str(self.config.uncertainty_mode).strip().lower()
        if mode in NO_UNCERTAINTY_MODES:
            return particles
        result = np.asarray(particles, dtype=float).copy()
        if self.system.name in ("kinematic_car", "pushing_object"):
            result[:, :2] += self._rng.normal(
                0.0,
                float(self.config.position_std),
                size=(len(result), 2),
            )
            result[:, 2] = wrap_angle_numpy(
                result[:, 2]
                + self._rng.normal(
                    0.0,
                    float(self.config.rotation_std),
                    size=len(result),
                )
            )
            return result
        if self.system.name == "double_integrator":
            result[:, 3:6] += self._rng.normal(
                0.0,
                float(self.config.velocity_std),
                size=(len(result), 3),
            )
            return result
        if self.system.name == "dubins_airplane":
            result[:, :3] += self._rng.normal(
                0.0,
                float(self.config.position_std),
                size=(len(result), 3),
            )
            result[:, 3:5] += self._rng.normal(
                0.0,
                float(self.config.rotation_std),
                size=(len(result), 2),
            )
            result[:, 5] += self._rng.normal(
                0.0,
                float(self.config.velocity_std),
                size=len(result),
            )
            result[:, 3] = wrap_angle_numpy(result[:, 3])
            lower = np.asarray([bound[0] for bound in self.system.state_bounds])
            upper = np.asarray([bound[1] for bound in self.system.state_bounds])
            return np.clip(result, lower, upper)
        raise ValueError(
            f"RandUP-RRT uncertainty is not configured for {self.system.name!r}"
        )

    def propagate_particle_batch(
        self, particles: np.ndarray, control: np.ndarray, duration: float
    ) -> np.ndarray:
        try:
            propagated = self.system.propagate(particles, control, duration)
            result = np.asarray(propagated, dtype=float)
            if result.shape == particles.shape:
                return result
        except Exception:
            pass
        return np.stack(
            [
                np.asarray(
                    self.system.propagate(particle, control, duration),
                    dtype=float,
                )
                for particle in particles
            ],
            axis=0,
        )

    def propagate_edge(
        self,
        parent: RandUpNode,
        control_values: np.ndarray,
        duration_steps: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None, str]:
        """Propagate and validate every primitive state along one candidate edge."""

        nominal = np.asarray(parent.nominal_values, dtype=float).copy()
        particles = np.asarray(parent.particles, dtype=float).copy()
        step_size = float(self.getSpaceInformation().getPropagationStepSize())
        for _ in range(int(duration_steps)):
            try:
                nominal = np.asarray(
                    self.system.propagate(nominal, control_values, step_size),
                    dtype=float,
                ).reshape(-1)
                particles = self.propagate_particle_batch(
                    particles, control_values, step_size
                )
                particles = self.apply_time_varying_uncertainty(particles)
                self.stats["dynamics_propagations"] += 1 + int(
                    self.config.num_particles
                )
                self.stats["particle_propagations"] += int(self.config.num_particles)
            except Exception:
                self.stats["numerical_propagation_failures"] += 1
                return None, None, "numerical_propagation_failure"
            if not np.all(np.isfinite(nominal)) or not np.all(np.isfinite(particles)):
                self.stats["numerical_propagation_failures"] += 1
                return None, None, "numerical_propagation_failure"
            if not self.state_valid(nominal):
                self.stats["nominal_collision_rejections"] += 1
                return None, None, "nominal_collision_during_expansion"
            particle_validity = are_state_arrays_valid(
                particles,
                system=self.system.name,
                config=self._validity_config,
                obstacle_config=self.obstacle_config,
                safety_radius_override=(
                    self._base_safety_radius + float(self.config.padding_epsilon)
                ),
            )
            if not bool(np.all(particle_validity)):
                self.stats["particle_collision_rejections"] += 1
                return None, None, "particle_collision_during_expansion"
        return nominal, particles, ""

    def particles_satisfy_goal(self, particles: np.ndarray) -> bool:
        """Evaluate the configured OMPL goal region for every particle."""

        satisfaction = [
            self.state_distance(particle, self._goal_state_values)
            <= self._goal_threshold
            for particle in np.asarray(particles, dtype=float)
        ]
        return bool(satisfaction) and all(satisfaction)

    def node_reaches_goal(self, node: RandUpNode) -> bool:
        if bool(self.config.goal_requires_all_particles):
            return self.particles_satisfy_goal(node.particles)
        return bool(
            self.state_distance(node.nominal_values, self._goal_state_values)
            <= self._goal_threshold
        )

    def solution_indices(self, goal_index: int) -> list[int]:
        indices = []
        current: int | None = int(goal_index)
        while current is not None:
            indices.append(current)
            current = self.nodes[current].parent
        indices.reverse()
        return indices

    def register_solution(self, goal_index: int) -> None:
        indices = self.solution_indices(goal_index)
        path = RandUpControlPath(
            states=tuple(
                tuple(float(value) for value in self.nodes[index].nominal_values)
                for index in indices
            ),
            controls=tuple(
                tuple(float(value) for value in self.nodes[index].control_values)
                for index in indices[1:]
            ),
            duration_steps=tuple(
                int(self.nodes[index].duration_steps) for index in indices[1:]
            ),
            duration_seconds=tuple(
                float(self.nodes[index].duration_seconds) for index in indices[1:]
            ),
        )
        self.goal_node_index = int(goal_index)
        self.last_solution_path = path
        self.last_solution_particle_sets = [
            np.asarray(self.nodes[index].particles, dtype=float).copy()
            for index in indices
        ]
        self.last_solution_persistent_parameters = [
            np.asarray(self.nodes[index].persistent_parameters, dtype=float).copy()
            for index in indices
        ]

    def solution_duration_steps(self, goal_index: int) -> int:
        """Return the physical primitive count from the root to a goal node."""

        return sum(
            int(self.nodes[index].duration_steps)
            for index in self.solution_indices(goal_index)[1:]
        )

    def consider_solution(self, goal_index: int, started: float) -> bool:
        """Retain a robust goal path when it improves control duration."""

        duration_steps = self.solution_duration_steps(goal_index)
        self.stats["solution_candidates"] += 1
        if self.stats["first_solution_time"] is None:
            self.stats["first_solution_time"] = time.monotonic() - started

        best_steps = self.stats["best_solution_duration_steps"]
        if best_steps is not None and duration_steps >= int(best_steps):
            return False

        self.register_solution(goal_index)
        step_size = float(self.getSpaceInformation().getPropagationStepSize())
        self.stats["solution_improvements"] += 1
        self.stats["best_solution_duration_steps"] = duration_steps
        self.stats["best_solution_duration_seconds"] = duration_steps * step_size
        return True

    def solution_info(self) -> dict[str, Any] | None:
        if self.last_solution_path is None:
            return None
        path = self.last_solution_path
        return {
            "state_count": path.getStateCount(),
            "control_count": path.getControlCount(),
            "states": [list(state) for state in path.states],
            "controls": [list(control) for control in path.controls],
            "time": list(path.duration_seconds),
            "time_steps": list(path.duration_steps),
            "cost": float(sum(path.duration_seconds)),
            "approximate": False,
            "solution_difference": 0.0,
            "randup": self.solution_metadata(),
        }

    def expand_tree_once(self, sampled_state, sampled_control) -> RandUpNode | None:
        """Sample and validate one tree expansion."""

        self.stats["iterations"] += 1
        self.stats["tree_expansions"] += 1
        if self._rng.random() < float(self.config.goal_bias):
            self.set_state(sampled_state, self._goal_state_values)
        else:
            self._state_sampler.sampleUniform(sampled_state)

        parent_index = self.nearest_index(sampled_state)
        parent = self.nodes[parent_index]
        control, duration_steps = self.sample_control(sampled_control)
        nominal, particles, rejection = self.propagate_edge(
            parent,
            control,
            duration_steps,
        )
        if rejection:
            self.last_expansion_rejection = rejection
            self.stats["last_expansion_rejection"] = rejection
            return None

        duration = duration_steps * float(
            self.getSpaceInformation().getPropagationStepSize()
        )
        node = RandUpNode(
            nominal_state=np.asarray(nominal, dtype=float).copy(),
            nominal_values=np.asarray(nominal, dtype=float).copy(),
            particles=np.asarray(particles, dtype=float),
            parent=parent_index,
            control=control.copy(),
            control_values=control.copy(),
            duration_steps=duration_steps,
            duration_seconds=duration,
            persistent_parameters=np.asarray(
                parent.persistent_parameters,
                dtype=float,
            ).copy(),
        )
        self.nodes.append(node)
        self.update_goal_statistics(node)
        self.stats["accepted_edges"] += 1
        self.stats["tree_size"] = len(self.nodes)
        return node

    def exact_solution_status(self, goal_index: int | None, started: float):
        if goal_index is not None:
            self.consider_solution(goal_index, started)
        self.failure_reason = ""
        self.stats["planning_success"] = True
        self.stats["failure_reason"] = ""
        self.stats["planning_time"] += time.monotonic() - started
        return ob.PlannerStatus(ob.PlannerStatus.EXACT_SOLUTION)

    def failure_status(self, started: float, termination_condition):
        elapsed = time.monotonic() - started
        self.stats["planning_time"] += elapsed
        if int(self.stats["iterations"]) >= int(self.config.max_iterations):
            reason = "no_robust_plan_found:max_iterations"
        elif elapsed >= float(self.config.planning_time) - 1e-6 or bool(
            termination_condition()
        ):
            reason = "planning_timeout"
        else:
            reason = "no_robust_plan_found"
        self.failure_reason = reason
        self.stats["failure_reason"] = reason
        self.stats["tree_size"] = len(self.nodes)
        return ob.PlannerStatus(ob.PlannerStatus.TIMEOUT)

    def solve(self, termination_condition):
        self.checkValidity()
        self.ensure_samplers()
        started = time.monotonic()
        if not self.nodes and not self.initialize_root():
            self.stats["planning_time"] += time.monotonic() - started
            self.stats["failure_reason"] = self.failure_reason
            return ob.PlannerStatus(ob.PlannerStatus.INVALID_START)

        if self.goal_node_index is not None:
            return ob.PlannerStatus(ob.PlannerStatus.EXACT_SOLUTION)
        if self.node_reaches_goal(self.nodes[0]):
            return self.exact_solution_status(0, started)

        si = self.getSpaceInformation()
        sampled_state = si.allocState()
        sampled_control = si.allocControl()
        try:
            while (
                not termination_condition()
                and int(self.stats["iterations"]) < int(self.config.max_iterations)
                and time.monotonic() - started < float(self.config.planning_time)
            ):
                node = self.expand_tree_once(sampled_state, sampled_control)
                if node is not None and self.node_reaches_goal(node):
                    goal_index = len(self.nodes) - 1
                    if not bool(self.config.optimize_after_first_solution):
                        return self.exact_solution_status(goal_index, started)
                    self.consider_solution(goal_index, started)
        finally:
            sampled_control = None
            sampled_state = None
        if self.goal_node_index is not None:
            return self.exact_solution_status(None, started)
        return self.failure_status(started, termination_condition)

    def solution_metadata(self) -> dict[str, Any]:
        """Return serializable planner statistics and recovered particle sets."""

        return {
            **deepcopy(self.stats),
            "failure_reason": str(self.failure_reason or self.stats["failure_reason"]),
            "particle_sets": [
                particles.tolist() for particles in self.last_solution_particle_sets
            ],
            "persistent_parameters": [
                parameters.tolist()
                for parameters in self.last_solution_persistent_parameters
            ],
        }
