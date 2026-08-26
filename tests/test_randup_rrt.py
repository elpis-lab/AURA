from __future__ import annotations

import numpy as np
from ompl import control as oc
from ompl import util as ou

from methods.plan import OMPLPlanner
from methods.RandUpRRT import RandUpNode, RandUpRRT, RandUpRRTConfig
from propagators import KinematicCar, get_system
from simulation.simulator import KinematicCarGaussianNoise
from utils.utils import arrayDistance, are_state_arrays_valid, is_state_array_valid


class _ProblemDefinitionStub:
    def getStartStateCount(self) -> int:
        return 1


def _randup(
    *,
    config: RandUpRRTConfig,
    obstacles: dict | None = None,
    start=(0.0, 0.0, 0.0),
    goal=(1.0, 0.0, 0.0),
    threshold=0.1,
) -> RandUpRRT:
    system = KinematicCar()
    system.configure_duration_contract(
        1.0,
        config.control_duration_min,
        config.control_duration_max,
    )
    si = oc.SpaceInformation(system.state_space, system.control_space)
    si.setPropagationStepSize(1.0)
    planner = RandUpRRT(
        si,
        system=system,
        config=config,
        start_state=np.asarray(start, dtype=float),
        goal_state=np.asarray(goal, dtype=float),
        goal_threshold=float(threshold),
        obstacle_config=obstacles,
    )
    planner.setProblemDefinition(_ProblemDefinitionStub())
    return planner


def _node(nominal, particles) -> RandUpNode:
    nominal = np.asarray(nominal, dtype=float)
    return RandUpNode(
        nominal_state=nominal.copy(),
        nominal_values=nominal.copy(),
        particles=np.asarray(particles, dtype=float),
        parent=None,
        control=None,
        control_values=None,
        duration_steps=0,
        duration_seconds=0.0,
        persistent_parameters=np.empty((len(particles), 0), dtype=float),
    )


def _tree_node(nominal, *, parent: int, duration_steps: int) -> RandUpNode:
    nominal = np.asarray(nominal, dtype=float)
    return RandUpNode(
        nominal_state=nominal.copy(),
        nominal_values=nominal.copy(),
        particles=nominal.reshape(1, -1).copy(),
        parent=parent,
        control=np.array([0.5, 0.0]),
        control_values=np.array([0.5, 0.0]),
        duration_steps=duration_steps,
        duration_seconds=float(duration_steps),
        persistent_parameters=np.empty((1, 0), dtype=float),
    )


def test_array_validity_checks_every_configured_state_dimension() -> None:
    config = {
        "state_bounds": [
            [-3.0, 3.0],
            [-3.0, 3.0],
            [0.0, 3.0],
            [-0.5, 0.5],
            [-0.5, 0.5],
            [-0.5, 0.5],
        ],
        "obstacles": {"enabled": False},
    }
    valid = np.zeros(6)
    invalid_z = valid.copy()
    invalid_z[2] = -0.01
    invalid_velocity = valid.copy()
    invalid_velocity[5] = 0.51
    assert is_state_array_valid(valid, config=config)
    assert not is_state_array_valid(invalid_z, config=config)
    assert not is_state_array_valid(invalid_velocity, config=config)
    np.testing.assert_array_equal(
        are_state_arrays_valid(
            np.stack([valid, invalid_z, invalid_velocity]), config=config
        ),
        [True, False, False],
    )


def test_deterministic_equivalence_m1_zero_uncertainty() -> None:
    config = RandUpRRTConfig(
        num_particles=1,
        planning_time=1.0,
        max_iterations=10,
        random_seed=17,
        control_duration_min=1,
        control_duration_max=3,
        uncertainty_mode="none",
    )
    planner = _randup(config=config)
    parent = _node([0.0, 0.0, 0.0], [[0.0, 0.0, 0.0]])
    control = np.array([0.5, 0.1], dtype=float)
    nominal, particles, rejection = planner.propagate_edge(parent, control, 3)
    expected = np.array([0.0, 0.0, 0.0], dtype=float)
    for _ in range(3):
        expected = planner.system.propagate(expected, control, 1.0)
    assert rejection == ""
    np.testing.assert_allclose(nominal, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(particles[0], expected, rtol=0.0, atol=1e-12)


def test_particle_propagation_is_seed_reproducible() -> None:
    config = RandUpRRTConfig(
        num_particles=4,
        planning_time=1.0,
        max_iterations=10,
        random_seed=829,
        control_duration_min=1,
        control_duration_max=2,
        uncertainty_mode="gaussian_process",
        position_std=0.02,
        rotation_std=0.03,
    )
    first = _randup(config=config)
    second = _randup(config=config)
    parent = _node([0.0, 0.0, 0.0], np.zeros((4, 3), dtype=float))
    control = np.array([0.4, -0.05], dtype=float)
    first_result = first.propagate_edge(parent, control, 2)
    second_result = second.propagate_edge(parent, control, 2)
    assert first_result[2] == second_result[2] == ""
    np.testing.assert_allclose(first_result[0], second_result[0])
    np.testing.assert_allclose(first_result[1], second_result[1])


def test_particle_collision_rejects_nominally_safe_edge() -> None:
    config = RandUpRRTConfig(
        num_particles=1,
        planning_time=1.0,
        max_iterations=10,
        random_seed=5,
        control_duration_min=1,
        control_duration_max=1,
        uncertainty_mode="none",
    )
    planner = _randup(
        config=config,
        obstacles={
            "enabled": True,
            "safety_radius": 0.0,
            "circles": [[1.0, 0.2, 0.05]],
        },
    )
    parent = _node([0.0, 0.0, 0.0], [[0.0, 0.2, 0.0]])
    nominal, particles, rejection = planner.propagate_edge(
        parent, np.array([1.0, 0.0]), 1
    )
    assert nominal is None
    assert particles is None
    assert rejection == "particle_collision_during_expansion"
    assert planner.stats["particle_collision_rejections"] == 1


def test_robust_goal_requires_every_particle() -> None:
    planner = _randup(
        config=RandUpRRTConfig(
            num_particles=2,
            planning_time=1.0,
            max_iterations=10,
            control_duration_min=1,
            control_duration_max=1,
            uncertainty_mode="none",
        ),
        goal=(1.0, 0.0, 0.0),
        threshold=0.1,
    )
    assert planner.particles_satisfy_goal(
        np.array([[1.0, 0.0, 0.0], [1.05, 0.0, 0.0]])
    )
    assert not planner.particles_satisfy_goal(
        np.array([[1.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    )


def test_anytime_mode_keeps_shortest_robust_solution(monkeypatch) -> None:
    planner = _randup(
        config=RandUpRRTConfig(
            num_particles=1,
            planning_time=10.0,
            max_iterations=2,
            control_duration_min=1,
            control_duration_max=1,
            optimize_after_first_solution=True,
            uncertainty_mode="none",
        ),
        goal=(1.0, 0.0, 0.0),
        threshold=0.1,
    )
    planner.nodes = [
        _node([0.0, 0.0, 0.0], [[0.0, 0.0, 0.0]]),
        _tree_node([0.3, 0.0, 0.0], parent=0, duration_steps=1),
        _tree_node([0.6, 0.0, 0.0], parent=1, duration_steps=1),
    ]
    candidates = [
        _tree_node([1.0, 0.0, 0.0], parent=2, duration_steps=1),
        _tree_node([1.0, 0.0, 0.0], parent=0, duration_steps=1),
    ]

    def expand_once(*_args):
        planner.stats["iterations"] += 1
        node = candidates.pop(0)
        planner.nodes.append(node)
        return node

    monkeypatch.setattr(planner, "expand_tree_once", expand_once)
    status = planner.solve(lambda: False)

    assert bool(status)
    solution = planner.solution_info()
    assert solution is not None
    assert solution["time_steps"] == [1]
    assert solution["cost"] == 1.0
    assert planner.stats["solution_candidates"] == 2
    assert planner.stats["solution_improvements"] == 2
    assert planner.stats["best_solution_duration_steps"] == 1


def test_mapping_enables_anytime_control_duration_optimization() -> None:
    config = RandUpRRTConfig.from_mapping(
        {"randup_optimize_after_first_solution": True}
    )
    assert config.optimize_after_first_solution is True


def test_zero_uncertainty_plan_executes_through_existing_simulator() -> None:
    seed = 61423
    ou.RNG.setSeed(seed)
    config = RandUpRRTConfig(
        num_particles=1,
        planning_time=1.0,
        max_iterations=2000,
        random_seed=seed,
        control_duration_min=1,
        control_duration_max=2,
        uncertainty_mode="none",
        goal_bias=0.2,
    )
    wrapper = OMPLPlanner(
        system=KinematicCar(),
        start_state=np.array([0.0, 0.0, 0.0]),
        goal_state=np.array([0.25, 0.0, 0.0]),
        planner_method="randup_rrt",
        goal_threshold=0.18,
        min_max_control_duration=(1, 2),
        propagation_step_size=0.2,
        initial_planning_time=1.0,
        goal_bias=0.2,
        obstacle_config={"enabled": False},
        randup_config=config,
    )
    wrapper.single_solve = True
    solutions, _ = wrapper.plan()
    assert solutions
    solution = solutions[0]

    simulator = KinematicCarGaussianNoise(
        {
            "start_state": [0.0, 0.0, 0.0],
            "propagation_step_size": 0.2,
            "min_control_duration": 1,
            "max_control_duration": 2,
            "sampling_position_std": 0.0,
            "sampling_rotation_std": 0.0,
        }
    )
    for control, duration in zip(solution["controls"], solution["time"]):
        simulator.execute_segment(control, duration)
    assert (
        arrayDistance(
            simulator.get_state(),
            np.array([0.25, 0.0, 0.0]),
            system="kinematic_car",
        )
        <= 0.18
    )


def test_wrapper_accepts_robust_randup_goal_when_nominal_endpoint_is_outside() -> None:
    """The all-particle goal predicate, not the nominal endpoint, is authoritative."""

    wrapper = OMPLPlanner(
        system=KinematicCar(),
        start_state=np.array([0.0, 0.0, 0.0]),
        goal_state=np.array([1.0, 0.0, 0.0]),
        planner_method="randup_rrt",
        goal_threshold=0.1,
        min_max_control_duration=(1, 1),
        propagation_step_size=1.0,
        initial_planning_time=1.0,
        obstacle_config={"enabled": False},
        randup_config=RandUpRRTConfig(
            num_particles=2,
            planning_time=1.0,
            max_iterations=10,
            control_duration_min=1,
            control_duration_max=1,
            uncertainty_mode="none",
        ),
    )
    robust_info = {
        "state_count": 2,
        "control_count": 1,
        "states": [[0.0, 0.0, 0.0], [1.11, 0.0, 0.0]],
        "controls": [[0.11, 0.0]],
        "time": [1.0],
        "time_steps": [1],
        "cost": 1.0,
        "approximate": False,
        "solution_difference": 0.0,
        "randup": {
            "planning_success": True,
            "particle_sets": [[[0.95, 0.0, 0.0], [1.05, 0.0, 0.0]]],
        },
    }

    class _RobustPlannerStub:
        def solution_info(self):
            return robust_info

        def solution_metadata(self):
            return robust_info["randup"]

    wrapper.randup_planner = _RobustPlannerStub()
    solutions = wrapper.get_solutions()

    assert solutions == [robust_info]
    assert solutions[0]["goal_distance"] > wrapper.goal_threshold


def test_dubins_airplane_randup_uses_compound_state_and_matched_uncertainty() -> None:
    system = get_system("dubins_airplane")
    system.configure_duration_contract(1.0, 1, 1)
    si = oc.SpaceInformation(system.state_space, system.control_space)
    si.setPropagationStepSize(1.0)
    config = RandUpRRTConfig(
        num_particles=3,
        planning_time=1.0,
        max_iterations=10,
        random_seed=421,
        control_duration_min=1,
        control_duration_max=1,
        uncertainty_mode="gaussian_process_oracle",
        position_std=0.003,
        rotation_std=0.05,
        velocity_std=0.003,
    )
    start = np.asarray([0.1, 0.1, 0.15, 0.0, 0.0, 0.15])
    planner = RandUpRRT(
        si,
        system=system,
        config=config,
        start_state=start,
        goal_state=np.asarray([0.85, 0.85, 0.75, 0.0, 0.0, 0.15]),
        goal_threshold=0.35,
        obstacle_config={"enabled": False},
    )
    planner.setProblemDefinition(_ProblemDefinitionStub())

    state = system.state_space.allocState()
    planner.set_state(state, start)
    np.testing.assert_allclose(planner.state_to_numpy(state), start)

    particles = np.repeat(start[None, :], 3, axis=0)
    disturbed = planner.apply_time_varying_uncertainty(particles)
    assert disturbed.shape == (3, 6)
    assert np.all(np.isfinite(disturbed))
    for index, (low, high) in enumerate(system.state_bounds):
        assert np.all(disturbed[:, index] >= low)
        assert np.all(disturbed[:, index] <= high)
