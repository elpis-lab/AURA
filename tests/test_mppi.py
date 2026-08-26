from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from experiment.task_time_efficiency import (
    _apply_bounds,
    load_system_panel,
    shared_environment_contract,
)
from methods.MPPI import MPPIController, default_parameters, parameters_from_config
from propagators import DoubleIntegrator, KinematicCar, get_system


class BiasedPushingModel(torch.nn.Module):
    """Deterministic test model with an intentionally nonzero intercept."""

    def forward(self, control: torch.Tensor) -> torch.Tensor:
        distance = control[:, 2]
        return torch.stack(
            [
                distance + 0.011,
                torch.full_like(distance, -0.004),
                torch.full_like(distance, 0.02),
            ],
            dim=1,
        )


def make_biased_pushing_controller(
    monkeypatch,
    *,
    short_push_cost_scale: float = 0.0,
) -> MPPIController:
    monkeypatch.setattr(
        "methods.MPPI.get_pushing_model",
        lambda *args, **kwargs: BiasedPushingModel(),
    )
    system = get_system("pushing_object")
    system.set_control_bounds([(0.0, 0.75), (-0.4, 0.4), (0.0, 0.13)])
    system.set_state_bounds([(-1.0, 1.0), (-1.0, 1.0)])
    system.object_shape = np.array([0.16, 0.21, 0.07])
    parameters = replace(
        default_parameters("pushing_object"),
        horizon_steps=3,
        zero_anchor_pushing_model=True,
        short_push_cost_scale=short_push_cost_scale,
        minimum_preferred_push_distance=0.02,
    )
    return MPPIController(
        system,
        [0.4, 0.0, 0.0],
        propagation_step_size=1.0,
        goal_threshold=0.075,
        parameters=parameters,
        seed=13,
        device="cpu",
    )


def test_mppi_command_is_seeded_finite_and_within_current_control_bounds() -> None:
    system = KinematicCar()
    system.configure_duration_contract(1.0, 1, 5)
    kwargs = dict(
        system=system,
        goal_state=[3.0, 3.0, 1.5708],
        propagation_step_size=1.0,
        parameters=default_parameters("kinematic_car"),
        seed=1234,
        device="cpu",
    )
    first = MPPIController(**kwargs)
    second = MPPIController(**kwargs)
    state = np.array([-0.2, 0.0, 0.0])
    u_first, info_first = first.command(state)
    u_second, info_second = second.command(state)
    np.testing.assert_allclose(u_first, u_second, rtol=0.0, atol=0.0)
    assert np.isfinite(u_first).all()
    assert system.control_bounds[0][0] <= u_first[0] <= system.control_bounds[0][1]
    assert system.control_bounds[1][0] <= u_first[1] <= system.control_bounds[1][1]
    assert info_first == info_second
    assert info_first["effective_sample_size"] >= 1.0


def test_mppi_prediction_uses_same_double_integrator_duration_contract() -> None:
    system = DoubleIntegrator()
    system.configure_duration_contract(1.0, 1, 5)
    controller = MPPIController(
        system,
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        propagation_step_size=1.0,
        parameters=default_parameters("double_integrator"),
        seed=9,
        device="cpu",
    )
    state = np.zeros(6)
    control = np.array([0.2, -0.1, 0.05])
    predicted = controller.predict_next(state, control)
    expected = system.propagate(state, control, 1.0)
    np.testing.assert_allclose(predicted, expected, rtol=1e-6, atol=1e-7)


def test_mppi_uses_one_tick_actions_not_ompl_variable_edge_durations() -> None:
    assert default_parameters("kinematic_car").action_duration_steps == 1
    assert default_parameters("double_integrator").action_duration_steps == 1
    assert default_parameters("pushing_object").action_duration_steps == 1
    assert default_parameters("dubins_airplane").action_duration_steps == 1


def test_zero_anchored_pushing_model_has_no_motion_at_zero_distance(
    monkeypatch,
) -> None:
    controller = make_biased_pushing_controller(monkeypatch)
    start = np.array([0.1, -0.2, 0.3])

    stationary = controller.predict_next(start, [0.25, 0.04, 0.0])
    moving = controller.predict_next(start, [0.25, 0.04, 0.05])

    np.testing.assert_allclose(stationary, start, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(
        moving,
        [
            start[0] + 0.05 * np.cos(start[2]),
            start[1] + 0.05 * np.sin(start[2]),
            start[2],
        ],
        rtol=0.0,
        atol=1e-6,
    )


def test_adaptive_short_push_cost_prefers_a_useful_distance(monkeypatch) -> None:
    controller = make_biased_pushing_controller(
        monkeypatch,
        short_push_cost_scale=1000.0,
    )
    states = torch.zeros((2, 3), dtype=torch.float32)
    preferred = controller.preferred_push_distance(states)
    controls = torch.tensor(
        [[0.25, 0.04, 0.0], [0.25, 0.04, float(preferred[1])]],
        dtype=torch.float32,
    )
    costs = controller.short_push_cost(states, controls)

    torch.testing.assert_close(
        preferred,
        torch.full_like(preferred, (0.4 - 0.075) / 3.0),
    )
    assert float(costs[0]) > 0.0
    assert float(costs[1]) == 0.0


def test_mppi_defaults_have_no_goal_bearing_funnel() -> None:
    car = default_parameters("kinematic_car")
    assert car.goal_cost_mode == "shared_state_metric"
    assert car.smoothness_cost_scale == 0.0
    for system_name in ("double_integrator", "pushing_object", "dubins_airplane"):
        parameters = default_parameters(system_name)
        assert parameters.goal_cost_mode == "direct_goal_region"
        assert parameters.smoothness_cost_scale == 0.0


def test_shared_car_goal_cost_is_not_shaped_by_goal_bearing() -> None:
    system = KinematicCar()
    system.configure_duration_contract(1.0, 1, 5)
    controller = MPPIController(
        system,
        [0.0, 0.0, 0.0],
        propagation_step_size=1.0,
        parameters=default_parameters("kinematic_car"),
        seed=17,
        device="cpu",
    )
    # Equal position distance and equal yaw error must have equal cost even
    # though the two states have opposite bearings to the goal.
    states = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    costs = controller.state_cost(states, terminal=False)
    torch.testing.assert_close(costs[0], costs[1], rtol=0.0, atol=0.0)


def test_continuous_mppi_update_uses_drawn_gaussian_at_control_bounds() -> None:
    system = KinematicCar()
    system.configure_duration_contract(1.0, 1, 5)
    controller = MPPIController(
        system,
        [3.0, 3.0, 1.5708],
        propagation_step_size=1.0,
        parameters=default_parameters("kinematic_car"),
        seed=1234,
        device="cpu",
    )
    controller.control_sequence[:] = controller.internal_control_high
    sampled, perturbation = controller.sample_controls()
    clipped_delta = sampled - controller.control_sequence.unsqueeze(0)
    # Some positive Gaussian samples are clipped at the upper control bound.
    # The standard MPPI update retains those drawn perturbations rather than
    # silently replacing them with a zero projected delta.
    clipped_positive = (clipped_delta == 0.0) & (perturbation > 0.0)
    assert torch.any(clipped_positive)


def test_mppi_shared_environment_contract_matches_system_config() -> None:
    _, config = load_system_panel("pushing_object", "gaussian")
    system = get_system(config["system_name"])
    _apply_bounds(system, config)
    system.configure_duration_contract(
        config["propagation_step_size"],
        config["min_control_duration"],
        config["max_control_duration"],
    )
    parameters = default_parameters(config["system_name"])
    contract = shared_environment_contract(config, system, parameters)
    assert contract["start_state"] == config["start_state"]
    assert contract["goal_state"] == config["goal_state"]
    assert contract["goal_threshold"] == config["goal_threshold"]
    assert contract["mppi_goal_threshold"] == config["goal_threshold"]
    assert contract["propagation_step_size"] == 1.0
    assert contract["effective_control_bounds"] == config["control_bounds"]
    assert contract["mppi_action_duration_steps"] == 1
    assert contract["mppi_action_duration_seconds"] == 1.0
    assert contract["mppi_physical_action_seconds"] == 2.0
    assert contract["physical_execution_step_seconds"] == 2.0
    assert contract["mppi_goal_bias"] == "not_applicable_no_state_sampler"


def test_dubins_airplane_experiment2_uses_tuned_compute_parameters() -> None:
    _, config = load_system_panel("dubins_airplane", "gaussian")
    parameters = parameters_from_config("dubins_airplane", config)
    assert parameters.horizon_steps == 5
    assert parameters.num_samples == 512
    assert parameters.temperature == 1.4
    assert parameters.control_noise_std == (0.15, 0.10, 0.10)
    assert get_system("dubins_airplane").control_bounds == [
        (-0.3, 0.3),
        (-np.pi / 4.0, np.pi / 4.0),
        (-np.pi / 4.0, np.pi / 4.0),
    ]
    defaults = default_parameters("dubins_airplane")
    assert defaults.horizon_steps == 5
    assert defaults.num_samples == 512
    assert defaults.temperature == 2.0


def test_dubins_airplane_mppi_uses_shared_goal_region() -> None:
    _, config = load_system_panel("dubins_airplane", "gaussian")
    system = get_system(config["system_name"])
    _apply_bounds(system, config)
    system.configure_duration_contract(
        config["propagation_step_size"],
        config["min_control_duration"],
        config["max_control_duration"],
    )
    contract = shared_environment_contract(
        config, system, default_parameters("dubins_airplane")
    )
    assert contract["goal_threshold"] == 0.35
    assert contract["mppi_goal_threshold"] == 0.35


def test_car_environments_use_the_same_mppi_settings() -> None:
    _, mujoco_config = load_system_panel("kinematic_car", "mujoco")
    _, gaussian_config = load_system_panel("kinematic_car", "gaussian")

    mujoco = parameters_from_config("kinematic_car", mujoco_config)
    gaussian = parameters_from_config("kinematic_car", gaussian_config)

    assert mujoco == gaussian == default_parameters("kinematic_car")
    assert mujoco.horizon_steps == 15
    assert mujoco.num_samples == 512
    assert mujoco.temperature == 2.0
    assert mujoco.control_noise_std == (0.23, 0.12)
    assert mujoco.goal_cost_mode == "shared_state_metric"


def test_shared_car_metric_weights_position_and_yaw_equally() -> None:
    _, config = load_system_panel("kinematic_car", "mujoco")
    system = get_system("kinematic_car")
    parameters = parameters_from_config("kinematic_car", config)
    controller = MPPIController(
        system,
        config["goal_state"],
        propagation_step_size=config["propagation_step_size"],
        parameters=parameters,
        seed=11,
        device="cpu",
    )
    goal = np.asarray(config["goal_state"], dtype=float)
    position_error = torch.as_tensor(
        np.asarray([goal + [1.0, 0.0, 0.0]]), dtype=torch.float32
    )
    yaw_error = torch.as_tensor(
        np.asarray([goal + [0.0, 0.0, 1.0]]), dtype=torch.float32
    )
    torch.testing.assert_close(
        controller.state_cost(position_error, terminal=False),
        controller.state_cost(yaw_error, terminal=False),
    )
    torch.testing.assert_close(
        controller.state_cost(position_error, terminal=True),
        controller.state_cost(yaw_error, terminal=True),
    )
