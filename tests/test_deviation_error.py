from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from methods.MPPI import default_parameters
from utils.deviation import (
    AURALocalTrackingAdapter,
    make_mppi_controller,
    run_aura_tracking,
    run_mppi_tracking,
    run_open_loop,
)
from scripts.plot_deviation_error import plot_tracking_statistics
from experiment.deviation_error import (
    _checkpoint,
    _condition_config,
    _make_plant,
    condition_output_dir,
    main as deviation_main,
)
from utils.deviation import (
    compute_tracking_statistics,
    method_result_to_step_rows,
    method_result_to_trial_row,
    write_summary_tables,
)
from utils.deviation import (
    ReferenceTrajectory,
)
from utils.utils import arrayDistance
from utils.experiment_io import read_csv, write_csv


def _reference(system, config, count: int = 10) -> ReferenceTrajectory:
    initial = np.array([0.2, -0.1, 1.2, 0.03, -0.02, 0.01])
    controls = np.tile(np.array([[0.08, -0.04, 0.02]]), (count, 1))
    states = [initial.copy()]
    for control in controls:
        states.append(system.propagate(states[-1], control, config["control_duration"]))
    return ReferenceTrajectory(
        system=system.name,
        environment="gaussian",
        initial_state=initial,
        states=np.asarray(states),
        controls=controls,
        control_duration=config["control_duration"],
    )


def _run_forced_methods(*, noise: bool):
    system, config = _condition_config("double_integrator", "gaussian", "cpu")
    if not noise:
        config["simulator_config"]["sampling_position_std"] = 0.0
        config["simulator_config"]["sampling_velocity_std"] = 0.0
    reference = _reference(system, config)
    disturbances = np.random.default_rng(123).standard_normal((10, 3))
    plants = [
        _make_plant(
            "double_integrator", "gaussian", config, reference.initial_state, disturbances, 55
        )
        for _ in range(3)
    ]
    aura = AURALocalTrackingAdapter(system, config)
    mppi = make_mppi_controller(
        system,
        reference,
        default_parameters("double_integrator"),
        config,
        seed=77,
    )
    try:
        open_result = run_open_loop(reference, plants[0], config)
        aura_result = run_aura_tracking(
            reference, aura, plants[1], config, seed=88, force_nominal=True
        )
        mppi_result = run_mppi_tracking(
            reference, mppi, plants[2], config, force_nominal=True
        )
    finally:
        for plant in plants:
            plant.stop()
    return reference, open_result, aura_result, mppi_result


def test_same_reference_initial_state_and_common_disturbance_schedule() -> None:
    reference, open_result, aura_result, mppi_result = _run_forced_methods(noise=True)
    assert not reference.states.flags.writeable
    np.testing.assert_array_equal(open_result.initial_state, reference.initial_state)
    np.testing.assert_array_equal(aura_result.initial_state, reference.initial_state)
    np.testing.assert_array_equal(mppi_result.initial_state, reference.initial_state)
    np.testing.assert_allclose(aura_result.states, open_result.states, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(mppi_result.states, open_result.states, rtol=0.0, atol=0.0)


def test_wrapped_se2_tracking_distance() -> None:
    epsilon = 1.0e-3
    distance = arrayDistance(
        [0.0, 0.0, np.pi - epsilon],
        [0.0, 0.0, -np.pi + epsilon],
        system="kinematic_car",
    )
    assert np.isclose(distance, epsilon, rtol=0.0, atol=1e-12)


def test_zero_uncertainty_and_forced_controllers_match_open_loop() -> None:
    reference, open_result, aura_result, mppi_result = _run_forced_methods(noise=False)
    for result in (open_result, aura_result, mppi_result):
        errors = [
            arrayDistance(result.states[i + 1], reference.states[i + 1], reference.system)
            for i in range(reference.num_controls)
        ]
        np.testing.assert_allclose(errors, 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(aura_result.states, open_result.states, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(mppi_result.states, open_result.states, rtol=0.0, atol=0.0)


def test_exactly_ten_rows_and_no_controller_disturbance_argument() -> None:
    reference, open_result, aura_result, mppi_result = _run_forced_methods(noise=True)
    for result in (open_result, aura_result, mppi_result):
        rows = method_result_to_step_rows(
            reference=reference, result=result, trial=0, seed=42
        )
        assert len(rows) == 10
        assert [int(row["step"]) for row in rows] == list(range(1, 11))
    for function in (run_open_loop, run_aura_tracking, run_mppi_tracking):
        assert "disturbances" not in inspect.signature(function).parameters


def test_aura_adapter_cannot_enable_global_replanning() -> None:
    system, config = _condition_config("double_integrator", "gaussian", "cpu")
    adapter = AURALocalTrackingAdapter(system, config)
    assert adapter.global_replanning_enabled is False
    assert config["aura"]["global_replanning_enabled"] is False


def test_method_failure_preserves_partial_steps_and_placeholders() -> None:
    system, config = _condition_config("double_integrator", "gaussian", "cpu")
    reference = _reference(system, config)

    class FailingPlant:
        def __init__(self):
            self.state = reference.initial_state.copy()
            self.calls = 0

        def reset(self):
            self.state = reference.initial_state.copy()

        def set_state(self, pose):
            self.state = np.asarray(pose, dtype=float).copy()

        def get_state(self):
            return self.state.tolist()

        def execute_segment(self, control, duration):
            if self.calls == 2:
                raise RuntimeError("synthetic plant failure")
            self.state = system.propagate(self.state, control, duration)
            self.calls += 1
            return self.state.tolist()

    result = run_open_loop(reference, FailingPlant(), config)
    rows = method_result_to_step_rows(reference=reference, result=result, trial=0, seed=42)
    assert not result.success and result.completed_steps == 2
    assert len(rows) == 10
    assert [row["success"] for row in rows[:2]] == [True, True]
    assert all(row["success"] is False for row in rows[2:])
    assert all("synthetic plant failure" in row["failure_reason"] for row in rows[2:])


def test_reference_centered_mppi_does_not_collapse_nominal_controls_to_zero() -> None:
    system, config = _condition_config("double_integrator", "gaussian", "cpu")
    config["simulator_config"]["sampling_position_std"] = 0.0
    config["simulator_config"]["sampling_velocity_std"] = 0.0
    reference = _reference(system, config)
    disturbances = np.zeros((reference.num_controls, 3))
    plant = _make_plant(
        "double_integrator",
        "gaussian",
        config,
        reference.initial_state,
        disturbances,
        55,
    )
    controller = make_mppi_controller(
        system,
        reference,
        default_parameters("double_integrator"),
        config,
        seed=77,
    )
    result = run_mppi_tracking(reference, controller, plant, config)
    nominal_norm = np.mean(np.linalg.norm(reference.controls, axis=1))
    executed = np.asarray(result.controls)
    assert result.success and result.completed_steps == reference.num_controls
    assert np.mean(np.linalg.norm(executed, axis=1)) > 0.75 * nominal_norm
    assert np.mean(np.linalg.norm(executed - reference.controls, axis=1)) < 0.05


def test_mppi_receives_exact_receding_reference_windows() -> None:
    system, config = _condition_config("double_integrator", "gaussian", "cpu")
    config["simulator_config"]["sampling_position_std"] = 0.0
    config["simulator_config"]["sampling_velocity_std"] = 0.0
    reference = _reference(system, config)
    disturbances = np.zeros((reference.num_controls, 3))
    plant = _make_plant(
        "double_integrator",
        "gaussian",
        config,
        reference.initial_state,
        disturbances,
        55,
    )

    class RecordingController:
        def __init__(self):
            self.parameters = SimpleNamespace(horizon_steps=5)
            self.device = torch.device("cpu")
            self.initialized_with = None
            self.calls = []

        def initialize_nominal(self, controls):
            self.initialized_with = np.asarray(controls, dtype=float).copy()

        def command_reference(self, state, state_references, control_references):
            self.calls.append(
                (
                    np.asarray(state, dtype=float).copy(),
                    np.asarray(state_references, dtype=float).copy(),
                    np.asarray(control_references, dtype=float).copy(),
                )
            )
            return np.asarray(control_references[0], dtype=float).copy(), {
                "active_horizon": float(len(control_references))
            }

    controller = RecordingController()
    result = run_mppi_tracking(reference, controller, plant, config)
    plant.stop()
    assert result.success and len(controller.calls) == reference.num_controls
    np.testing.assert_array_equal(
        controller.initialized_with, reference.controls[:5]
    )
    for index, (state, state_window, control_window) in enumerate(controller.calls):
        horizon = min(5, reference.num_controls - index)
        np.testing.assert_allclose(state, reference.states[index], rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(
            state_window,
            reference.states[index + 1 : index + 1 + horizon],
        )
        np.testing.assert_array_equal(
            control_window,
            reference.controls[index : index + horizon],
        )


def test_mppi_zero_perturbation_warm_start_stays_reference_aligned() -> None:
    system, config = _condition_config("double_integrator", "gaussian", "cpu")
    reference = _reference(system, config)
    controller = make_mppi_controller(
        system,
        reference,
        default_parameters("double_integrator"),
        config,
        seed=77,
    )
    controller.initialize_nominal(reference.controls[:5])

    def zero_perturbation_samples():
        samples = controller.control_sequence.unsqueeze(0).expand(
            controller.parameters.num_samples, -1, -1
        ).clone()
        return samples, torch.zeros_like(samples)

    controller.sample_controls = zero_perturbation_samples
    for index in range(reference.num_controls):
        horizon = min(5, reference.num_controls - index)
        command, diagnostic = controller.command_reference(
            reference.states[index],
            reference.states[index + 1 : index + 1 + horizon],
            reference.controls[index : index + horizon],
        )
        np.testing.assert_allclose(
            command, reference.controls[index], rtol=0.0, atol=1e-7
        )
        assert int(diagnostic["active_horizon"]) == horizon


def test_statistics_csv_tables_and_plots_on_tiny_dataset(tmp_path: Path) -> None:
    reference, open_result, aura_result, mppi_result = _run_forced_methods(noise=True)
    step_rows = []
    trial_rows = []
    for result in (open_result, aura_result, mppi_result):
        step_rows.extend(
            method_result_to_step_rows(reference=reference, result=result, trial=0, seed=42)
        )
        trial_rows.append(
            method_result_to_trial_row(reference=reference, result=result, trial=0, seed=42)
        )
    stats = compute_tracking_statistics(
        step_rows,
        trial_rows,
        num_controls=10,
        bootstrap_resamples=10_000,
        bootstrap_seed=123,
    )
    write_csv(tmp_path / "tracking_step_metrics.csv", step_rows)
    write_csv(tmp_path / "trial_tracking_metrics.csv", trial_rows)
    write_summary_tables(tmp_path, stats["summary"])
    paths = plot_tracking_statistics(stats["step_statistics"], tmp_path)
    assert len(read_csv(tmp_path / "tracking_step_metrics.csv")) == 30
    for name in (
        "tracking_summary.csv",
        "tracking_summary.tex",
        "tracking_summary.md",
    ):
        assert (tmp_path / name).is_file()
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
    assert (
        tmp_path
        / "tracking_error_by_step__double_integrator__gaussian.pdf"
    ).is_file()


def test_checkpoint_groups_raw_results_by_condition(tmp_path: Path) -> None:
    step_rows = [
        {"system": "double_integrator", "environment": "gaussian", "step": 1},
        {"system": "kinematic_car", "environment": "mujoco", "step": 1},
    ]
    trial_rows = [
        {"system": "double_integrator", "environment": "gaussian", "trial": 0},
        {"system": "kinematic_car", "environment": "mujoco", "trial": 0},
    ]
    references = [
        {"system": "double_integrator", "environment": "gaussian", "trial": 0},
        {"system": "kinematic_car", "environment": "mujoco", "trial": 0},
    ]

    _checkpoint(tmp_path, step_rows, trial_rows, references)

    for system, environment in (
        ("double_integrator", "gaussian"),
        ("kinematic_car", "mujoco"),
    ):
        condition_dir = condition_output_dir(tmp_path, system, environment)
        assert len(read_csv(condition_dir / "tracking_step_metrics.csv")) == 1
        assert len(read_csv(condition_dir / "trial_tracking_metrics.csv")) == 1
        assert (condition_dir / "reference_trajectories.jsonl").is_file()


def test_dry_run_does_not_replace_saved_results(tmp_path: Path) -> None:
    condition_dir = condition_output_dir(
        tmp_path, "double_integrator", "gaussian"
    )
    condition_dir.mkdir(parents=True)
    marker = condition_dir / "saved-result.csv"
    marker.write_text("keep\n", encoding="utf-8")

    assert deviation_main(
        [
            "--system",
            "double_integrator",
            "--environment",
            "gaussian",
            "--num-trials",
            "1",
            "--output-dir",
            str(tmp_path),
            "--dry-run",
            "--device",
            "cpu",
        ]
    ) == 0
    assert marker.read_text(encoding="utf-8") == "keep\n"
