from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts.plot_task_time import (
    BOX_CENTER_SPACING,
    BOXPLOT_WIDTH,
    PUBLICATION_FONT_SIZE,
    DEFAULT_METHOD_SEPARATORS,
    FIGURE_7_PANEL_ORDER,
    METHOD_GROUP_GAP,
    METHOD_LABELS,
    METHOD_LABEL_ROTATION,
    METHOD_LABEL_SIZE,
    METHOD_SEPARATOR_DASH_PATTERN,
    PANEL_TITLE_SIZE,
    PANEL_TITLE_Y,
    PANEL_SUBTITLE_Y,
    TICK_LABEL_SIZE,
    AXIS_LABEL_SIZE,
    PANEL_HEIGHT,
    PANEL_HORIZONTAL_PAD,
    PANEL_WSPACE,
    PANEL_WIDTH,
    group_positions,
    horizontal_limits,
    method_boundaries,
    PANEL_PLOT_TIME_CAP_SECONDS,
    Y_AXIS_TICK_COUNT,
    aggregate,
    collect_rows,
    create_figure,
    draw_panel,
    expected_mppi_keys,
    figure_panels,
    paired_comparisons,
    paired_success_tracking_summary,
    validate_matrix,
    validate_real_world_matrix,
    write_outputs,
)
from scripts.plot_trajectory_cost import _read_planner_rows
from experiment.task_time_efficiency import (
    plan_until_solution,
)
from utils.experiment_io import (
    METHODS,
    PLANNERS,
    SIMULATION_PANEL_ORDER,
    import_successful_results,
    job_complete,
    legacy_time_correction,
    result_path,
    serialize_result,
    task_time_seconds,
    validate_result,
    write_paired_results,
)


def _manifest() -> dict:
    panels = []
    for panel_id in (*SIMULATION_PANEL_ORDER, "pushing_real"):
        real = panel_id == "pushing_real"
        panels.append(
            {
                "execute": not real,
                "config_hash": f"hash-{panel_id}",
                "config": {
                    "panel_id": panel_id,
                    "panel_title": panel_id.replace("_", " ").title(),
                    "panel_subtitle": "Real-World Hardware" if real else "Simulation",
                    "simulator_mode": (
                        "real"
                        if real
                        else ("mujoco" if "mujoco" in panel_id else "gaussian")
                    ),
                },
            }
        )
    return {
        "manifest_id": "synthetic",
        "num_simulation_trials": 100,
        "selected_runs": [1],
        "num_real_trials": 20,
        "planners": list(PLANNERS),
        "methods": list(METHODS),
        "panels": panels,
    }


def _row(panel_id: str, planner: str, method: str, config_hash: str) -> dict:
    state_dim = 6 if panel_id.startswith("double") else 3
    return {
        "schema_version": 3,
        "panel_id": panel_id,
        "system": (
            "double_integrator"
            if panel_id.startswith("double")
            else (
                "kinematic_car"
                if panel_id.startswith("kinematic")
                else "pushing_object"
            )
        ),
        "environment": "mujoco" if "mujoco" in panel_id else "gaussian",
        "planner": planner,
        "method": method,
        "run_number": 1,
        "seed": 101,
        "rng_streams": {"execution": 7},
        "config_hash": config_hash,
        "initial_planning_seconds": 1.0,
        "initial_plan_hash": "same-plan",
        "status": "success",
        "failure_reason": "",
        "nominal_execution_seconds": 5.0,
        "actual_execution_seconds": 0.01,
        "online_replanning_seconds": 4.9 if method == "aura" else 0.0,
        "optimizer_seconds": 4.8 if method == "aura" else 0.0,
        "blocking_replanning_seconds": 0.0 if method == "aura" else 5.0,
        "compute_overrun_seconds": 0.0,
        "wall_time_definition": (
            (
                "initial_offline_planning + nominal_control_execution "
                "+ blocking_aura_restart_planning"
            )
            if method == "aura"
            else (
                "initial_offline_planning + nominal_control_execution "
                "+ blocking_restart_replanning"
            )
        ),
        "task_time_seconds": 6.0 if method == "aura" else 11.0,
        "raw_process_wall_seconds": 0.1,
        "setup_reset_operator_pause_seconds": 0.01,
        "num_controls": 1,
        "num_replanning": 1,
        "cost": 0.5,
        "tracking_error_mean": 0.02,
        "tracking_error_list": [0.02],
        "goal_distance": 0.01,
        "final_state": [0.0] * state_dim,
        "planned_final_state": [0.0] * state_dim,
        "controls": [[0.1, 0.0]],
        "control_duration_steps": [5],
        "control_duration_seconds": [5.0],
        "duration_audit_initial_tree": {
            "duration_step_histogram": {str(k): 1 for k in range(1, 6)},
            "range_steps": [1, 5],
            "propagation_step_size_seconds": 1.0,
        },
        "disturbance_schedule_hash": "paired-noise",
    }


def _write_complete_synthetic(root: Path) -> dict:
    manifest = _manifest()
    (root / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    panel_hash = {
        panel["config"]["panel_id"]: panel["config_hash"]
        for panel in manifest["panels"]
    }
    for panel in SIMULATION_PANEL_ORDER:
        for planner in PLANNERS:
            rows = [
                _row(panel, planner, method, panel_hash[panel])
                for method in METHODS
            ]
            write_paired_results(result_path(root, panel, planner, 1), rows)
    return manifest


def test_complete_matrix_validates_and_aggregates_compact_csv(tmp_path: Path) -> None:
    manifest = _write_complete_synthetic(tmp_path)
    rows = collect_rows(tmp_path, manifest)
    validation = validate_matrix(rows, manifest, require_complete_simulation=True)
    assert validation["complete"] is True
    assert validation["observed_simulation_method_rows"] == (
        len(SIMULATION_PANEL_ORDER) * len(PLANNERS) * len(METHODS)
    )
    summary = aggregate(rows, manifest)
    paired = paired_comparisons(rows)
    paired_tracking = paired_success_tracking_summary(rows)
    assert len(summary["groups"]) == (
        len(SIMULATION_PANEL_ORDER) * len(PLANNERS) * len(METHODS)
    )
    assert len(paired) == len(SIMULATION_PANEL_ORDER) * len(PLANNERS)
    assert paired_tracking[0]["scope"] == "overall"
    assert paired_tracking[0]["n_paired_successes"] == (
        len(SIMULATION_PANEL_ORDER) * len(PLANNERS)
    )
    assert (
        paired_tracking[0]["aura_step_weighted_tracking_error"]
        == paired_tracking[0]["replanning_step_weighted_tracking_error"]
    )
    assert all(row["replanning_minus_aura_time"] == 5.0 for row in paired)
    output = tmp_path / "aggregate"
    write_outputs(output, validation, summary, paired, paired_tracking)
    assert (output / "summary.csv").is_file()
    assert (output / "paired_comparisons.csv").is_file()
    assert (output / "paired_success_tracking.csv").is_file()


def test_complete_matrix_accepts_one_standalone_mppi_row_per_panel(
    tmp_path: Path,
) -> None:
    manifest = _write_complete_synthetic(tmp_path)
    panel_hashes = {
        panel["config"]["panel_id"]: panel["config_hash"]
        for panel in manifest["panels"]
    }
    for panel in SIMULATION_PANEL_ORDER:
        row = _row(panel, "mppi", "mppi", panel_hashes[panel])
        row["initial_planning_seconds"] = 0.0
        row["optimizer_seconds"] = 0.25
        row["blocking_replanning_seconds"] = 0.0
        row["wall_time_definition"] = (
            "blocking_mppi_optimization + nominal_control_execution"
        )
        row["task_time_seconds"] = 5.25
        row["control_duration_steps"] = [1]
        row["control_duration_seconds"] = [1.0]
        row["duration_audit_initial_tree"] = {
            "duration_step_histogram": {},
            "range_steps": [1, 1],
            "propagation_step_size_seconds": 1.0,
        }
        write_paired_results(result_path(tmp_path, panel, "mppi", 1), [row])
    rows = collect_rows(tmp_path, manifest)
    validation = validate_matrix(rows, manifest, require_complete_simulation=True)
    assert validation["complete"] is True
    assert validation["mppi_expected_rows"] == len(SIMULATION_PANEL_ORDER)
    assert validation["mppi_observed_rows"] == len(SIMULATION_PANEL_ORDER)
    assert validation["observed_simulation_method_rows"] == (
        len(SIMULATION_PANEL_ORDER) * (len(PLANNERS) * len(METHODS) + 1)
    )
    assert len(paired_comparisons(rows)) == (
        len(SIMULATION_PANEL_ORDER) * len(PLANNERS)
    )
    assert len(aggregate(rows)["groups"]) == (
        len(SIMULATION_PANEL_ORDER) * (len(PLANNERS) * len(METHODS) + 1)
    )


def test_mppi_matrix_can_use_independent_runs_per_panel() -> None:
    manifest = _manifest()
    manifest["mppi_selected_runs_by_panel"] = {
        "double_integrator_gaussian": [101],
        "pushing_gaussian": [201],
    }
    keys = expected_mppi_keys(manifest)
    assert (
        "double_integrator_gaussian",
        "mppi",
        101,
        "mppi",
    ) in keys
    assert ("pushing_gaussian", "mppi", 201, "mppi") in keys
    assert (
        "kinematic_car_gaussian",
        "mppi",
        1,
        "mppi",
    ) in keys


def test_failure_is_penalized_at_panel_maximum() -> None:
    row = _row(
        "pushing_gaussian", "aorrt", "restartReplanning", "hash"
    )
    row["status"] = "failure"
    row["failure_reason"] = "no_solution"
    row["task_time_seconds"] = 3.0
    row["num_controls"] = 4
    csv_row = serialize_result(row)
    # The historical paper plot added 2*num_controls for fake pushing. The
    # value therefore reconstructs exactly the shared 300-second task cap.
    assert csv_row["time"] + 2 * csv_row["num_controls"] == 300.0
    assert csv_row["plot_time_seconds"] == 300.0
    assert csv_row["overall_time"] == 3.0


def test_serialize_result_accepts_numpy_state_and_control_arrays() -> None:
    row = _row("pushing_real", "mppi", "mppi", "hash-pushing_real")
    row["final_state"] = np.array([0.38524, -0.59372, 0.04095])
    row["planned_final_state"] = np.array([0.36, -0.59, 0.13])
    row["controls"] = [
        np.array([0.5, -0.055, 0.101]),
        np.array([0.5, 0.009, 0.115]),
    ]
    row["tracking_error_list"] = np.array([0.02, 0.03])

    csv_row = serialize_result(row)

    assert csv_row["final_state_x"] == pytest.approx(0.38524)
    assert csv_row["final_state_y"] == pytest.approx(-0.59372)
    assert csv_row["final_state_theta"] == pytest.approx(0.04095)
    assert csv_row["planned_final_theta"] == pytest.approx(0.13)
    assert csv_row["controls"] == "0.5;-0.055;0.101|0.5;0.009;0.115"
    assert csv_row["tracking_errors"] == "0.02;0.03"


def test_plot_caps_are_panel_specific_and_do_not_change_raw_csv_time(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    panel = "pushing_gaussian"
    row = _row(panel, "mppi", "mppi", f"hash-{panel}")
    row["status"] = "timeout"
    row["failure_reason"] = "task_time_limit"
    row["task_time_seconds"] = 300.0
    row["control_duration_steps"] = [1]
    row["control_duration_seconds"] = [1.0]
    write_paired_results(result_path(tmp_path, panel, "mppi", 1), [row])

    collected = collect_rows(tmp_path, manifest)
    assert len(collected) == 1
    assert collected[0]["overall_time"] == 300.0
    assert collected[0]["plot_time_seconds"] == 300.0
    assert collected[0]["display_time_seconds"] == 100.0
    group = aggregate(collected)["groups"][0]
    assert group["plot_cap_seconds"] == 100.0
    assert group["penalized_time"]["mean"] == 100.0


def test_requested_panel_plot_caps() -> None:
    assert PANEL_PLOT_TIME_CAP_SECONDS == {
        "double_integrator_gaussian": 250.0,
        "kinematic_car_gaussian": 300.0,
        "pushing_gaussian": 100.0,
        "dubins_airplane_gaussian": 300.0,
        "kinematic_car_mujoco": 250.0,
        "pushing_mujoco": 200.0,
        "pushing_real": 300.0,
    }


def test_figure_7_includes_all_simulation_panels_and_real_world() -> None:
    assert FIGURE_7_PANEL_ORDER == (
        "double_integrator_gaussian",
        "kinematic_car_gaussian",
        "dubins_airplane_gaussian",
        "pushing_gaussian",
        "kinematic_car_mujoco",
        "pushing_mujoco",
        "pushing_real",
    )
    rows = [
        {"panel_id": panel}
        for panel in (*SIMULATION_PANEL_ORDER, "pushing_real")
    ]
    assert figure_panels(rows) == list(FIGURE_7_PANEL_ORDER)
    assert "dubins_airplane_gaussian" in figure_panels(rows)


def test_figure_7_restores_plot_exp2_style_and_one_row(tmp_path: Path) -> None:
    _write_complete_synthetic(tmp_path)
    paths = create_figure(tmp_path, include_empty_panels=True)
    image = plt.imread(next(path for path in paths if path.suffix == ".png"))

    assert image.shape[1] > image.shape[0] * 2.2
    assert plt.rcParams["font.serif"][0] == "Times New Roman"
    assert METHOD_LABELS["randup"] == "RobRRT"
    assert BOXPLOT_WIDTH == pytest.approx(0.34)
    assert BOX_CENTER_SPACING == pytest.approx(0.38)
    assert BOX_CENTER_SPACING > BOXPLOT_WIDTH
    assert METHOD_GROUP_GAP == pytest.approx(0.15)
    assert METHOD_LABEL_SIZE == PUBLICATION_FONT_SIZE
    assert PANEL_TITLE_SIZE == PUBLICATION_FONT_SIZE
    assert TICK_LABEL_SIZE == PUBLICATION_FONT_SIZE
    assert AXIS_LABEL_SIZE == PUBLICATION_FONT_SIZE
    assert METHOD_LABEL_ROTATION == 90
    assert METHOD_SEPARATOR_DASH_PATTERN == (4.0, 6.0)
    assert DEFAULT_METHOD_SEPARATORS is True
    assert Y_AXIS_TICK_COUNT == 6
    assert PANEL_WIDTH == pytest.approx(2.0)
    assert PANEL_HEIGHT == pytest.approx(4.2)
    assert PANEL_HORIZONTAL_PAD == pytest.approx(0.05)
    assert PANEL_WSPACE == pytest.approx(0.30)
    assert PANEL_TITLE_Y > PANEL_SUBTITLE_Y


def test_panel_uses_six_y_labels_and_no_box_tick_marks() -> None:
    panel = "double_integrator_gaussian"
    rows = [
        _row(panel, planner, method, f"hash-{panel}")
        for planner in PLANNERS
        for method in METHODS
    ]
    figure, axis = plt.subplots()

    draw_panel(
        axis,
        rows,
        panel,
        show_ylabel=True,
        dash=True,
        black=False,
    )

    assert len(axis.get_yticks()) == Y_AXIS_TICK_COUNT
    assert all(tick.tick1line.get_markersize() == 0 for tick in axis.xaxis.majorTicks)
    labels = {
        text.get_text(): text
        for text in axis.texts
        if text.get_text() in METHOD_LABELS.values()
    }
    assert labels
    assert all(text.get_rotation() == METHOD_LABEL_ROTATION for text in labels.values())
    aura_positions = [
        patch.get_path().vertices[:, 0].mean()
        for patch in axis.patches[:3]
    ]
    rr_positions = [
        patch.get_path().vertices[:, 0].mean()
        for patch in axis.patches[3:6]
    ]
    assert np.diff(aura_positions) == pytest.approx(np.diff(rr_positions))
    plt.close(figure)


def test_single_method_boxes_are_centered_in_their_sections() -> None:
    groups = [
        ((method, planner), [1.0])
        for method, planner in (
            ("aura", "aorrt"),
            ("aura", "aoest"),
            ("aura", "sststar"),
            ("replanning", "aorrt"),
            ("replanning", "aoest"),
            ("replanning", "sststar"),
            ("mppi", "mppi"),
            ("randup", "randup"),
        )
    ]
    positions, method_positions = group_positions(groups)
    boundaries = method_boundaries(method_positions)
    _, right = horizontal_limits(positions, method_positions, boundaries)

    assert method_positions["mppi"][0] == pytest.approx(
        (boundaries[-2] + boundaries[-1]) / 2.0
    )
    assert method_positions["randup"][0] == pytest.approx(
        (boundaries[-1] + right) / 2.0
    )


def test_preserved_real_world_csv_schema_is_plotable(tmp_path: Path) -> None:
    panel = tmp_path / "pushing_real"
    panel.mkdir()
    (panel / "pushing_aorrt_01.csv").write_text(
        "run_number,planner,method,time,cost,actual_final_x,actual_final_y,"
        "planned_final_x,planned_final_y,num_controls\n"
        "1,aorrt,aura,191.0,2.2,-0.30,-0.72,-0.29,-0.71,8\n"
        "1,aorrt,replanning,222.0,4.3,-0.29,-0.69,-0.29,-0.71,10\n",
        encoding="utf-8",
    )

    rows = collect_rows(tmp_path)

    assert len(rows) == 2
    assert {row["panel_id"] for row in rows} == {"pushing_real"}
    assert {row["method"] for row in rows} == {"aura", "restartReplanning"}
    assert all(row["status"] == "success" for row in rows)
    assert all(row["display_time_seconds"] < 300.0 for row in rows)


def test_preserved_real_world_run_number_comes_from_filename(tmp_path: Path) -> None:
    panel = tmp_path / "pushing_real"
    panel.mkdir()
    (panel / "pushing_aorrt_11.csv").write_text(
        "run_number,planner,method,time\n"
        "1,aorrt,aura,200.0\n"
        "1,aorrt,replanning,210.0\n",
        encoding="utf-8",
    )

    assert {row["run_number"] for row in collect_rows(tmp_path)} == {11}


def test_real_world_plot_accepts_independent_trial_counts() -> None:
    rows = [
        {
            "panel_id": "pushing_real",
            "planner": method,
            "method": method,
            "run_number": run,
        }
        for method, count in (("mppi", 7), ("randup_rrt_m50", 3))
        for run in range(1, count + 1)
    ]
    validation = validate_real_world_matrix(rows)
    assert validation["real_world_mppi_observed_trials"] == 7
    assert validation["real_world_randup_observed_trials"] == 3
    assert validation["real_world_mppi_run_numbers"] == list(range(1, 8))
    assert validation["real_world_randup_run_numbers"] == [1, 2, 3]


def test_legacy_plot_does_not_apply_aura_pushing_correction_to_mppi() -> None:
    assert legacy_time_correction("pushing_gaussian", "mppi", 7) == 0.0


def test_wall_time_contract_adds_blocking_aura_restart_planning() -> None:
    assert task_time_seconds(
        "aura",
        initial_planning_seconds=4.0,
        nominal_execution_seconds=17.0,
        blocking_replanning_seconds=99.0,
    ) == 120.0


def test_wall_time_contract_adds_only_blocking_restart_replanning() -> None:
    assert task_time_seconds(
        "restartReplanning",
        initial_planning_seconds=4.0,
        nominal_execution_seconds=17.0,
        blocking_replanning_seconds=13.5,
    ) == 34.5


def test_result_validation_rejects_background_overrun_in_task_time() -> None:
    row = _row("double_integrator_gaussian", "aorrt", "aura", "hash")
    row["compute_overrun_seconds"] = 3.0
    validate_result(row, expected_config_hash="hash")
    row["task_time_seconds"] += row["compute_overrun_seconds"]
    with pytest.raises(ValueError, match="wall-time accounting contract"):
        validate_result(row, expected_config_hash="hash")


def test_missing_matrix_fails_strict_validation(tmp_path: Path) -> None:
    manifest = _write_complete_synthetic(tmp_path)
    result_path(tmp_path, SIMULATION_PANEL_ORDER[0], PLANNERS[0], 1).unlink()
    rows = collect_rows(tmp_path, manifest)
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_matrix(rows, manifest, require_complete_simulation=True)


def test_incomplete_matrix_is_plotable_by_default(tmp_path: Path) -> None:
    manifest = _write_complete_synthetic(tmp_path)
    result_path(tmp_path, SIMULATION_PANEL_ORDER[0], PLANNERS[0], 1).unlink()
    validation = validate_matrix(
        collect_rows(tmp_path, manifest),
        manifest,
        require_complete_simulation=False,
    )
    assert validation["complete"] is False


def test_trajectory_plot_discovers_unequal_available_run_counts(
    tmp_path: Path,
) -> None:
    for planner, runs in (("aorrt", (1, 4)), ("aoest", (2,))):
        for run in runs:
            (tmp_path / f"{planner}_{run:03d}.csv").write_text(
                "planner,planning_time,status,initial_cost,final_cost\n"
                f"{planner},1.0,success,2.0,1.0\n",
                encoding="utf-8",
            )
    rows = _read_planner_rows(tmp_path)
    assert len(rows) == 3


def test_resume_only_accepts_valid_hash_and_paired_methods(tmp_path: Path) -> None:
    panel = SIMULATION_PANEL_ORDER[0]
    planner = PLANNERS[0]
    write_paired_results(
        result_path(tmp_path, panel, planner, 1),
        [_row(panel, planner, method, "correct") for method in METHODS],
    )
    assert job_complete(
        tmp_path,
        panel,
        planner,
        1,
        list(METHODS),
        expected_config_hash="correct",
    )
    assert not job_complete(
        tmp_path,
        panel,
        planner,
        1,
        list(METHODS),
        expected_config_hash="wrong",
    )


def test_initial_planning_continues_same_tree_until_solution() -> None:
    solution = {
        "controls": [[0.1, 0.0]],
        "states": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    }

    class Planner:
        initial_planning_time = 1.0

        def __init__(self):
            self.replan_calls = 0

        def plan(self):
            return [], None

        def retry_initial_plan(self, *, time_budget):
            assert time_budget > 0.0
            self.replan_calls += 1
            return ([solution] if self.replan_calls == 2 else []), None

    planner = Planner()
    observed, elapsed, attempts, error = plan_until_solution(
        planner,
        attempt_budget_seconds=1.0,
        total_budget_seconds=10.0,
    )
    assert observed is solution
    assert elapsed >= 0.0
    assert attempts == 3
    assert planner.replan_calls == 2
    assert error == ""


def test_initial_planning_stops_pathological_immediate_retry_spin() -> None:
    class Planner:
        initial_planning_time = 1.0

        def plan(self):
            return [], None

        def retry_initial_plan(self, *, time_budget):
            assert time_budget > 0.0
            return [], None

    observed, elapsed, attempts, error = plan_until_solution(
        Planner(),
        attempt_budget_seconds=1.0,
        total_budget_seconds=10.0,
    )
    assert observed is None
    assert elapsed < 1.0
    assert attempts == 3
    assert error == "planner_returned_immediately_without_exact_solution"


def test_initial_time_control_duration_is_a_maximum() -> None:
    from experiment.initial_time_sensitivity import trial_config

    observed = trial_config(
        {
            "motion_validation_step_size": 1.0,
            "minimum_control_duration_steps": 1,
        },
        duration=5.0,
        seed=7,
    )

    assert observed["propagation_step_size"] == 1.0
    assert observed["min_control_duration"] == 1
    assert observed["max_control_duration"] == 5


def test_import_successes_preserves_only_successful_method(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    manifest = _manifest()
    panel = SIMULATION_PANEL_ORDER[0]
    planner = PLANNERS[0]
    aura = _row(panel, planner, "aura", "old-hash")
    replanning = _row(
        panel, planner, "restartReplanning", "old-hash"
    )
    replanning["status"] = "failure"
    replanning["failure_reason"] = "old_one_shot_failure"
    write_paired_results(
        result_path(source, panel, planner, 1),
        [aura, replanning],
    )
    imported = import_successful_results(source, destination, manifest)
    assert imported == 1
    assert job_complete(
        destination,
        panel,
        planner,
        1,
        ["aura"],
        expected_config_hash=f"hash-{panel}",
        accepted_statuses={"success", "timeout"},
    )
    assert not job_complete(
        destination,
        panel,
        planner,
        1,
        ["restartReplanning"],
        expected_config_hash=f"hash-{panel}",
        accepted_statuses={"success", "timeout"},
    )
