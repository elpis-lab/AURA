from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.plot_task_time import aggregate, collect_rows, validate_matrix
from experiment.task_time_efficiency import (
    aggregate_randup_rows,
    run_randup_trial,
)
from utils.experiment_io import result_path, write_paired_results, write_randup_result
from scripts.plot_task_time import create_figure


def _existing_row(method: str) -> dict:
    return {
        "schema_version": 3,
        "panel_id": "kinematic_car_gaussian",
        "system": "kinematic_car",
        "environment": "gaussian",
        "planner": "aorrt",
        "method": method,
        "run_number": 1,
        "seed": 7,
        "rng_streams": {"execution": 8},
        "config_hash": "panel-hash",
        "initial_planning_seconds": 1.0,
        "initial_plan_hash": "actual-plan",
        "initial_plan": {"duration_steps": [1], "duration_seconds": [1.0]},
        "status": "success",
        "failure_reason": "",
        "nominal_execution_seconds": 1.0,
        "actual_execution_seconds": 0.01,
        "online_replanning_seconds": 0.0,
        "optimizer_seconds": 0.0,
        "blocking_replanning_seconds": 0.0,
        "compute_overrun_seconds": 0.0,
        "wall_time_definition": "test",
        "task_time_seconds": 2.0,
        "raw_process_wall_seconds": 0.1,
        "setup_reset_operator_pause_seconds": 0.0,
        "num_controls": 1,
        "num_replanning": 0,
        "cost": 1.0,
        "tracking_error_mean": 0.01,
        "tracking_error_list": [0.01],
        "goal_distance": 0.05,
        "final_state": [3.0, 3.0, 1.57],
        "planned_final_state": [3.0, 3.0, 1.57],
        "controls": [[0.5, 0.1]],
        "control_duration_steps": [1],
        "control_duration_seconds": [1.0],
        "duration_audit_initial_tree": {
            "range_steps": [1, 5],
            "propagation_step_size_seconds": 1.0,
            "duration_step_histogram": {"1": 1},
        },
        "disturbance_schedule_hash": "shared",
    }


def _randup_row(*, planning_success: bool = True) -> dict:
    success = bool(planning_success)
    return {
        **_existing_row("oracle_randup_rrt"),
        "schema_version": 1,
        "planner": "randup_rrt_m10_u1",
        "method": "oracle_randup_rrt",
        "method_label": "Oracle RandUP-RRT",
        "planning_seed": 9,
        "execution_seed": 10,
        "uncertainty_level": 1.0,
        "planning_success": success,
        "execution_success": success,
        "collision": False,
        "status": "success" if success else "failure",
        "failure_reason": "" if success else "planning_timeout",
        "last_expansion_rejection": "particle_collision_during_expansion",
        "execution_time_seconds": 1.0 if success else 0.0,
        "minimum_obstacle_clearance": None,
        "tree_expansions": 12,
        "tree_size": 10,
        "dynamics_propagations": 100,
        "particle_propagations": 80,
        "particle_collision_rejections": 2,
        "nominal_collision_rejections": 1,
        "numerical_propagation_failures": 0,
        "num_particles": 10,
        "padding_epsilon": 0.02,
        "uncertainty_mode": "gaussian_process_oracle",
        "goal_requires_all_particles": True,
        "executed_primitive_steps": 1 if success else 0,
        "route_signature": "not_applicable",
        "condition_hash": "condition",
        "source_inventory": {},
        "planner_stats": {},
        "condition": {},
        "primitive_states": [[-0.2, 0.0, 0.0]],
    }


def test_randup_writer_and_task_time_plot_integrate(tmp_path: Path) -> None:
    manifest = {
        "selected_runs": [1],
        "num_simulation_trials": 1,
        "planners": ["aorrt"],
        "methods": ["aura", "restartReplanning"],
        "panels": [
            {
                "execute": True,
                "config_hash": "panel-hash",
                "config": {
                    "panel_id": "kinematic_car_gaussian",
                    "simulator_mode": "gaussian",
                    "obstacles": {"enabled": False},
                },
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    write_paired_results(
        result_path(tmp_path, "kinematic_car_gaussian", "aorrt", 1),
        [_existing_row("aura"), _existing_row("restartReplanning")],
    )
    artifact = write_randup_result(tmp_path, _randup_row())
    assert artifact == (
        tmp_path
        / "kinematic_car_gaussian"
        / "artifacts"
        / "randup_rrt_m10_u1_trial-001.json"
    )
    assert (tmp_path / "randup_trials.csv").is_file()

    rows = collect_rows(tmp_path)
    validation = validate_matrix(rows, manifest, require_complete_simulation=True)
    assert validation["complete"]
    groups = aggregate(rows)["groups"]
    assert any(group["method"] == "oracle_randup_rrt" for group in groups)

    paths = create_figure(tmp_path)
    assert all(path.is_file() for path in paths)


def test_randup_aggregate_keeps_failures_and_reports_valid_counts() -> None:
    success = _randup_row(planning_success=True)
    failure = _randup_row(planning_success=False)
    failure["run_number"] = 2
    failure["cost"] = None
    rows = aggregate_randup_rows([success, failure])
    assert len(rows) == 1
    assert rows[0]["n_trials"] == 2
    assert rows[0]["planning_success_rate"] == 0.5
    assert rows[0]["overall_task_success_rate"] == 0.5
    assert rows[0]["cost_valid_n"] == 1


def test_randup_aggregate_accepts_legacy_real_world_rows() -> None:
    row = _existing_row("randup_rrt")
    row.update(
        {
            "panel_id": "pushing_real",
            "planner": "randup_rrt_m50",
            "initial_plan": {"controls": [[0.5, 0.0, 0.1]]},
        }
    )

    summary = aggregate_randup_rows([row])[0]

    assert summary["method"] == "RandUP-RRT"
    assert summary["num_particles"] == 50
    assert summary["n_plans_found"] == 1
    assert summary["n_execution_success"] == 1


def test_failed_randup_legacy_row_preserves_raw_and_capped_times(tmp_path: Path) -> None:
    row = _randup_row(planning_success=False)
    row["task_time_seconds"] = 2.5
    write_randup_result(tmp_path, row)

    path = result_path(
        tmp_path,
        "kinematic_car_gaussian",
        "randup_rrt_m10_u1",
        1,
    )
    with path.open(newline="", encoding="utf-8") as stream:
        written = next(csv.DictReader(stream))

    assert float(written["time"]) == 300.0
    assert float(written["plot_time_seconds"]) == 300.0
    assert float(written["overall_time"]) == 2.5


def test_randup_trial_adds_fresh_replanning_to_task_time(monkeypatch) -> None:
    solutions = [
        {
            "states": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            "controls": [[0.5, 0.0]],
            "time_steps": [1],
            "time": [1.0],
            "cost": 1.0,
        },
        {
            "states": [[0.4, 0.0, 0.0], [2.0, 0.0, 0.0]],
            "controls": [[0.5, 0.0]],
            "time_steps": [1],
            "time": [1.0],
            "cost": 1.0,
        },
    ]
    planning_times = [1.25, 0.75]

    class Internal:
        def __init__(self, planning_time: float):
            self.failure_reason = ""
            self.last_expansion_rejection = ""
            self.stats = {"planning_time": planning_time, "tree_size": 2}

    class Planner:
        def __init__(self, index: int):
            self.index = index
            self.randup_planner = Internal(planning_times[index])

        def plan(self):
            return [solutions[self.index]], None

    planner_count = {"value": 0}
    observed_budgets = []

    def make_planner(*args, **kwargs):
        index = planner_count["value"]
        planner_count["value"] += 1
        observed_budgets.append(float(args[2].planning_time))
        return Planner(index)

    class Simulator:
        def __init__(self):
            self.states = iter(([0.4, 0.0, 0.0], [2.0, 0.0, 0.0]))

        def reset(self):
            return [0.0, 0.0, 0.0]

        def set_state(self, pose):
            return pose

        def execute_segment(self, control, duration):
            return next(self.states)

        def close(self):
            return None

    monkeypatch.setattr(
        "experiment.task_time_efficiency._planner_from_config", make_planner
    )
    monkeypatch.setattr(
        "experiment.task_time_efficiency.create_simulator",
        lambda *args, **kwargs: Simulator(),
    )
    config = {
        "panel_id": "kinematic_car_gaussian",
        "system_name": "kinematic_car",
        "simulator_mode": "gaussian",
        "start_state": [0.0, 0.0, 0.0],
        "goal_state": [2.0, 0.0, 0.0],
        "state_bounds": [[-10.0, 10.0], [-10.0, 10.0]],
        "goal_threshold": 0.1,
        "planning_time": 5.0,
        "propagation_step_size": 1.0,
        "min_control_duration": 1,
        "max_control_duration": 1,
        "max_steps": 10,
        "task_time_limit_seconds": 20.0,
        "sampling_position_std": 0.01,
        "sampling_rotation_std": 0.01,
        "obstacles": {"enabled": False},
    }
    row = run_randup_trial(
        {"config": config, "config_hash": "test-hash"},
        1,
        base_seed=9,
        num_particles=2,
        uncertainty_level=1.0,
    )
    assert row["status"] == "success"
    assert row["num_replanning"] == 1
    assert row["initial_planning_seconds"] == 1.25
    assert row["blocking_replanning_seconds"] == 0.75
    assert row["execution_time_seconds"] == 2.0
    assert row["task_time_seconds"] == 4.0
    assert observed_budgets == [5.0, 5.0]
