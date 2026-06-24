from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from utils.utils import arrayDistance, sample_control_curve, state2list


def wrap_to_pi(angle: float) -> float:
    return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi


def se2_breakdown(a: np.ndarray, b: np.ndarray, system_name: str) -> tuple[float, float, float]:
    """OMPL SE(2) distance plus planar and yaw deltas (yaw in [-pi, pi])."""
    aa = np.asarray(a, dtype=float).reshape(-1)
    bb = np.asarray(b, dtype=float).reshape(-1)
    d_ompl = float(arrayDistance(aa, bb, system=system_name))
    dxy = float(np.linalg.norm(aa[:2] - bb[:2]))
    dth = wrap_to_pi(float(aa[2] - bb[2]))
    return d_ompl, dxy, dth


def fmt_scalar(x: float | None, *, prec: int = 6) -> str:
    if x is None:
        return "—"
    xf = float(x)
    if math.isnan(xf):
        return "—"
    ax = abs(xf)
    if ax != 0 and (ax < 1e-3 or ax >= 1e4):
        return f"{xf:.{prec}e}"
    return f"{xf:.{prec}f}"


def print_step_table(step: int, rows: list[tuple[str, str]]) -> None:
    width = 82
    bar = "=" * width
    print()
    print(bar)
    print(f" AURA step {step} ".center(width, "="))
    print(bar)
    if not rows:
        print(bar)
        print()
        return
    c0 = min(max(len(r[0]) for r in rows) + 1, 34)
    for label, value in rows:
        if "\n" in value:
            print(f"  {label:<{c0}}")
            for line in value.split("\n"):
                print(f"  {'':<{c0}}  {line}")
        else:
            print(f"  {label:<{c0}}  {value}")
    print(bar)
    print()


def snapshot_plan(plan_dict: dict) -> dict:
    """Immutable-ish copy for logging/visualization of the current best plan."""
    snap = dict(plan_dict)
    if "states" in snap and snap["states"] is not None:
        snap["states"] = [np.asarray(s, dtype=float).copy() for s in snap["states"]]
    if "_display_reference_states" in snap and snap["_display_reference_states"] is not None:
        snap["_display_reference_states"] = [
            np.asarray(s, dtype=float).copy()
            for s in snap["_display_reference_states"]
        ]
    if "controls" in snap and snap["controls"] is not None:
        snap["controls"] = [np.asarray(u, dtype=float).copy() for u in snap["controls"]]
    if "time" in snap and snap["time"] is not None:
        snap["time"] = [float(t) for t in snap["time"]]
    return snap


def align_plan_root_to_pose(plan: dict, pose: np.ndarray) -> dict:
    """Set plan.states[0] to the true pose so tracking metrics match execution."""
    if not plan or not plan.get("states"):
        return plan
    out = dict(plan)
    states = [np.asarray(s, dtype=float).copy() for s in plan["states"]]
    states[0] = np.asarray(pose, dtype=float).reshape(-1).copy()
    out["states"] = states
    return out


def plan_control_duration(
    plan: dict,
    control_index: int = 0,
    fallback: float | None = None,
    *,
    default_duration: float,
) -> float:
    """Return the OMPL path duration for a control, falling back to the step size."""
    default = default_duration if fallback is None else fallback
    times = plan.get("time") or plan.get("durations") or []
    if control_index < len(times):
        try:
            duration = float(times[control_index])
            if duration > 0.0:
                return duration
        except (TypeError, ValueError):
            pass
    return float(default)


def suffix_plan_from_index(
    plan: dict,
    start_index: int,
    pose: np.ndarray,
    *,
    continuity_dist: float,
    source: str,
    propagation_step_size: float,
) -> dict | None:
    states_raw = plan.get("states") or []
    controls_raw = plan.get("controls") or []
    times_raw = plan.get("time") or []
    if start_index < 0 or start_index >= len(controls_raw):
        return None
    out = dict(plan)
    reference_states = [np.asarray(s, dtype=float).copy() for s in states_raw]
    states = [np.asarray(s, dtype=float).copy() for s in states_raw[start_index:]]
    controls = [np.asarray(u, dtype=float).copy() for u in controls_raw[start_index:]]
    times = [float(t) for t in times_raw[start_index:]] if times_raw else []
    if not states or not controls:
        return None
    states[0] = np.asarray(pose, dtype=float).reshape(-1).copy()
    out["states"] = states
    out["controls"] = controls
    out["time"] = times
    out["state_count"] = len(states)
    out["control_count"] = len(controls)
    duration_cost = float(np.sum(times)) if times else float(len(controls)) * float(propagation_step_size)
    out["cost"] = duration_cost + float(continuity_dist)
    out["_continuity_dist"] = float(continuity_dist)
    out["_trim_index"] = int(start_index)
    out["_selection_source"] = source
    out["_display_reference_states"] = reference_states
    return out


def reachable_plan_candidates(
    solutions: list[dict],
    previous_plan: dict,
    pose: np.ndarray,
    *,
    system_name: str,
    propagation_step_size: float,
    continuity_max_distance: float,
) -> tuple[list[dict], dict]:
    max_dist = float(continuity_max_distance)
    candidates: list[dict] = []
    rejected = 0
    best_rejected_dist = float("inf")

    for solution in solutions or []:
        states = solution.get("states") or []
        controls = solution.get("controls") or []
        n = min(len(states) - 1, len(controls))
        if n <= 0:
            rejected += 1
            continue
        dists = [
            float(arrayDistance(pose, states[i], system=system_name))
            for i in range(n)
        ]
        idx = int(np.argmin(dists))
        closest = float(dists[idx])
        if closest <= max_dist:
            candidate = suffix_plan_from_index(
                solution,
                idx,
                pose,
                continuity_dist=closest,
                source="resolved tree",
                propagation_step_size=propagation_step_size,
            )
            if candidate is not None:
                candidates.append(candidate)
        else:
            rejected += 1
            best_rejected_dist = min(best_rejected_dist, closest)

    previous_states = previous_plan.get("states") or []
    previous_controls = previous_plan.get("controls") or []
    if len(previous_controls) > 1 and len(previous_states) > 1:
        prev_dist = float(arrayDistance(pose, previous_states[1], system=system_name))
        previous_candidate = suffix_plan_from_index(
            previous_plan,
            1,
            pose,
            continuity_dist=prev_dist,
            source="previous plan suffix",
            propagation_step_size=propagation_step_size,
        )
        if previous_candidate is not None:
            candidates.append(previous_candidate)

    candidates.sort(key=lambda x: x["cost"])
    info = {
        "max_dist": max_dist,
        "accepted": len(candidates),
        "rejected": rejected,
        "best_rejected_dist": best_rejected_dist,
    }
    return candidates, info


def visual_candidate_paths(candidates: list[dict]) -> list[dict]:
    visual: list[dict] = []
    for idx, candidate in enumerate((candidates or [])[:5]):
        snap = snapshot_plan(candidate)
        snap["_is_selected"] = idx == 0
        snap["_selection_source"] = candidate.get("_selection_source", "")
        snap["_trim_index"] = candidate.get("_trim_index", -1)
        snap["_continuity_dist"] = candidate.get("_continuity_dist", float("nan"))
        snap["cost"] = candidate.get("cost", float("nan"))
        visual.append(snap)
    return visual


def visual_goal_solution_paths(solutions: list[dict]) -> list[dict]:
    visual: list[dict] = []
    for solution in (solutions or [])[:50]:
        snap = snapshot_plan(solution)
        snap["_selection_source"] = "resolved goal solution"
        snap["cost"] = solution.get("cost", float("nan"))
        visual.append(snap)
    return visual


def plan_change_caption(
    selected_plan: dict,
    continuity_info: dict,
    candidates: list[dict],
    *,
    recovery_info: dict | None = None,
    goal_solution_count: int = 0,
    initial: bool = False,
    final: bool = False,
) -> str:
    if initial:
        return (
            "Initial solution is shown dashed in orange. Later frames call resolve on the "
            "existing tree and choose the lowest-cost continuous candidate from the executed pose."
        )
    if final:
        return (
            "Final frame after executing the selected last control; success is checked against "
            "both the goal region and AURA's final planned state."
        )
    source = str(selected_plan.get("_selection_source", "selected plan"))
    accepted = continuity_info.get("accepted", len(candidates or []))
    rejected = continuity_info.get("rejected", "—")
    trim_idx = selected_plan.get("_trim_index", "—")
    cont = fmt_scalar(selected_plan.get("_continuity_dist"))
    alt_count = max(0, len(candidates or []) - 1)
    if source == "resolved tree":
        reason = (
            "resolve continued the existing tree and produced the lowest-cost continuous "
            "candidate"
        )
    elif source == "previous plan suffix":
        reason = (
            "the previous plan suffix stayed as the lowest-cost continuous option from "
            "the executed pose"
        )
    elif source == "recovery fresh plan":
        reason = (
            "a bounded recovery plan was used because the selected next-control curve "
            "was invalid"
        )
    else:
        reason = f"{source} was selected"
    caption = (
        f"Plan update: {reason} (trim idx {trim_idx}, continuity {cont}; "
        f"accepted {accepted}, rejected {rejected})."
    )
    if recovery_info and recovery_info.get("attempted"):
        caption += f" Recovery status: {recovery_info.get('status')}."
    if alt_count > 0:
        caption += (
            f" Faint blue curves show {min(alt_count, 4)} other accepted candidate"
            f"{'' if min(alt_count, 4) == 1 else 's'}; solid blue is the plan followed next."
        )
    if goal_solution_count > 0:
        shown = min(int(goal_solution_count), 50)
        cap_note = " (capped)" if int(goal_solution_count) > shown else ""
        caption += (
            f" Faint gray curves show {shown} exact goal-reaching resolve solution"
            f"{'' if shown == 1 else 's'}{cap_note}; not all of them are continuous "
            "from the executed pose."
        )
    return caption


def trajectory_cost(states: list[np.ndarray], *, system_name: str) -> float:
    total = 0.0
    for i in range(len(states) - 1):
        total += float(
            arrayDistance(
                np.asarray(states[i], dtype=float),
                np.asarray(states[i + 1], dtype=float),
                system=system_name,
            )
        )
    return total


def remaining_plan_path_cost(
    plan_dict: dict,
    *,
    system: Any,
    propagation_step_size: float,
) -> float:
    states = plan_dict.get("states") or []
    controls = plan_dict.get("controls") or []
    if not states or not controls:
        return 0.0
    curve = [np.asarray(states[0], dtype=float).reshape(-1).copy()]
    for control_index, control in enumerate(controls):
        duration = plan_control_duration(
            plan_dict,
            control_index,
            default_duration=propagation_step_size,
        )
        curve.extend(
            sample_control_curve(
                system,
                curve[-1],
                np.asarray(control, dtype=float),
                float(duration),
                float(propagation_step_size),
                include_start=False,
            )
        )
    return trajectory_cost(curve, system_name=system.name)


def annotate_visual_cost(
    plan_dict: dict,
    *,
    controls_trajectory: list[np.ndarray],
    dense_actual_trajectory: list[np.ndarray],
    system: Any,
    propagation_step_size: float,
    planner: Any,
    final: bool = False,
) -> dict:
    executed_controls = float(len(controls_trajectory))
    remaining_controls = 0.0 if final else float(len(plan_dict.get("controls") or []))
    executed_path_cost = trajectory_cost(dense_actual_trajectory, system_name=system.name)
    remaining_path_cost = 0.0 if final else remaining_plan_path_cost(
        plan_dict,
        system=system,
        propagation_step_size=propagation_step_size,
    )
    cost_mode = str(getattr(planner, "cost_mode", "control_count")).lower()
    if cost_mode in ("control_count", "controls", "num_controls"):
        total_cost = executed_controls + remaining_controls
        label = getattr(planner, "cost_label", "best total controls")
    elif cost_mode in ("path_cost", "path_length", "length"):
        total_cost = executed_path_cost + remaining_path_cost
        label = getattr(planner, "cost_label", "best total path cost")
    else:
        try:
            total_cost = executed_path_cost + float(plan_dict.get("cost"))
        except (TypeError, ValueError):
            total_cost = executed_path_cost + remaining_path_cost
        label = getattr(planner, "cost_label", "best total cost")

    plan_dict["total_cost"] = float(total_cost)
    plan_dict["cost_label"] = str(label)
    plan_dict["cost_mode"] = cost_mode
    plan_dict["executed_control_count"] = executed_controls
    plan_dict["remaining_control_count"] = remaining_controls
    plan_dict["executed_path_cost"] = float(executed_path_cost)
    plan_dict["remaining_path_cost"] = float(remaining_path_cost)
    return plan_dict


def display_states_for_plan(
    plan_dict: dict,
    initial_reference_plan: dict,
) -> list[np.ndarray]:
    source_states = (
        plan_dict.get("_display_reference_states")
        or plan_dict.get("states")
        or initial_reference_plan.get("states", [])
    )
    return [np.asarray(s, dtype=float).copy() for s in source_states]


def add_visual_reference(
    plan_dict: dict,
    *,
    initial_reference_plan: dict,
    planner: Any,
    goal_state: np.ndarray,
    goal_threshold: float,
) -> dict:
    plan_dict["initial_states"] = [
        np.asarray(s, dtype=float).copy()
        for s in initial_reference_plan.get("states", [])
    ]
    plan_dict["display_states"] = display_states_for_plan(plan_dict, initial_reference_plan)
    plan_dict["initial_controls"] = [
        np.asarray(u, dtype=float).copy()
        for u in initial_reference_plan.get("controls", [])
    ]
    plan_dict["initial_time"] = [
        float(t) for t in initial_reference_plan.get("time", [])
    ]
    plan_dict["planner_name"] = getattr(planner, "planner_method", "")
    plan_dict["goal_state"] = np.asarray(goal_state, dtype=float).copy()
    plan_dict["goal_threshold"] = float(goal_threshold)
    return plan_dict


def to_numpy_state_control(x: Any, system_name: str) -> np.ndarray | None:
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy(), dtype=float)
    if isinstance(x, np.ndarray):
        return x.astype(float)
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=float)
    return np.asarray(state2list(x, system_name), dtype=float)


def planner_fallback_control(
    children_controls: list[np.ndarray],
    fallback_control: np.ndarray | None = None,
) -> np.ndarray | None:
    if fallback_control is not None:
        return np.asarray(fallback_control, dtype=float)
    return (
        np.asarray(children_controls[0], dtype=float)
        if len(children_controls) > 0
        else None
    )


def store_optimizer_partial_result(slot: dict, partial_result: dict) -> None:
    slot["partial_result"] = partial_result
    slot["partial_steps_completed"] = partial_result.get("steps_completed")


# Single live window for optimizer MSE vs Adam step (updated each AURA step).
_OPTIMIZER_LOSS_LIVE: dict = {"fig": None, "ax": None}


def reset_optimizer_loss_live_figure() -> None:
    """Close the live optimizer-loss window so the next run starts fresh."""
    fig = _OPTIMIZER_LOSS_LIVE.get("fig")
    if fig is not None:
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass
    _OPTIMIZER_LOSS_LIVE["fig"] = None
    _OPTIMIZER_LOSS_LIVE["ax"] = None


def plot_optimization_loss_history(
    loss_history: list[float] | np.ndarray | None,
    *,
    step_index: int,
    save_path: str | None = None,
    show_live: bool = False,
) -> None:
    """
    Plot optimizer MSE over Adam iterations for one AURA step.
    Low final MSE means the torch dynamics batch fit the sampled targets; it does not
    guarantee a lower OMPL distance for the executed pose under `system.propagate`.
    """
    if loss_history is None:
        return
    if hasattr(loss_history, "detach"):
        loss_history = loss_history.detach().cpu().numpy()
    y = np.asarray(loss_history, dtype=float).reshape(-1)
    if y.size == 0:
        return
    x = np.arange(len(y), dtype=float)
    y0, y1 = float(y[0]), float(y[-1])
    rel = (y0 - y1) / max(abs(y0), 1e-30)
    subtitle = f"init MSE={y0:.6e}  final={y1:.6e}  rel. drop={100.0 * rel:.1f}%"
    use_log = (y.max() / max(y.min(), 1e-30)) > 15.0

    import matplotlib.pyplot as plt

    def _style_axis(ax) -> None:
        ax.plot(x, y, "b.-", markersize=5, linewidth=1.0)
        if use_log:
            ax.set_yscale("log")
        ax.set_xlabel("Adam step")
        ax.set_ylabel("MSE loss (torch dynamics vs targets)")
        ax.set_title(f"AURA step {step_index} — optimizer convergence\n{subtitle}", fontsize=10)
        ax.grid(True, which="both", ls="-", alpha=0.3)

    if save_path:
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        _style_axis(ax)
        fig.tight_layout()
        loss_dir = os.path.dirname(os.path.abspath(save_path))
        if loss_dir:
            os.makedirs(loss_dir, exist_ok=True)
        fig.savefig(save_path, dpi=140)
        plt.close(fig)

    if show_live:
        try:
            if _OPTIMIZER_LOSS_LIVE["fig"] is None:
                raise ValueError("recreate")
            fig = _OPTIMIZER_LOSS_LIVE["fig"]
            ax = _OPTIMIZER_LOSS_LIVE["ax"]
            if fig is None or ax is None:
                raise ValueError("recreate")
            _ = fig.canvas
        except Exception:
            plt.ion()
            fig, ax = plt.subplots(figsize=(7.5, 4.0))
            try:
                fig.canvas.manager.set_window_title("AURA optimizer loss")
            except Exception:
                pass
            _OPTIMIZER_LOSS_LIVE["fig"] = fig
            _OPTIMIZER_LOSS_LIVE["ax"] = ax
        ax.clear()
        _style_axis(ax)
        fig.tight_layout()
        fig.canvas.draw_idle()
        try:
            fig.canvas.flush_events()
        except Exception:
            pass
        plt.pause(0.02)
