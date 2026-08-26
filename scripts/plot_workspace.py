#!/usr/bin/env python3
"""Plot top-down workspaces, paths, safety margins, and vehicle poses.

The command-line renderer lives in ``scripts/render_workspace_replay.py``.
"""

from __future__ import annotations

import os
import sys
import textwrap
from typing import Callable, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aura_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter, writers
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

from propagators import get_system
from utils.utils import normalize_obstacle_config, sample_piecewise_control_curve

PLANNER_METHOD_COLORS = {
    ("sst", "fusion"): "#EB5B00",
    ("sst", "replanning"): "#EB5B00",
    ("sststar", "fusion"): "#EB5B00",
    ("sststar", "replanning"): "#EB5B00",
    ("aorrt", "fusion"): "#A0C878",
    ("aorrt", "replanning"): "#A0C878",
    ("aoest", "fusion"): "#143D60",
    ("aoest", "replanning"): "#143D60",
    ("mppi", "mppi"): "#9B59B6",
}
PALETTE_ORANGE = "#EB5B00"
PALETTE_GREEN = "#A0C878"
PALETTE_BLUE = "#143D60"
PALETTE_PURPLE = "#9B59B6"
CURRENT_PLAN_COLOR = PALETTE_BLUE
INITIAL_PLAN_COLOR = "#8A949E"
EXECUTED_PATH_COLOR = PALETTE_PURPLE
SAFETY_MARGIN_COLOR = "#8FA9C9"
POSE_COLOR = PALETTE_ORANGE
POSE_ARROW_COLOR = "#D62828"
GOAL_REGION_COLOR = PALETTE_GREEN
GOAL_CENTER_COLOR = "#3b5f2a"
MAX_ALTERNATIVE_CANDIDATES = 5
MAX_GOAL_SOLUTION_PATHS = 50
GOAL_SOLUTION_COLOR = "#6f7f87"
WORKSPACE_BG = "#F7F8FA"
WORKSPACE_EDGE = "#AEB6BE"
AXIS_COLOR = "#000000"
OBSTACLE_FILL = "#343A40"
OBSTACLE_EDGE = "#111827"
TEXT_DARK = "#17202A"
TEXT_MUTED = "#55616D"
WHITE_STROKE = [pe.Stroke(linewidth=4.8, foreground="white", alpha=0.92), pe.Normal()]
SOFT_STROKE = [pe.Stroke(linewidth=5.2, foreground="white", alpha=0.82), pe.Normal()]
WORKSPACE_FONT_FAMILY = "Times New Roman"
WORKSPACE_BASE_FONT_SIZE = 14
WORKSPACE_AXIS_FONT_SIZE = 18
WORKSPACE_TICK_FONT_SIZE = 15
WORKSPACE_TITLE_FONT_SIZE = 18
WORKSPACE_LEGEND_FONT_SIZE = 12
WORKSPACE_BADGE_FONT_SIZE = 13
WORKSPACE_CAPTION_FONT_SIZE = 11
WORKSPACE_SIDE_INFO_X = 1.10
WORKSPACE_CAPTION_Y = -0.32


matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [WORKSPACE_FONT_FAMILY, "Times", "DejaVu Serif"],
        "font.size": WORKSPACE_BASE_FONT_SIZE,
        "axes.titlesize": WORKSPACE_TITLE_FONT_SIZE,
        "axes.labelsize": WORKSPACE_AXIS_FONT_SIZE,
        "xtick.labelsize": WORKSPACE_TICK_FONT_SIZE,
        "ytick.labelsize": WORKSPACE_TICK_FONT_SIZE,
        "legend.fontsize": WORKSPACE_LEGEND_FONT_SIZE,
        "axes.linewidth": 0.9,
    }
)


def _new_agg_figure(figsize: tuple[float, float] = (9.2, 6.0)):
    fig = Figure(figsize=figsize, facecolor="white")
    FigureCanvasAgg(fig)
    return fig, fig.add_subplot(111)


def planner_color(planner_name: str | None, mode: str = "fusion") -> str:
    key = (str(planner_name or "").lower(), str(mode).lower())
    if key in PLANNER_METHOD_COLORS:
        return PLANNER_METHOD_COLORS[key]
    simple_key = str(planner_name or "").lower()
    for (method, _), color in PLANNER_METHOD_COLORS.items():
        if method == simple_key:
            return color
    return PALETTE_BLUE


def format_frame_title(title: str) -> str:
    parts = [p.strip() for p in str(title or "").split("|") if p.strip()]
    if len(parts) <= 1:
        return parts[0] if parts else ""
    return f"{parts[0]}\n{' | '.join(parts[1:])}"


def _obstacle_safety(occ: dict) -> float:
    occ = occ or {}
    if not occ:
        return 0.0
    return float(occ.get("safety_radius", 0.0))


def draw_state_bounds(ax, state_bounds: list[tuple[float, float]], **kwargs) -> None:
    """2D box [xmin,xmax] x [ymin,ymax] from the first two bounds entries."""
    if len(state_bounds) < 2:
        return
    x0, x1 = state_bounds[0]
    y0, y1 = state_bounds[1]
    w, h = float(x1) - float(x0), float(y1) - float(y0)
    r = Rectangle(
        (x0, y0),
        w,
        h,
        facecolor="white",
        edgecolor=AXIS_COLOR,
        linewidth=1.4,
        zorder=0.8,
        **kwargs,
    )
    ax.add_patch(r)


def draw_obstacles_and_margins(
    ax, obstacle_config: Optional[dict], *, zobstacle: int = 2, zmargin: int = 1
) -> None:
    """
    Draw physical geometry and an inflated region for the planner safety margin
    (matches the collision test: circles use r+margin, AABBs/boxes expanded in position).
    """
    occ = normalize_obstacle_config(obstacle_config) if obstacle_config is not None else None
    if not occ or not occ.get("enabled", False):
        return
    m = _obstacle_safety(occ)
    ocolor, mcolor = OBSTACLE_FILL, SAFETY_MARGIN_COLOR
    a_obstacle, a_margin = 0.86, 0.20

    for cx, cy, r in occ.get("circles", []):
        cx, cy, r = float(cx), float(cy), float(r)
        if m > 0.0 and r + m > r + 1e-9:
            c_out = Circle(
                (cx, cy),
                r + m,
                facecolor=mcolor,
                edgecolor=mcolor,
                linewidth=0.8,
                alpha=a_margin,
                zorder=zmargin,
            )
            ax.add_patch(c_out)
            ax.add_patch(
                Circle(
                    (cx, cy),
                    r + m,
                    facecolor="none",
                    edgecolor=mcolor,
                    linewidth=0.9,
                    linestyle=(0, (4, 3)),
                    alpha=0.72,
                    zorder=zmargin + 0.05,
                )
            )
        c_in = Circle(
            (cx, cy),
            r,
            facecolor=ocolor,
            edgecolor=OBSTACLE_EDGE,
            linewidth=0.9,
            alpha=a_obstacle,
            zorder=zobstacle,
        )
        ax.add_patch(c_in)

    for item in occ.get("aabbs", []):
        xmin, ymin, xmax, ymax = (float(t) for t in item)
        w, h = xmax - xmin, ymax - ymin
        if m > 0.0:
            r_out = Rectangle(
                (xmin - m, ymin - m),
                w + 2 * m,
                h + 2 * m,
                facecolor=mcolor,
                edgecolor=mcolor,
                linewidth=0.8,
                alpha=a_margin,
                zorder=zmargin,
            )
            ax.add_patch(r_out)
            ax.add_patch(
                Rectangle(
                    (xmin - m, ymin - m),
                    w + 2 * m,
                    h + 2 * m,
                    facecolor="none",
                    edgecolor=mcolor,
                    linewidth=0.9,
                    linestyle=(0, (4, 3)),
                    alpha=0.72,
                    zorder=zmargin + 0.05,
                )
            )
        r_in = Rectangle(
            (xmin, ymin),
            w,
            h,
            facecolor=ocolor,
            edgecolor=OBSTACLE_EDGE,
            linewidth=0.9,
            alpha=a_obstacle,
            zorder=zobstacle,
        )
        ax.add_patch(r_in)

    for item in occ.get("boxes", []):
        cx, cy, hx, hy, yaw = (float(t) for t in item)
        mloc = m
        hx_e, hy_e = hx + mloc, hy + mloc
        c, s = np.cos(yaw), np.sin(yaw)
        for half_x, half_y, alpha, zz in (
            (hx_e, hy_e, 0.22, zmargin),
            (hx, hy, 0.7, zobstacle),
        ):
            L = half_x
            T = half_y
            v = np.array(
                [
                    [L, T],
                    [L, -T],
                    [-L, -T],
                    [-L, T],
                ],
                dtype=float,
            )
            v[:, 0], v[:, 1] = c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]
            v[:, 0] += cx
            v[:, 1] += cy
            if alpha < 0.3:
                poly = Polygon(
                    v,
                    closed=True,
                    facecolor=mcolor,
                    edgecolor=mcolor,
                    alpha=alpha,
                    zorder=zz,
                )
                ax.add_patch(poly)
                outline = Polygon(
                    v,
                    closed=True,
                    facecolor="none",
                    edgecolor=mcolor,
                    alpha=0.72,
                    linewidth=0.9,
                    linestyle=(0, (4, 3)),
                    zorder=zz + 0.05,
                )
                ax.add_patch(outline)
                continue
            else:
                poly = Polygon(
                    v,
                    closed=True,
                    facecolor=ocolor,
                    edgecolor=OBSTACLE_EDGE,
                    alpha=alpha,
                    linewidth=0.9,
                    zorder=zz,
                )
            ax.add_patch(poly)


def draw_goal_region(
    ax,
    goal_state: Optional[np.ndarray | list | tuple],
    goal_threshold: Optional[float],
) -> None:
    """Draw a slightly enlarged XY projection of the goal tolerance for visibility."""
    if goal_state is None or goal_threshold is None:
        return
    g = np.asarray(goal_state, dtype=float).reshape(-1)
    if g.size < 2:
        return
    radius = 2.0 * float(goal_threshold)
    if not np.isfinite(radius) or radius <= 0.0:
        return
    ax.add_patch(
        Circle(
            (float(g[0]), float(g[1])),
            radius,
            facecolor=GOAL_REGION_COLOR,
            edgecolor=GOAL_CENTER_COLOR,
            linewidth=1.4,
            alpha=0.28,
            zorder=1.7,
            label="Goal region (2x threshold)",
        )
    )
    ax.add_patch(
        Circle(
            (float(g[0]), float(g[1])),
            radius,
            facecolor="none",
            edgecolor=GOAL_CENTER_COLOR,
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            alpha=0.75,
            zorder=1.8,
        )
    )
    ax.plot(
        [float(g[0])],
        [float(g[1])],
        marker="*",
        markersize=12,
        markerfacecolor=GOAL_CENTER_COLOR,
        markeredgecolor="white",
        markeredgewidth=1.0,
        linestyle="none",
        zorder=6.5,
        path_effects=[pe.Stroke(linewidth=3.2, foreground="white", alpha=0.95), pe.Normal()],
    )


def _format_cost_value(value) -> Optional[str]:
    try:
        cost = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(cost):
        return None
    rounded = round(cost)
    if abs(cost - rounded) <= 1e-9:
        return str(int(rounded))
    if cost != 0.0 and (abs(cost) < 1e-3 or abs(cost) >= 1e4):
        return f"{cost:.3e}"
    return f"{cost:.3f}"


def draw_cost_badge(ax, best_trajectory: dict) -> None:
    cost_value = best_trajectory.get("total_cost", best_trajectory.get("cost"))
    cost_text = _format_cost_value(cost_value)
    if cost_text is None:
        candidate_paths = best_trajectory.get("candidate_paths") or []
        if candidate_paths:
            cost_text = _format_cost_value(
                candidate_paths[0].get("total_cost", candidate_paths[0].get("cost"))
            )
    if cost_text is None:
        return
    label = str(best_trajectory.get("cost_label") or "best cost")
    ax.text(
        WORKSPACE_SIDE_INFO_X,
        1.00,
        f"{label.upper()}\n{cost_text}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=WORKSPACE_BADGE_FONT_SIZE,
        linespacing=1.35,
        color=TEXT_DARK,
        fontweight="semibold",
        bbox={
            "boxstyle": "round,pad=0.36,rounding_size=0.12",
            "facecolor": "white",
            "edgecolor": CURRENT_PLAN_COLOR,
            "alpha": 0.94,
            "linewidth": 1.15,
        },
        zorder=9,
        clip_on=False,
    )


def states_xy_yaw(states: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, y, yaw) arrays from a list of flat states [x,y,yaw] or [x,y,...]."""
    if not states:
        return (
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
        )
    arr = np.array([np.asarray(s, dtype=float).ravel()[:3] for s in states], dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2] if arr.shape[1] >= 3 else np.zeros(len(arr), dtype=float)


def draw_se2_path(
    ax,
    states: list,
    color: str = "C0",
    label: str = "Planned path",
    zorder: int = 4,
    *,
    linestyle: str = "-",
    linewidth: float = 1.5,
    alpha: float = 0.9,
    show_nodes: bool = True,
    path_effects=None,
) -> None:
    x, y, _ = states_xy_yaw(states)
    if len(x) > 0:
        ax.plot(
            x,
            y,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
            label=label,
            alpha=alpha,
            solid_capstyle="round",
            dash_capstyle="round",
            path_effects=path_effects,
        )
        if show_nodes and len(x) <= 80:
            ax.plot(
                x,
                y,
                linestyle="none",
                marker="o",
                markersize=2.5,
                color=color,
                alpha=min(0.75, alpha),
                zorder=zorder + 0.1,
            )


def draw_path_endpoints(
    ax,
    states: list,
    *,
    start: bool = False,
    final: bool = True,
    color: str = CURRENT_PLAN_COLOR,
    zorder: int = 6,
) -> None:
    x, y, _ = states_xy_yaw(states)
    if len(x) == 0:
        return
    if start:
        ax.plot(
            [x[0]],
            [y[0]],
            marker="o",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.4,
            linestyle="none",
            zorder=zorder,
            label="Plan start",
        )
    if final:
        ax.plot(
            [x[-1]],
            [y[-1]],
            marker="D",
            markersize=5.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.9,
            linestyle="none",
            zorder=zorder,
            label="Planned final state",
            path_effects=[pe.Stroke(linewidth=3.1, foreground="white", alpha=0.9), pe.Normal()],
        )


def draw_se2_pose(
    ax,
    state: np.ndarray,
    *,
    color: str = "C3",
    scale: float = 0.18,
    zorder: int = 5,
    label: Optional[str] = "Current state",
    arrow_color: str = POSE_ARROW_COLOR,
) -> None:
    """Clear SE(2) vehicle glyph plus a red heading arrow."""
    s = np.asarray(state, dtype=float).ravel()
    if len(s) < 3:
        return
    x, y, th = float(s[0]), float(s[1]), float(s[2])
    direction = np.array([np.cos(th), np.sin(th)], dtype=float)
    normal = np.array([-np.sin(th), np.cos(th)], dtype=float)

    nose = np.array([x, y]) + direction * scale * 0.95
    rear = np.array([x, y]) - direction * scale * 0.55
    left = rear + normal * scale * 0.42
    right = rear - normal * scale * 0.42
    body = Polygon(
        np.vstack([nose, left, right]),
        closed=True,
        facecolor=color,
        edgecolor="white",
        linewidth=1.2,
        alpha=0.95,
        zorder=zorder,
        label=label,
        path_effects=[pe.Stroke(linewidth=2.4, foreground="#2D2D2D", alpha=0.60), pe.Normal()],
    )
    ax.add_patch(body)

    start = np.array([x, y]) + direction * scale * 0.20
    end = np.array([x, y]) + direction * scale * 1.95
    outline = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=24,
        linewidth=5.4,
        color="white",
        zorder=zorder + 0.8,
        shrinkA=0,
        shrinkB=0,
    )
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=22,
        linewidth=3.1,
        color=arrow_color,
        zorder=zorder + 1.0,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(outline)
    ax.add_patch(arrow)
    ax.add_patch(
        Circle(
            (x, y),
            radius=scale * 0.16,
            facecolor="white",
            edgecolor=arrow_color,
            linewidth=1.0,
            zorder=zorder + 1.1,
        )
    )


def plot_planning_frame(
    ax,
    state_bounds: list[tuple[float, float]],
    obstacle_config: Optional[dict],
    best_trajectory: dict,
    current_state: np.ndarray,
    *,
    title: str = "",
    system=None,
    curve_step_size: float = 0.05,
    planner_name: str | None = None,
    goal_state: Optional[np.ndarray | list | tuple] = None,
    goal_threshold: Optional[float] = None,
) -> None:
    ax.set_facecolor(WORKSPACE_BG)
    draw_state_bounds(ax, state_bounds)
    draw_obstacles_and_margins(ax, obstacle_config)
    goal_state = goal_state if goal_state is not None else best_trajectory.get("goal_state")
    if goal_threshold is None:
        goal_threshold = best_trajectory.get("goal_threshold")
    draw_goal_region(ax, goal_state, goal_threshold)
    active_color = CURRENT_PLAN_COLOR
    initial_color = INITIAL_PLAN_COLOR

    initial_states = best_trajectory.get("initial_states") or []
    initial_controls = best_trajectory.get("initial_controls") or []
    initial_durations = best_trajectory.get("initial_time") or []
    if system is not None and initial_states and initial_controls:
        initial_curve = sample_piecewise_control_curve(
            system,
            initial_states,
            initial_controls,
            initial_durations,
            curve_step_size,
        )
        draw_se2_path(
            ax,
            initial_curve,
            color=initial_color,
            label="Initial plan",
            zorder=2,
            linestyle="-",
            linewidth=2.0,
            alpha=0.70,
            show_nodes=False,
            path_effects=[pe.Stroke(linewidth=4.0, foreground="white", alpha=0.72), pe.Normal()],
        )
    elif initial_states:
        draw_se2_path(
            ax,
            initial_states,
            color=initial_color,
            label="Initial plan",
            zorder=2,
            linestyle="-",
            linewidth=2.0,
            alpha=0.70,
            show_nodes=False,
            path_effects=[pe.Stroke(linewidth=4.0, foreground="white", alpha=0.72), pe.Normal()],
        )

    goal_solution_paths = best_trajectory.get("goal_solution_paths") or []
    goal_solution_label_used = False
    for solution_path in goal_solution_paths[:MAX_GOAL_SOLUTION_PATHS]:
        sol_states = solution_path.get("states") or []
        sol_controls = solution_path.get("controls") or []
        sol_durations = solution_path.get("time") or []
        if not sol_states:
            continue
        if system is not None and sol_controls:
            sol_curve = sample_piecewise_control_curve(
                system,
                sol_states,
                sol_controls,
                sol_durations,
                curve_step_size,
            )
        else:
            sol_curve = sol_states
        draw_se2_path(
            ax,
            sol_curve,
            color=GOAL_SOLUTION_COLOR,
            label="Goal solutions" if not goal_solution_label_used else "_nolegend_",
            zorder=2.6,
            linewidth=1.0,
            alpha=0.16,
            show_nodes=False,
        )
        goal_solution_label_used = True

    candidate_paths = best_trajectory.get("candidate_paths") or []
    candidate_label_used = False
    for idx, candidate in enumerate(candidate_paths[:MAX_ALTERNATIVE_CANDIDATES]):
        if candidate.get("_is_selected", idx == 0):
            continue
        alt_states = candidate.get("states") or []
        alt_controls = candidate.get("controls") or []
        alt_durations = candidate.get("time") or []
        if not alt_states:
            continue
        if system is not None and alt_controls:
            alt_curve = sample_piecewise_control_curve(
                system,
                alt_states,
                alt_controls,
                alt_durations,
                curve_step_size,
            )
        else:
            alt_curve = alt_states
        draw_se2_path(
            ax,
            alt_curve,
            color=active_color,
            label="Accepted alternatives" if not candidate_label_used else "_nolegend_",
            zorder=3,
            linewidth=1.25,
            alpha=0.18,
            show_nodes=False,
        )
        candidate_label_used = True

    states = best_trajectory.get("states") or []
    controls = best_trajectory.get("controls") or []
    durations = best_trajectory.get("time") or []
    current_curve = []
    if system is not None and states and controls:
        curve_states = sample_piecewise_control_curve(
            system,
            states,
            controls,
            durations,
            curve_step_size,
        )
        current_curve = curve_states
        draw_se2_path(
            ax,
            curve_states,
            color=active_color,
            label="Current plan",
            zorder=4,
            linewidth=3.0,
            alpha=0.98,
            show_nodes=False,
            path_effects=SOFT_STROKE,
        )
    else:
        current_curve = states
        draw_se2_path(
            ax,
            states,
            color=active_color,
            label="Current plan",
            zorder=3,
            linewidth=3.0,
            show_nodes=False,
            path_effects=SOFT_STROKE,
        )
    if len(current_curve) > 0:
        draw_path_endpoints(
            ax,
            current_curve,
            start=False,
            final=True,
            color=active_color,
            zorder=5.7,
        )
    actual_states = best_trajectory.get("actual_states") or []
    if actual_states:
        draw_se2_path(
            ax,
            actual_states,
            color=EXECUTED_PATH_COLOR,
            label="Executed control curve",
            zorder=5,
            linewidth=2.25,
            alpha=0.95,
            show_nodes=False,
            path_effects=[pe.Stroke(linewidth=4.1, foreground="white", alpha=0.82), pe.Normal()],
        )
    draw_se2_pose(
        ax,
        current_state,
        color=POSE_COLOR,
        arrow_color=POSE_ARROW_COLOR,
        label="Actual pose",
        zorder=6,
    )
    draw_cost_badge(ax, best_trajectory)
    ax.set_title(
        format_frame_title(title),
        fontsize=WORKSPACE_TITLE_FONT_SIZE,
        fontweight="semibold",
        color=TEXT_DARK,
        pad=12,
        linespacing=1.18,
        loc="left",
    )
    if len(state_bounds) >= 2:
        x0, x1 = state_bounds[0]
        y0, y1 = state_bounds[1]
        ax.set_xlim(float(x0), float(x1))
        ax.set_ylim(float(y0), float(y1))
        workspace_ticks = np.arange(1, 6, dtype=int)
        ax.set_xticks([tick for tick in workspace_ticks if float(x0) <= tick <= float(x1)])
        ax.set_yticks([tick for tick in workspace_ticks if float(y0) <= tick <= float(y1)])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x", labelpad=14)
    ax.set_ylabel("y", labelpad=10)
    ax.xaxis.label.set_size(WORKSPACE_AXIS_FONT_SIZE)
    ax.yaxis.label.set_size(WORKSPACE_AXIS_FONT_SIZE)
    ax.tick_params(
        colors=AXIS_COLOR,
        width=1.1,
        length=4.2,
        labelsize=WORKSPACE_TICK_FONT_SIZE,
    )
    ax.xaxis.label.set_color(AXIS_COLOR)
    ax.yaxis.label.set_color(AXIS_COLOR)
    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(1.1)
    ax.grid(True, color="#DDE3EA", linewidth=0.75, zorder=0)
    ax.set_axisbelow(True)
    caption = (
        best_trajectory.get("plan_caption")
        or best_trajectory.get("_plan_change_caption")
        or ""
    )
    if caption:
        wrapped = "\n".join(textwrap.wrap(str(caption), width=104))
        ax.text(
            0.5,
            WORKSPACE_CAPTION_Y,
            wrapped,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=WORKSPACE_CAPTION_FONT_SIZE,
            color=TEXT_MUTED,
            linespacing=1.25,
            bbox={
                "boxstyle": "round,pad=0.35,rounding_size=0.10",
                "facecolor": "white",
                "edgecolor": "#D4DAE2",
                "linewidth": 0.8,
                "alpha": 0.92,
            },
            clip_on=False,
            zorder=10,
        )
    occ = normalize_obstacle_config(obstacle_config) if obstacle_config is not None else None
    margin = _obstacle_safety(occ or {})
    if margin > 0.0:
        ax.text(
            WORKSPACE_SIDE_INFO_X,
            0.02,
            f"safety margin {margin:.3g} m",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=WORKSPACE_LEGEND_FONT_SIZE,
            color=TEXT_MUTED,
            bbox={
                "boxstyle": "round,pad=0.26,rounding_size=0.10",
                "facecolor": "white",
                "edgecolor": SAFETY_MARGIN_COLOR,
                "alpha": 0.90,
                "linewidth": 0.8,
            },
            zorder=8,
            clip_on=False,
        )
    h, l = ax.get_legend_handles_labels()
    if h:
        unique = {}
        for handle, label in zip(h, l):
            if not label or label == "_nolegend_" or label.startswith("_"):
                continue
            unique.setdefault(label, handle)
        if unique:
            legend = ax.legend(
                unique.values(),
                unique.keys(),
                loc="upper left",
                bbox_to_anchor=(WORKSPACE_SIDE_INFO_X, 0.68),
                bbox_transform=ax.transAxes,
                borderaxespad=0.0,
                framealpha=0.92,
                fontsize=WORKSPACE_LEGEND_FONT_SIZE,
                ncol=1,
                borderpad=0.55,
                labelspacing=0.45,
                handlelength=2.4,
                handletextpad=0.7,
            )
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_edgecolor("#D4DAE2")
            legend.get_frame().set_linewidth(0.8)


def _state_list(states) -> list[np.ndarray]:
    return [np.asarray(s, dtype=float).reshape(-1).copy() for s in (states or [])]


def _control_list(controls) -> list[np.ndarray]:
    return [np.asarray(c, dtype=float).reshape(-1).copy() for c in (controls or [])]


def _duration_list(durations) -> list[float]:
    return [float(t) for t in (durations or [])]


def _compact_candidate_paths(candidates, *, limit: int = MAX_ALTERNATIVE_CANDIDATES) -> list[dict]:
    out: list[dict] = []
    for candidate in (candidates or [])[: int(limit)]:
        out.append(
            {
                "states": _state_list(candidate.get("states")),
                "controls": _control_list(candidate.get("controls")),
                "time": _duration_list(candidate.get("time") or candidate.get("durations")),
                "_selection_source": candidate.get("_selection_source", ""),
                "_trim_index": int(candidate.get("_trim_index", -1)),
                "_continuity_dist": float(candidate.get("_continuity_dist", np.nan)),
                "cost": float(candidate.get("cost", np.nan)),
                "_is_selected": bool(candidate.get("_is_selected", False)),
            }
        )
    return out


class WorkspaceReplayRecorder:
    """Collect compact per-update workspace data for later video rendering."""

    def __init__(
        self,
        replay_path: str,
        state_bounds: list,
        obstacle_config: Optional[dict],
        *,
        system_name: str = "kinematic_car",
        planner_name: str | None = None,
        curve_step_size: float = 0.05,
        title_prefix: str = "",
        goal_state: Optional[np.ndarray | list | tuple] = None,
        goal_threshold: Optional[float] = None,
    ):
        self.replay_path = replay_path
        self.frames: list[dict] = []
        goal_list = None
        if goal_state is not None:
            goal_list = np.asarray(goal_state, dtype=float).reshape(-1).tolist()
        self.metadata = {
            "version": 1,
            "system_name": system_name,
            "planner_name": planner_name,
            "state_bounds": [(float(a), float(b)) for a, b in state_bounds],
            "obstacles": obstacle_config,
            "curve_step_size": float(curve_step_size),
            "title_prefix": str(title_prefix),
            "goal_state": goal_list,
            "goal_threshold": None if goal_threshold is None else float(goal_threshold),
        }

    def __call__(self, step_id: int, best_tr: dict, pose: np.ndarray) -> None:
        self.frames.append(
            {
                "step": int(step_id),
                "pose": np.asarray(pose, dtype=float).reshape(-1).copy(),
                "states": _state_list(best_tr.get("states")),
                "controls": _control_list(best_tr.get("controls")),
                "time": _duration_list(best_tr.get("time") or best_tr.get("durations")),
                "initial_states": _state_list(best_tr.get("initial_states")),
                "initial_controls": _control_list(best_tr.get("initial_controls")),
                "initial_time": _duration_list(best_tr.get("initial_time")),
                "planner_name": best_tr.get("planner_name", self.metadata.get("planner_name")),
                "actual_states": _state_list(best_tr.get("actual_states")),
                "goal_solution_paths": _compact_candidate_paths(
                    best_tr.get("goal_solution_paths"),
                    limit=MAX_GOAL_SOLUTION_PATHS,
                ),
                "candidate_paths": _compact_candidate_paths(best_tr.get("candidate_paths")),
                "cost": float(best_tr.get("cost", np.nan)),
                "total_cost": float(best_tr.get("total_cost", np.nan)),
                "cost_label": best_tr.get("cost_label", ""),
                "cost_mode": best_tr.get("cost_mode", ""),
                "executed_control_count": float(
                    best_tr.get("executed_control_count", np.nan)
                ),
                "remaining_control_count": float(
                    best_tr.get("remaining_control_count", np.nan)
                ),
                "executed_path_cost": float(best_tr.get("executed_path_cost", np.nan)),
                "remaining_path_cost": float(best_tr.get("remaining_path_cost", np.nan)),
                "plan_caption": (
                    best_tr.get("plan_caption")
                    or best_tr.get("_plan_change_caption")
                    or ""
                ),
                "selection_source": best_tr.get("_selection_source", ""),
                "trim_index": int(best_tr.get("_trim_index", -1)),
                "continuity_dist": float(best_tr.get("_continuity_dist", np.nan)),
                "is_final": bool(best_tr.get("_final_update", False)),
            }
        )

    def save(self) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(self.replay_path)), exist_ok=True)
        np.savez_compressed(
            self.replay_path,
            metadata=np.array(self.metadata, dtype=object),
            frames=np.array(self.frames, dtype=object),
        )
        print(f"[replay] saved {len(self.frames)} workspace updates -> {self.replay_path}")
        return self.replay_path


def load_workspace_replay(replay_path: str) -> tuple[dict, list[dict]]:
    data = np.load(replay_path, allow_pickle=True)
    metadata = data["metadata"].item()
    frames = list(data["frames"])
    return metadata, frames


def render_workspace_replay(
    replay_path: str,
    *,
    video_path: Optional[str] = None,
    frames_dir: Optional[str] = None,
    fps: float = 3.0,
    dpi: int = 400,
    show: bool = False,
) -> Optional[str]:
    """Render a saved workspace replay to MP4/GIF and/or PNG frames."""
    metadata, frames = load_workspace_replay(replay_path)
    if not frames:
        raise ValueError(f"No frames in replay: {replay_path}")

    system = get_system(str(metadata.get("system_name", "kinematic_car")))
    state_bounds = metadata["state_bounds"]
    obstacle_config = metadata.get("obstacles")
    curve_step_size = float(metadata.get("curve_step_size", 0.05))
    replay_planner_name = metadata.get("planner_name")
    title_prefix = str(metadata.get("title_prefix", "AURA replay"))
    goal_state = metadata.get("goal_state")
    goal_threshold = metadata.get("goal_threshold")

    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)

    if show:
        fig, ax = plt.subplots(figsize=(9.2, 6.0))
    else:
        fig, ax = _new_agg_figure(figsize=(7.5, 6.0))

    def draw_frame(frame: dict) -> None:
        ax.clear()
        plan = {
            "states": frame.get("states") or [],
            "controls": frame.get("controls") or [],
            "time": frame.get("time") or [],
            "initial_states": frame.get("initial_states") or [],
            "initial_controls": frame.get("initial_controls") or [],
            "initial_time": frame.get("initial_time") or [],
            "planner_name": frame.get("planner_name") or replay_planner_name,
            "actual_states": frame.get("actual_states") or [],
            "goal_solution_paths": frame.get("goal_solution_paths") or [],
            "candidate_paths": frame.get("candidate_paths") or [],
            "cost": frame.get("cost", np.nan),
            "total_cost": frame.get("total_cost", np.nan),
            "cost_label": frame.get("cost_label", ""),
            "cost_mode": frame.get("cost_mode", ""),
            "executed_control_count": frame.get("executed_control_count", np.nan),
            "remaining_control_count": frame.get("remaining_control_count", np.nan),
            "executed_path_cost": frame.get("executed_path_cost", np.nan),
            "remaining_path_cost": frame.get("remaining_path_cost", np.nan),
            "plan_caption": frame.get("plan_caption") or "",
        }
        suffix = "final" if frame.get("is_final") else f"step {int(frame.get('step', 0))}"
        title = f"{title_prefix} | {suffix}"
        source = frame.get("selection_source")
        if source:
            title = f"{title} | {source}"
        plot_planning_frame(
            ax,
            state_bounds,
            obstacle_config,
            plan,
            np.asarray(frame["pose"], dtype=float),
            title=title,
            system=system,
            curve_step_size=curve_step_size,
            planner_name=frame.get("planner_name") or replay_planner_name,
            goal_state=goal_state,
            goal_threshold=goal_threshold,
        )
        fig.subplots_adjust(left=0.08, right=0.66, top=0.90, bottom=0.38)

    rendered_video: Optional[str] = None
    writer = None
    if video_path:
        os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
        ext = os.path.splitext(video_path)[1].lower()
        if ext == ".mp4" and writers.is_available("ffmpeg"):
            writer = FFMpegWriter(fps=float(fps), bitrate=20000)
        elif ext == ".gif":
            writer = PillowWriter(fps=float(fps))
        else:
            print(
                "[replay] No suitable writer available for "
                f"{video_path!r}; writing PNG frames instead."
            )
            if frames_dir is None:
                base = os.path.splitext(os.path.abspath(video_path))[0]
                frames_dir = f"{base}_frames"
                os.makedirs(frames_dir, exist_ok=True)

    if writer is not None and video_path:
        with writer.saving(fig, video_path, dpi=int(dpi)):
            for frame in frames:
                draw_frame(frame)
                writer.grab_frame()
        rendered_video = video_path
        print(f"[replay] video -> {video_path}")

    if frames_dir:
        for frame in frames:
            draw_frame(frame)
            p = os.path.join(frames_dir, f"workspace_{int(frame.get('step', 0)):04d}.png")
            fig.savefig(p, dpi=int(dpi), bbox_inches="tight")
        print(f"[replay] frames -> {frames_dir}")

    if show:
        draw_frame(frames[-1])
        plt.show(block=True)
    else:
        fig.clear()
    return rendered_video


def workspace_planning_callback(
    frames_dir: str,
    state_bounds: list,
    obstacle_config: Optional[dict],
    *,
    system=None,
    planner_name: str | None = None,
    curve_step_size: float = 0.05,
    title_prefix: str = "",
    goal_state: Optional[np.ndarray | list | tuple] = None,
    goal_threshold: Optional[float] = None,
) -> Callable[[int, dict, np.ndarray], None]:
    """
    For use with AURA.run(on_planning_update=...): saves a workspace PNG per replan
    (bounds, obstacles+margin, nominal path, actual pose).
    """
    os.makedirs(frames_dir, exist_ok=True)
    ocfg = dict(obstacle_config) if isinstance(obstacle_config, dict) else obstacle_config

    def on_planning_update(step_id: int, best_tr: dict, pose: np.ndarray) -> None:
        fig, ax = _new_agg_figure(figsize=(9.2, 6.0))
        t = f"{title_prefix} | step {step_id}" if title_prefix else f"Planning update {step_id}"
        plot_planning_frame(
            ax,
            state_bounds,
            ocfg,
            best_tr,
            np.asarray(pose, dtype=float),
            title=t,
            system=system,
            curve_step_size=curve_step_size,
            planner_name=planner_name,
            goal_state=goal_state,
            goal_threshold=goal_threshold,
        )
        p = os.path.join(frames_dir, f"planning_{step_id:04d}.png")
        fig.savefig(p, dpi=120, bbox_inches="tight")
        fig.clear()
        print(f"[plotting] workspace frame -> {p}")

    return on_planning_update

def workspace_planning_callback_interactive(
    state_bounds: list,
    obstacle_config: Optional[dict],
    *,
    system=None,
    planner_name: str | None = None,
    curve_step_size: float = 0.05,
    frames_dir: Optional[str] = None,
    title_prefix: str = "",
    goal_state: Optional[np.ndarray | list | tuple] = None,
    goal_threshold: Optional[float] = None,
) -> Callable[[int, dict, np.ndarray], None]:
    """
    One workspace figure that updates on each AURA replan; optionally also saves PNGs
    to *frames_dir* if set. Requires a non-Agg backend (no-op draw if not available).
    """
    ocfg = dict(obstacle_config) if isinstance(obstacle_config, dict) else obstacle_config
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
    if not hasattr(workspace_planning_callback_interactive, "_live_state"):
        workspace_planning_callback_interactive._live_state = {"fig": None, "ax": None}  # type: ignore[attr-defined]
    st = workspace_planning_callback_interactive._live_state  # type: ignore[attr-defined]

    def on_planning_update(step_id: int, best_tr: dict, pose: np.ndarray) -> None:
        if st["ax"] is None:
            st["fig"], st["ax"] = plt.subplots(figsize=(9.2, 6.0))
        t = f"{title_prefix} | step {step_id}" if title_prefix else f"Planning {step_id}"
        st["ax"].clear()
        plot_planning_frame(
            st["ax"],
            state_bounds,
            ocfg,
            best_tr,
            np.asarray(pose, dtype=float),
            title=t,
            system=system,
            curve_step_size=curve_step_size,
            planner_name=planner_name,
            goal_state=goal_state,
            goal_threshold=goal_threshold,
        )
        st["fig"].subplots_adjust(left=0.08, right=0.66, top=0.90, bottom=0.38)
        if frames_dir:
            p = os.path.join(frames_dir, f"planning_{step_id:04d}.png")
            st["fig"].savefig(p, dpi=120, bbox_inches="tight")
            print(f"[plotting] workspace -> {p}")
        if matplotlib.get_backend() != "Agg" and st["fig"] is not None:
            if not plt.isinteractive():
                plt.ion()
            st["fig"].canvas.draw()
            st["fig"].canvas.flush_events()
            plt.show(block=False)
            plt.pause(0.1)

    return on_planning_update
