#!/usr/bin/env python3
"""Presentation-quality Gaussian-noise dynamics demos for AURA systems.

Examples:
  python3.10 scripts/simulation_demo.py kinematic_car
  python3.10 scripts/simulation_demo.py double_integrator --output results/simulation_demo/double.mp4
  python3.10 scripts/simulation_demo.py pushing_object --duration 10 --fps 30 --dpi 400
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/aura_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter, writers
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from simulation.simulator import create_simulator
from propagators import get_system


BLUE = "#143D60"
ORANGE = "#EB5B00"
GREEN = "#A0C878"
PURPLE = "#9B59B6"
RED = "#D62828"
DARK = "#17202A"
MUTED = "#59636E"
GRID = "#DDE3EA"
PANEL = "#F7F8FA"

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 18,
        "axes.labelsize": 23,
        "axes.titlesize": 25,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "axes.linewidth": 1.2,
    }
)


@dataclass
class DemoData:
    system_name: str
    states: np.ndarray
    nominal_states: np.ndarray
    controls: np.ndarray
    dt: float
    frame_dt: float
    control_duration: float


def _wrap_angle(theta: float) -> float:
    return float((theta + np.pi) % (2.0 * np.pi) - np.pi)


def _control_kinematic_car(t: float) -> np.ndarray:
    velocity = 0.62 + 0.08 * np.sin(0.65 * t)
    steering = 0.16 * np.sin(0.55 * t + 0.40)
    return np.array([velocity, steering], dtype=float)


def _control_double_integrator(t: float) -> np.ndarray:
    return np.array(
        [
            0.165 * np.sin(0.55 * t + 0.25),
            0.170 * np.sin(0.48 * t + 1.40),
            0.125 * np.cos(0.70 * t + 0.25),
        ],
        dtype=float,
    )


def _control_pushing_object(t: float) -> np.ndarray:
    # Pushing controls are interpreted by the learned pushing model as
    # [push mode/strength, side offset, push duration/scale].
    # Keep the push almost centered so the learned model moves smoothly across
    # the view instead of turning out of frame.
    side = 0.004 * np.sin(0.50 * t)
    push_scale = 0.070
    return np.array([0.0, side, push_scale], dtype=float)


def _sample_pushing_control(system, state: np.ndarray, segment: int) -> np.ndarray:
    """Sample varied pushing controls and keep one that mostly advances right."""
    state = np.asarray(state, dtype=float).reshape(-1)
    best_control = None
    best_score = -np.inf
    desired_y = -0.60 + 0.070 * np.sin(0.95 * float(segment) + 0.45)

    for _ in range(96):
        control = np.array(
            [
                np.random.uniform(0.0, 0.04),
                np.random.uniform(-0.120, 0.095),
                np.random.uniform(0.045, 0.090),
            ],
            dtype=float,
        )
        next_state = np.asarray(system.propagate(state, control, 2.0), dtype=float)
        dx = float(next_state[0] - state[0])
        dy_target = abs(float(next_state[1] - desired_y))
        dtheta = abs(_wrap_angle(float(next_state[2] - state[2])))

        if next_state[0] < -0.20 or next_state[0] > 0.78 or next_state[1] < -0.82 or next_state[1] > -0.34:
            continue

        # Prefer rightward progress, but leave enough randomness that the
        # learned propagation produces a visibly physical, non-straight path.
        score = dx - 0.30 * dy_target - 0.050 * dtheta + np.random.normal(0.0, 0.014)
        if dx > 0.012 and score > best_score:
            best_control = control
            best_score = score

    if best_control is not None:
        return best_control
    return _control_pushing_object(0.0)


def _interpolate_pushing_pose(
    start: np.ndarray,
    end: np.ndarray,
    control: np.ndarray,
    alpha: float,
) -> np.ndarray:
    start = np.asarray(start, dtype=float).reshape(-1)
    end = np.asarray(end, dtype=float).reshape(-1)
    control = np.asarray(control, dtype=float).reshape(-1)
    a = float(np.clip(alpha, 0.0, 1.0))
    smooth = a * a * (3.0 - 2.0 * a)

    pose = start + smooth * (end - start)
    delta_xy = end[:2] - start[:2]
    length = float(np.linalg.norm(delta_xy))
    if length > 1e-9:
        normal = np.array([-delta_xy[1], delta_xy[0]], dtype=float) / length
        side = float(control[1]) if control.shape[0] > 1 else 0.0
        curvature = np.clip(1.8 * side, -0.11, 0.11)
        pose[:2] += normal * curvature * np.sin(np.pi * a)

    pose[2] = _wrap_angle(start[2] + smooth * _wrap_angle(float(end[2] - start[2])))
    return pose


def _demo_config(system_name: str, dt: float) -> dict:
    if system_name == "kinematic_car":
        return {
            "start_state": [0.25, 0.25, 0.0],
            "propagation_step_size": dt,
            "sampling_position_std": 0.0045,
            "sampling_rotation_std": 0.030,
            "state_bounds": [(0.0, 5.0), (0.0, 5.0)],
        }
    if system_name == "double_integrator":
        return {
            "start_state": [-2.25, -1.80, 0.35, 0.60, 0.38, 0.30],
            "propagation_step_size": dt,
            "sampling_position_std": 0.0,
            "sampling_velocity_std": 0.018,
            "state_bounds": [(-2.5, 7.1), (-2.0, 3.25), (0.15, 2.95)],
        }
    if system_name == "pushing_object":
        return {
            "start_state": [0.25, -0.58, np.pi],
            "propagation_step_size": dt,
            "sampling_position_std": 0.00025,
            "sampling_rotation_std": 0.002,
            "state_bounds": [(-0.25, 0.85), (-0.88, -0.28)],
        }
    raise ValueError(f"Unsupported system: {system_name}")


def _control_fn(system_name: str) -> Callable[[float], np.ndarray]:
    if system_name == "kinematic_car":
        return _control_kinematic_car
    if system_name == "double_integrator":
        return _control_double_integrator
    if system_name == "pushing_object":
        return _control_pushing_object
    raise ValueError(f"Unsupported system: {system_name}")


def simulate_demo(system_name: str, duration: float, fps: float, seed: int) -> DemoData:
    rng_state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        frame_dt = 1.0 / float(fps)
        control_duration = 2.0 if system_name == "pushing_object" else frame_dt
        total_frames = max(2, int(round(float(duration) * float(fps))))
        system = get_system(system_name)
        config = _demo_config(system_name, control_duration)
        simulator = create_simulator(system_name, "gaussian", config=config)
        simulator.reset()

        current_nominal = np.asarray(config["start_state"], dtype=float)
        states = [np.asarray(simulator.get_state(), dtype=float)]
        nominal_states = [current_nominal.copy()]
        controls = []
        control_fn = _control_fn(system_name)

        if system_name == "pushing_object":
            num_segments = max(1, int(np.ceil(float(duration) / control_duration)))
            segment_frames = max(1, int(round(control_duration * float(fps))))
            for segment in range(num_segments):
                t = segment * control_duration
                control = _sample_pushing_control(system, current_nominal, segment)
                controls.append(control.copy())
                prev_state = states[-1].copy()
                prev_nominal = current_nominal.copy()
                executed_end = np.asarray(
                    simulator.execute_segment(control, control_duration),
                    dtype=float,
                )
                nominal_end = np.asarray(
                    system.propagate(current_nominal, control, control_duration),
                    dtype=float,
                )
                nominal_end[2] = _wrap_angle(float(nominal_end[2]))

                for frame_in_segment in range(1, segment_frames + 1):
                    if len(states) >= total_frames + 1:
                        break
                    alpha = float(frame_in_segment) / float(segment_frames)
                    interp_state = _interpolate_pushing_pose(
                        prev_state,
                        executed_end,
                        control,
                        alpha,
                    )
                    interp_nominal = _interpolate_pushing_pose(
                        prev_nominal,
                        nominal_end,
                        control,
                        alpha,
                    )
                    states.append(interp_state.copy())
                    nominal_states.append(interp_nominal.copy())
                current_nominal = nominal_end.copy()
                if len(states) >= total_frames + 1:
                    break
        else:
            for step in range(total_frames):
                t = step * frame_dt
                control = control_fn(t)
                controls.append(control.copy())
                executed = np.asarray(simulator.execute_segment(control, frame_dt), dtype=float)
                current_nominal = np.asarray(
                    system.propagate(current_nominal, control, frame_dt),
                    dtype=float,
                )
                if system_name == "kinematic_car":
                    current_nominal[2] = _wrap_angle(float(current_nominal[2]))
                states.append(executed.copy())
                nominal_states.append(current_nominal.copy())

        return DemoData(
            system_name=system_name,
            states=np.asarray(states, dtype=float),
            nominal_states=np.asarray(nominal_states, dtype=float),
            controls=np.asarray(controls, dtype=float),
            dt=frame_dt,
            frame_dt=frame_dt,
            control_duration=control_duration,
        )
    finally:
        np.random.set_state(rng_state)


def _sample_mujoco_car_control(t: float) -> np.ndarray:
    velocity = np.random.uniform(0.42, 0.72)
    steering_center = 0.12 * np.sin(0.70 * t + 0.35)
    steering = np.clip(steering_center + np.random.normal(0.0, 0.045), -0.22, 0.22)
    return np.array([velocity, steering], dtype=float)


def run_mujoco_demo(system_name: str, duration: float, seed: int) -> None:
    if system_name == "double_integrator":
        raise ValueError("MuJoCo mode is available for kinematic_car and pushing_object demos only.")

    rng_state = np.random.get_state()
    np.random.seed(int(seed))
    simulator = None
    try:
        if system_name == "kinematic_car":
            control_duration = 0.50
            config = _demo_config(system_name, 0.02)
            simulator = create_simulator(system_name, "mujoco", config=config)
            print("[mujoco demo] viewer launched for kinematic_car")
            print("[mujoco demo] waiting 5.0s before applying controls")
            time.sleep(5.0)
            print("[mujoco demo] applying sampled forward controls; close the viewer or press Ctrl+C to stop")
            num_segments = max(1, int(np.ceil(float(duration) / control_duration)))
            for segment in range(num_segments):
                t = segment * control_duration
                control = _sample_mujoco_car_control(t)
                print(
                    f"[mujoco demo] car push {segment + 1:02d}/{num_segments}: "
                    f"v={control[0]:.3f}, steer={control[1]:+.3f}"
                )
                simulator.execute_segment(control, control_duration)

        elif system_name == "pushing_object":
            control_duration = 2.0
            system = get_system(system_name)
            config = _demo_config(system_name, control_duration)
            simulator = create_simulator(system_name, "mujoco", config=config)
            simulator.set_state(config["start_state"])
            current_nominal = np.asarray(config["start_state"], dtype=float)
            print("[mujoco demo] viewer launched for UR10 pushing")
            print("[mujoco demo] waiting 5.0s before applying controls")
            time.sleep(5.0)
            print("[mujoco demo] applying sampled 2-second pushes; close the viewer or press Ctrl+C to stop")
            num_segments = max(1, int(np.ceil(float(duration) / control_duration)))
            for segment in range(num_segments):
                control = _sample_pushing_control(system, current_nominal, segment)
                print(
                    f"[mujoco demo] robot push {segment + 1:02d}/{num_segments}: "
                    f"rotation={control[0]:.3f}, side={control[1]:+.3f}, distance={control[2]:.3f}"
                )
                simulator.execute_segment(control, control_duration)
                current_nominal = np.asarray(simulator.get_state(), dtype=float)
                current_nominal[2] = _wrap_angle(float(current_nominal[2]))
        else:
            raise ValueError(f"Unsupported MuJoCo demo system: {system_name}")

        time.sleep(1.0)
    finally:
        np.random.set_state(rng_state)
        if simulator is not None:
            try:
                simulator.stop()
            except Exception:
                pass
            try:
                simulator.close()
            except Exception:
                pass


def _draw_arrow_pose(ax, state, *, scale: float, color: str = ORANGE, label: str | None = None):
    x, y, theta = float(state[0]), float(state[1]), float(state[2])
    direction = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    normal = np.array([-np.sin(theta), np.cos(theta)], dtype=float)
    nose = np.array([x, y]) + direction * scale * 0.95
    rear = np.array([x, y]) - direction * scale * 0.55
    left = rear + normal * scale * 0.42
    right = rear - normal * scale * 0.42
    ax.add_patch(
        Polygon(
            np.vstack([nose, left, right]),
            closed=True,
            facecolor=color,
            edgecolor="white",
            linewidth=1.4,
            label=label,
            zorder=8,
            path_effects=[pe.Stroke(linewidth=2.8, foreground="#222222", alpha=0.50), pe.Normal()],
        )
    )
    start = np.array([x, y]) + direction * scale * 0.18
    end = np.array([x, y]) + direction * scale * 1.9
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=30,
            linewidth=6.0,
            color="white",
            zorder=8.5,
            shrinkA=0,
            shrinkB=0,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=27,
            linewidth=3.4,
            color=RED,
            zorder=9,
            shrinkA=0,
            shrinkB=0,
        )
    )


def _draw_pushing_box(ax, state, *, size=(0.1628, 0.2139), label: str | None = None):
    x, y, theta = float(state[0]), float(state[1]), float(state[2])
    half = np.array([size[0] / 2.0, size[1] / 2.0])
    corners = np.array(
        [
            [-half[0], -half[1]],
            [half[0], -half[1]],
            [half[0], half[1]],
            [-half[0], half[1]],
        ],
        dtype=float,
    )
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=float,
    )
    world = corners @ rot.T + np.array([x, y])
    ax.add_patch(
        Polygon(
            world,
            closed=True,
            facecolor=ORANGE,
            edgecolor="white",
            linewidth=1.6,
            label=label,
            zorder=7,
            path_effects=[pe.Stroke(linewidth=3.0, foreground="#222222", alpha=0.45), pe.Normal()],
        )
    )
    direction = np.array([np.cos(theta), np.sin(theta)])
    ax.add_patch(
        FancyArrowPatch(
            np.array([x, y]),
            np.array([x, y]) + direction * (max(size) * 0.78),
            arrowstyle="-|>",
            mutation_scale=22,
            linewidth=2.8,
            color=RED,
            zorder=8,
        )
    )


def _setup_2d_axes(ax, xlim, ylim):
    ax.set_facecolor(PANEL)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=GRID, linewidth=0.85)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)
    ax.tick_params(
        colors="black",
        width=1.1,
        length=0.0,
        labelbottom=False,
        labelleft=False,
    )


def _render_kinematic_frame(fig, ax, data: DemoData, frame: int):
    ax.clear()
    states = data.states[: frame + 1]
    nominal = data.nominal_states[: frame + 1]
    _setup_2d_axes(ax, (0.0, 5.30), (0.0, 5.0))
    ax.plot(nominal[:, 0], nominal[:, 1], color="#8A949E", linewidth=2.4)
    ax.plot(states[:, 0], states[:, 1], color=BLUE, linewidth=4.0)
    _draw_arrow_pose(ax, states[-1], scale=0.18)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.98, bottom=0.04)


def _cuboid_vertices(center, size):
    cx, cy, cz = center
    sx, sy, sz = np.asarray(size, dtype=float) / 2.0
    return np.array(
        [
            [cx - sx, cy - sy, cz - sz],
            [cx + sx, cy - sy, cz - sz],
            [cx + sx, cy + sy, cz - sz],
            [cx - sx, cy + sy, cz - sz],
            [cx - sx, cy - sy, cz + sz],
            [cx + sx, cy - sy, cz + sz],
            [cx + sx, cy + sy, cz + sz],
            [cx - sx, cy + sy, cz + sz],
        ],
        dtype=float,
    )


def _draw_particle_cube(ax, center, size=(0.12, 0.12, 0.12)):
    v = _cuboid_vertices(center, size)
    faces = [
        [v[i] for i in [0, 1, 2, 3]],
        [v[i] for i in [4, 5, 6, 7]],
        [v[i] for i in [0, 1, 5, 4]],
        [v[i] for i in [2, 3, 7, 6]],
        [v[i] for i in [1, 2, 6, 5]],
        [v[i] for i in [0, 3, 7, 4]],
    ]
    collection = Poly3DCollection(
        faces,
        facecolors=ORANGE,
        edgecolors="white",
        linewidths=0.8,
        alpha=0.96,
    )
    collection.set_path_effects([pe.Stroke(linewidth=2.0, foreground="#222222", alpha=0.35), pe.Normal()])
    ax.add_collection3d(collection)


def _render_double_frame(fig, ax, data: DemoData, frame: int):
    ax.clear()
    states = data.states[: frame + 1]
    nominal = data.nominal_states[: frame + 1]
    ax.set_facecolor("white")
    ax.plot(nominal[:, 0], nominal[:, 1], nominal[:, 2], color="#8A949E", linewidth=2.5)
    ax.plot(states[:, 0], states[:, 1], states[:, 2], color=BLUE, linewidth=4.0)
    ax.scatter(states[-1, 0], states[-1, 1], states[-1, 2], s=90, color=ORANGE, edgecolor="white", linewidth=1.4, depthshade=False)
    _draw_particle_cube(ax, states[-1, :3])
    vel = states[-1, 3:6]
    speed = float(np.linalg.norm(vel))
    if speed > 1e-9:
        direction = vel / speed
        ax.quiver(
            states[-1, 0],
            states[-1, 1],
            states[-1, 2],
            direction[0],
            direction[1],
            direction[2],
            length=0.45,
            color=RED,
            linewidth=2.6,
            arrow_length_ratio=0.25,
            normalize=True,
        )
    ax.set_xlim(-2.45, 7.05)
    ax.set_ylim(-1.95, 3.25)
    ax.set_zlim(0.15, 2.95)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(length=0)
    ax.view_init(elev=25, azim=38 + 10.0 * frame / max(1, len(data.states) - 1))
    ax.grid(True, color=GRID)
    fig.subplots_adjust(left=0.00, right=1.00, top=1.00, bottom=0.00)


def _render_pushing_frame(fig, ax, data: DemoData, frame: int):
    ax.clear()
    states = data.states[: frame + 1]
    nominal = data.nominal_states[: frame + 1]
    _setup_2d_axes(ax, (-0.25, 0.85), (-0.88, -0.28))
    ax.plot(nominal[:, 0], nominal[:, 1], color="#8A949E", linewidth=2.4)
    ax.plot(states[:, 0], states[:, 1], color=BLUE, linewidth=4.0)
    _draw_pushing_box(ax, states[-1])
    fig.subplots_adjust(left=0.04, right=0.98, top=0.98, bottom=0.04)


def render_demo(data: DemoData, output: str, fps: float, dpi: int, show_final: bool = False) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    if data.system_name == "double_integrator":
        fig = plt.figure(figsize=(10.5, 7.2), facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        render_frame = _render_double_frame
    else:
        figsize = (10.5, 6.0) if data.system_name == "pushing_object" else (10.5, 7.2)
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        render_frame = _render_kinematic_frame if data.system_name == "kinematic_car" else _render_pushing_frame

    ext = os.path.splitext(output)[1].lower()
    if ext == ".gif":
        writer = PillowWriter(fps=float(fps))
    elif writers.is_available("ffmpeg"):
        writer = FFMpegWriter(
            fps=float(fps),
            bitrate=24000,
            extra_args=["-pix_fmt", "yuv420p"],
        )
    else:
        raise RuntimeError("ffmpeg is not available; use an output ending in .gif instead.")

    with writer.saving(fig, output, dpi=int(dpi)):
        for frame in range(len(data.states)):
            render_frame(fig, ax, data, frame)
            writer.grab_frame()

    if show_final:
        render_frame(fig, ax, data, len(data.states) - 1)
        plt.show()
    plt.close(fig)
    return output


def default_output(system_name: str, ext: str = ".mp4") -> str:
    return os.path.join(
        "results", "simulation_demo", f"{system_name}_gaussian_demo{ext}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a Gaussian-noise video or launch a live MuJoCo dynamics demo."
    )
    parser.add_argument(
        "system_name",
        choices=["kinematic_car", "double_integrator", "pushing_object"],
        help="System to animate.",
    )
    parser.add_argument("--output", default=None, help="Output .mp4 or .gif path.")
    parser.add_argument(
        "--mode",
        choices=["gaussian", "mujoco"],
        default="gaussian",
        help="gaussian renders a video; mujoco launches the live MuJoCo viewer.",
    )
    parser.add_argument("--duration", type=float, default=10.0, help="Animation duration in seconds.")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second.")
    parser.add_argument("--dpi", type=int, default=350, help="Render DPI.")
    parser.add_argument("--seed", type=int, default=7, help="Gaussian-noise random seed.")
    parser.add_argument("--show-final", action="store_true", help="Open the final frame after rendering.")
    args = parser.parse_args()

    if float(args.duration) <= 0.0:
        raise ValueError("--duration must be positive")
    if float(args.fps) <= 0.0:
        raise ValueError("--fps must be positive")
    if int(args.dpi) <= 0:
        raise ValueError("--dpi must be positive")

    if args.mode == "mujoco":
        print(f"[demo] launching MuJoCo {args.system_name} demo for {float(args.duration):.2f}s")
        run_mujoco_demo(args.system_name, float(args.duration), int(args.seed))
        print("[demo] MuJoCo demo finished")
        return

    output = args.output or default_output(args.system_name)
    print(f"[demo] simulating {args.system_name} for {float(args.duration):.2f}s at {float(args.fps):.1f} fps")
    data = simulate_demo(args.system_name, float(args.duration), float(args.fps), int(args.seed))
    print(f"[demo] rendering -> {output}")
    rendered = render_demo(data, output, float(args.fps), int(args.dpi), show_final=bool(args.show_final))
    print(f"[demo] saved {rendered}")


if __name__ == "__main__":
    main()
