#!/usr/bin/env python3
"""Render recorded MuJoCo snapshots in a separate process.

Keeping offscreen rendering out of the live viewer process avoids random native
crashes from shared GLFW/OpenGL/MuJoCo state during teardown.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import subprocess
import sys
import tempfile


def save_recording_in_subprocess(
    *,
    kind: str,
    xml_path: str,
    snapshots: list,
    output_path: str,
    fps: float,
    width: int,
    height: int,
    goal_region=None,
    table_height: float = 0.0,
    camera_config: dict | None = None,
) -> bool:
    import imageio.v2 as imageio

    temp_dir = tempfile.mkdtemp(prefix="aura_mujoco_recording_", dir="/tmp")
    payload_path = os.path.join(temp_dir, "payload.pkl")
    frame_dir = os.path.join(temp_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    payload = {
        "kind": kind,
        "xml_path": xml_path,
        "snapshots": snapshots,
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "goal_region": goal_region,
        "table_height": float(table_height),
        "camera_config": camera_config,
    }
    try:
        with open(payload_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        frame_paths = [
            os.path.join(frame_dir, f"frame_{idx + 1:06d}.png")
            for idx in range(len(snapshots))
        ]
        chunk_size = max(1, int(round(float(fps))))

        def _render_range(start: int, end: int) -> bool:
            result = subprocess.run(
                [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--payload",
                    payload_path,
                    "--frame-dir",
                    frame_dir,
                    "--start",
                    str(start),
                    "--end",
                    str(end),
                ],
                check=False,
            )
            expected = frame_paths[start:end]
            frames_exist = all(os.path.exists(path) for path in expected)
            if result.returncode != 0 and frames_exist:
                print(
                    "[mujoco video] worker exited nonzero after writing frames; "
                    "continuing",
                    flush=True,
                )
            return frames_exist

        for chunk_start in range(0, len(snapshots), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(snapshots))
            if not _render_range(chunk_start, chunk_end):
                print(
                    f"[mujoco video] retrying frames {chunk_start + 1}-{chunk_end} one at a time",
                    flush=True,
                )
                for frame_idx in range(chunk_start, chunk_end):
                    if not os.path.exists(frame_paths[frame_idx]) and not _render_range(
                        frame_idx, frame_idx + 1
                    ):
                        print(
                            f"[WARNING] missing MuJoCo frame {frame_idx + 1}; "
                            "video save failed",
                            flush=True,
                        )
                        return False
            print(
                f"[mujoco video] rendered {chunk_end}/{len(snapshots)} frames",
                flush=True,
            )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        writer_kwargs = {"fps": float(fps)}
        if not str(output_path).lower().endswith(".gif"):
            writer_kwargs["macro_block_size"] = 1
            writer_kwargs["codec"] = "libx264"
            writer_kwargs["ffmpeg_params"] = ["-pix_fmt", "yuv420p"]
        writer = None
        try:
            print(f"[mujoco video] encoding {len(frame_paths)} frames", flush=True)
            writer = imageio.get_writer(output_path, **writer_kwargs)
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))
        finally:
            if writer is not None:
                writer.close()
        return True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _add_plan_path_to_scene(mujoco, scene, plan_path, *, kind: str, table_height: float):
    if not plan_path or len(plan_path) < 2:
        return
    color = [0.02, 0.25, 0.78, 0.92]
    z = float(table_height) + 0.035 if kind == "pushing" else 0.035
    radius = 0.012 if kind == "pushing" else 0.018
    import numpy as np

    for start, end in zip(plan_path[:-1], plan_path[1:]):
        if scene.ngeom >= scene.maxgeom:
            return
        a = np.asarray(start, dtype=float).reshape(-1)
        b = np.asarray(end, dtype=float).reshape(-1)
        if a.size < 2 or b.size < 2:
            continue
        from_pt = np.array([a[0], a[1], z], dtype=float)
        to_pt = np.array([b[0], b[1], z], dtype=float)
        if np.linalg.norm(to_pt - from_pt) < 1e-9:
            continue
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            radius,
            from_pt,
            to_pt,
        )
        geom.rgba[:] = color
        scene.ngeom += 1


def _add_goal_region_to_scene(mujoco, scene, goal_region, *, kind: str, table_height: float):
    if goal_region is None:
        return
    import numpy as np

    x, y, radius = goal_region
    segments = 48
    half_arc = np.pi * float(radius) / float(segments)
    z = float(table_height) + 0.008 if kind == "pushing" else 0.010
    thickness = 0.008 if kind == "pushing" else 0.012
    height = 0.003 if kind == "pushing" else 0.004
    for i in range(segments):
        if scene.ngeom >= scene.maxgeom:
            return
        angle = 2.0 * np.pi * float(i) / float(segments)
        geom = scene.geoms[scene.ngeom]
        mat = np.array(
            [
                [np.cos(angle + np.pi / 2.0), -np.sin(angle + np.pi / 2.0), 0.0],
                [np.sin(angle + np.pi / 2.0), np.cos(angle + np.pi / 2.0), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_BOX,
            np.array([half_arc, thickness, height], dtype=float),
            np.array([x + radius * np.cos(angle), y + radius * np.sin(angle), z], dtype=float),
            mat.reshape(-1),
            np.array([0.05, 0.85, 0.12, 0.85], dtype=float),
        )
        scene.ngeom += 1


def _render_payload(payload: dict, frame_dir: str, start: int, end: int) -> int:
    import imageio.v2 as imageio
    import mujoco
    import numpy as np

    kind = str(payload["kind"])
    width = int(payload["width"])
    height = int(payload["height"])
    snapshots = list(payload["snapshots"])
    goal_region = payload.get("goal_region")
    table_height = float(payload.get("table_height", 0.0))
    camera_config = payload.get("camera_config") or {}
    start = max(0, int(start))
    end = min(len(snapshots), int(end))

    if start >= end:
        return 0
    os.makedirs(frame_dir, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(payload["xml_path"]))
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height, width)
    try:
        for frame_idx, (qpos, qvel, ctrl, lookat, plan_path) in enumerate(
            snapshots[start:end], start=start + 1
        ):
            qpos = np.asarray(qpos, dtype=float).reshape(-1)
            qvel = np.asarray(qvel, dtype=float).reshape(-1)
            ctrl = np.asarray(ctrl, dtype=float).reshape(-1)
            data.qpos[: min(qpos.size, data.qpos.size)] = qpos[: data.qpos.size]
            data.qvel[: min(qvel.size, data.qvel.size)] = qvel[: data.qvel.size]
            data.ctrl[: min(ctrl.size, data.ctrl.size)] = ctrl[: data.ctrl.size]
            mujoco.mj_forward(model, data)
            cam = mujoco.MjvCamera()
            camera_lookat = camera_config.get("lookat", lookat)
            cam.lookat[:] = np.asarray(camera_lookat, dtype=float).reshape(-1)[:3]
            cam.distance = float(
                camera_config.get("distance", 1.3 if kind == "pushing" else 3.0)
            )
            cam.azimuth = float(camera_config.get("azimuth", 135))
            cam.elevation = float(camera_config.get("elevation", -55))
            renderer.update_scene(data, camera=cam)
            _add_goal_region_to_scene(
                mujoco,
                renderer.scene,
                goal_region,
                kind=kind,
                table_height=table_height,
            )
            _add_plan_path_to_scene(
                mujoco,
                renderer.scene,
                plan_path,
                kind=kind,
                table_height=table_height,
            )
            frame = np.ascontiguousarray(renderer.render())
            frame_path = os.path.join(frame_dir, f"frame_{frame_idx:06d}.png")
            imageio.imwrite(frame_path, frame)
        print(
            f"[mujoco video worker] rendered {start + 1}-{end}/{len(snapshots)}",
            flush=True,
        )
    finally:
        renderer.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--frame-dir", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()
    with open(args.payload, "rb") as handle:
        payload = pickle.load(handle)
    return _render_payload(payload, args.frame_dir, args.start, args.end)


if __name__ == "__main__":
    raise SystemExit(main())
