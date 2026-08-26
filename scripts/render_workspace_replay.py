#!/usr/bin/env python3
"""Render AURA workspace replay .npz files to MP4/GIF or PNG frames."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/aura_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


def _video_path_for_replay(replay_path: str, output_dir: str, ext: str) -> str:
    stem = os.path.splitext(os.path.basename(replay_path))[0]
    clean_ext = ext if ext.startswith(".") else f".{ext}"
    return os.path.join(output_dir, f"{stem}{clean_ext}")


def _render_batch(args) -> None:
    replay_dir = os.path.abspath(args.replay)
    pattern = os.path.join(replay_dir, args.pattern)
    replay_paths = sorted(glob.glob(pattern))
    if args.limit is not None:
        replay_paths = replay_paths[: max(0, int(args.limit))]
    if not replay_paths:
        raise SystemExit(f"No replay files matched {pattern!r}.")

    output_dir = os.path.abspath(args.output_dir or replay_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[batch] replay dir: {replay_dir}")
    print(f"[batch] output dir: {output_dir}")
    print(f"[batch] matched replay files: {len(replay_paths)}")
    print(f"[batch] video extension: {args.video_ext}")

    rendered = 0
    skipped = 0
    failed = 0
    for index, replay_path in enumerate(replay_paths, start=1):
        video_path = _video_path_for_replay(replay_path, output_dir, args.video_ext)
        if (
            not bool(args.overwrite)
            and os.path.exists(video_path)
            and os.path.getsize(video_path) > 0
        ):
            skipped += 1
            print(f"[batch] {index}/{len(replay_paths)} skip existing: {video_path}")
            continue

        frames_dir = None
        if args.frames_dir:
            frames_dir = os.path.join(
                os.path.abspath(args.frames_dir),
                os.path.splitext(os.path.basename(replay_path))[0],
            )

        print(f"[batch] {index}/{len(replay_paths)} render: {replay_path}")
        if bool(args.dry_run):
            print(f"[batch] dry-run video -> {video_path}")
            skipped += 1
            continue

        try:
            from scripts.plot_workspace import render_workspace_replay

            render_workspace_replay(
                replay_path,
                video_path=video_path,
                frames_dir=frames_dir,
                fps=float(args.fps),
                dpi=int(args.dpi),
                show=False,
            )
            rendered += 1
        except Exception as exc:
            failed += 1
            print(f"[batch][ERROR] {replay_path}: {exc}")

    print(
        "[batch] finished: "
        f"rendered={rendered}, skipped={skipped}, failed={failed}, total={len(replay_paths)}"
    )
    if failed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render compact AURA workspace replay data into a video or frame directory."
    )
    parser.add_argument(
        "replay",
        help="Path to a workspace replay .npz file, or a directory of .npz replay files.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Output video path for a single replay. Use .mp4 when ffmpeg is installed, or .gif.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Batch output directory. Default: the replay directory itself.",
    )
    parser.add_argument(
        "--pattern",
        default="*.npz",
        help="Batch replay filename pattern inside a replay directory. Default: *.npz.",
    )
    parser.add_argument(
        "--video-ext",
        default=".mp4",
        choices=[".mp4", "mp4", ".gif", "gif"],
        help="Batch video extension. Default: .mp4.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render videos that already exist. Default: skip existing non-empty videos.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Batch debug helper: render only the first N matching replay files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print batch render targets without writing videos.",
    )
    parser.add_argument(
        "--frames-dir",
        default=None,
        help=(
            "Optional directory for PNG frames. For batch mode, one frame subdirectory "
            "is created per replay. If omitted and video writing is unavailable, a "
            "<video>_frames directory is used."
        ),
    )
    parser.add_argument("--fps", type=float, default=3.0, help="Video frames per second.")
    parser.add_argument("--dpi", type=int, default=400, help="Render DPI.")
    parser.add_argument(
        "--show-final",
        action="store_true",
        help="Also open the final frame in a matplotlib window if a GUI backend is available.",
    )
    args = parser.parse_args()

    if float(args.fps) <= 0.0:
        raise ValueError("--fps must be positive")
    if int(args.dpi) <= 0:
        raise ValueError("--dpi must be positive")

    if os.path.isdir(args.replay):
        if args.video:
            raise ValueError("--video is only valid when rendering one replay file.")
        if args.show_final:
            raise ValueError("--show-final is only valid when rendering one replay file.")
        _render_batch(args)
        return

    if args.output_dir:
        raise ValueError("--output-dir is only valid when rendering a replay directory.")

    from scripts.plot_workspace import render_workspace_replay

    render_workspace_replay(
        args.replay,
        video_path=args.video,
        frames_dir=args.frames_dir,
        fps=float(args.fps),
        dpi=int(args.dpi),
        show=bool(args.show_final),
    )


if __name__ == "__main__":
    main()
