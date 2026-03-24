#!/usr/bin/env python3
"""
Parallel-render all scenes using multiprocessing, then concatenate.

Usage:
    python render_all.py              # 1080p 60fps, parallel
    python render_all.py --quality l  # fast preview
    python render_all.py --quality k  # 4K
    python render_all.py --workers 4  # limit parallelism
"""
import subprocess, sys, os, time
from pathlib import Path
from multiprocessing import Pool, cpu_count

SCENES = [
    "TitleScene",
    "WhatIsAGraphScene",
    "MapRepresentationsScene",
    "FrontierConceptScene",
    "BFSScene",
    "PathReconstructionScene",
    "EarlyExitScene",
    "MovementCostsScene",
    "DijkstraScene",
    "HeuristicScene",
    "GreedyBFSScene",
    "AStarScene",
    "TripleComparisonScene",
    "SummaryScene",
]

QUALITY = {
    "l": ("480p15",  "-ql"),
    "m": ("720p30",  "-qm"),
    "h": ("1080p60", "-qh --fps 60"),
    "k": ("2160p60", "-qk --fps 60"),
}


def render_scene(args):
    """Render a single scene. Called from worker pool."""
    idx, scene, flags, src = args
    t0 = time.time()
    cmd = f"manim {flags} {src} {scene}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - t0
    status = "✓" if result.returncode == 0 else "✗"
    print(f"  [{idx+1:2d}/{len(SCENES)}] {status} {scene:<32s} ({elapsed:.1f}s)")
    if result.returncode != 0:
        # Print last 3 lines of stderr for debugging
        err_lines = result.stderr.strip().split('\n')[-3:]
        for line in err_lines:
            print(f"       {line}")
    return (idx, scene, result.returncode)


def main():
    # Parse arguments
    q = "h"
    workers = max(1, cpu_count() - 1)  # leave 1 core free

    args = sys.argv[1:]
    if "--quality" in args:
        i = args.index("--quality")
        q = args[i+1] if i+1 < len(args) else "k"
    if "--workers" in args:
        i = args.index("--workers")
        workers = int(args[i+1]) if i+1 < len(args) else workers

    folder, flags = QUALITY.get(q, QUALITY["h"])
    src = "main.py"

    print(f"\n{'='*62}")
    print(f"  Rendering {len(SCENES)} scenes at {folder}")
    print(f"  Workers: {workers}  (CPUs: {cpu_count()})")
    print(f"{'='*62}\n")

    t_start = time.time()

    # Render all scenes in parallel
    tasks = [(i, scene, flags, src) for i, scene in enumerate(SCENES)]

    with Pool(processes=workers) as pool:
        results = pool.map(render_scene, tasks)

    t_render = time.time() - t_start
    print(f"\n  Render time: {t_render:.1f}s "
          f"({t_render/len(SCENES):.1f}s/scene avg, "
          f"{t_render/60:.1f} min total)")

    # Collect rendered files in scene order
    video_dir = Path(f"media/videos/main/{folder}")

    rendered = [video_dir / f"{s}.mp4" for s in SCENES]

    print(rendered)

    # rendered = []
    # for idx, scene, rc in sorted(results, key=lambda x: x[0]):
    #     if rc != 0:
    #         continue
    #     candidates = sorted(video_dir.glob(f"{scene}.*"),
    #                        key=os.path.getmtime, reverse=True)
    #     if candidates:
    #         rendered.append(candidates[0])
    #
    # if not rendered:
    #     print("\n  No scenes rendered successfully.")
    #     return

    # Concatenate
    print(f"\n{'='*62}")
    print(f"  Concatenating {len(rendered)}/{len(SCENES)} scenes...")
    print(f"{'='*62}")

    fl = Path("filelist.txt")
    fl.write_text("\n".join(f"file '{p.resolve()}'" for p in rendered) + "\n")
    out = f"color_spaces_full_{folder}.mp4"
    r = subprocess.run(
        f"ffmpeg -y -f concat -safe 0 -i {fl} -c copy {out}",
        shell=True, capture_output=True)
    fl.unlink(missing_ok=True)

    if r.returncode == 0:
        size_mb = os.path.getsize(out) / 1e6
        print(f"\n  ✅ {out}  ({size_mb:.1f} MB)")
    else:
        print(f"\n  ⚠  ffmpeg failed. Individual files in {video_dir}")

    total = time.time() - t_start
    print(f"  Total time: {total:.1f}s ({total/60:.1f} min)\n")


if __name__ == "__main__":
    main()