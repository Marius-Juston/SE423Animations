"""Offline replay of lidar_slam_2d.py that produces the data behind the
interactive SLAM walkthrough (site/lidar_slam/data/).

For each noise level it simulates the same diff-drive run, corrupts the
odometry with the ProducerSim noise model, simulates rolling LiDAR sweeps
(every beam from its own true pose, like ProducerSim) and feeds them through
the real undistort_scan / LidarSLAM2D / pl_icp_2d / search_loop_closure code.
Differences from the threaded pipeline, on purpose:
  - no threads or shared memory; odometry is read straight from the array;
  - the loop search for keyframe i runs synchronously right after i is added
    (the real LoopClosureWorker does the same work on another thread);
  - the SLAM's wheel-odometry noise model is set to the level's true noise
    parameters (a real robot would use calibrated values).

Loop candidates come from the current SLAM estimates, exactly like the
worker. If a level produced no loop closure at all, a ground-truth-assisted
search is used as a fallback and every such edge is flagged `gtAssisted`.

Each loop edge is [new kf, old kf, gtAssisted (0/1), |ICP measurement −
true relative pose| in metres]; the last field is an offline diagnostic the
real system cannot know, used by the site to mark wrong (aliased) loops.

Writes <out>/index.json plus one <out>/level_XX.json per noise level.

Usage:  uv run python src/slam/generate_cache.py --out <site>/lidar_slam/data
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import gtsam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lidar_slam_2d import (  # noqa: E402
    N_BEAMS, LIDAR_FOV,
    _build_world,
    _make_diff_drive_trajectory,
    _simulate_scan,
    _sweep_beam_poses,
    LidarSLAM2D,
    SLAMConfig,
    voxel_downsample,
    undistort_scan,
    search_loop_closure,
)

HZ = 50
LIDAR_EVERY = 5          # 50 Hz / 5 = 10 Hz
N_LEVELS = 12
WRONG_LOOP_M = 0.15      # loop measurement off by more than this = wrong


def noise_for(t: float) -> dict:
    """Noise mapping shared with the site's slider (app.jsx)."""
    return {"k_d": 0.5 * t, "k_th": 0.4 * t, "arw": 0.05 * t, "lidar": 0.10 * t}


def severity_for(n: dict) -> float:
    return n["k_d"] + n["k_th"] + 10.0 * n["arw"] + 5.0 * n["lidar"]   # = 1.9 t


def simulate_odometry(gt_traj, k_d, k_th, arw, rng):
    dt = 1.0 / HZ
    odom = [np.array(gt_traj[0], dtype=float)]
    noisy = np.array(gt_traj[0], dtype=float)
    for i in range(1, len(gt_traj)):
        gt, gp = gt_traj[i], gt_traj[i - 1]
        true_d = np.hypot(gt[0] - gp[0], gt[1] - gp[1])
        true_dth = (gt[2] - gp[2] + np.pi) % (2 * np.pi) - np.pi
        noisy_d = true_d + rng.normal(0, np.sqrt(k_d ** 2 * true_d))
        noisy_dth = true_dth + rng.normal(0, np.sqrt(k_th ** 2 * abs(true_dth) + arw ** 2 * dt))
        mid = noisy[2] + noisy_dth / 2.0
        noisy[0] += noisy_d * np.cos(mid)
        noisy[1] += noisy_d * np.sin(mid)
        noisy[2] = (noisy[2] + noisy_dth + np.pi) % (2 * np.pi) - np.pi
        odom.append(noisy.copy())
    return odom


def config_for(noise: dict) -> SLAMConfig:
    cfg = SLAMConfig()
    cfg.wheel_k_d, cfg.wheel_k_th, cfg.wheel_arw = noise["k_d"], noise["k_th"], noise["arw"]
    return cfg


def run_level(gt_traj, walls, noise, cfg, seed=42, use_gt_search=False):
    np.random.seed(seed)                   # lidar noise (_simulate_scan)
    rng = np.random.default_rng(seed)      # odometry noise
    odom_traj = simulate_odometry(gt_traj, noise["k_d"], noise["k_th"], noise["arw"], rng)
    n = len(gt_traj)
    ts = np.arange(n) / HZ
    odom_buf = np.column_stack([ts, np.asarray(odom_traj)])
    sweep = LIDAR_EVERY / HZ

    slam = LidarSLAM2D(cfg)
    kfTrajIdx, kfGt, kfOdom, kfOptOnline, kfCands, loopEdges = [], [], [], [], [], []
    last_pose = None
    delta_since_kf = gtsam.Pose2(0, 0, 0)
    last_loop_i = -10**9
    stats = {"icp_rejected": 0}

    for i in range(0, n, LIDAR_EVERY):
        # Rolling sweep ending at sample i, then motion compensation from the
        # (noisy) odometry available up to i — as in the threaded pipeline.
        beams = _sweep_beam_poses(gt_traj, i, LIDAR_EVERY)
        scan = _simulate_scan(gt_traj[i], walls, noise_std=noise["lidar"], beam_poses=beams)
        if cfg.enable_undistort:
            scan = undistort_scan(scan, ts[i] - sweep, sweep, tuple(odom_traj[i]),
                                  odom_buf[max(0, i - 60): i + 1], cfg)
        odom_pose = gtsam.Pose2(*odom_traj[i])

        if slam.n_kf == 0:
            slam.add_first_scan(scan, odom_pose)
            kfTrajIdx.append(i); kfGt.append(gt_traj[i]); kfOdom.append(odom_traj[i])
            kfOptOnline.append(list(slam.trajectory()[0])); kfCands.append([])
            last_pose = odom_pose
            continue

        delta_since_kf = delta_since_kf.compose(last_pose.between(odom_pose))
        last_pose = odom_pose
        d = delta_since_kf
        if not (np.hypot(d.x(), d.y()) > cfg.kf_trans or abs(d.theta()) > cfg.kf_rot):
            continue

        # --- same logic as RealtimeLidarSLAM._handle_keyframe --------------
        scan_ds = voxel_downsample(scan, cfg.voxel_size)
        submap = slam.get_submap(slam.n_kf - 1)
        dt = ts[i] - ts[kfTrajIdx[-1]]
        icp = slam.match_keyframe(scan_ds, delta_since_kf, submap, dt)
        if icp is None:
            stats["icp_rejected"] += 1
        j_new = slam.add_keyframe(scan_ds, delta_since_kf, icp, dt)
        slam.flush()
        delta_since_kf = gtsam.Pose2(0, 0, 0)

        kfTrajIdx.append(i); kfGt.append(gt_traj[i]); kfOdom.append(odom_traj[i])
        est = slam.trajectory()
        kfOptOnline.append(list(est[j_new]))

        # --- same logic as LoopClosureWorker (synchronous here) ------------
        if j_new - last_loop_i < cfg.loop_cooldown_kf:
            kfCands.append([])
            continue
        search_poses = np.asarray(kfGt, dtype=float) if use_gt_search else est
        best, tried = search_loop_closure(search_poses, slam.kf_scans, slam.kf_normals,
                                          j_new, cfg, slam.threshold.sigma)
        kfCands.append([[int(k), float(search_poses[k][0]), float(search_poses[k][1])]
                        for k in tried])
        if best is not None:
            last_loop_i = j_new
            slam.inject_loop_factor(best.j, best.i, best.rel, best.rmse, best.cov)
            slam.flush(cfg.isam_extra_iters_after_loop)
            # Offline-only diagnostic: how far is the ICP measurement from the
            # true relative pose? (perceptual aliasing at high noise)
            true_rel = gtsam.Pose2(*kfGt[best.j]).between(gtsam.Pose2(*kfGt[j_new]))
            e = true_rel.between(best.rel)
            loopEdges.append([j_new, int(best.j), 1 if use_gt_search else 0,
                              round(float(np.hypot(e.x(), e.y())), 3)])

    stats["sigma"] = slam.threshold.sigma
    return {
        "odomTraj": odom_traj,
        "kfTrajIdx": kfTrajIdx,
        "kfGt": kfGt,
        "kfOdom": kfOdom,
        "kfOpt": slam.trajectory().tolist(),
        "kfOptOnline": kfOptOnline,
        "kfCands": kfCands,
        "kfScans": slam.kf_scans,
        "loopEdges": loopEdges,
    }, stats


def rmse_xy(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(np.sum((a[:, :2] - b[:, :2]) ** 2, axis=1))))


def r4(x):
    return np.round(np.asarray(x, dtype=float), 4).tolist()


def main():
    here = Path(__file__).resolve()
    default_out = here.parents[3] / "SE-423---Class-Material" / "site" / "lidar_slam" / "data"
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=default_out,
                    help=f"output directory (default: {default_out})")
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    walls = _build_world()
    gt_traj = _make_diff_drive_trajectory(hz=HZ, side=8.0, laps=2)
    cfg = SLAMConfig()          # for the shared config fields in index.json

    levels = []
    print(f"{'lvl':>3} {'t':>5} {'sev':>6} | {'odom RMSE':>9} {'opt RMSE':>9} "
          f"{'final err':>9} | {'KF':>4} {'loops':>5} {'wrong':>5} {'gt-assist':>9} {'icp rej':>7}")
    ok = True
    for li, t in enumerate(np.linspace(0.0, 1.0, N_LEVELS)):
        noise = noise_for(float(t))
        lcfg = config_for(noise)
        data, stats = run_level(gt_traj, walls, noise, lcfg)
        gt_assisted = False
        if not data["loopEdges"]:
            data, stats = run_level(gt_traj, walls, noise, lcfg, use_gt_search=True)
            gt_assisted = True

        odom_rmse = rmse_xy(data["kfOdom"], data["kfGt"])
        opt_rmse = rmse_xy(data["kfOpt"], data["kfGt"])
        final_err = float(np.hypot(data["kfOpt"][-1][0] - data["kfGt"][-1][0],
                                   data["kfOpt"][-1][1] - data["kfGt"][-1][1]))
        n_loops = len(data["loopEdges"])
        n_wrong = sum(1 for e in data["loopEdges"] if e[3] > WRONG_LOOP_M)
        print(f"{li:3d} {t:5.2f} {severity_for(noise):6.3f} | {odom_rmse:9.4f} {opt_rmse:9.4f} "
              f"{final_err:9.4f} | {len(data['kfGt']):4d} {n_loops:5d} {n_wrong:5d} {str(gt_assisted):>9} "
              f"{stats['icp_rejected']:7d}")
        if li == 0:
            ok &= opt_rmse < 0.02
        else:
            ok &= opt_rmse <= odom_rmse
        ok &= n_loops > 0

        fname = f"level_{li:02d}.json"
        payload = {
            "odomTraj": r4(data["odomTraj"]),
            "kfTrajIdx": data["kfTrajIdx"],
            "kfGt": r4(data["kfGt"]),
            "kfOdom": r4(data["kfOdom"]),
            "kfOpt": r4(data["kfOpt"]),
            "kfOptOnline": r4(data["kfOptOnline"]),
            "kfCands": [[[c[0], round(c[1], 4), round(c[2], 4)] for c in cs]
                        for cs in data["kfCands"]],
            "kfScans": [r4(s) for s in data["kfScans"]],
            "loopEdges": data["loopEdges"],
        }
        with open(out / fname, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        levels.append({
            "index": li,
            "t": round(float(t), 6),
            "severity": round(severity_for(noise), 4),
            "file": fname,
            "noise": {k: round(v, 6) for k, v in noise.items()},
            "stats": {
                "odomRmse": round(odom_rmse, 4),
                "optRmse": round(opt_rmse, 4),
                "finalErr": round(final_err, 4),
                "nKf": len(data["kfGt"]),
                "nLoops": n_loops,
                "nWrongLoops": n_wrong,
                "gtAssisted": gt_assisted,
                "icpRejected": stats["icp_rejected"],
                "icpSigma": round(stats["sigma"], 4),
            },
        })

    index = {
        "levels": levels,
        "config": {
            "hz": HZ,
            "lidarEvery": LIDAR_EVERY,
            "nBeams": N_BEAMS,
            "lidarFov": float(LIDAR_FOV),
            "voxelSize": cfg.voxel_size,
            "submapSize": cfg.submap_size,
            "kfTrans": cfg.kf_trans,
            "kfRot": cfg.kf_rot,
            "minGap": cfg.loop_min_kf_gap,
            "loopRadius": cfg.loop_search_radius,
            "headingLimit": float(cfg.loop_max_heading_diff),
            "maxCandidates": cfg.loop_max_candidates,
            "wrongLoopM": WRONG_LOOP_M,
            "cauchyK": cfg.loop_cauchy_k,
            "loopCooldown": cfg.loop_cooldown_kf,
        },
        "nTraj": len(gt_traj),
        "gtTraj": r4(gt_traj),
    }
    with open(out / "index.json", "w") as f:
        json.dump(index, f, separators=(",", ":"))

    total = sum(p.stat().st_size for p in out.glob("*.json"))
    print(f"wrote {len(levels) + 1} files to {out}  ({total / 1e6:.2f} MB total)")
    print("ACCEPTANCE", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
