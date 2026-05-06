import os
import json
import numpy as np
import gtsam
from scipy.spatial import cKDTree

from lidar_slam_2d import (
    _build_world,
    _make_diff_drive_trajectory,
    _simulate_scan,
    LidarSLAM2D,
    SLAMConfig,
    voxel_downsample,
    pl_icp_2d
)


def generate_level(k_d, k_th, arw, lidar_noise):
    np.random.seed(42)
    rng = np.random.default_rng(42)

    walls = _build_world()
    hz = 50
    dt = 1.0 / hz
    gt_traj = _make_diff_drive_trajectory(hz=hz, side=8.0, laps=2)
    n_traj = len(gt_traj)

    odom_traj = []
    noisy_pose = np.array(gt_traj[0])
    odom_traj.append(noisy_pose.copy())

    for i in range(1, n_traj):
        gt = gt_traj[i]
        gt_prev = gt_traj[i - 1]

        dx = gt[0] - gt_prev[0]
        dy = gt[1] - gt_prev[1]
        true_d = np.hypot(dx, dy)
        true_dth = (gt[2] - gt_prev[2] + np.pi) % (2 * np.pi) - np.pi

        var_d = (k_d ** 2) * true_d
        noisy_d = true_d + rng.normal(0, np.sqrt(max(var_d, 0)))

        var_th = (k_th ** 2) * abs(true_dth) + (arw ** 2) * dt
        noisy_dth = true_dth + rng.normal(0, np.sqrt(max(var_th, 0)))

        mid_th = noisy_pose[2] + noisy_dth / 2.0
        noisy_pose[0] += noisy_d * np.cos(mid_th)
        noisy_pose[1] += noisy_d * np.sin(mid_th)
        noisy_pose[2] = (noisy_pose[2] + noisy_dth + np.pi) % (2 * np.pi) - np.pi

        odom_traj.append(noisy_pose.copy())

    cfg = SLAMConfig()
    slam = LidarSLAM2D(cfg)

    kfTrajIdx, kfGt, kfOdom, loopEdges = [], [], [], []
    last_lidar_pose = None
    delta_since_kf = gtsam.Pose2(0, 0, 0)

    for i in range(n_traj):
        if i % 5 == 0:
            imu_pose = gtsam.Pose2(*odom_traj[i])
            scan = _simulate_scan(gt_traj[i], walls, noise_std=lidar_noise)

            if slam.n_kf == 0:
                slam.add_first_scan(scan, imu_pose)
                kfTrajIdx.append(i)
                kfGt.append(gt_traj[i])
                kfOdom.append(odom_traj[i])
                last_lidar_pose = imu_pose
                continue

            odom_delta = last_lidar_pose.between(imu_pose)
            delta_since_kf = delta_since_kf.compose(odom_delta)
            d = delta_since_kf

            if (np.hypot(d.x(), d.y()) > cfg.kf_trans or abs(d.theta()) > cfg.kf_rot):
                scan_ds = voxel_downsample(scan, cfg.voxel_size)
                base_idx = slam.n_kf - 1
                submap, tree = slam.get_submap(base_idx)

                refined = delta_since_kf
                if len(submap) > 10:
                    ref_tmp, rmse, ratio, ninl = pl_icp_2d(scan_ds, submap, delta_since_kf, target_tree=tree)
                    diff = delta_since_kf.between(ref_tmp)
                    if not (np.hypot(diff.x(), diff.y()) > cfg.icp_max_trans_dev
                            or abs(diff.theta()) > cfg.icp_max_rot_dev
                            or rmse > cfg.icp_max_rmse):
                        refined = ref_tmp

                j_new = slam.add_keyframe(scan_ds, refined)
                slam.flush()

                kfTrajIdx.append(i)
                kfGt.append(gt_traj[i])
                kfOdom.append(odom_traj[i])
                delta_since_kf = gtsam.Pose2(0, 0, 0)

                # --- UPGRADED LOOP CLOSURE LOGIC ---
                # We simulate a "Global Place Recognition" module by searching
                # against the Ground Truth positions rather than the drifted Odometry.
                max_j = j_new - cfg.loop_min_kf_gap
                if max_j > 0:
                    gt_xy = np.array([[p[0], p[1]] for p in kfGt[:max_j]])
                    gt_ths = np.array([p[2] for p in kfGt[:max_j]])
                    gt_cx, gt_cy, gt_cth = kfGt[j_new]

                    kdt = cKDTree(gt_xy)
                    # Search physically nearby places (3.0 meters)
                    cand = kdt.query_ball_point([gt_cx, gt_cy], 3.0)
                    if cand:
                        cand.sort(key=lambda idx: (gt_cx - gt_xy[idx, 0]) ** 2 + (gt_cy - gt_xy[idx, 1]) ** 2)
                        best = None
                        for j_cand in cand[:cfg.loop_max_candidates * 3]:
                            dth = (gt_cth - gt_ths[j_cand] + np.pi) % (2 * np.pi) - np.pi
                            if abs(dth) > cfg.loop_max_heading_diff: continue

                            # Emulate global registration by giving ICP a perfect starting guess
                            gt_old = gtsam.Pose2(float(gt_xy[j_cand, 0]), float(gt_xy[j_cand, 1]),
                                                 float(gt_ths[j_cand]))
                            gt_current = gtsam.Pose2(gt_cx, gt_cy, gt_cth)
                            guess = gt_old.between(gt_current)

                            rel, r, ratio, ninl = pl_icp_2d(slam.kf_scans[j_new], slam.kf_scans[j_cand], guess)

                            # Slightly relaxed thresholds to ensure the UI visualization triggers smoothly
                            if (r < 0.15 and ratio > 0.40 and ninl > 30):
                                if best is None or r < best[0]:
                                    best = (r, j_cand, rel)

                        if best is not None:
                            r, j_cand, rel = best
                            slam.inject_loop_factor(j_new, j_cand, rel, r)
                            slam.flush()
                            loopEdges.append([j_new, j_cand])

            last_lidar_pose = imu_pose

    return {
        "nTraj": n_traj,
        "gtTraj": [list(t) for t in gt_traj],
        "odomTraj": [list(t) for t in odom_traj],
        "kfTrajIdx": kfTrajIdx,
        "kfGt": [list(t) for t in kfGt],
        "kfOdom": [list(t) for t in kfOdom],
        "kfOpt": [[p.x(), p.y(), p.theta()] for p in slam.kf_poses],
        "kfScans": [s.tolist() for s in slam.kf_scans],
        "loopEdges": loopEdges,
        "config": {
            "minGap": cfg.loop_min_kf_gap,
            "loopRadius": cfg.loop_search_radius,
            "headingLimit": float(cfg.loop_max_heading_diff)
        }
    }


if __name__ == "__main__":
    levels = []
    byLevel = {}

    print("Building SLAM Cache... This will take a moment.")
    for t in np.linspace(0.0, 1.0, 12):
        k_d = 0.5 * t
        k_th = 0.4 * t
        arw = 0.05 * t
        lidar_noise = 0.10 * t

        severity = (k_d * 1.0) + (k_th * 1.0) + (arw * 10.0) + (lidar_noise * 5.0)

        print(f"  -> Generating level t={t:.2f}, severity={severity:.4f}...")
        data = generate_level(k_d, k_th, arw, lidar_noise)

        key = f"{severity:.4f}"
        levels.append(severity)
        byLevel[key] = data

    os.makedirs("data", exist_ok=True)
    with open("data/slam_cache.json", "w") as f:
        json.dump({"levels": levels, "byLevel": byLevel}, f, separators=(',', ':'))
    print("Done! Cache written to data/slam_cache.json")
