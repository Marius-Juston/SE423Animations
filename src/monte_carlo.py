import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib.patches as patches
import copy

# ==========================================
# 1. Matplotlib Configuration (Safe Math Text)
# ==========================================
plt.rc('font', family='serif')
plt.rc('mathtext', fontset='cm')  # Computer Modern (LaTeX font)
plt.rc('axes', titlesize=18, labelsize=14)

# ==========================================
# 2. Simulation Parameters & Environment
# ==========================================
NUM_PARTICLES = 1000
WORLD_SIZE = 10.0  # 10x10 meters
LANDMARKS = np.array([[2.0, 8.0], [8.0, 8.0], [8.0, 2.0]])  # [x, y] coordinates

# Reduced Noise parameters for "not a lot of noise" assumption
ODOM_NOISE_TRANS = 0.2  # meters
ODOM_NOISE_ROT = 0.2  # radians
SENSOR_NOISE = 0.5  # meters
BEARING_NOISE = 0.2  # radians


# ==========================================
# 3. Helper Functions
# ==========================================
def create_uniform_particles(x_range, y_range, theta_range, N):
    particles = {
        "state": np.empty((N, 3)),
        "history": [[] for _ in range(N)]
    }

    particles["state"][:, 0] = np.random.uniform(*x_range, size=N)
    particles["state"][:, 1] = np.random.uniform(*y_range, size=N)
    particles["state"][:, 2] = np.random.uniform(*theta_range, size=N)

    for i in range(N):
        particles["history"][i].append(particles["state"][i, :2].copy())

    return particles


def motion_model(particles, u, dt=1.0, sub_steps=10):
    v, w = u
    state = particles["state"]
    N = len(state)

    v_noise = np.random.randn(N) * ODOM_NOISE_TRANS
    w_noise = np.random.randn(N) * ODOM_NOISE_ROT

    v_hat = v + v_noise
    w_hat = w + w_noise

    dt_step = dt / sub_steps

    for _ in range(sub_steps):
        tol = 1e-6
        straight_idx = np.abs(w_hat) < tol
        curve_idx = ~straight_idx

        state[straight_idx, 0] += v_hat[straight_idx] * np.cos(state[straight_idx, 2]) * dt_step
        state[straight_idx, 1] += v_hat[straight_idx] * np.sin(state[straight_idx, 2]) * dt_step
        state[straight_idx, 2] += w_hat[straight_idx] * dt_step

        theta = state[curve_idx, 2]
        w_c = w_hat[curve_idx]
        v_c = v_hat[curve_idx]

        state[curve_idx, 0] += (v_c / w_c) * (np.sin(theta + w_c * dt_step) - np.sin(theta))
        state[curve_idx, 1] += (v_c / w_c) * (np.cos(theta) - np.cos(theta + w_c * dt_step))
        state[curve_idx, 2] += w_c * dt_step

        for i in range(N):
            particles["history"][i].append(state[i, :2].copy())

    state[:, 2] %= (2 * np.pi)
    return particles


def update_true_pose(pose, u, dt=1.0, sub_steps=10):
    v, w = u
    arc = [pose[:2].copy()]
    current_pose = pose.copy()
    dt_step = dt / sub_steps

    for _ in range(sub_steps):
        if abs(w) < 1e-6:
            current_pose[0] += v * np.cos(current_pose[2]) * dt_step
            current_pose[1] += v * np.sin(current_pose[2]) * dt_step
            current_pose[2] += w * dt_step
        else:
            current_pose[0] += (v / w) * (np.sin(current_pose[2] + w * dt_step) - np.sin(current_pose[2]))
            current_pose[1] += (v / w) * (np.cos(current_pose[2]) - np.cos(current_pose[2] + w * dt_step))
            current_pose[2] += w * dt_step
        arc.append(current_pose[:2].copy())

    current_pose[2] %= (2 * np.pi)
    return current_pose, np.array(arc)


def measurement_model(particles, true_pose, landmarks):
    """
    Robust Lidar/Directional Likelihood Model using Log-Sum-Exp.
    Properly handles angular wrap-around and prevents floating-point underflow.
    """
    state = particles["state"]
    N = len(state)

    # Use log weights to accumulate probabilities safely
    log_weights = np.zeros(N)
    true_measurements = []

    for lx, ly in landmarks:
        # True Robot Measurement Generation
        dx = lx - true_pose[0]
        dy = ly - true_pose[1]

        true_r = np.hypot(dx, dy) + np.random.randn() * SENSOR_NOISE
        true_phi = np.arctan2(dy, dx) - true_pose[2]
        true_phi += np.random.randn() * BEARING_NOISE

        # Normalize true reading strictly to [-pi, pi]
        true_phi = (true_phi + np.pi) % (2 * np.pi) - np.pi

        true_measurements.append((true_r, true_phi))

        # Particle Predictions
        pdx = lx - state[:, 0]
        pdy = ly - state[:, 1]

        pr = np.hypot(pdx, pdy)
        pphi = np.arctan2(pdy, pdx) - state[:, 2]

        # Calculate ERRORS, not raw probabilities
        diff_r = true_r - pr
        diff_phi = true_phi - pphi

        # VERY IMPORTANT: Normalize angular error for directional sensors!
        diff_phi = (diff_phi + np.pi) % (2 * np.pi) - np.pi

        # Add Log-Likelihoods (evaluating the error around a mean of 0)
        log_w_r = scipy.stats.norm.logpdf(diff_r, loc=0, scale=SENSOR_NOISE)
        log_w_phi = scipy.stats.norm.logpdf(diff_phi, loc=0, scale=BEARING_NOISE)

        log_weights += log_w_r + log_w_phi

    # Log-Sum-Exp Trick to convert back to normal weights without underflow
    max_log_weight = np.max(log_weights)
    weights = np.exp(log_weights - max_log_weight)  # Shift highest weight to 1.0 safely
    weights += 1e-300  # Avoid strict zeros
    weights /= np.sum(weights)  # Normalize so they sum to 1.0

    return weights, true_measurements


def resample(particles, weights):
    N = len(weights)
    indices = np.zeros(N, dtype=int)

    r = np.random.uniform(0, 1.0 / N)
    c = weights[0]
    i = 0

    for m in range(N):
        U = r + m * (1.0 / N)
        while U > c:
            i += 1
            c += weights[i]
        indices[m] = i

    new_particles = {
        "state": particles["state"][indices].copy(),
        "history": [list(particles["history"][idx]) for idx in indices]
    }

    return new_particles, np.ones(N) / N


def get_estimated_pose(particles_state, weights):
    x_mean = np.average(particles_state[:, 0], weights=weights)
    y_mean = np.average(particles_state[:, 1], weights=weights)

    sin_sum = np.sum(weights * np.sin(particles_state[:, 2]))
    cos_sum = np.sum(weights * np.cos(particles_state[:, 2]))
    theta_mean = np.arctan2(sin_sum, cos_sum)

    return np.array([x_mean, y_mean, theta_mean])


# ==========================================
# 4. Plotting Function for Slides
# ==========================================
def render_slide(particles, weights, true_pose, step_name, filename, eq_text,
                 measurements=None, true_arc=None):
    fig, ax = plt.subplots(figsize=(9, 7))
    state = particles["state"]

    for traj in particles["history"]:
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], color='#95a5a6', alpha=0.75, linewidth=1)

    if true_arc is not None:
        ax.plot(true_arc[:, 0], true_arc[:, 1],
                color='#c0392b', linestyle='--', linewidth=2, zorder=4, label='True Path (1s Arc)')

    if measurements is not None:
        for i, (lx, ly) in enumerate(LANDMARKS):
            circle = patches.Circle((lx, ly), measurements[i][0], color='#2ecc71', fill=False,
                                    linestyle='--', linewidth=2, alpha=0.6, zorder=1)
            ax.add_patch(circle)
        ax.plot([], [], color='#2ecc71', linestyle='--', linewidth=2, label='Sensor Reading $z_t$')

    ax.scatter(LANDMARKS[:, 0], LANDMARKS[:, 1], c='#f1c40f', s=300, marker='*',
               edgecolor='black', zorder=4, label='Landmarks')

    if measurements is not None:
        scatter = ax.scatter(state[:, 0], state[:, 1], c=weights, cmap='viridis',
                             s=10 + (weights / weights.max()) * 60, alpha=0.7, zorder=2)
        fig.colorbar(scatter, ax=ax, label='Particle Weight $w_t$')

        ax.quiver(state[:, 0], state[:, 1], np.cos(state[:, 2]), np.sin(state[:, 2]),
                  color='#2c3e50', alpha=0.4, scale=40, width=0.0025, headwidth=4, zorder=3, label='Particles')
    else:
        ax.scatter(state[:, 0], state[:, 1], c='#3498db', s=15, alpha=0.5, zorder=2)
        ax.quiver(state[:, 0], state[:, 1], np.cos(state[:, 2]), np.sin(state[:, 2]),
                  color='#2980b9', alpha=0.6, scale=40, width=0.0025, headwidth=4, zorder=3, label='Particles')

    est_pose = get_estimated_pose(state, weights)
    ax.quiver(est_pose[0], est_pose[1], np.cos(est_pose[2]), np.sin(est_pose[2]),
              color='#8e44ad', scale=12, width=0.02, zorder=6, label='Estimated Pose (Mean)')

    ax.quiver(true_pose[0], true_pose[1], np.cos(true_pose[2]), np.sin(true_pose[2]),
              color='#e74c3c', scale=12, width=0.02, zorder=5, label='True Robot Pose')

    ax.set_xlim(0, WORLD_SIZE)
    ax.set_ylim(0, WORLD_SIZE)
    ax.set_xlabel(r'$X$-Position (m)')
    ax.set_ylabel(r'$Y$-Position (m)')

    plt.title(f"{step_name}\n{eq_text}", pad=15, fontweight='bold')

    ax.legend(loc='lower left', framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig(filename, format='svg', transparent=True)
    plt.close()
    print(f"Generated: {filename}")


# ==========================================
# 5. Execute Simulation and Generate Slides
# ==========================================

DT = 0.5


def main():
    np.random.seed(42)

    true_pose = np.array([2.0, 2.0, np.pi / 4])

    particles = create_uniform_particles((0, WORLD_SIZE), (0, WORLD_SIZE), (0, 2 * np.pi), NUM_PARTICLES)
    weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES
    eq1 = r"Belief initialization: $bel(x_0) = \frac{1}{|X|}$"
    render_slide(particles, weights, true_pose, "Slide 1: Global Initialization", "slide1_init.svg", eq1)

    u = [3.0, 0.4]
    true_pose, true_arc = update_true_pose(true_pose, u, dt=DT)
    particles = motion_model(particles, u, dt=DT)
    eq2 = r"Prediction: $\overline{bel}(x_t) = \int p(x_t | u_t, x_{t-1}) bel(x_{t-1}) \, dx_{t-1}$"

    render_slide(particles, weights, true_pose, "Slide 2: Action / Prediction Step", "slide2_predict.svg", eq2,
                 true_arc=true_arc)

    weights, true_measurements = measurement_model(particles, true_pose, LANDMARKS)
    eq3 = r"Correction: $w_t^{[m]} = \eta \exp \left( \sum \log p(z_t | x_t^{[m]}) \right)$"
    render_slide(particles, weights, true_pose, "Slide 3: Measurement / Update Step", "slide3_update.svg", eq3,
                 measurements=true_measurements)

    particles, weights = resample(particles, weights)
    eq4 = r"Resampling: $x_t^{[m]} \sim \langle x_t^{[i]}, w_t^{[i]} \rangle$"
    render_slide(particles, weights, true_pose, "Slide 4: Resampling", "slide4_resample.svg", eq4)

    u2 = [2.0, -0.3]
    true_pose, true_arc2 = update_true_pose(true_pose, u2, dt=DT)
    particles = motion_model(particles, u2, dt=DT)

    render_slide(
        particles, weights, true_pose,
        "Slide 5: Second Prediction",
        "slide5_predict2.svg",
        r"Second Prediction Step",
        true_arc=true_arc2
    )

    weights, measurements = measurement_model(particles, true_pose, LANDMARKS)

    render_slide(
        particles, weights, true_pose,
        "Slide 6: Second Update",
        "slide6_update2.svg",
        r"Second Measurement Update",
        measurements=measurements
    )

    particles, weights = resample(particles, weights)

    render_slide(
        particles, weights, true_pose,
        "Slide 7: Second Resampling",
        "slide7_resample2.svg",
        r"Second Resampling Step"
    )


if __name__ == '__main__':
    main()