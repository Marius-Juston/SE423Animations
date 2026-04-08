import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch

# ==========================================
# 1. Apply Requested Styling
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'lines.linewidth': 3
})

# ==========================================
# 2. System and Filter Parameters (adjust as needed)
# ==========================================
# Initializing using specific values from previous context:
# Pose (0,0, 45°), nominal v 5.0, straight driving, sigma_v 0.5, sigma_theta 0.6
np.random.seed(42) # For consistent data generation

dt = 1.0
v_nom = 5.0      # Forward velocity
omega_nom = 0.0  # Driving straight

# Initial State
mu_x, mu_y = 0.0, 0.0
mu_theta = np.pi / 4  # 45 degrees

# Noise Standard Deviations
sigma_v = 0.5      # Velocity noise
sigma_theta = 0.6  # Heading noise (large enough to show curvature)

# Define distinct styling for different elements:
mc_style = {'color': '#1f77b4', 's': 3, 'alpha': 0.15}
ekf_ellipse_style = {'edgecolor': '#d62728', 'facecolor': 'none', 'lw': 3, 'linestyle': '--', 'zorder': 4}
ekf_mean_style = {'color': '#d62728', 'marker': 'x', 's': 100, 'zorder': 5, 'lw': 3}
lie_bound_style = {'color': '#2ca02c', 'lw': 4, 'zorder': 4}
lie_mean_style = {'color': '#2ca02c', 'marker': '+', 's': 150, 'zorder': 6, 'lw': 3}
start_pose_style = {'color': 'black', 's': 300, 'marker': '*', 'zorder': 10}
start_heading_arrow_style = {'fc': 'black', 'ec': 'black', 'zorder': 10, 'lw': 2, 'head_width': 0.15, 'head_length': 0.25}
start_annotation_style = {'fontsize': 16, 'fontweight': 'bold'}
update_arrow_style = {'mutation_scale': 25, 'color': 'dimgray', 'lw': 2.5, 'linestyle': '--', 'zorder': 3}


# ==========================================
# 3. Data Generation & Calculation (Perform Once)
# ==========================================

# 3a. Monte Carlo Particles
N = 10000
v_samples = np.random.normal(v_nom, sigma_v, N)
theta_samples = np.random.normal(mu_theta, sigma_theta, N)
mc_x = mu_x + v_samples * dt * np.cos(theta_samples)
mc_y = mu_y + v_samples * dt * np.sin(theta_samples)

# 3b. EKF Calculations & Ellipse Patch Creation
ekf_mu_x = mu_x + v_nom * dt * np.cos(mu_theta)
ekf_mu_y = mu_y + v_nom * dt * np.sin(mu_theta)

J = np.array([
    [dt * np.cos(mu_theta), -v_nom * dt * np.sin(mu_theta)],
    [dt * np.sin(mu_theta),  v_nom * dt * np.cos(mu_theta)]
])
Cov_noise = np.diag([sigma_v**2, sigma_theta**2])
ekf_cov = J @ Cov_noise @ J.T

eigvals, eigvecs = np.linalg.eigh(ekf_cov)
order = eigvals.argsort()[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]
ekf_angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
ekf_width, ekf_height = 2 * 2.0 * np.sqrt(eigvals) # 2 standard deviations for full width/height

# 3c. Lie Group SE(2) Bound Calculations
alpha = np.linspace(0, 2 * np.pi, 200)
u_err = 2.0 * sigma_v * np.cos(alpha)
omega_err = 2.0 * sigma_theta * np.sin(alpha)
u_total = (v_nom + u_err) * dt
omega_total = omega_err  # Nominally zero

x_local = np.where(np.abs(omega_total) > 1e-5,
                   (u_total / omega_total) * np.sin(omega_total),
                   u_total)
y_local = np.where(np.abs(omega_total) > 1e-5,
                   (u_total / omega_total) * (1 - np.cos(omega_total)),
                   0.0)

se2_bound_x = mu_x + x_local * np.cos(mu_theta) - y_local * np.sin(mu_theta)
se2_bound_y = mu_y + x_local * np.sin(mu_theta) + y_local * np.cos(mu_theta)


# ==========================================
# 4. Figure Creation and Saving (building up and saving separately)
# ==========================================

# Use unique Ellipse patches for each axes
def create_ekf_ellipse(ekf_mu_x, ekf_mu_y, ekf_width, ekf_height, ekf_angle, style):
    return Ellipse((ekf_mu_x, ekf_mu_y), ekf_width, ekf_height, angle=ekf_angle, **style)

# Function to handle legend alpha looping
def adjust_legend_alpha(leg):
    for handle in leg.legend_handles:
        if hasattr(handle, 'set_alpha'):
            handle.set_alpha(1.0)


start_loc = (mu_x - 0.55, mu_y + 0.25)

# --- FIGURE 2: Monte Carlo + EKF (including visual context) ---
fig2, ax2 = plt.subplots(figsize=(10, 10))
ax2.set_title("Monte Carlo + EKF Bound", pad=15, fontweight='bold')

# Plot Initial Pose and Heading (for clarity in Figures 2 and 3)
ax2.scatter(mu_x, mu_y, **start_pose_style, label='Initial Pose $(x_0, y_0)$')
heading_dx = 0.8 * np.cos(mu_theta)
heading_dy = 0.8 * np.sin(mu_theta)
ax2.arrow(mu_x, mu_y, heading_dx, heading_dy, **start_heading_arrow_style)
ax2.annotate('Start', start_loc, **start_annotation_style)

# Plot State Prediction Update Arrow (for Figures 2 and 3)
update_arrow2 = FancyArrowPatch((mu_x, mu_y), (ekf_mu_x, ekf_mu_y), arrowstyle='-|>', label='Nominal State Update ($v \\cdot \\Delta t$)', **update_arrow_style)
ax2.add_patch(update_arrow2)

# Plot Monte Carlo Particles
ax2.scatter(mc_x, mc_y, **mc_style, label='Monte Carlo Particles')

# Plot EKF
ekf_ellipse2 = create_ekf_ellipse(ekf_mu_x, ekf_mu_y, ekf_width, ekf_height, ekf_angle, ekf_ellipse_style)
ekf_ellipse2.set_label('EKF Cartesian Bound (2$\\sigma$)')
ax2.add_patch(ekf_ellipse2)
ax2.scatter(ekf_mu_x, ekf_mu_y, **ekf_mean_style) # Scatter is not explicitly styleable with label and linestyle, so we add a patch and scatter separate

# Formatting and Legend
ax2.set_aspect('equal')
ax2.set_xlabel('Global X Position')
ax2.set_ylabel('Global Y Position')
leg2 = ax2.legend(loc='lower left', frameon=True, framealpha=0.95)
adjust_legend_alpha(leg2)
plt.tight_layout()
fig2.savefig('monte_carlo_ekf.svg', format='svg', transparent=True)
print("Saved 'monte_carlo_ekf.svg'")


# --- FIGURE 3: Monte Carlo + EKF + Lie (full comparison) ---
fig3, ax3 = plt.subplots(figsize=(10, 10))
ax3.set_title("Monte Carlo + EKF + Lie Group SE(2) Bounds", pad=15, fontweight='bold')

# Plot Initial Pose and Heading
ax3.scatter(mu_x, mu_y, **start_pose_style, label='Initial Pose $(x_0, y_0)$')
ax3.arrow(mu_x, mu_y, heading_dx, heading_dy, **start_heading_arrow_style)
ax3.annotate('Start', start_loc, **start_annotation_style)

# Plot State Prediction Update Arrow
update_arrow3 = FancyArrowPatch((mu_x, mu_y), (ekf_mu_x, ekf_mu_y), arrowstyle='-|>', label='Nominal State Update ($v \\cdot \\Delta t$)', **update_arrow_style)
ax3.add_patch(update_arrow3)

# Plot Monte Carlo Particles
ax3.scatter(mc_x, mc_y, **mc_style, label='Monte Carlo Particles')

# Plot EKF
ekf_ellipse3 = create_ekf_ellipse(ekf_mu_x, ekf_mu_y, ekf_width, ekf_height, ekf_angle, ekf_ellipse_style)
ekf_ellipse3.set_label('EKF Cartesian Bound (2$\\sigma$)')
ax3.add_patch(ekf_ellipse3)
ax3.scatter(ekf_mu_x, ekf_mu_y, **ekf_mean_style)

# Plot Lie Group SE(2) Bound
ax3.plot(se2_bound_x, se2_bound_y, **lie_bound_style, label='Lie Group SE(2) Bound (2$\\sigma$)')
ax3.scatter(ekf_mu_x, ekf_mu_y, **lie_mean_style) # Lie Group uses standard mean marker and scatter doesn't support + style easily

# Formatting and Complete Legend
ax3.set_aspect('equal')
ax3.set_xlabel('Global X Position')
ax3.set_ylabel('Global Y Position')
leg3 = ax3.legend(loc='lower left', frameon=True, framealpha=0.95)
adjust_legend_alpha(leg3)
plt.tight_layout()
fig3.savefig('monte_carlo_ekf_lie.svg', format='svg', transparent=True)
print("Saved 'monte_carlo_ekf_lie.svg'")


# --- FIGURE 1: Monte Carlo Only ---
fig1, ax1 = plt.subplots(figsize=(10, 10))
ax1.set_title("Monte Carlo Particles", pad=15, fontweight='bold')
ax1.scatter(mc_x, mc_y, **mc_style, label='Monte Carlo (True Distribution)')
ax1.set_xlim(ax2.get_xlim())
ax1.set_ylim(ax2.get_ylim())
ax1.set_aspect('equal')
ax1.set_xlabel('Global X Position')
ax1.set_ylabel('Global Y Position')

leg1 = ax1.legend(loc='lower left', frameon=True, framealpha=0.95)
adjust_legend_alpha(leg1)


plt.tight_layout()
fig1.savefig('monte_carlo_only.svg', format='svg', transparent=True)
print("Saved 'monte_carlo_only.svg'")

# ==========================================
# 5. Clean-up
# ==========================================
plt.close('all')
print("All figures closed. Figures are successfully saved to separate SVG files.")