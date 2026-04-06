import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# --- Settings and Style (Seaborn-like and Slide-Ready) ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'lines.linewidth': 3,
    'figure.autolayout': True  # Equivalent to tight_layout()
})


def draw_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
    """
    Helper function to draw an error ellipse representing the covariance.
    Standardized to 2-sigma (approx 95% confidence).
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Ellipse radii are sqrt(1 + pearson) and sqrt(1 - pearson)
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)

    # Scaling and rotating the ellipse patch
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    trans = transforms.Affine2D().scale(scale_x, scale_y).rotate_deg(45).translate(mean[0], mean[1])

    ellipse.set_transform(trans + ax.transData)
    return ax.add_patch(ellipse)


def setup_2d_plot(title):
    """Helper function to set up the slide-ready 2D plot formatting."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title, pad=15, fontweight='bold')
    ax.set_xlim(-2, 16)
    ax.set_ylim(-2, 10)
    ax.set_xlabel('Position X')
    ax.set_ylabel('Position Y')
    ax.set_aspect('equal', adjustable='box')  # Preserve shape
    return fig, ax


# ==========================================
# 0. System Parameters and Variables
# ==========================================

# 1. INITIAL STATE (Initial Belief)
x0 = np.array([2.0, 2.0])  # Mean [x, y]
P0 = np.array([[1.5, 0.8], [0.8, 1.0]])  # Large, tilted initial covariance

# 2. MOTION MODEL (Prediction)
F = np.array([[1.1, 0.0], [0.0, 1.0]])  # State Transition: slight X expansion
u = np.array([6.0, 3.0])  # Commanded move vector
B = np.eye(2)  # Control matrix (identity)
Q = np.array([[1.5, 0.0], [0.0, 1.5]])  # Process noise matrix (add spread)

# Calculate Prediction State (Prior)
x_pred = F @ x0 + B @ u
P_pred = F @ P0 @ F.T + Q

# 3. MEASUREMENT MODEL (Update)
H = np.array([[1.0, 0.0], [0.0, 1.0]])  # Observation matrix
z_k = np.array([12.0, 8.0])  # Sensor reading
R = np.array([[1.2, -0.4], [-0.4, 0.8]])  # Sensor noise covariance

# The Measurement in State Space
H_inv = np.linalg.inv(H)
x_meas_state_space = H_inv @ z_k
P_meas_state_space = H_inv @ R @ H_inv.T

# 4. FUSION (Kalman Gain and Final State)
S = H @ P_pred @ H.T + R  # Innovation covariance
K = P_pred @ H.T @ np.linalg.inv(S)

innovation = z_k - H @ x_pred
x_final = x_pred + K @ innovation

I = np.eye(2)
P_final = (I - K @ H) @ P_pred

# ==========================================
# Figure 1: Initial Belief (2D)
# ==========================================
fig1, ax1 = setup_2d_plot('MD Initial Belief (The State Vector)')
draw_ellipse(ax1, x0, P0, color='#1f77b4', alpha=0.3)
draw_ellipse(ax1, x0, P0, color='#1f77b4', linewidth=3, fill=False)
ax1.scatter(x0[0], x0[1], color='#1f77b4', s=100, label=r'Initial Belief $(\mathbf{x}_0)$')

# FIXED: Removed \begin{bmatrix} and used transposed bracket notation
ax1.text(0.5, 9.0, r'$\mathbf{x} = [x, y]^T$', fontsize=16, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
ax1.text(0.5, 7.5, r'$\mathbf{P} = [[\sigma_x^2, \sigma_{xy}], [\sigma_{yx}, \sigma_y^2]]$', fontsize=16,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

ax1.legend()
plt.tight_layout()
fig1.savefig('MD_01_initial_belief.svg', format='svg')

# ==========================================
# Figure 2: Prediction (Shift, Stretch & Spread)
# ==========================================
# ==========================================
# Figure 2: Prediction (Shift, Stretch & Spread)
# ==========================================
fig2, ax2 = setup_2d_plot('MD Prediction (Linear Transformation)')

draw_ellipse(ax2, x0, P0, color='#1f77b4', linestyle='--', linewidth=2, fill=False, alpha=0.5)
ax2.scatter(x0[0], x0[1], color='#1f77b4', s=100, alpha=0.5, label='Old Belief')

P_transformed = F @ P0 @ F.T
draw_ellipse(ax2, F @ x0, P_transformed, color='#9467bd', linestyle=':', linewidth=2, fill=False, alpha=0.7)

# FIXED: Added parentheses around (F @ x0) to ensure matrix math happens before indexing
ax2.text((F @ x0)[0] - 0.5, (F @ x0)[1] + 1.5, r'Motion $(\mathbf{F}\mathbf{P}\mathbf{F}^T)$')

draw_ellipse(ax2, x_pred, P_pred, color='#9467bd', alpha=0.3)
draw_ellipse(ax2, x_pred, P_pred, color='#9467bd', linewidth=3, fill=False)
ax2.scatter(x_pred[0], x_pred[1], color='#9467bd', s=100, label=r'Predicted Prior $(\mathbf{x}_{pred}, \mathbf{P}_{pred})$')

ax2.annotate('', xy=F @ x0, xytext=x0, arrowprops=dict(arrowstyle='-|>', mutation_scale=20, shrinkA=0, shrinkB=0, linewidth=2, color='gray'))
ax2.annotate('', xy=x_pred, xytext=F @ x0, arrowprops=dict(facecolor='black', arrowstyle='-|>', mutation_scale=20, shrinkA=0, shrinkB=0, linewidth=2))

ax2.text(8, 2.5, r'$\mathbf{x}_{pred} = \mathbf{F}\mathbf{x}_{k-1} + \mathbf{B}\mathbf{u}$', fontsize=14, color='#9467bd')
ax2.text(8, 1.0, r'$\mathbf{P}_{pred} = \mathbf{F}\mathbf{P}_{k-1}\mathbf{F}^T + \mathbf{Q}$', fontsize=14, color='#9467bd')

ax2.legend()
plt.tight_layout()
fig2.savefig('MD_02_prediction.svg', format='svg')

# ==========================================
# Figure 3: The Sensor's Opinion (Mapped to State Space)
# ==========================================
fig3, ax3 = setup_2d_plot('MD Update (Transforming the Measurement)')

draw_ellipse(ax3, x_pred, P_pred, color='#9467bd', fill=False, alpha=0.3, label='Prediction')
ax3.scatter(x_pred[0], x_pred[1], color='#9467bd', s=100, alpha=0.3)

draw_ellipse(ax3, x_meas_state_space, P_meas_state_space, color='#ff7f0e', alpha=0.3)
draw_ellipse(ax3, x_meas_state_space, P_meas_state_space, color='#ff7f0e', linewidth=3, fill=False)
ax3.scatter(x_meas_state_space[0], x_meas_state_space[1], color='#ff7f0e', s=100,
            label=r'Effective Sensor Belief $(\mathbf{H}^{-1}\mathbf{z}_k, \mathbf{P}_{meas})$')

ax3.plot([x_pred[0], x_meas_state_space[0]], [x_pred[1], x_meas_state_space[1]], color='gray', linestyle=':')
ax3.text(10.2, 5.0, 'Residual\n'r'$(\mathbf{z}_k - \mathbf{H}\mathbf{x}_{pred})$')

ax3.text(0, 0.5, r'Innovation Covariance (S):', fontsize=14)
ax3.text(0, -1.0, r'$\mathbf{S} = \mathbf{H}\mathbf{P}_{pred}\mathbf{H}^T + \mathbf{R}$', fontsize=14, color='#ff7f0e')

ax3.legend()
plt.tight_layout()
fig3.savefig('MD_03_sensor_opinion.svg', format='svg')

# ==========================================
# Figure 4: The Magic of Fusion (Shrinking Uncertainty)
# ==========================================
fig4, ax4 = setup_2d_plot('Fusing Data (Shrinking the State Ellipse)')

draw_ellipse(ax4, x_pred, P_pred, color='#9467bd', fill=False, alpha=0.3, linestyle='--', label='Prediction (Faded)')
draw_ellipse(ax4, x_meas_state_space, P_meas_state_space, color='#ff7f0e', fill=False, alpha=0.3, linestyle='--',
             label='Effective Measurement (Faded)')

draw_ellipse(ax4, x_final, P_final, color='#2ca02c', alpha=0.4)
draw_ellipse(ax4, x_final, P_final, color='#2ca02c', linewidth=4, fill=False)
ax4.scatter(x_final[0], x_final[1], color='#2ca02c', s=120, label=r'Fused Truth $(\mathbf{x}_k, \mathbf{P}_k)$')

ax4.annotate('', xy=x_final, xytext=x_pred, arrowprops=dict(arrowstyle='-|>', mutation_scale=20, linewidth=2, color='#2ca02c'))
ax4.text(x_final[0] - 2, x_final[1] + 1, r'$\mathbf{K}(\mathbf{z}_k - \mathbf{H}\mathbf{x}_{pred})$', color='#2ca02c')

ax4.text(-1, 9.2, f'Kalman Gain K[0,0] = {K[0, 0]:.2f}', fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
ax4.text(-1, 8.5, r'$\mathbf{K} = \mathbf{P}_{pred}\mathbf{H}^T\mathbf{S}^{-1}$', fontsize=14)

ax4.text(-1, -1.0, r'$\mathbf{P}_k = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}_{pred}$', fontsize=14,
         color='#2ca02c')

ax4.legend(loc='lower right')
plt.tight_layout()
fig4.savefig('MD_04_fused_update.svg', format='svg')

print("Successfully generated 4 Multi-Dimension (2D) Kalman Filter SVG files:")
print("- MD_01_initial_belief.svg")
print("- MD_02_prediction.svg")
print("- MD_03_sensor_opinion.svg")
print("- MD_04_fused_update.svg")
