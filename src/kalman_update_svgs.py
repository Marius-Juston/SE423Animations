import numpy as np
import matplotlib.pyplot as plt

# Set a clean, modern style for presentation slides
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'lines.linewidth': 3
})

def gaussian(x, mu, var):
    """Calculates the 1D Gaussian probability density function."""
    return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * ((x - mu) ** 2) / var)

def setup_plot(title):
    """Helper function to set up the slide-ready plot formatting."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(title, pad=15, fontweight='bold')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 0.5)
    ax.set_yticks([]) # Hide y-axis numbers to focus on the shape/concept
    ax.set_xlabel('Position (x)')
    return fig, ax

# --- Define the Domain ---
x = np.linspace(0, 15, 1000)

# --- Process Parameters ---
# 1. Initial State
mu_0 = 4.0
var_0 = 1.0

# 2. Prediction (Move)
u = 4.0        # Commanded movement distance
var_move = 1.5 # Process noise variance

mu_pred = mu_0 + u
var_pred = var_0 + var_move

# 3. Update (Measurement)
z = 9.5          # Sensor reading
var_sensor = 1.2 # Sensor noise variance

# Calculate Kalman Gain
K = var_pred / (var_pred + var_sensor)

# Calculate final state
mu_new = mu_pred + K * (z - mu_pred)
var_new = (1 - K) * var_pred

# ==========================================
# Figure 1: Initial Belief
# ==========================================
fig1, ax1 = setup_plot('Initial Belief')
ax1.plot(x, gaussian(x, mu_0, var_0), label=fr'$\mu={mu_0}$, $\sigma^2={var_0}$', color='#1f77b4')
ax1.fill_between(x, gaussian(x, mu_0, var_0), alpha=0.3, color='#1f77b4')
ax1.legend()
plt.tight_layout()
fig1.savefig('01_initial_belief.svg', format='svg', transparent=True)

# ==========================================
# Figure 2: Prediction (Motion and Spreading)
# ==========================================
fig2, ax2 = setup_plot('1D Prediction (Shift & Spread)')
# Old state (faded to show history)
ax2.plot(x, gaussian(x, mu_0, var_0), linestyle='--', alpha=0.5, color='#1f77b4', label='Old Belief')
# New predicted state
ax2.plot(x, gaussian(x, mu_pred, var_pred), color='#9467bd', label=rf'Prediction ($\mu={mu_pred}$, $\sigma^2={var_pred}$)')
ax2.fill_between(x, gaussian(x, mu_pred, var_pred), alpha=0.3, color='#9467bd')
# Arrow to show movement
ax2.annotate('', xy=(mu_pred, 0.1), xytext=(mu_0, 0.1),
            arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10))
ax2.text((mu_0 + mu_pred)/2, 0.12, f'+u ({u})', ha='center', fontsize=12)
ax2.legend()
plt.tight_layout()
fig2.savefig('02_prediction.svg', format='svg', transparent=True)

# ==========================================
# Figure 3: The Sensor's Opinion
# ==========================================
fig3, ax3 = setup_plot('1D Update (Conflicting Opinions)')
# Prediction curve
ax3.plot(x, gaussian(x, mu_pred, var_pred), color='#9467bd', label='Prediction')
ax3.fill_between(x, gaussian(x, mu_pred, var_pred), alpha=0.2, color='#9467bd')
# Sensor curve
ax3.plot(x, gaussian(x, z, var_sensor), color='#ff7f0e', label=rf'Sensor ($z={z}$, $\sigma^2={var_sensor}$)')
ax3.fill_between(x, gaussian(x, z, var_sensor), alpha=0.2, color='#ff7f0e')
ax3.legend()
plt.tight_layout()
fig3.savefig('03_sensor_opinion.svg', format='svg', transparent=True)

# ==========================================
# Figure 4: The Magic of Multiplying Gaussians
# ==========================================
fig4, ax4 = setup_plot('Fusing Data (Multiplying Gaussians)')
# Plot original two curves
ax4.plot(x, gaussian(x, mu_pred, var_pred), linestyle='--', color='#9467bd', alpha=0.6, label='Prediction')
ax4.plot(x, gaussian(x, z, var_sensor), linestyle='--', color='#ff7f0e', alpha=0.6, label='Sensor')
# Plot the new fused curve
ax4.plot(x, gaussian(x, mu_new, var_new), color='#2ca02c', label=rf'Fused Truth ($\mu={mu_new:.2f}$, $\sigma^2={var_new:.2f}$)')
ax4.fill_between(x, gaussian(x, mu_new, var_new), alpha=0.4, color='#2ca02c')

# Annotate Kalman Gain
ax4.text(1, 0.45, f'Kalman Gain (K) = {K:.2f}', fontsize=14, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

ax4.legend()
plt.tight_layout()
fig4.savefig('04_fused_update.svg', format='svg', transparent=True)

print("Successfully generated 4 SVG files:")
print("- 01_initial_belief.svg")
print("- 02_prediction.svg")
print("- 03_sensor_opinion.svg")
print("- 04_fused_update.svg")
