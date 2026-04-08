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

x_bounds = 5

# --- Define the Domain ---
x = np.linspace(-x_bounds, x_bounds, 1000)

# --- Process Parameters ---
# 1. Initial State
mu_0 = 0.0
var_0 = 1.0

fig, (ax, ax2, ax3) = plt.subplots(figsize=(5 * 3, 5), ncols=3)
ax.set_title("Gaussian", pad=15, fontweight='bold')
# ax.set_xlim(-3, 3)
# ax.set_ylim(0, 1)
ax2.set_title("Linear transform ($2x$)", pad=15, fontweight='bold')
ax3.set_title("Nonlinear transform ($sin(x)$)", pad=15, fontweight='bold')
# ax2.set_xlim(-3, 3)
# ax3.set_ylim(0, 1)

ax.set_yticks([])  # Hide y-axis numbers to focus on the shape/concept
ax2.set_yticks([])  # Hide y-axis numbers to focus on the shape/concept
ax3.set_yticks([])  # Hide y-axis numbers to focus on the shape/concept

N = 1_000_000
samples = np.random.normal(mu_0, var_0, N)

bins = 100

ax.hist(samples, bins = bins, density = True, color='#1f77b4', alpha = 0.5 )
ax2.hist(2 * samples, bins = bins, density = True, color='#9467bd', alpha = 0.5 )
ax3.hist(np.sin(samples), bins = bins, density = True, color='#ff7f0e', alpha = 0.5 )

#
# ax.plot(x, gaussian(x, mu_0, var_0), color='#1f77b4')
# ax.fill_between(x, gaussian(x, mu_0, var_0), alpha=0.3, color='#1f77b4')
#
# ax2.plot(x,2 * gaussian(x, mu_0, var_0), color='#1f77b4')
# ax2.fill_between(x,2 * gaussian(x, mu_0, var_0), alpha=0.3, color='#1f77b4')
#
# ax3.plot(x, gaussian(np.sin(x), mu_0, var_0), color='#1f77b4')
# ax3.fill_between(x, gaussian( np.sin( x), mu_0, var_0), alpha=0.3, color='#1f77b4')

plt.tight_layout()
fig.savefig('gaussian_transform.svg', format='svg', transparent=True)
