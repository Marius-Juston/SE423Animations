import numpy as np
import matplotlib.pyplot as plt

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

x = np.linspace(-x_bounds, x_bounds, 1000)

mu_0 = 0.0
var_0 = 1.0

fig, (ax, ax2, ax3) = plt.subplots(figsize=(5 * 3, 5), ncols=3)
ax.set_title("Gaussian", pad=15, fontweight='bold')
ax2.set_title("Linear transform ($2x$)", pad=15, fontweight='bold')
ax3.set_title("Nonlinear transform ($sin(x)$)", pad=15, fontweight='bold')

ax.set_yticks([])
ax2.set_yticks([])
ax3.set_yticks([])

N = 1_000_000
samples = np.random.normal(mu_0, var_0, N)

bins = 100

ax.hist(samples, bins=bins, density=True, color='#1f77b4', alpha=0.5)
ax2.hist(2 * samples, bins=bins, density=True, color='#9467bd', alpha=0.5)
ax3.hist(np.sin(samples), bins=bins, density=True, color='#ff7f0e', alpha=0.5)

plt.tight_layout()
fig.savefig('gaussian_transform.svg', format='svg', transparent=True)
