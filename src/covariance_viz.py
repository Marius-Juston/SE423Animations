import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Set a clean, modern style for presentation slides
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'mathtext.fontset': 'cm'  # Use Computer Modern for elegant LaTeX math
})


def generate_covariance_svg():
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(r"Unpacking the 2D Covariance Matrix ($\Sigma$)", pad=20, fontweight='bold')

    # 1. Define the System
    mean = np.array([5.0, 5.0])
    # A covariance matrix with positive correlation
    cov = np.array([[4.0, 2.5],
                    [2.5, 3.0]])

    # 2. Mathematical Decomposition (Spectral Theorem)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort by descending eigenvalue so v1 is the major axis
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Calculate the angle of rotation for the ellipse (in degrees)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    # 3. Plot the Confidence Ellipses (1, 2, and 3 Standard Deviations)
    colors = ['#1f77b4', '#1f77b4', '#1f77b4']
    alphas = [0.4, 0.2, 0.1]
    for i, n_std in enumerate([1, 2, 3]):
        ell = Ellipse(xy=mean,
                      width=2 * n_std * np.sqrt(eigenvalues[0]),
                      height=2 * n_std * np.sqrt(eigenvalues[1]),
                      angle=angle, facecolor=colors[i], edgecolor='black',
                      linewidth=1, alpha=alphas[i],
                      label=rf'{n_std}$\sigma$ Contour' if i == 0 else "")
        ax.add_patch(ell)

    # 4. Draw the Eigenvectors (Principal Components)
    # Scaled by 2 standard deviations for visual clarity
    v1 = eigenvectors[:, 0] * np.sqrt(eigenvalues[0]) * 2
    v2 = eigenvectors[:, 1] * np.sqrt(eigenvalues[1]) * 2

    # Major Axis
    ax.annotate('', xy=mean + v1, xytext=mean,
                arrowprops=dict(facecolor='#d62728', shrink=0, width=3, headwidth=10))
    ax.text(mean[0] + v1[0] * 1.1, mean[1] + v1[1] * 1.1, r'$\sqrt{\lambda_1}\vec{v_1}$ (Major Axis)',
            color='#d62728', fontsize=16, fontweight='bold', ha='left')

    # Minor Axis
    ax.annotate('', xy=mean + v2, xytext=mean,
                arrowprops=dict(facecolor='#2ca02c', shrink=0, width=3, headwidth=10))
    ax.text(mean[0] + v2[0] * 1.1, mean[1] + v2[1] * 1.1, r'$\sqrt{\lambda_2}\vec{v_2}$ (Minor Axis)',
            color='#2ca02c', fontsize=16, fontweight='bold', ha='right')

    # 5. Draw the Marginal Variances (\sigma_x and \sigma_y)
    sigma_x = np.sqrt(cov[0, 0])
    sigma_y = np.sqrt(cov[1, 1])

    # X-axis spread projection
    ax.hlines(y=mean[1], xmin=mean[0] - sigma_x, xmax=mean[0] + sigma_x,
              colors='black', linestyles='dashed', linewidth=2)
    ax.annotate('', xy=(mean[0] + sigma_x, mean[1]), xytext=(mean[0], mean[1]),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(mean[0] + sigma_x / 2, mean[1] - 0.4, r'$\sigma_x$', fontsize=18, ha='center')

    # Y-axis spread projection
    ax.vlines(x=mean[0], ymin=mean[1] - sigma_y, ymax=mean[1] + sigma_y,
              colors='black', linestyles='dashed', linewidth=2)
    ax.annotate('', xy=(mean[0], mean[1] + sigma_y), xytext=(mean[0], mean[1]),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(mean[0] - 0.5, mean[1] + sigma_y / 2, r'$\sigma_y$', fontsize=18, va='center')

    # 6. Add the Covariance Matrix Math Box
    # 6. Add the Covariance Matrix Math Box
    # Bypassing the mathtext 2D parser limitations by stacking 1D math strings.
    # We use \quad for standard spacing and \qquad for double spacing to align the columns.
    matrix_text = (
            r"$\mathbf{\Sigma} \ = \ \ [ \ \ \sigma_x^2 \qquad \sigma_{xy} \ \ ] \ \ = \ \ [ \ \ 4.0 \qquad 2.5 \ \ ]$" + "\n" +
            r"$\ \ \ \ \ \ \ \ \ [ \ \ \sigma_{yx} \qquad \sigma_y^2 \ \ ] \ \ \ \ \ \ \ \ [ \ \ 2.5 \qquad 3.0 \ \ ]$"
    )
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.05, 0.95, matrix_text, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', bbox=props)

    # 7. Final Formatting
    ax.plot(*mean, 'ko', markersize=8, label=r'Mean ($\mu$)')  # Mark the mean
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel('Robot X Position', fontsize=16)
    ax.set_ylabel('Robot Y Position', fontsize=16)
    ax.set_aspect('equal')  # Crucial so the ellipse isn't distorted
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('05_covariance_matrix.svg', format='svg')
    print("Successfully generated: 05_covariance_matrix.svg")


if __name__ == "__main__":
    generate_covariance_svg()
