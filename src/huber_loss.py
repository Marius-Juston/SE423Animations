import numpy as np
import matplotlib.pyplot as plt

import course_style

# Squared loss vs. Huber loss (delta = 1), used for robust kernels in the SLAM lecture (L24-25).
plt.style.use('classic')
course_style.apply()  # Atkinson Hyperlegible Next (course typeface)

DELTA = 1.0


def huber(a, delta=DELTA):
    """Quadratic within +/-delta, linear outside (continuous value and slope at |a| = delta)."""
    return np.where(np.abs(a) <= delta, 0.5 * a ** 2, delta * (np.abs(a) - 0.5 * delta))


if __name__ == '__main__':
    a = np.linspace(-4, 4, 801)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(a, 0.5 * a ** 2, color='b')
    ax.plot(a, huber(a), color='g')
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 8)
    fig.savefig('huber_loss.svg', format='svg', transparent=True)
    print("Saved 'huber_loss.svg'")
