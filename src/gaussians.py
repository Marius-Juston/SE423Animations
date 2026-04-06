import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 7, 1000)

def gaussian(x, mu, sigma):
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*sigma**2))

line1 = plt.plot(x, gaussian(x, 0, 1.5), label="High uncertainty", color="red")
line2 = plt.plot(x, gaussian(x, 2, 0.5), label="Low uncertainty", color="blue")

plt.plot([0, 0], [0, gaussian(0, 0, 1.5)], color='red')
plt.plot([2, 2], [0, gaussian(2, 2, 0.5)], color='blue')

plt.title("Gaussian Distribution")
plt.xlabel("Robot's x position")
plt.ylabel("Probability")
plt.legend()
plt.tight_layout()
plt.savefig("gaussians.svg", transparent=True)
