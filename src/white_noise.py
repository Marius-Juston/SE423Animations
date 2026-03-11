import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

PALETTE_NAME = 'flare'


def main():
    np.random.seed(0)

    dt = 0.1

    n = 1000
    steps = int((60*5)/dt)

    sigma = 1

    x = np.zeros(n)

    data = [x.copy()]



    for t in range(steps):
        x += np.sqrt(dt) * np.random.randn(n) * sigma

        data.append(x.copy())

    data = np.array(data)

    t = np.arange(steps + 1) * dt



    sub_sample = 100

    sub_sample_indices = np.random.choice(np.arange(n), sub_sample, replace=False)

    palette = sns.color_palette(PALETTE_NAME, n_colors=sub_sample)
    plt.gca().set_prop_cycle('color', palette)

    plt.plot(t, data[:, sub_sample_indices])
    plt.title(rf'White Noise ($\sigma = {sigma}$) vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Radians (degrees)')

    plt.savefig('white_noise.svg')
    plt.show()

    palette = sns.color_palette(PALETTE_NAME, n_colors=2)

    plt.gca().set_prop_cycle('color', palette)

    plt.plot(t, np.zeros(steps + 1), label=r'Expected Noise Mean ($\mu=0$)')
    plt.plot(t, np.mean(data, axis=1), label='White Noise Mean')
    plt.title('White Noise Mean vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Radians (degrees)')
    plt.legend()

    plt.savefig('white_noise_mean.svg')
    plt.show()

    palette = sns.color_palette(PALETTE_NAME, n_colors=2)

    plt.gca().set_prop_cycle('color', palette)

    plt.plot(t, np.sqrt(t) * sigma, label=fr'Expected Noise STD ($\sigma \sqrt{{t}}, \sigma = {sigma}$)')
    plt.plot(t, np.std(data, axis=1), label='White Noise STD')
    plt.title('White Noise Standard Deviation vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Radians (degrees)')
    plt.legend()

    plt.savefig('white_std.svg')
    plt.show()

if __name__ == '__main__':
    main()
