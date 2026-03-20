# https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer
# https://cie.co.at/datatable/cie-standard-illuminant-d65

import numpy as np
import matplotlib.pyplot as plt

# --- Load CIE 1931 CMFs ---
cmf = np.loadtxt("../resource/CIE_xyz_1931_2deg.csv", delimiter=",")
wavelengths = cmf[:, 0]
x_bar = cmf[:, 1]
y_bar = cmf[:, 2]
z_bar = cmf[:, 3]

# --- Load D65 SPD (must match wavelength grid) ---
spd = np.loadtxt("../resource/CIE_std_illum_D65.csv", delimiter=",")

# Handle offset if needed (assuming 1nm steps)
start_offset = int(wavelengths[0] - spd[0, 0])
# Ensure we don't slice negatively if spd starts after wavelengths
if start_offset > 0:
    spd_data = spd[start_offset:, 1]
    spd_waves = spd[start_offset:, 0]
else:
    spd_data = spd[:, 1]
    spd_waves = spd[:, 0]

# --- Ensure alignment ---
if not np.allclose(spd_waves, wavelengths):
    from scipy.interpolate import interp1d

    interp = interp1d(spd[:, 0], spd[:, 1], kind='linear', bounds_error=False, fill_value=0.0)
    S = interp(wavelengths)
else:
    S = spd_data

# --- Compute white point (proper normalization) ---
Xn = np.trapezoid(x_bar * S, wavelengths)
Yn = np.trapezoid(y_bar * S, wavelengths)
Zn = np.trapezoid(z_bar * S, wavelengths)


# --- Lab conversion helpers ---
def f(t):
    delta = 6 / 29
    # Add a small clip to prevent exact 0 passing to cbrt
    t = np.clip(t, 1e-8, None)
    return np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4 / 29)


def xyz_to_lab(X, Y, Z):
    # This function natively handles the division by the white point
    fx = f(X / Xn)
    fy = f(Y / Yn)
    fz = f(Z / Zn)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return L, a, b


# --- Generate optimal spectra ---
def generate_optimal_spectra(wavelengths):
    n = len(wavelengths)
    spectra = []

    # Absolute black and white
    spectra.append(np.zeros(n))
    spectra.append(np.ones(n))

    # Single step (High-pass and Low-pass)
    for i in range(n):
        R_up = np.zeros(n)
        R_up[i:] = 1
        spectra.append(R_up)

        R_down = np.ones(n)
        R_down[i:] = 0
        spectra.append(R_down)

    # Band-pass and Band-stop
    for i in range(n):
        for j in range(i + 1, n):
            # Type 1: Band-pass
            R_bp = np.zeros(n)
            R_bp[i:j] = 1
            spectra.append(R_bp)

            # Type 2: Band-stop
            R_bs = np.ones(n)
            R_bs[i:j] = 0
            spectra.append(R_bs)

    return np.array(spectra)


spectra = generate_optimal_spectra(wavelengths)

# --- Compute raw XYZ with illuminant ---
X = np.trapezoid(spectra * x_bar * S, wavelengths, axis=1)
Y = np.trapezoid(spectra * y_bar * S, wavelengths, axis=1)
Z = np.trapezoid(spectra * z_bar * S, wavelengths, axis=1)

# --- Convert to Lab (Using RAW XYZ) ---
L, a, b = xyz_to_lab(X, Y, Z)

# --- Normalize XYZ for sRGB matrix ---
# The standard sRGB matrix requires Y=1 for the reference white
X_srgb = X / Yn
Y_srgb = Y / Yn
Z_srgb = Z / Yn

# --- XYZ to linear sRGB ---
M = np.array([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570]
])

XYZ_matrix = np.stack([X_srgb, Y_srgb, Z_srgb], axis=1)
RGB_linear = XYZ_matrix @ M.T


# --- Gamma encoding ---
def gamma_encode(c):
    return np.where(
        c <= 0.0031308,
        12.92 * c,
        1.055 * np.power(np.clip(c, 0, None), 1 / 2.4) - 0.055
    )


# Clip before gamma encoding to prevent warnings with negative linear values
RGB = gamma_encode(RGB_linear)
RGB = np.clip(RGB , 0, 1)

# --- Plot Top View (a*, b*) ---
fig = plt.figure(figsize=(7, 7))

step_size = 1

order = np.argsort(L)

plt.scatter(a[order][::step_size], b[order][::step_size], c=RGB[order][::step_size], s=1, alpha=0.8)
plt.xlabel("a*")
plt.ylabel("b*")
plt.title("Optimal Colors (D65) - Top View")
plt.axis("equal")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

fig.savefig("optimal_colors_3d_top.svg")
fig.savefig("optimal_colors_3d_top.png", dpi=300, transparent=True)

# --- Plot Left View (L*, a*) ---
fig = plt.figure(figsize=(7, 7))

order = np.argsort(b)
plt.scatter(a[order][::step_size], L[order][::step_size], c=RGB[order][::step_size], s=1, alpha=0.8)
plt.xlabel("a*")
plt.ylabel("L*")
plt.title("Optimal Colors (D65) - Left View")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()


fig.savefig("optimal_colors_3d_left.svg")
fig.savefig("optimal_colors_3d_left.png", dpi=300, transparent=True)

plt.show()


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')

sc = ax.scatter(a, b, L, c=RGB, s=1)

ax.set_xlabel('a*')
ax.set_ylabel('b*')
ax.set_zlabel('L*')
ax.set_title('Optimal Color Solid, CIELAB (D65)')

# Set consistent limits
ax.set_xlim(a.min(), a.max())
ax.set_ylim(b.min(), b.max())
ax.set_zlim(L.min(), L.max())
fig.tight_layout()

# Rotation function
def update(angle):
    print(angle)
    ax.view_init(elev=30, azim=angle)
    return fig,



# Create animation
ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, 360, 300), interval=50)

# Save GIF
ani.save("optimal_colors_3d.gif", fps=60)

plt.close(fig)
