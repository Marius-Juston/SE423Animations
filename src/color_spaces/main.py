"""
COLOR SPACES: From Light to OKLab
══════════════════════════════════════════════════════════════════════════
A comprehensive Manim CE animation covering the full journey of color
science — from the electromagnetic spectrum, through human vision, CIE 1931,
RGB, HSV, CIELAB, to Björn Ottosson's OKLab.

Inspired by Reducible, 3Blue1Brown, and CIE optimal-color-solid work.

Run:
    manim -pql color_spaces.py <SceneName>   # preview one scene
    python render_all.py                      # render & concat all

Requires: manim (community edition ≥ 0.18), numpy
Install:  pip install manim
══════════════════════════════════════════════════════════════════════════
"""

from manim import *
import numpy as np
import math
from colorsys import hsv_to_rgb, rgb_to_hsv

# ═══════════════════════════════════════════════════════════════════════
#  GLOBAL PALETTE  (dark-theme, Reducible-inspired)
# ═══════════════════════════════════════════════════════════════════════
BG = "#0b0e17"
PANEL = "#111827"
ACCENT_BLUE = "#58c4dd"
ACCENT_TEAL = "#5ce1e6"
ACCENT_PINK = "#ff6b9d"
ACCENT_PURPLE = "#b388ff"
ACCENT_ORANGE = "#ffab40"
ACCENT_YELLOW = "#ffee58"
ACCENT_GREEN = "#69f0ae"
TEXT_PRI = "#e8eaf6"
TEXT_SEC = "#9fa8da"
GRID_COL = "#1e2640"
MUTED_RED = "#ef5350"
MUTED_GREEN = "#66bb6a"
MUTED_BLUE = "#42a5f5"


# ═══════════════════════════════════════════════════════════════════════
#  COLOR MATH ENGINE
# ═══════════════════════════════════════════════════════════════════════

# ── sRGB gamma ──────────────────────────────────────────────────────
def srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92,
                    ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.0031308, 12.92 * c,
                    1.055 * np.power(np.clip(c, 0, None), 1.0 / 2.4) - 0.055)


# ── XYZ ↔ linear sRGB (D65) ────────────────────────────────────────
M_XYZ_TO_LRGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
])
M_LRGB_TO_XYZ = np.linalg.inv(M_XYZ_TO_LRGB)


def xyz_to_srgb(X, Y, Z):
    """XYZ (D65, Y=1 white) → sRGB [0,1] clamped."""
    v = M_XYZ_TO_LRGB @ np.array([X, Y, Z])
    return np.clip(linear_to_srgb(np.clip(v, 0, None)), 0, 1)


# ── CIELAB ──────────────────────────────────────────────────────────
Xn, Yn, Zn = 0.95047, 1.00000, 1.08883  # D65 white point


def _f_lab(t):
    delta = 6.0 / 29.0
    t = np.clip(t, 1e-10, None)
    return np.where(t > delta ** 3, np.cbrt(t),
                    t / (3 * delta ** 2) + 4.0 / 29.0)


def xyz_to_cielab(X, Y, Z):
    fx = _f_lab(X / Xn)
    fy = _f_lab(Y / Yn)
    fz = _f_lab(Z / Zn)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return L, a, b


def srgb_to_cielab(r, g, b):
    lin = srgb_to_linear(np.array([r, g, b]))
    XYZ = M_LRGB_TO_XYZ @ lin
    return xyz_to_cielab(*XYZ)


def cielab_to_xyz(L, a, b):
    """Inverse CIELAB → XYZ."""
    delta = 6.0 / 29.0
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    def inv_f(t):
        return t ** 3 if t > delta else (t - 4.0 / 29.0) * 3 * delta ** 2

    X = Xn * inv_f(fx)
    Y = Yn * inv_f(fy)
    Z = Zn * inv_f(fz)
    return X, Y, Z


# ── OKLab ───────────────────────────────────────────────────────────
def linear_srgb_to_oklab(r, g, b):
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_ = np.cbrt(np.clip(l, 0, None))
    m_ = np.cbrt(np.clip(m, 0, None))
    s_ = np.cbrt(np.clip(s, 0, None))
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return L, A, B


def oklab_to_linear_srgb(L, a, b):
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l = l_ ** 3;
    m = m_ ** 3;
    s = s_ ** 3
    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bx = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return r, g, bx


def srgb_to_oklab(r, g, b):
    lin = srgb_to_linear(np.array([r, g, b]))
    return linear_srgb_to_oklab(*lin)


# ── Helpers ─────────────────────────────────────────────────────────
def rgb_to_hex(rgb):
    r, g, b = [int(np.clip(x, 0, 1) * 255) for x in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def oklab_to_hex(L, a, b):
    r, g, bx = oklab_to_linear_srgb(L, a, b)
    return rgb_to_hex(linear_to_srgb(np.clip([r, g, bx], 0, 1)))


def wavelength_to_rgb(wl):
    """Attempt to map visible wavelength (nm) → sRGB."""
    gamma = 0.8
    if 380 <= wl < 440:
        R, G, B = -(wl - 440) / (440 - 380), 0, 1
    elif 440 <= wl < 490:
        R, G, B = 0, (wl - 440) / (490 - 440), 1
    elif 490 <= wl < 510:
        R, G, B = 0, 1, -(wl - 510) / (510 - 490)
    elif 510 <= wl < 580:
        R, G, B = (wl - 510) / (580 - 510), 1, 0
    elif 580 <= wl < 645:
        R, G, B = 1, -(wl - 645) / (645 - 580), 0
    elif 645 <= wl <= 780:
        R, G, B = 1, 0, 0
    else:
        R, G, B = 0, 0, 0
    if 380 <= wl < 420:
        f = 0.3 + 0.7 * (wl - 380) / 40
    elif 700 < wl <= 780:
        f = 0.3 + 0.7 * (780 - wl) / 80
    else:
        f = 1.0
    return np.array([f * abs(R) ** gamma, f * abs(G) ** gamma, f * abs(B) ** gamma])


# ── CIE 1931 CMF approximation (Wyman et al. 2013 Gaussian fit) ───
def xbar(w):
    return (1.056 * np.exp(-0.5 * ((w - 599.8) / 37.9) ** 2)
            + 0.362 * np.exp(-0.5 * ((w - 442.0) / 16.0) ** 2)
            - 0.065 * np.exp(-0.5 * ((w - 501.1) / 20.4) ** 2))


def ybar(w):
    return (0.821 * np.exp(-0.5 * ((w - 568.8) / 46.9) ** 2)
            + 0.286 * np.exp(-0.5 * ((w - 530.9) / 16.3) ** 2))


def zbar(w):
    return (1.217 * np.exp(-0.5 * ((w - 437.0) / 11.8) ** 2)
            + 0.681 * np.exp(-0.5 * ((w - 459.0) / 26.0) ** 2))


# ── Gamut surface generators ───────────────────────────────────────
def generate_gamut_surface(res=16):
    """sRGB cube surface points."""
    pts = []
    vals = np.linspace(0, 1, res)
    for r in vals:
        for g in vals:
            for b in vals:
                if np.isclose(r, 0) or np.isclose(r, 1) or \
                        np.isclose(g, 0) or np.isclose(g, 1) or \
                        np.isclose(b, 0) or np.isclose(b, 1):
                    pts.append([r, g, b])
    return np.array(pts)


def generate_gamut_volume(res=10):
    pts = []
    for r in np.linspace(0, 1, res):
        for g in np.linspace(0, 1, res):
            for b in np.linspace(0, 1, res):
                pts.append([r, g, b])
    return np.array(pts)


# ── Blend functions ────────────────────────────────────────────────
def srgb_blend(c1, c2, t):
    return np.clip(np.array(c1) * (1 - t) + np.array(c2) * t, 0, 1)


def hsv_blend_fn(c1, c2, t):
    h1, s1, v1 = rgb_to_hsv(*c1)
    h2, s2, v2 = rgb_to_hsv(*c2)
    dh = h2 - h1
    if abs(dh) > 0.5:
        if dh > 0:
            h1 += 1
        else:
            h2 += 1
    h = ((h1 + t * (h2 - h1)) % 1.0)
    s = s1 + t * (s2 - s1)
    v = v1 + t * (v2 - v1)
    return np.array(hsv_to_rgb(h, s, v))


def oklab_blend_fn(c1, c2, t):
    lin1 = srgb_to_linear(np.array(c1))
    lin2 = srgb_to_linear(np.array(c2))
    L1, a1, b1 = linear_srgb_to_oklab(*lin1)
    L2, a2, b2 = linear_srgb_to_oklab(*lin2)
    L = L1 + t * (L2 - L1)
    a = a1 + t * (a2 - a1)
    b = b1 + t * (b2 - b1)
    ro, go, bo = oklab_to_linear_srgb(L, a, b)
    return np.clip(linear_to_srgb(np.clip([ro, go, bo], 0, 1)), 0, 1)


def make_gradient_bar(blend_fn, c1_rgb, c2_rgb, n=80, width=9.5, height=0.5):
    bar = VGroup()
    sw = width / n
    for i in range(n):
        t = i / (n - 1)
        col = blend_fn(c1_rgb, c2_rgb, t)
        rect = Rectangle(width=sw + 0.015, height=height,
                         fill_color=rgb_to_hex(col), fill_opacity=1,
                         stroke_width=0)
        rect.move_to(LEFT * width / 2 + RIGHT * (i * sw + sw / 2))
        bar.add(rect)
    return bar


# ── LCh / OKLch (cylindrical representations) ──────────────────────
def cielab_to_lch(L, a, b):
    """CIELAB (L*, a*, b*) → LCh (L*, C*, h°).  h in [0, 360)."""
    C = np.sqrt(a ** 2 + b ** 2)
    h = np.degrees(np.arctan2(b, a)) % 360
    return L, C, h


def lch_to_cielab(L, C, h_deg):
    """LCh → CIELAB."""
    h = np.radians(h_deg)
    return L, C * np.cos(h), C * np.sin(h)


def oklab_to_oklch(L, a, b):
    """OKLab → OKLch (L, C, h°).  h in [0, 360)."""
    C = np.sqrt(a ** 2 + b ** 2)
    h = np.degrees(np.arctan2(b, a)) % 360
    return L, C, h


def oklch_to_oklab(L, C, h_deg):
    """OKLch → OKLab."""
    h = np.radians(h_deg)
    return L, C * np.cos(h), C * np.sin(h)


def oklch_blend_fn(c1, c2, t):
    """Blend two sRGB colors through OKLch (short-arc hue interpolation)."""
    lin1 = srgb_to_linear(np.array(c1))
    lin2 = srgb_to_linear(np.array(c2))
    L1, a1, b1 = linear_srgb_to_oklab(*lin1)
    L2, a2, b2 = linear_srgb_to_oklab(*lin2)
    _, C1, h1 = oklab_to_oklch(L1, a1, b1)
    _, C2, h2 = oklab_to_oklch(L2, a2, b2)
    dh = h2 - h1
    if dh > 180:  dh -= 360
    if dh < -180: dh += 360
    L = L1 + t * (L2 - L1)
    C = C1 + t * (C2 - C1)
    h = (h1 + t * dh) % 360
    Lo, ao, bo = oklch_to_oklab(L, C, h)
    ro, go, bxo = oklab_to_linear_srgb(Lo, ao, bo)
    return np.clip(linear_to_srgb(np.clip([ro, go, bxo], 0, 1)), 0, 1)


# ── ΔE color difference ────────────────────────────────────────────
def delta_e76(lab1, lab2):
    """CIE76 Euclidean distance in CIELAB.  lab1, lab2: (L*, a*, b*) tuples."""
    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2))))


def delta_e2000(lab1, lab2, kL=1, kC=1, kH=1):
    """CIEDE2000 color difference (Sharma et al. 2005)."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    C_avg = (C1 + C2) / 2.0
    C_avg7 = C_avg ** 7
    G = 0.5 * (1 - np.sqrt(C_avg7 / (C_avg7 + 25 ** 7)))
    a1p = a1 * (1 + G);
    a2p = a2 * (1 + G)
    C1p = np.sqrt(a1p ** 2 + b1 ** 2)
    C2p = np.sqrt(a2p ** 2 + b2 ** 2)

    def hp(ap, bv):
        if ap == 0 and bv == 0: return 0.0
        return np.degrees(np.arctan2(bv, ap)) % 360

    h1p = hp(a1p, b1);
    h2p = hp(a2p, b2)
    dLp = L2 - L1;
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p - h1p > 180:
        dhp = h2p - h1p - 360
    else:
        dhp = h2p - h1p + 360
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))
    Lbar = (L1 + L2) / 2
    Cbar_p = (C1p + C2p) / 2
    if C1p * C2p == 0:
        hbar_p = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbar_p = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        hbar_p = (h1p + h2p + 360) / 2
    else:
        hbar_p = (h1p + h2p - 360) / 2
    T = (1 - 0.17 * np.cos(np.radians(hbar_p - 30))
         + 0.24 * np.cos(np.radians(2 * hbar_p))
         + 0.32 * np.cos(np.radians(3 * hbar_p + 6))
         - 0.20 * np.cos(np.radians(4 * hbar_p - 63)))
    SL = 1 + 0.015 * (Lbar - 50) ** 2 / np.sqrt(20 + (Lbar - 50) ** 2)
    SC = 1 + 0.045 * Cbar_p
    SH = 1 + 0.015 * Cbar_p * T
    Cbar7 = Cbar_p ** 7
    RC = 2 * np.sqrt(Cbar7 / (Cbar7 + 25 ** 7))
    d_theta = 30 * np.exp(-((hbar_p - 275) / 25) ** 2)
    RT = -np.sin(np.radians(2 * d_theta)) * RC
    return float(np.sqrt(
        (dLp / (kL * SL)) ** 2 + (dCp / (kC * SC)) ** 2 + (dHp / (kH * SH)) ** 2
        + RT * (dCp / (kC * SC)) * (dHp / (kH * SH))))


# ── MacAdam 1942 ellipses (25 JND loci, Brown & MacAdam 1949) ────────
# Format: (x_chrom, y_chrom, semi_minor, semi_major, angle_deg_from_x_axis)
MACADAM_ELLIPSES = [
    (0.160, 0.057, 0.0030, 0.0085, 62.5),
    (0.187, 0.118, 0.0027, 0.0102, 77.0),
    (0.253, 0.125, 0.0021, 0.0076, 55.5),
    (0.150, 0.680, 0.0048, 0.0198, 8.0),
    (0.131, 0.521, 0.0039, 0.0158, 11.0),
    (0.212, 0.550, 0.0068, 0.0295, 26.0),
    (0.258, 0.450, 0.0062, 0.0201, 30.0),
    (0.152, 0.365, 0.0032, 0.0196, 4.0),
    (0.280, 0.385, 0.0048, 0.0184, 30.0),
    (0.380, 0.498, 0.0080, 0.0307, 41.0),
    (0.160, 0.200, 0.0023, 0.0100, 34.5),
    (0.228, 0.250, 0.0032, 0.0143, 44.0),
    (0.305, 0.323, 0.0040, 0.0136, 50.0),
    (0.385, 0.393, 0.0043, 0.0151, 36.0),
    (0.472, 0.399, 0.0058, 0.0159, 38.0),
    (0.527, 0.350, 0.0079, 0.0183, 68.0),
    (0.475, 0.300, 0.0058, 0.0141, 47.0),
    (0.510, 0.236, 0.0040, 0.0104, 60.0),
    (0.596, 0.283, 0.0063, 0.0134, 68.0),
    (0.344, 0.284, 0.0037, 0.0118, 48.0),
    (0.390, 0.237, 0.0039, 0.0101, 53.0),
    (0.441, 0.198, 0.0044, 0.0103, 60.0),
    (0.278, 0.223, 0.0027, 0.0082, 37.0),
    (0.240, 0.290, 0.0028, 0.0092, 44.0),
    (0.300, 0.255, 0.0032, 0.0097, 49.0),
]

# ── Color vision deficiency simulation (Viénot et al. 1999) ─────────
# Hunt-Pointer-Estevez matrix (D65 adapted)
M_RGB_TO_LMS = np.array([
    [0.4002, 0.7076, -0.0808],
    [-0.2263, 1.1653, 0.0457],
    [0.0000, 0.0000, 0.9182],
])
M_LMS_TO_RGB = np.linalg.inv(M_RGB_TO_LMS)

# LMS projection matrices for each deficiency type
M_DEUTERANOPIA = np.array([  # M-cones absent (red-green)
    [1.0, 0.0, 0.0],
    [0.4942, 0.0, 1.2483],
    [0.0, 0.0, 1.0],
])
M_PROTANOPIA = np.array([  # L-cones absent (red-blind)
    [0.0, 2.0234, -2.5258],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])
M_TRITANOPIA = np.array([  # S-cones absent (blue-yellow)
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [-0.3959, 0.8011, 0.0],
])


def _simulate_cvd(r, g, b, cvd_matrix):
    """Apply color vision deficiency simulation via LMS space."""
    lin = srgb_to_linear(np.array([r, g, b]))
    lms = M_RGB_TO_LMS @ lin
    lms_sim = cvd_matrix @ lms
    lin_sim = M_LMS_TO_RGB @ lms_sim
    return np.clip(linear_to_srgb(np.clip(lin_sim, 0, None)), 0, 1)


def simulate_deuteranopia(r, g, b):
    """sRGB → deuteranopia-simulated sRGB (no M cones, red-green blind)."""
    return _simulate_cvd(r, g, b, M_DEUTERANOPIA)


def simulate_protanopia(r, g, b):
    """sRGB → protanopia-simulated sRGB (no L cones, red-blind)."""
    return _simulate_cvd(r, g, b, M_PROTANOPIA)


def simulate_tritanopia(r, g, b):
    """sRGB → tritanopia-simulated sRGB (no S cones, blue-yellow blind)."""
    return _simulate_cvd(r, g, b, M_TRITANOPIA)


# ── Planckian locus (Kang et al. 2002 polynomial) ────────────────────
def planckian_xy(T):
    """Blackbody chromaticity (x, y) for color temperature T in Kelvin.
    Valid range: ~1667 K – 25000 K.  Kang et al. 2002 polynomial fit."""
    if T <= 4000:
        x = (-0.2661239e9 / T ** 3 - 0.2343580e6 / T ** 2
             + 0.8776956e3 / T + 0.179910)
    else:
        x = (-3.0258469e9 / T ** 3 + 2.1070379e6 / T ** 2
             + 0.2226347e3 / T + 0.240390)
    if T <= 2222:
        y = (-1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683)
    elif T <= 4000:
        y = (-0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867)
    else:
        y = (3.0817580 * x ** 3 - 5.87338670 * x ** 2 + 3.75112997 * x - 0.37001483)
    return x, y


# ── Bradford chromatic adaptation ────────────────────────────────────
M_BRADFORD = np.array([
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
])
_M_BRADFORD_INV = np.linalg.inv(M_BRADFORD)


def bradford_adapt(X, Y, Z, src_white, dst_white):
    """Adapt XYZ from src_white to dst_white using Bradford method.
    src_white, dst_white: (Xw, Yw, Zw) tuples (e.g., D65, D50)."""
    rho_s = M_BRADFORD @ np.array(src_white)
    rho_d = M_BRADFORD @ np.array(dst_white)
    scale = np.diag(rho_d / rho_s)
    M_adapt = _M_BRADFORD_INV @ scale @ M_BRADFORD
    return M_adapt @ np.array([X, Y, Z])


# D65 and D50 white points
D65_WHITE = (0.95047, 1.00000, 1.08883)
D50_WHITE = (0.96422, 1.00000, 0.82521)

# ── PQ transfer function (SMPTE ST 2084) ─────────────────────────────
_PQ_M1 = 0.1593017578125
_PQ_M2 = 78.84375
_PQ_C1 = 0.8359375
_PQ_C2 = 18.8515625
_PQ_C3 = 18.6875
_PQ_L_PEAK = 10000.0  # cd/m²


def pq_eotf(N):
    """PQ EOTF: normalized [0,1] → absolute luminance (cd/m²).
    SMPTE ST 2084 / ITU-R BT.2100."""
    N = np.asarray(N, dtype=float)
    Np = N ** (1.0 / _PQ_M2)
    num = np.maximum(Np - _PQ_C1, 0.0)
    den = _PQ_C2 - _PQ_C3 * Np
    return _PQ_L_PEAK * (num / den) ** (1.0 / _PQ_M1)


def pq_oetf(L):
    """PQ OETF: absolute luminance (cd/m²) → normalized [0,1]."""
    L = np.asarray(L, dtype=float)
    Lp = (np.clip(L, 0, None) / _PQ_L_PEAK) ** _PQ_M1
    return (((_PQ_C1 + _PQ_C2 * Lp) / (1.0 + _PQ_C3 * Lp)) ** _PQ_M2)


# ── WCAG 2.x contrast ────────────────────────────────────────────────
def wcag_relative_luminance(r, g, b):
    """sRGB [0,1] → WCAG 2.x relative luminance (= linear Y)."""
    lin = srgb_to_linear(np.array([r, g, b]))
    return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])


def wcag_contrast(rgb1, rgb2):
    """WCAG 2.x contrast ratio (≥ 1.0) between two sRGB colors."""
    L1 = wcag_relative_luminance(*rgb1)
    L2 = wcag_relative_luminance(*rgb2)
    lighter = max(L1, L2)
    darker = min(L1, L2)
    return (lighter + 0.05) / (darker + 0.05)


# ── Gamut mapping (OKLch chroma reduction) ────────────────────────────
def is_in_srgb(lin_r, lin_g, lin_b, tol=1e-4):
    """Return True if linear sRGB values are all within [0, 1] (with tol)."""
    return (lin_r >= -tol and lin_r <= 1 + tol and
            lin_g >= -tol and lin_g <= 1 + tol and
            lin_b >= -tol and lin_b <= 1 + tol)


def gamut_map_oklch(L, a, b, tol=1e-4):
    """Reduce chroma of OKLab color until it fits in sRGB gamut.
    Returns clamped (L, a, b) in OKLab."""
    _, C, h = oklab_to_oklch(L, a, b)
    # Already in gamut?
    r0, g0, b0 = oklab_to_linear_srgb(L, a, b)
    if is_in_srgb(r0, g0, b0, tol):
        return L, a, b
    # Binary search on chroma
    C_lo, C_hi = 0.0, C
    for _ in range(30):
        C_mid = (C_lo + C_hi) / 2
        Lo, ao, bo = oklch_to_oklab(L, C_mid, h)
        rm, gm, bm = oklab_to_linear_srgb(Lo, ao, bo)
        if is_in_srgb(rm, gm, bm, tol):
            C_lo = C_mid
        else:
            C_hi = C_mid
        if C_hi - C_lo < tol:
            break
    return oklch_to_oklab(L, C_lo, h)


# ── OKLch palette generation ──────────────────────────────────────────
def oklch_palette(n, L=0.65, C=0.15):
    """Return n equally-spaced OKLch hues as list of sRGB [0,1] arrays."""
    colors = []
    for i in range(n):
        h = (360.0 * i / n) % 360
        Lo, ao, bo = oklch_to_oklab(L, C, h)
        ro, go, bxo = oklab_to_linear_srgb(Lo, ao, bo)
        colors.append(np.clip(linear_to_srgb(np.clip([ro, go, bxo], 0, 1)), 0, 1))
    return colors


# ── IPT color space (Ebner & Fairchild 1998) ─────────────────────────
# XYZ → LMS (Hunt-Pointer-Estevez, D65 adapted)
M_XYZ_TO_LMS_IPT = np.array([
    [0.4002, 0.7076, -0.0808],
    [-0.2263, 1.1653, 0.0457],
    [0.0000, 0.0000, 0.9182],
])
# LMS^0.43 → IPT
M_LMS_TO_IPT = np.array([
    [0.4000, 0.4000, 0.2000],
    [4.4550, -4.8510, 0.3960],
    [0.8056, 0.3572, -1.1628],
])


def xyz_to_ipt(X, Y, Z):
    """CIE XYZ (D65) → IPT color space (Ebner & Fairchild 1998).
    Returns (I, P, T)."""
    lms = M_XYZ_TO_LMS_IPT @ np.array([X, Y, Z])
    # Non-linearity: component-wise power 0.43 (preserving sign)
    lms_p = np.sign(lms) * np.abs(lms) ** 0.43
    return M_LMS_TO_IPT @ lms_p


# ── CIE RGB CMF approximations (r̄ḡb̄, for negative-lobe visualization) ──
# Based on Stiles & Burch 1955 / CIE 1931 2° primaries: R=700nm, G=546.1nm, B=435.8nm
# Simple Gaussian-sum analytic fits to the tabulated CMFs
def rbar(w):
    """CIE r̄(λ) color matching function (Stiles & Burch 2° approx).
    Has a negative lobe in the cyan ~480-520nm region."""
    pos = (0.84 * np.exp(-0.5 * ((w - 600.8) / 33.0) ** 2)
           + 0.10 * np.exp(-0.5 * ((w - 445.0) / 20.0) ** 2))
    neg = -0.25 * np.exp(-0.5 * ((w - 494.0) / 27.0) ** 2)
    return pos + neg


def gbar(w):
    """CIE ḡ(λ) color matching function. Small negative lobe ~380-440nm."""
    pos = (0.76 * np.exp(-0.5 * ((w - 555.0) / 36.0) ** 2)
           + 0.22 * np.exp(-0.5 * ((w - 530.0) / 16.0) ** 2))
    neg = -0.05 * np.exp(-0.5 * ((w - 410.0) / 20.0) ** 2)
    return pos + neg


def bbar(w):
    """CIE b̄(λ) color matching function. Mostly positive, sharp blue peak."""
    return (1.12 * np.exp(-0.5 * ((w - 450.0) / 18.0) ** 2)
            + 0.14 * np.exp(-0.5 * ((w - 430.0) / 9.0) ** 2))


# ── Simple illuminant/SPD models ──────────────────────────────────────
def illuminant_d65_spd(w):
    """Approximate CIE D65 relative spectral power distribution."""
    w = np.asarray(w, dtype=float)
    return np.where((w >= 380) & (w <= 780),
                    100.0 * (0.5 + 0.5 * np.exp(-((w - 560) / 180) ** 2) + 0.08),
                    0.0)


def illuminant_a_spd(w):
    """CIE Illuminant A: blackbody at 2856K (normalized to 1 at 560nm)."""
    w = np.asarray(w, dtype=float)
    # Planck radiation ratio relative to 560nm
    c2 = 1.435e7  # nm·K
    ref = np.exp(c2 / (560 * 2856)) - 1
    return np.where((w > 0), (w / 560) ** (-5) * (ref / (np.exp(c2 / (w * 2856)) - 1)), 0.0)


def fluorescent_spd(w):
    """Simplified fluorescent lamp SPD: broad continuum + line peaks at 546nm and 611nm."""
    w = np.asarray(w, dtype=float)
    broad = 0.3 * np.exp(-0.5 * ((w - 560) / 80) ** 2)
    line1 = 0.6 * np.exp(-0.5 * ((w - 546) / 4) ** 2)
    line2 = 0.5 * np.exp(-0.5 * ((w - 611) / 4) ** 2)
    return np.where((w >= 380) & (w <= 780), broad + line1 + line2, 0.0)


def spd_to_xyz(spd_fn, illum_fn, wl=None):
    """Integrate SPD × illuminant × CMFs over visible range → XYZ.
    Returns normalized (X, Y, Z) where Y_white = 1."""
    if wl is None:
        wl = np.linspace(380, 780, 401)
    spd = np.array([spd_fn(w) for w in wl])
    illum = np.array([illum_fn(w) for w in wl])
    xb = np.array([xbar(w) for w in wl])
    yb = np.array([ybar(w) for w in wl])
    zb = np.array([zbar(w) for w in wl])
    k = 1.0 / np.trapezoid(illum * yb, wl)
    X = k * np.trapezoid(spd * illum * xb, wl)
    Y = k * np.trapezoid(spd * illum * yb, wl)
    Z = k * np.trapezoid(spd * illum * zb, wl)
    return X, Y, Z


# ── Linear-light blending ─────────────────────────────────────────────
def linear_blend(c1_rgb, c2_rgb, t):
    """Blend two sRGB colors in linear light, then gamma-encode output."""
    lin1 = srgb_to_linear(np.array(c1_rgb))
    lin2 = srgb_to_linear(np.array(c2_rgb))
    lin_mid = lin1 * (1 - t) + lin2 * t
    return np.clip(linear_to_srgb(np.clip(lin_mid, 0, 1)), 0, 1)


# ── Y'CbCr (BT.709) ──────────────────────────────────────────────────
def rgb_to_ycbcr_bt709(r, g, b):
    """sRGB (gamma-encoded) → Y'CbCr full-range, BT.709."""
    Yp = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Cb = (b - Yp) / 1.8556
    Cr = (r - Yp) / 1.5748
    return Yp, Cb, Cr


def ycbcr_to_rgb_bt709(Yp, Cb, Cr):
    """Y'CbCr (BT.709) → sRGB (gamma-encoded)."""
    r = np.clip(Yp + 1.5748 * Cr, 0, 1)
    g = np.clip(Yp - 0.1873 * Cb - 0.4681 * Cr, 0, 1)
    b = np.clip(Yp + 1.8556 * Cb, 0, 1)
    return r, g, b


# ── Tone mapping operators ────────────────────────────────────────────
def reinhard_tonemap(L, L_white=1.0):
    """Reinhard (2002) tone map: L_out = L(1+L/Lw²)/(1+L)."""
    L = np.asarray(L, dtype=float)
    return L * (1.0 + L / L_white ** 2) / (1.0 + L)


def aces_tonemap(x):
    """ACES filmic tone mapping approximation (Narkowicz 2015)."""
    x = np.asarray(x, dtype=float)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


def agx_tonemap(x):
    """AgX-like S-curve tone mapping (simplified version)."""
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0, None)
    return x / (x + 0.5)  # simplified sigmoid for visualization


# ── Spectral power distribution → XYZ (already in spd_to_xyz) ────────
def spectral_reflectance_to_xyz(reflect_fn, illum_fn, wl=None):
    """Integrate object reflectance × illuminant × CMFs → XYZ."""
    if wl is None:
        wl = np.linspace(380, 780, 401)
    refl = np.array([float(reflect_fn(w)) for w in wl])
    illum = np.array([float(illum_fn(w)) for w in wl])
    xb = np.array([xbar(w) for w in wl])
    yb = np.array([ybar(w) for w in wl])
    zb = np.array([zbar(w) for w in wl])
    k = 1.0 / np.trapz(illum * yb, wl)
    X = k * np.trapz(refl * illum * xb, wl)
    Y = k * np.trapz(refl * illum * yb, wl)
    Z = k * np.trapz(refl * illum * zb, wl)
    return X, Y, Z


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 01 — TITLE
# ═══════════════════════════════════════════════════════════════════════
class TitleScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        # Particle field — 220 OKLab hue-circle dots, capped within safe frame bounds
        dots = VGroup()
        np.random.seed(7)
        for _ in range(220):
            h = np.random.random() * TAU
            L = 0.60 + 0.30 * np.random.random()
            C = 0.10 + 0.14 * np.random.random()  # higher C → more vivid hues
            col = oklab_to_hex(L, C * np.cos(h), C * np.sin(h))
            d = Dot(point=[np.random.uniform(-6.2, 6.2),
                           np.random.uniform(-3.6, 3.6), 0],
                    radius=np.random.uniform(0.02, 0.10),
                    color=col, fill_opacity=np.random.uniform(0.15, 0.55))
            dots.add(d)
        self.play(FadeIn(dots, lag_ratio=0.012), run_time=1.5)

        title = Text("Color Spaces", font_size=84, weight=BOLD, color=TEXT_PRI)
        sub = Text("Light · Eyes · Math · Color", font_size=38, color=ACCENT_TEAL)
        sub.next_to(title, DOWN, buff=0.4)
        VGroup(title, sub).move_to(ORIGIN)

        rainbow = Line(LEFT * 4.5, RIGHT * 4.5, stroke_width=6)
        rainbow.set_color(color=[MUTED_RED, ACCENT_ORANGE, ACCENT_YELLOW,
                                 ACCENT_GREEN, ACCENT_BLUE, ACCENT_PURPLE, ACCENT_PINK])
        rainbow.next_to(sub, DOWN, buff=0.35)

        scope_lbl = Text("complete beginner-friendly",
                         font_size=22, color=TEXT_SEC)
        scope_lbl.next_to(rainbow, DOWN, buff=0.3)

        self.play(Write(title, run_time=1.6),
                  FadeIn(sub, shift=UP * 0.3, run_time=1.4))
        self.play(Create(rainbow, run_time=2.2))
        self.play(FadeIn(scope_lbl, shift=UP * 0.15), run_time=1.2)
        self.play(dots.animate.set_opacity(0.18), run_time=2.0, rate_func=smooth)
        self.wait(3.5)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 02 — ELECTROMAGNETIC SPECTRUM → VISIBLE LIGHT
# ═══════════════════════════════════════════════════════════════════════
class ElectromagneticSpectrumScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("I.  What Is Light?", font_size=44, color=ACCENT_BLUE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        # Hook: beginner intro before the spectrum
        hook = VGroup(
            Text("Light is electromagnetic radiation — energy travelling as waves.",
                 font_size=22, color=TEXT_PRI),
            Text("The wavelength of the wave determines what your eye perceives as colour.",
                 font_size=20, color=TEXT_PRI),
        ).arrange(DOWN, buff=0.35)
        hook.move_to(ORIGIN)
        self.play(FadeIn(hook, lag_ratio=0.3), run_time=1.5)
        self.wait(3.5)
        self.play(FadeOut(hook), run_time=1.0)

        # Full EM spectrum bar
        em_data = [
            ("γ-rays", "#7b1fa2", 0.8),
            ("X-rays", "#6a1b9a", 1.0),
            ("UV", "#4a148c", 0.8),
            ("VISIBLE", None, 2.0),
            ("IR", "#b71c1c", 1.2),
            ("Micro", "#880e4f", 1.0),
            ("Radio", "#4a0033", 2.0),
        ]
        em_bar = VGroup()
        x = -5.4
        vis_start_x = None
        vis_end_x = None
        non_vis_idx = 0
        for name, col, w in em_data:
            if name == "VISIBLE":
                vis_start_x = x
                n_vis = 50
                sw = w / n_vis
                for i in range(n_vis):
                    wl = 380 + (780 - 380) * i / (n_vis - 1)
                    rgb = wavelength_to_rgb(wl)
                    r = Rectangle(width=sw + 0.005, height=0.8,
                                  fill_color=rgb_to_hex(rgb), fill_opacity=1,
                                  stroke_width=0)
                    r.move_to([x + i * sw + sw / 2, 0.8, 0])
                    em_bar.add(r)
                vis_end_x = x + w
            else:
                r = Rectangle(width=w, height=0.8, fill_color=col,
                              fill_opacity=0.6, stroke_width=0.5,
                              stroke_color="#ffffff15")
                r.move_to([x + w / 2, 0.8, 0])
                lbl = Text(name, font_size=15, color="#ffffffaa")
                # Alternate: even indices above bar, odd indices below
                if non_vis_idx % 2 == 0:
                    lbl.next_to(r, UP, buff=0.08)
                else:
                    lbl.next_to(r, DOWN, buff=0.08)
                em_bar.add(VGroup(r, lbl))
                non_vis_idx += 1
            x += w

        self.play(FadeIn(em_bar, lag_ratio=0.015), run_time=2)

        # Brace around visible
        vis_brace = BraceBetweenPoints(
            [vis_start_x, 0.35, 0], [vis_end_x, 0.35, 0],
            direction=DOWN, color=ACCENT_YELLOW)
        vis_lbl = Text("~380–780 nm — the only light we see!", font_size=18,
                       color=ACCENT_YELLOW)
        vis_lbl.next_to(vis_brace, DOWN, buff=0.15)
        self.play(Create(vis_brace), FadeIn(vis_lbl))
        self.wait(3.5)

        # Zoom into visible spectrum
        self.play(*[FadeOut(m) for m in [em_bar, vis_brace, vis_lbl]], run_time=1.4)

        zoom_title = Text("The Visible Spectrum", font_size=28, color=TEXT_PRI)
        zoom_title.move_to(UP * 1.8)
        self.play(FadeIn(zoom_title))

        spectrum = VGroup()
        n = 250
        bw = 11
        for i in range(n):
            wl = 380 + (780 - 380) * i / (n - 1)
            rgb = wavelength_to_rgb(wl)
            rect = Rectangle(width=bw / n + 0.005, height=1.0,
                             fill_color=rgb_to_hex(rgb), fill_opacity=1,
                             stroke_width=0)
            rect.move_to(LEFT * bw / 2 + RIGHT * (i * bw / n + bw / n / 2) + UP * 0.4)
            spectrum.add(rect)
        self.play(FadeIn(spectrum, lag_ratio=0.003, run_time=2))

        # Colour name labels above visible spectrum
        colour_name_data = [
            ("Violet", 405), ("Blue", 455), ("Green", 530),
            ("Yellow", 575), ("Orange", 610), ("Red", 680),
        ]
        colour_labels = VGroup()
        for cname, wl in colour_name_data:
            frac = (wl - 380) / (780 - 380)
            xp = -bw / 2 + frac * bw
            clbl = Text(cname, font_size=13, color=rgb_to_hex(wavelength_to_rgb(wl)))
            clbl.move_to([xp, 1.1, 0])
            colour_labels.add(clbl)
        self.play(FadeIn(colour_labels, lag_ratio=0.2), run_time=1.2)

        # Wavelength ticks
        ticks = VGroup()
        for wl in [400, 450, 500, 550, 600, 650, 700, 750]:
            frac = (wl - 380) / (780 - 380)
            xp = -bw / 2 + frac * bw
            tick = Line([xp, -0.15, 0], [xp, -0.35, 0],
                        color=TEXT_SEC, stroke_width=1.5)
            lbl = Text(str(wl), font_size=14, color=TEXT_SEC)
            lbl.next_to(tick, DOWN, buff=0.05)
            ticks.add(VGroup(tick, lbl))
        nm = Text("wavelength (nm)", font_size=16, color=TEXT_SEC).move_to(DOWN * 0.9)
        self.play(FadeIn(ticks), FadeIn(nm))

        # Wave frequency comparison: violet (short λ) vs red (long λ)
        wave_y = 1.8
        violet_axes = Axes(
            x_range=[0, 4, 1], y_range=[-1.2, 1.2, 1],
            x_length=4.5, y_length=1.2,
            axis_config={"include_ticks": False, "include_tip": False,
                         "stroke_width": 1.0, "stroke_color": "#ffffff22"},
        ).move_to(LEFT * 2.8 + DOWN * wave_y)
        violet_wave = violet_axes.plot(
            lambda t: np.sin(t * 6 * PI), color=ACCENT_PURPLE, stroke_width=2.5)
        violet_lbl = Text("Violet  λ ≈ 400 nm · short · energetic",
                          font_size=14, color=ACCENT_PURPLE)
        violet_lbl.next_to(violet_axes, DOWN, buff=0.12)

        red_axes = Axes(
            x_range=[0, 4, 1], y_range=[-1.2, 1.2, 1],
            x_length=4.5, y_length=1.2,
            axis_config={"include_ticks": False, "include_tip": False,
                         "stroke_width": 1.0, "stroke_color": "#ffffff22"},
        ).move_to(RIGHT * 2.8 + DOWN * wave_y)
        red_wave = red_axes.plot(
            lambda t: np.sin(t * 2 * PI), color=MUTED_RED, stroke_width=2.5)
        red_lbl = Text("Red  λ ≈ 700 nm · long · less energy",
                       font_size=14, color=MUTED_RED)
        red_lbl.next_to(red_axes, DOWN, buff=0.12)

        self.play(Create(violet_wave), run_time=1.8)
        self.play(Create(red_wave), run_time=1.8)
        self.play(FadeIn(violet_lbl), FadeIn(red_lbl))

        insight = Text(
            "Each wavelength triggers a unique response in our eyes",
            font_size=20, color=ACCENT_TEAL)
        insight.to_edge(DOWN, buff=0.55)
        self.play(FadeIn(insight, shift=UP * 0.2))
        self.wait(5)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 03 — GAMMA AND TRANSFER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════
class GammaTransferScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("II.  Gamma and Transfer Functions",
                  font_size=40, color=ACCENT_ORANGE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        # ── Act 0: hook slide ─────────────────────────────────────────
        hook_title = Text("The Brightness Problem", font_size=34,
                          color=ACCENT_ORANGE, weight=BOLD)
        hook_title.next_to(ch, DOWN, buff=0.55)
        hook_body = VGroup(
            Text("Question: if you store the value 128 in a pixel (out of 255),",
                 font_size=21, color=TEXT_PRI),
            Text("does that look like 50% brightness on screen?",
                 font_size=21, color=TEXT_PRI),
            Text("Answer: No — it looks like only ~21%.",
                 font_size=21, color=ACCENT_ORANGE),
            Text("Understanding why requires a short trip into CRT history.",
                 font_size=21, color=TEXT_SEC),
        ).arrange(DOWN, buff=0.3)
        hook_body.next_to(hook_title, DOWN, buff=0.45)
        self.play(FadeIn(hook_title, shift=UP * 0.2))
        for line in hook_body:
            self.play(FadeIn(line, shift=UP * 0.1), run_time=0.9)
        self.wait(5)
        self.play(FadeOut(hook_title), FadeOut(hook_body), run_time=1.0)

        # ── Act 1: Why gamma exists ───────────────────────────────────
        crt_lbl = Text("Old CRTs (cathode-ray tubes) had a physical quirk:",
                       font_size=24, color=TEXT_PRI)
        crt_lbl.next_to(ch, DOWN, buff=0.5)
        crt_lbl2 = Text("doubling the voltage produced  4× the brightness  (a power-law response)",
                        font_size=20, color=TEXT_SEC)
        crt_lbl2.next_to(crt_lbl, DOWN, buff=0.25)
        self.play(FadeIn(crt_lbl))
        self.play(FadeIn(crt_lbl2))

        gamma_axes = Axes(
            x_range=[0, 1, 0.25], y_range=[0, 1, 0.25],
            x_length=5.5, y_length=4.5, tips=False,
            axis_config={"color": GRID_COL, "stroke_width": 1.5,
                         "include_numbers": True, "font_size": 16})
        gamma_axes.move_to(LEFT * 2.5 + DOWN * 0.8)
        gx_lbl = Text("Signal (V)", font_size=15, color=TEXT_SEC)
        gx_lbl.next_to(gamma_axes, DOWN, buff=0.15)
        gy_lbl = Text("Brightness", font_size=15, color=TEXT_SEC).rotate(PI / 2)
        gy_lbl.next_to(gamma_axes, LEFT, buff=0.15)

        crt_plot = gamma_axes.plot(
            lambda v: v ** 2.2, color=MUTED_RED, stroke_width=3,
            x_range=[0, 1, 0.01])
        ideal_plot = gamma_axes.plot(
            lambda v: v, color=GRID_COL, stroke_width=1.5,
            x_range=[0, 1, 0.01])
        crt_plot_lbl = MathTex("CRT: V^2.2", font_size=16, color=MUTED_RED)
        crt_plot_lbl.move_to(gamma_axes.c2p(0.75, 0.35))
        ideal_lbl = Text("ideal 1:1", font_size=14, color=GRID_COL)
        ideal_lbl.move_to(gamma_axes.c2p(0.7, 0.75))
        # Callout: 50% signal → 21% light
        callout_dot = Dot(gamma_axes.c2p(0.5, 0.5 ** 2.2), color=ACCENT_YELLOW, radius=0.08)
        callout_arr = Arrow(gamma_axes.c2p(0.68, 0.5), gamma_axes.c2p(0.52, 0.5 ** 2.2 + 0.03),
                            color=ACCENT_YELLOW, stroke_width=2, buff=0.05, max_tip_length_to_length_ratio=0.2)
        callout_txt = Text("50% signal → only 21% light!", font_size=14, color=ACCENT_YELLOW)
        callout_txt.move_to(gamma_axes.c2p(0.72, 0.52))

        self.play(Create(gamma_axes), FadeIn(gx_lbl), FadeIn(gy_lbl), run_time=1.4)
        self.play(Create(ideal_plot), FadeIn(ideal_lbl), run_time=1.2)
        self.play(Create(crt_plot), FadeIn(crt_plot_lbl), run_time=1.0)
        self.play(FadeIn(callout_dot), GrowArrow(callout_arr), FadeIn(callout_txt))

        # Right panel: explanation
        explain_lines = VGroup(
            Text("Cameras measure linear light.", font_size=19, color=TEXT_PRI),
            Text("Old CRTs applied γ≈2.2 to voltage.", font_size=19, color=TEXT_PRI),
            Text("Fix: pre-encode image with 1/2.2", font_size=19, color=TEXT_PRI),
            Text("  before sending to the display.", font_size=19, color=TEXT_PRI),
        )
        explain_lines.arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        explain_lines.move_to(RIGHT * 2.8 + DOWN * 0.6)
        for line in explain_lines:
            self.play(FadeIn(line), run_time=1.0)
        self.wait(3.5)

        # ── Act 2: sRGB piecewise EOTF/OETF ─────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.2)
        self.play(FadeIn(ch))

        srgb_title = Text("sRGB Transfer Function  (IEC 61966-2-1)",
                          font_size=26, color=ACCENT_ORANGE)
        srgb_title.next_to(ch, DOWN, buff=0.45)
        self.play(FadeIn(srgb_title))

        eotf_axes = Axes(
            x_range=[0, 1, 0.2], y_range=[0, 1, 0.2],
            x_length=5.5, y_length=4.5, tips=False,
            axis_config={"color": GRID_COL, "stroke_width": 1.5,
                         "include_numbers": True, "font_size": 16})
        eotf_axes.move_to(LEFT * 2.5 + DOWN * 0.8)
        ex_lbl = Text("Encoded (non-linear)", font_size=14, color=TEXT_SEC)
        ex_lbl.next_to(eotf_axes, DOWN, buff=0.12)
        ey_lbl = Text("Linear light", font_size=14, color=TEXT_SEC).rotate(PI / 2)
        ey_lbl.next_to(eotf_axes, LEFT, buff=0.12)

        eotf_plot = eotf_axes.plot(
            lambda x: float(srgb_to_linear(np.array([x, x, x]))[0]),
            color=ACCENT_ORANGE, stroke_width=3.5, x_range=[0, 1, 0.005])
        eotf_lbl = Text("EOTF (decode)", font_size=15, color=ACCENT_ORANGE)
        eotf_lbl.move_to(eotf_axes.c2p(0.65, 0.35))

        # Highlight linear segment
        lin_seg = eotf_axes.plot(
            lambda x: x / 12.92, color=ACCENT_YELLOW, stroke_width=2.5,
            x_range=[0, 0.04045, 0.001])
        lin_seg_ann = VGroup(
            Text("linear", font_size=13, color=ACCENT_YELLOW),
            Text("below 0.04045", font_size=13, color=ACCENT_YELLOW),
        ).arrange(DOWN, buff=0.08)
        lin_seg_ann.move_to(eotf_axes.c2p(0.10, 0.25))

        self.play(Create(eotf_axes), FadeIn(ex_lbl), FadeIn(ey_lbl), run_time=1.4)
        self.play(Create(eotf_plot), FadeIn(eotf_lbl), run_time=1.0)
        self.play(Create(lin_seg), FadeIn(lin_seg_ann), run_time=1.5)

        # 50% gray comparison
        gray_lines = VGroup(
            Text("50% gray  (encoded = 0.5):", font_size=19, color=TEXT_PRI),
            Text("  sRGB decoded  → 0.214  (linear)", font_size=18, color=ACCENT_GREEN),
            Text("  Naïve 0.5      → 0.500  (13× too bright!)", font_size=18, color=MUTED_RED),
        )
        gray_lines.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        gray_lines.move_to(RIGHT * 2.8 + DOWN * 0.5)
        for line in gray_lines:
            self.play(FadeIn(line), run_time=1.2)

        # Visual swatches: naive 0.5 vs correct 0.214
        swatch_naive = Rectangle(width=1.8, height=1.0,
                                 fill_color=rgb_to_hex([0.5, 0.5, 0.5]), fill_opacity=1,
                                 stroke_width=0)
        swatch_correct = Rectangle(width=1.8, height=1.0,
                                   fill_color=rgb_to_hex([0.214, 0.214, 0.214]), fill_opacity=1,
                                   stroke_width=0)
        swatches = VGroup(swatch_naive, swatch_correct).arrange(RIGHT, buff=0.5)
        swatches.move_to(RIGHT * 2.8 + UP * 1.3)
        swatch_naive_lbl = VGroup(
            Text("0.5", font_size=15, color=MUTED_RED),
            Text("(as stored)", font_size=13, color=MUTED_RED),
        ).arrange(DOWN, buff=0.06)
        swatch_naive_lbl.next_to(swatch_naive, DOWN, buff=0.1)
        swatch_correct_lbl = VGroup(
            Text("0.214", font_size=15, color=ACCENT_GREEN),
            Text("(actual light)", font_size=13, color=ACCENT_GREEN),
        ).arrange(DOWN, buff=0.06)
        swatch_correct_lbl.next_to(swatch_correct, DOWN, buff=0.1)
        self.play(FadeIn(swatches), FadeIn(swatch_naive_lbl), FadeIn(swatch_correct_lbl))
        self.wait(4)

        # ── Act 3: Quantization and banding ──────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.2)
        self.play(FadeIn(ch))

        quant_title = Text("Quantization: Why 8-bit gamma is enough",
                           font_size=26, color=ACCENT_YELLOW)
        quant_title.next_to(ch, DOWN, buff=0.45)
        self.play(FadeIn(quant_title))

        lbl_lin = Text("8-bit LINEAR (bands visible)", font_size=18, color=MUTED_RED)
        lbl_lin.move_to(UP * 0.7 + LEFT * 3.0)
        lbl_gam = Text("8-bit GAMMA-ENCODED (smooth)", font_size=18, color=ACCENT_GREEN)
        lbl_gam.move_to(DOWN * 2.05 + LEFT * 3.0)
        self.play(FadeIn(lbl_lin), FadeIn(lbl_gam))

        # Linear 8-bit gradient bar (show posterization by quantizing to 8 bits linearly)
        n_seg = 80
        bar_w = 9.0
        lin_bar = VGroup()
        gam_bar = VGroup()
        sw = bar_w / n_seg
        for i in range(n_seg):
            t = i / (n_seg - 1)
            # Linear: quantize in linear space → obvious bands in dark region
            lin_q = round(t * 255) / 255.0
            lin_col = rgb_to_hex([lin_q, lin_q, lin_q])
            r1 = Rectangle(width=sw + 0.02, height=0.7,
                           fill_color=lin_col, fill_opacity=1, stroke_width=0)
            r1.move_to(LEFT * bar_w / 2 + RIGHT * (i * sw + sw / 2) + UP * 0.1)
            lin_bar.add(r1)
            # Gamma-encoded: quantize in gamma space → perceptually uniform
            gam_encoded = float(linear_to_srgb(np.array([t]))[0])
            gam_col = rgb_to_hex([gam_encoded, gam_encoded, gam_encoded])
            r2 = Rectangle(width=sw + 0.02, height=0.7,
                           fill_color=gam_col, fill_opacity=1, stroke_width=0)
            r2.move_to(LEFT * bar_w / 2 + RIGHT * (i * sw + sw / 2) + DOWN * 1.5)
            gam_bar.add(r2)

        self.play(FadeIn(lin_bar), run_time=1.5)
        self.play(FadeIn(gam_bar), run_time=1.5)

        quant_box = RoundedRectangle(corner_radius=0.12, width=11, height=0.9,
                                     fill_color=PANEL, fill_opacity=0.9,
                                     stroke_color=ACCENT_YELLOW, stroke_width=1.5)
        quant_box.to_edge(DOWN, buff=0.45)
        quant_insight = Text(
            "Gamma ≈ perceptual quantization: equal bit steps ≈ equal JNDs  →  8-bit sRGB is sufficient",
            font_size=18, color=ACCENT_YELLOW)
        quant_insight.move_to(quant_box)
        self.play(FadeIn(quant_box), FadeIn(quant_insight))
        self.wait(7)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 04 — HUMAN VISION: CONE CELLS
# ═══════════════════════════════════════════════════════════════════════
class HumanVisionScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("III.  How We See Color", font_size=44,
                  color=ACCENT_BLUE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        # Hook before eye diagram
        hook_eye = VGroup(
            Text("To understand colour, we first need to understand the sensor — your eye.",
                 font_size=20, color=TEXT_PRI),
            Text("Light enters, hits the back wall (the retina), and triggers one of three detectors.",
                 font_size=20, color=TEXT_PRI),
        ).arrange(DOWN, buff=0.35)
        hook_eye.next_to(ch, DOWN, buff=0.5)
        self.play(FadeIn(hook_eye, lag_ratio=0.3), run_time=1.5)
        self.wait(3)

        # Eye cross-section — labelled anatomy built from Manim primitives
        eye_cx = LEFT * 4 + UP * 0.2  # centre of the eye group

        sclera = Ellipse(width=3.4, height=2.8,
                         color=TEXT_SEC, fill_color="#1a1a2e", fill_opacity=0.6,
                         stroke_width=2)
        sclera.move_to(eye_cx)

        iris = Annulus(inner_radius=0.22, outer_radius=0.52,
                       color="#7b5c2e", fill_color="#7b5c2e", fill_opacity=0.9,
                       stroke_width=0)
        iris.move_to(sclera.get_left() + RIGHT * 0.75)

        pupil = Circle(radius=0.22, color=BG, fill_color=BG, fill_opacity=1,
                       stroke_width=0)
        pupil.move_to(iris)

        lens = Ellipse(width=0.28, height=0.72,
                       color=ACCENT_TEAL, fill_color="#163d4f", fill_opacity=0.7,
                       stroke_width=1.5)
        lens.next_to(iris, RIGHT, buff=0.15)

        retina = Arc(radius=1.25, start_angle=-PI / 3.2, angle=2 * PI / 3.2,
                     color=ACCENT_ORANGE, stroke_width=4)
        retina.move_to(sclera.get_right() + LEFT * 0.22)

        fovea = Dot(radius=0.09, color=ACCENT_YELLOW)
        # Place fovea at the midpoint of the retina arc
        fovea.move_to(retina.get_top())

        optic = Line(ORIGIN, RIGHT * 0.55, color=TEXT_SEC, stroke_width=3)
        optic.next_to(sclera, RIGHT, buff=-0.4)
        optic.shift(DOWN * 0.5)

        rays = VGroup(*[
            Line(eye_cx + LEFT * 2.8 + UP * (0.3 * (i - 1)),
                 sclera.get_left() + RIGHT * 0.12,
                 color=ACCENT_YELLOW, stroke_width=1.5, stroke_opacity=0.6)
            for i in range(3)
        ])

        lbl_pupil = Text("Pupil", font_size=13, color=TEXT_SEC)
        lbl_lens = Text("Lens", font_size=13, color=ACCENT_TEAL)
        lbl_retina = Text("Retina", font_size=13, color=ACCENT_ORANGE)
        lbl_fovea = Text("Fovea", font_size=13, color=ACCENT_YELLOW)
        lbl_optic = Text("Optic nerve", font_size=12, color=TEXT_SEC)
        lbl_pupil.next_to(pupil, DOWN, buff=0.12)
        lbl_lens.next_to(lens, UP, buff=0.12)
        lbl_retina.next_to(retina, RIGHT, buff=0.12)
        lbl_fovea.next_to(fovea, UP, buff=0.12)
        lbl_optic.next_to(optic, RIGHT, buff=0.08)

        eye_group = VGroup(sclera, iris, pupil, lens, retina, fovea, optic, rays,
                           lbl_pupil, lbl_lens, lbl_retina, lbl_fovea, lbl_optic)

        self.play(FadeOut(hook_eye), run_time=0.8)
        self.play(Create(sclera), run_time=1.0)
        self.play(FadeIn(iris), FadeIn(pupil), run_time=0.8)
        self.play(Create(lens), FadeIn(lbl_lens), run_time=0.8)
        self.play(Create(retina), FadeIn(lbl_retina), run_time=0.8)
        self.play(FadeIn(fovea), FadeIn(lbl_fovea), run_time=0.6)
        self.play(Create(optic), FadeIn(lbl_optic), run_time=0.6)
        self.play(FadeIn(lbl_pupil), run_time=0.5)
        self.play(FadeIn(rays), run_time=0.8)

        # Arrow: from eye to cone curves area
        arrow = Arrow(sclera.get_right() + RIGHT * 0.15, RIGHT * 0.2 + UP * 0.5,
                      color=TEXT_SEC, stroke_width=2, buff=0.1)
        self.play(GrowArrow(arrow))

        # Cone sensitivity curves
        axes = Axes(
            x_range=[380, 780, 50], y_range=[0, 1.15, 0.25],
            x_length=7, y_length=3.5, tips=False,
            axis_config={"color": GRID_COL, "stroke_width": 1.5})
        axes.move_to(RIGHT * 2.3 + UP * 0.4)
        x_lbl = Text("λ (nm)", font_size=15, color=TEXT_SEC)
        x_lbl.next_to(axes, DOWN, buff=0.12)
        y_lbl = Text("response", font_size=15, color=TEXT_SEC).rotate(PI / 2)
        y_lbl.next_to(axes, LEFT, buff=0.12)

        def cone(wl, peak, sigma):
            return np.exp(-0.5 * ((wl - peak) / sigma) ** 2)

        s_plot = axes.plot(lambda w: cone(w, 445, 24), color="#5c6bc0",
                           stroke_width=3.5, x_range=[380, 550])
        m_plot = axes.plot(lambda w: cone(w, 543, 38), color="#66bb6a",
                           stroke_width=3.5, x_range=[420, 700])
        l_plot = axes.plot(lambda w: cone(w, 570, 48), color="#ef5350",
                           stroke_width=3.5, x_range=[440, 780])

        s_area = axes.get_area(s_plot, x_range=[380, 550],
                               color="#5c6bc0", opacity=0.12)
        m_area = axes.get_area(m_plot, x_range=[420, 700],
                               color="#66bb6a", opacity=0.10)
        l_area = axes.get_area(l_plot, x_range=[440, 780],
                               color="#ef5350", opacity=0.10)

        s_lbl = Text("S", font_size=20, color="#5c6bc0", weight=BOLD)
        s_lbl.move_to(axes.c2p(435, 1.08))
        m_lbl = Text("M", font_size=20, color="#66bb6a", weight=BOLD)
        m_lbl.move_to(axes.c2p(535, 1.08))
        l_lbl = Text("L", font_size=20, color="#ef5350", weight=BOLD)
        l_lbl.move_to(axes.c2p(600, 1.08))

        self.play(Create(axes), FadeIn(x_lbl), FadeIn(y_lbl), run_time=1.5)
        self.play(Create(s_plot), FadeIn(s_area), FadeIn(s_lbl), run_time=0.9)
        self.play(Create(m_plot), FadeIn(m_area), FadeIn(m_lbl), run_time=0.9)
        self.play(Create(l_plot), FadeIn(l_area), FadeIn(l_lbl), run_time=0.9)

        bridging = Text("These 3 curves explain why a camera sensor has R·G·B channels.",
                        font_size=19, color=TEXT_SEC)
        bridging.next_to(axes, DOWN, buff=0.3)
        self.play(FadeIn(bridging, shift=UP * 0.1))
        self.wait(2)

        insight = Text(
            "3 cone types → every color we see is described by just 3 numbers  (trichromacy)",
            font_size=20, color=ACCENT_YELLOW)
        box = RoundedRectangle(corner_radius=0.12, width=insight.width + 0.1, height=0.9,
                               fill_color=PANEL, fill_opacity=0.9,
                               stroke_color=ACCENT_YELLOW, stroke_width=1.5)

        box.to_edge(DOWN, buff=0.45)
        insight.move_to(box)
        self.play(FadeIn(box), FadeIn(insight))
        self.wait(6)

        # ── Act 2: Opponent Process Theory ───────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.4)
        self.play(FadeIn(ch))

        # Bridge before Opponent Process
        bridge = VGroup(
            Text("But wait — if you have 3 sensors,",
                 font_size=21, color=TEXT_PRI),
            Text("why do we talk about red/green and blue/yellow?",
                 font_size=21, color=TEXT_PRI),
            Text("Your brain rewires the cone signals into opponent channels.",
                 font_size=21, color=TEXT_PRI),
        ).arrange(DOWN, buff=0.35)
        bridge.move_to(ORIGIN)
        self.play(FadeIn(bridge, lag_ratio=0.3), run_time=1.5)
        self.wait(3.5)
        self.play(FadeOut(bridge), run_time=1.0)

        op_title = Text("Opponent Process Theory  (Hering 1892)",
                        font_size=38, color=ACCENT_PURPLE, weight=BOLD)
        op_title.to_edge(UP, buff=0.45)
        self.play(FadeOut(ch), FadeIn(op_title, shift=DOWN * 0.2))

        wls = np.linspace(380, 780, 200)

        def s_cone(w): return np.exp(-0.5 * ((w - 445) / 24) ** 2)

        def m_cone(w): return np.exp(-0.5 * ((w - 543) / 38) ** 2)

        def l_cone(w): return np.exp(-0.5 * ((w - 570) / 48) ** 2)

        op_axes = Axes(
            x_range=[380, 780, 50], y_range=[-0.6, 1.1, 0.4],
            x_length=10, y_length=5.0, tips=False,
            axis_config={"color": GRID_COL, "stroke_width": 1.5})
        op_axes.move_to(ORIGIN + DOWN * 0.3)
        zero_line = op_axes.plot(lambda w: 0, color=GRID_COL,
                                 stroke_width=1.0, x_range=[380, 780])
        op_x_lbl = Text("λ (nm)", font_size=15, color=TEXT_SEC)
        op_x_lbl.next_to(op_axes, DOWN, buff=0.12)
        self.play(Create(op_axes), Create(zero_line), FadeIn(op_x_lbl), run_time=1.4)

        # Red-green channel: L − M
        rg_plot = op_axes.plot(
            lambda w: l_cone(w) - m_cone(w),
            color=MUTED_RED, stroke_width=3.5, x_range=[380, 780])
        rg_lbl = Text("L − M  (red / green)", font_size=18, color=MUTED_RED, weight=BOLD)
        rg_lbl.move_to(op_axes.c2p(650, 0.6))
        rg_lbl.shift(RIGHT * 1.0)

        # Blue-yellow channel: (L + M) − S
        by_plot = op_axes.plot(
            lambda w: (l_cone(w) + m_cone(w)) * 0.5 - s_cone(w) * 0.9,
            color=ACCENT_YELLOW, stroke_width=3.5, x_range=[380, 780])
        by_lbl = Text("(L+M) − S  (blue / yellow)", font_size=18,
                      color=ACCENT_YELLOW, weight=BOLD)
        by_lbl.move_to(op_axes.c2p(550, -0.45))
        by_lbl.shift(RIGHT * 0.8)

        self.play(Create(rg_plot), FadeIn(rg_lbl), run_time=1.0)
        self.play(Create(by_plot), FadeIn(by_lbl), run_time=1.0)

        # Annotate zero crossing of L-M ≈ unique yellow
        yellow_x = 6 / 215 * (17833 + 76 * np.sqrt(729))
        yellow_dot = Dot(op_axes.c2p(yellow_x, 0), color=ACCENT_YELLOW, radius=0.10)
        yellow_ann = VGroup(
            Text("Unique yellow", font_size=15, color=ACCENT_YELLOW),
            Text("(L−M = 0)", font_size=15, color=ACCENT_YELLOW),
        ).arrange(DOWN, buff=0.1)
        yellow_ann.next_to(yellow_dot, UP, buff=0.15)
        yellow_ann.shift(RIGHT * 0.5)
        self.play(FadeIn(yellow_dot), FadeIn(yellow_ann))

        op_insight = Text(
            "a* ≈ red/green axis · b* ≈ blue/yellow axis  →  CIELAB and OKLab reflect opponent biology",
            font_size=19, color=ACCENT_PURPLE)
        op_box = RoundedRectangle(corner_radius=0.12, width=op_insight.width + 0.1, height=1.0,
                                  fill_color=PANEL, fill_opacity=0.9,
                                  stroke_color=ACCENT_PURPLE, stroke_width=1.5)
        op_box.to_edge(DOWN, buff=0.4)
        op_insight.move_to(op_box)
        self.play(FadeIn(op_box), FadeIn(op_insight))
        self.wait(7)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 04 — CIE 1931: COLOR MATCHING FUNCTIONS & XYZ
# ═══════════════════════════════════════════════════════════════════════
class CIE1931Scene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("IV.  The CIE 1931 Standard", font_size=44,
                  color=ACCENT_PURPLE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        # ── Act 0: Animated XYZ hook — "what does colour math do?" ──────
        def _green_spd(w):
            return 0.8 * np.exp(-0.5 * ((w - 540) / 40) ** 2)
        X_g, Y_g, Z_g = spd_to_xyz(_green_spd, illuminant_d65_spd)
        green_srgb = np.clip(xyz_to_srgb(X_g, Y_g, Z_g), 0, 1)

        patch = Rectangle(width=2.5, height=1.5,
                           fill_color=rgb_to_hex(green_srgb), fill_opacity=1.0,
                           stroke_width=0)
        patch.move_to(LEFT * 3.0 + UP * 1.8)

        patch_lbl = Text("Green (540 nm)", font_size=14, color=TEXT_SEC)
        patch_lbl.next_to(patch, DOWN, buff=0.12)

        q_mark = Text("?", font_size=64, color=TEXT_PRI, weight=BOLD)
        q_cap = Text("What numbers describe this?", font_size=18, color=TEXT_SEC)
        q_grp = VGroup(q_mark, q_cap).arrange(DOWN, buff=0.22)
        q_grp.move_to(RIGHT * 2.8 + UP * 1.8)

        self.play(FadeIn(patch), FadeIn(patch_lbl), run_time=1.0)
        self.play(FadeIn(q_grp, shift=UP * 0.2), run_time=1.0)
        self.wait(1.5)

        # Three bars growing — tops at bar_top_y so they don't overlap patch
        max_xyz = max(X_g, Y_g, Z_g, 1e-9)
        bar_scale = 1.5 / max_xyz
        bar_w = 0.52
        bar_spacing = 0.72
        bar_top_y = patch.get_bottom()[1] - 0.5
        bars_data_xyz = [
            ("X", X_g, MUTED_RED, -bar_spacing),
            ("Y", Y_g, MUTED_GREEN, 0.0),
            ("Z", Z_g, MUTED_BLUE, bar_spacing),
        ]
        hook_bars = VGroup()
        hook_lbls = VGroup()
        for (label, val, col, x_off) in bars_data_xyz:
            h = max(val * bar_scale, 0.1)
            b = Rectangle(width=bar_w, height=h,
                           fill_color=col, fill_opacity=0.85, stroke_width=0)
            b.move_to([patch.get_center()[0] + x_off, bar_top_y - h / 2, 0])
            lbl = Text(label, font_size=18, color=col, weight=BOLD)
            lbl.next_to(b, DOWN, buff=0.1)
            hook_bars.add(b)
            hook_lbls.add(lbl)

        self.play(
            GrowFromEdge(hook_bars[0], DOWN),
            GrowFromEdge(hook_bars[1], DOWN),
            GrowFromEdge(hook_bars[2], DOWN),
            FadeIn(hook_lbls), run_time=1.5)

        answer = Text("X, Y, Z", font_size=48, color=ACCENT_PURPLE, weight=BOLD)
        answer.move_to(q_mark.get_center())
        answer_cap = Text("CIE 1931 — computed from any light spectrum",
                          font_size=17, color=TEXT_SEC)
        answer_cap.move_to(q_cap.get_center())
        self.play(Transform(q_mark, answer), Transform(q_cap, answer_cap), run_time=1.0)
        self.wait(3.5)
        self.play(FadeOut(VGroup(patch, patch_lbl, q_mark, q_cap, hook_bars, hook_lbls)),
                  run_time=1.2)

        # ── Act 1: Animated matching-experiment ───────────────────────
        exp_panel = RoundedRectangle(corner_radius=0.15, width=11.0, height=2.2,
                                     fill_color=PANEL, fill_opacity=0.9,
                                     stroke_color=ACCENT_PURPLE, stroke_width=1.5)
        exp_panel.move_to(UP * 0.5)

        # Left half: test light (540 nm green)
        test_rect = Rectangle(width=3.6, height=1.5,
                               fill_color=rgb_to_hex(wavelength_to_rgb(540)),
                               fill_opacity=1.0, stroke_width=0)
        test_rect.move_to(exp_panel.get_left() + RIGHT * 2.1)
        test_lbl2 = Text("Test  λ = 540 nm", font_size=13, color=TEXT_PRI)
        test_lbl2.next_to(test_rect, UP, buff=0.1)

        # Dividing wall
        div_wall = Line(
            exp_panel.get_top() + DOWN * 0.15 + LEFT * 0.5,
            exp_panel.get_bottom() + UP * 0.15 + LEFT * 0.5,
            color=TEXT_SEC, stroke_width=1.5)

        # Observer (head + body)
        obs_head = Circle(radius=0.22, fill_color=TEXT_PRI, fill_opacity=0.85, stroke_width=0)
        obs_body = Line(DOWN * 0.0, DOWN * 0.55, stroke_width=2.5, color=TEXT_PRI)
        obs_body.next_to(obs_head, DOWN, buff=0.0)
        obs = VGroup(obs_head, obs_body)
        obs.move_to(exp_panel.get_center() + LEFT * 0.5 + DOWN * 0.1)

        # Right half: match light — starts white
        match_rect = Rectangle(width=3.6, height=1.5,
                                fill_color="#ffffff", fill_opacity=1.0, stroke_width=0)
        match_rect.move_to(exp_panel.get_right() + LEFT * 2.1)

        # R, G, B slider rows
        slider_track_w = 2.0
        slider_x0 = match_rect.get_left()[0] + 0.3
        slider_rows_y = match_rect.get_top()[1] - 0.2
        sliders = VGroup()
        slider_data = [
            ("R", MUTED_RED,    0.35, 0.30),
            ("G", MUTED_GREEN,  0.35, 0.75),
            ("B", MUTED_BLUE,   0.35, 0.28),
        ]
        knob_dots = []
        for i, (s_lbl, s_col, s_start, s_end) in enumerate(slider_data):
            track_y = slider_rows_y - i * 0.5
            track = Line([slider_x0, track_y, 0],
                         [slider_x0 + slider_track_w, track_y, 0],
                         color=s_col, stroke_width=2.0, stroke_opacity=0.45)
            dot = Dot([slider_x0 + s_start * slider_track_w, track_y, 0],
                      radius=0.09, color=s_col)
            lbl_s = Text(s_lbl, font_size=11, color=s_col)
            lbl_s.next_to(track, LEFT, buff=0.08)
            sliders.add(VGroup(track, dot, lbl_s))
            knob_dots.append((dot, slider_x0 + s_end * slider_track_w, track_y))

        self.play(Create(exp_panel), run_time=1.0)
        self.play(FadeIn(test_rect), FadeIn(test_lbl2), FadeIn(div_wall), FadeIn(obs),
                  run_time=0.8)
        self.play(FadeIn(match_rect), FadeIn(sliders), run_time=0.7)

        # Animate sliders moving + match rect filling
        self.play(
            *[dot.animate.move_to([ex, ey, 0]) for (dot, ex, ey) in knob_dots],
            match_rect.animate.set_fill(color=rgb_to_hex(wavelength_to_rgb(540))),
            run_time=1.8)

        check = Text("✓", font_size=36, color=ACCENT_GREEN)
        check.move_to(exp_panel.get_center() + LEFT * 0.5 + UP * 0.5)
        self.play(FadeIn(check, scale=1.4), run_time=0.6)

        # Two-line repeat label
        repeat_line1 = Text("Repeat for every wavelength 380–780 nm → get matching curves",
                            font_size=15, color=TEXT_SEC)
        repeat_line2 = Text("These are called r̄(λ) ḡ(λ) b̄(λ) — the Color Matching Functions (CMFs)",
                            font_size=15, color=TEXT_SEC)
        repeat_lbl = VGroup(repeat_line1, repeat_line2).arrange(DOWN, buff=0.12)
        repeat_lbl.next_to(exp_panel, DOWN, buff=0.25)
        self.play(FadeIn(repeat_lbl, shift=UP * 0.1))
        self.wait(2.5)

        self.play(*[FadeOut(m) for m in self.mobjects if m is not ch], run_time=1.2)

        cmf_title = Text("Color Matching Functions  x̄(λ), ȳ(λ), z̄(λ)", font_size=24,
                         color=TEXT_PRI)
        cmf_title.next_to(ch, DOWN, buff=0.4)
        self.play(FadeIn(cmf_title))

        axes = Axes(
            x_range=[380, 780, 50], y_range=[-0.1, 1.85, 0.5],
            x_length=10, y_length=3.8, tips=False,
            axis_config={"color": GRID_COL, "stroke_width": 1.5})
        axes.move_to(DOWN * 0.6)

        # Axis labels for CMF axes
        cmf_x_lbl = Text("Wavelength  λ (nm)", font_size=15, color=TEXT_SEC)
        cmf_x_lbl.next_to(axes, DOWN, buff=0.2)
        cmf_y_lbl = Text("Response", font_size=15, color=TEXT_SEC)
        cmf_y_lbl.rotate(PI / 2)
        cmf_y_lbl.next_to(axes, LEFT, buff=0.25)

        # X-tick labels at key wavelengths
        cmf_ticks = VGroup()
        for wl_tick in [400, 500, 600, 700]:
            t = Text(str(wl_tick), font_size=12, color=TEXT_SEC)
            t.move_to(axes.c2p(wl_tick, -0.08) + DOWN * 0.18)
            cmf_ticks.add(t)

        x_plot = axes.plot(xbar, color=MUTED_RED, stroke_width=3,
                           x_range=[380, 780, 2])
        y_plot = axes.plot(ybar, color=MUTED_GREEN, stroke_width=3,
                           x_range=[380, 780, 2])
        z_plot = axes.plot(zbar, color=MUTED_BLUE, stroke_width=3,
                           x_range=[380, 780, 2])

        xl = MathTex(r"\bar{x}", font_size=26, color=MUTED_RED)
        xl.move_to(axes.c2p(620, 1.1))
        yl = MathTex(r"\bar{y}", font_size=26, color=MUTED_GREEN)
        yl.move_to(axes.c2p(555, 1.15))
        zl = MathTex(r"\bar{z}", font_size=26, color=MUTED_BLUE)
        zl.move_to(axes.c2p(445, 1.55))

        self.play(Create(axes, run_time=1.4))
        self.play(FadeIn(cmf_x_lbl), FadeIn(cmf_y_lbl), FadeIn(cmf_ticks))
        self.play(Create(x_plot, run_time=1.2), FadeIn(xl))
        self.play(Create(y_plot, run_time=1.2), FadeIn(yl))
        self.play(Create(z_plot, run_time=1.2), FadeIn(zl))

        # Sweep animation: dashed line + tracking dots move across CMF curves
        wl_tracker = ValueTracker(380)
        sweep_line = always_redraw(lambda: DashedLine(
            axes.c2p(wl_tracker.get_value(), -0.05),
            axes.c2p(wl_tracker.get_value(), 1.82),
            color=ACCENT_YELLOW, stroke_width=2.0, dash_length=0.10))
        xd = always_redraw(lambda: Dot(
            axes.c2p(wl_tracker.get_value(), xbar(wl_tracker.get_value())),
            radius=0.07, color=MUTED_RED))
        yd = always_redraw(lambda: Dot(
            axes.c2p(wl_tracker.get_value(), ybar(wl_tracker.get_value())),
            radius=0.07, color=MUTED_GREEN))
        zd = always_redraw(lambda: Dot(
            axes.c2p(wl_tracker.get_value(), zbar(wl_tracker.get_value())),
            radius=0.07, color=MUTED_BLUE))

        self.add(sweep_line, xd, yd, zd)
        self.play(wl_tracker.animate.set_value(780), run_time=4.0, rate_func=linear)
        sweep_line.clear_updaters(); xd.clear_updaters()
        yd.clear_updaters(); zd.clear_updaters()
        self.play(FadeOut(VGroup(sweep_line, xd, yd, zd)), run_time=0.6)

        # Filled areas under CMF curves — show integration visually
        x_fill = axes.get_area(x_plot, x_range=[380, 780], color=MUTED_RED, opacity=0.0)
        y_fill = axes.get_area(y_plot, x_range=[380, 780], color=MUTED_GREEN, opacity=0.0)
        z_fill = axes.get_area(z_plot, x_range=[380, 780], color=MUTED_BLUE, opacity=0.0)
        self.add(x_fill, y_fill, z_fill)
        self.play(x_fill.animate.set_fill(opacity=0.22),
                  y_fill.animate.set_fill(opacity=0.22),
                  z_fill.animate.set_fill(opacity=0.22), run_time=1.2)

        # Brief callout before integral formula
        area_callout_txt = Text("These shaded areas = the X, Y, Z values",
                                font_size=18, color=ACCENT_YELLOW)
        area_callout_box = RoundedRectangle(corner_radius=0.10,
            width=area_callout_txt.width + 0.5, height=area_callout_txt.height + 0.4,
            fill_color=PANEL, fill_opacity=0.9,
            stroke_color=ACCENT_YELLOW, stroke_width=1.5)
        area_callout_box.to_edge(DOWN, buff=0.45)
        area_callout_txt.move_to(area_callout_box)
        self.play(FadeIn(area_callout_box), FadeIn(area_callout_txt))
        self.wait(2.0)
        self.play(FadeOut(area_callout_box), FadeOut(area_callout_txt))

        formula = MathTex(
            r"X = \int S(\lambda)\,\bar{x}(\lambda)\,d\lambda",
            r"\quad Y = \int S(\lambda)\,\bar{y}(\lambda)\,d\lambda",
            r"\quad Z = \int S(\lambda)\,\bar{z}(\lambda)\,d\lambda",
            font_size=22, color=TEXT_PRI)
        formula_box = RoundedRectangle(corner_radius=0.12,
            width=formula.width + 0.6, height=formula.height + 0.4,
            fill_color=PANEL, fill_opacity=0.9,
            stroke_color=ACCENT_PURPLE, stroke_width=1.5)
        formula_box.to_edge(DOWN, buff=0.45)
        formula.move_to(formula_box)
        self.play(FadeIn(formula_box), Write(formula, run_time=1.5))
        self.wait(4)

        # ── Act 2: r̄ḡb̄ negative lobes → why XYZ was needed ──────────
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.4)

        rgb_ch = Text("The Original CIE RGB Experiment: r̄ḡb̄", font_size=34,
                      color=ACCENT_ORANGE, weight=BOLD)
        rgb_ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(rgb_ch, shift=DOWN * 0.2))

        # Brief intro explaining the problem
        intro_problem = Text("But the CIE RGB experiment found a problem...",
                             font_size=22, color=ACCENT_ORANGE)
        intro_problem.next_to(rgb_ch, DOWN, buff=0.45)
        self.play(FadeIn(intro_problem, shift=UP * 0.1))
        self.wait(2.0)
        self.play(FadeOut(intro_problem))

        rgb_axes = Axes(
            x_range=[380, 780, 50], y_range=[-0.3, 1.1, 0.4],
            x_length=10, y_length=3.8, tips=False,
            axis_config={"color": GRID_COL, "stroke_width": 1.5})
        rgb_axes.move_to(DOWN * 0.6)
        rgb_zero = rgb_axes.plot(lambda w: 0, color=GRID_COL,
                                 stroke_width=1.0, x_range=[380, 780])

        # Axis labels for r̄ḡb̄ axes
        rgb_x_lbl = Text("λ (nm)", font_size=15, color=TEXT_SEC)
        rgb_x_lbl.next_to(rgb_axes, DOWN, buff=0.2)
        rgb_y_lbl = Text("Response", font_size=15, color=TEXT_SEC)
        rgb_y_lbl.rotate(PI / 2)
        rgb_y_lbl.next_to(rgb_axes, LEFT, buff=0.25)

        self.play(Create(rgb_axes), Create(rgb_zero),
                  FadeIn(rgb_x_lbl), FadeIn(rgb_y_lbl), run_time=1.4)

        r_plot = rgb_axes.plot(rbar, color=MUTED_RED, stroke_width=3.5,
                               x_range=[380, 780, 2])
        g_plot = rgb_axes.plot(gbar, color=MUTED_GREEN, stroke_width=3.5,
                               x_range=[380, 780, 2])
        b_plot = rgb_axes.plot(bbar, color=MUTED_BLUE, stroke_width=3.5,
                               x_range=[380, 780, 2])

        rl = MathTex(r"\bar{r}", font_size=24, color=MUTED_RED)
        rl.move_to(rgb_axes.c2p(640, 0.78))
        gl = MathTex(r"\bar{g}", font_size=24, color=MUTED_GREEN)
        gl.move_to(rgb_axes.c2p(555, 0.90))
        bl = MathTex(r"\bar{b}", font_size=24, color=MUTED_BLUE)
        bl.move_to(rgb_axes.c2p(455, 0.95))

        self.play(Create(r_plot), FadeIn(rl), run_time=1.0)
        self.play(Create(g_plot), FadeIn(gl), run_time=1.0)
        self.play(Create(b_plot), FadeIn(bl), run_time=1.0)

        # Highlight negative region of r̄ — animated fill area
        neg_area = rgb_axes.get_area(r_plot, x_range=[455, 532],
                                      color=MUTED_RED, opacity=0.0)
        self.add(neg_area)
        self.play(neg_area.animate.set_fill(opacity=0.28), run_time=1.5)

        # Annotation: "r̄ < 0 here"
        neg_ann_simple = Text("r̄ < 0 here", font_size=15, color=MUTED_RED)
        neg_ann_simple.move_to(rgb_axes.c2p(495, -0.18))
        self.play(FadeIn(neg_ann_simple, shift=UP * 0.1))

        # Explanation panel for why r̄ goes negative
        neg_exp_content = VGroup(
            Text("Normally:  match = R·red + G·green + B·blue", font_size=13, color=TEXT_PRI),
            Text("At ~500 nm:  you had to add red to the TEST side", font_size=13, color=ACCENT_ORANGE),
            Text("test + R·red = G·green + B·blue", font_size=13, color=TEXT_SEC),
            Text("∴  test = –R·red + G·green + B·blue   (r̄ is negative)", font_size=13, color=MUTED_RED),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        neg_exp_panel = RoundedRectangle(corner_radius=0.12,
            width=neg_exp_content.width + 0.5, height=neg_exp_content.height + 0.4,
            fill_color=PANEL, fill_opacity=0.92,
            stroke_color=MUTED_RED, stroke_width=1.5)
        neg_exp_panel.to_edge(DOWN, buff=0.3)
        neg_exp_content.move_to(neg_exp_panel)
        self.play(FadeIn(neg_exp_panel), FadeIn(neg_exp_content))
        self.wait(4.0)
        self.play(FadeOut(neg_exp_panel), FadeOut(neg_exp_content), FadeOut(neg_ann_simple))

        # XYZ vs RGB explanation panel
        xyz_vs_rgb_content = VGroup(
            Text("CIE RGB (r̄ḡb̄): real primaries, but values go negative", font_size=16, color=TEXT_PRI),
            Text("CIE XYZ (x̄ȳz̄): imaginary primaries, always ≥ 0", font_size=16, color=TEXT_PRI),
            Text("The CIE applied a 3×3 matrix to rotate r̄ḡb̄ → x̄ȳz̄", font_size=16, color=ACCENT_ORANGE),
        ).arrange(DOWN, buff=0.14, aligned_edge=LEFT)
        xyz_vs_rgb_panel = RoundedRectangle(corner_radius=0.12,
            width=xyz_vs_rgb_content.width + 0.5, height=xyz_vs_rgb_content.height + 0.4,
            fill_color=PANEL, fill_opacity=0.92,
            stroke_color=ACCENT_ORANGE, stroke_width=1.5)
        xyz_vs_rgb_panel.to_edge(DOWN, buff=0.35)
        xyz_vs_rgb_content.move_to(xyz_vs_rgb_panel)
        self.play(FadeIn(xyz_vs_rgb_panel), FadeIn(xyz_vs_rgb_content))
        self.wait(4.0)
        self.play(FadeOut(xyz_vs_rgb_panel), FadeOut(xyz_vs_rgb_content))

        # Expanded transformation callout — 2 lines
        transform_line1 = Text(
            "A 3×3 matrix rotates r̄ḡb̄ → x̄ȳz̄.  XYZ 'primaries' are outside the visible gamut",
            font_size=16, color=ACCENT_ORANGE)
        transform_line2 = Text(
            "— they don't exist as real lights — but that makes the math clean: X, Y, Z ≥ 0 always.",
            font_size=16, color=TEXT_PRI)
        transform_txt = VGroup(transform_line1, transform_line2).arrange(DOWN, buff=0.12)
        transform_box = RoundedRectangle(corner_radius=0.10,
            width=transform_txt.width + 0.5, height=transform_txt.height + 0.4,
            fill_color=PANEL, fill_opacity=0.9,
            stroke_color=ACCENT_ORANGE, stroke_width=1.5)
        transform_box.to_edge(DOWN, buff=0.45)
        transform_txt.move_to(transform_box)
        self.play(FadeIn(transform_box), FadeIn(transform_txt))
        self.wait(6)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 06 — SPECTRAL RENDERING
# ═══════════════════════════════════════════════════════════════════════
class SpectralRenderingScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("V.  Spectral Rendering and Physically Based Color",
                  font_size=36, color=ACCENT_TEAL, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        # ── Act 0: Animated RGB-vs-spectral visual hook ───────────────
        # Left side: RGB renderer (flat colour, instant)
        rgb_side_lbl = Text("RGB renderer", font_size=18, color=TEXT_SEC)
        rgb_side_lbl.move_to(LEFT * 3.5 + UP * 2.2)
        rgb_surface = Rectangle(width=3.0, height=1.5,
                                 fill_color=rgb_to_hex([0.85, 0.42, 0.10]), fill_opacity=1.0,
                                 stroke_width=0)
        rgb_surface.move_to(LEFT * 3.5 + UP * 0.7)

        # Normalize bars to max height 2.0
        hook_bar_raw = [2.0, 1.1, 0.28]
        hook_bar_max = max(hook_bar_raw)
        hook_bar_scale = 2.0 / hook_bar_max
        hook_bar_w = 0.32
        hook_bar_data = [(MUTED_RED, hook_bar_raw[0] * hook_bar_scale),
                         (MUTED_GREEN, hook_bar_raw[1] * hook_bar_scale),
                         (MUTED_BLUE, hook_bar_raw[2] * hook_bar_scale)]
        hook_rgb_bars = VGroup()
        hook_rgb_lbls = VGroup()
        bar_top_ref = rgb_surface.get_bottom()[1] - 0.28
        for i, (col, bh) in enumerate(hook_bar_data):
            xb = rgb_surface.get_center()[0] + (i - 1) * 0.6
            b = Rectangle(width=hook_bar_w, height=bh,
                           fill_color=col, fill_opacity=0.85, stroke_width=0)
            b.move_to([xb, bar_top_ref - bh / 2, 0])
            hook_rgb_bars.add(b)
        for col, label in [(MUTED_RED, "R"), (MUTED_GREEN, "G"), (MUTED_BLUE, "B")]:
            t = Text(label, font_size=12, color=col)
            hook_rgb_lbls.add(t)
        for bar, lbl in zip(hook_rgb_bars, hook_rgb_lbls):
            lbl.next_to(bar, DOWN, buff=0.06)

        rgb_done_lbl = Text("Three numbers. Done.", font_size=13, color=TEXT_SEC)
        rgb_done_lbl.next_to(hook_rgb_bars, DOWN, buff=0.18)

        # Right side: spectral renderer (curve → integrate → swatch)
        spec_side_lbl = Text("Spectral renderer", font_size=18, color=ACCENT_TEAL)
        spec_side_lbl.move_to(RIGHT * 3.5 + UP * 2.2)
        hook_axes = Axes(
            x_range=[380, 780, 200], y_range=[0, 1.0, 0.5],
            x_length=3.5, y_length=1.5, tips=False,
            axis_config={"color": GRID_COL, "stroke_width": 1.0,
                         "include_ticks": False, "include_tip": False})
        hook_axes.move_to(RIGHT * 3.5 + UP * 0.7)

        # Axis labels for hook axes
        hook_x_lbl = Text("λ (nm)", font_size=12, color=TEXT_SEC)
        hook_x_lbl.next_to(hook_axes, DOWN, buff=0.12)
        hook_y_lbl = Text("Intensity", font_size=12, color=TEXT_SEC)
        hook_y_lbl.rotate(PI / 2)
        hook_y_lbl.next_to(hook_axes, LEFT, buff=0.15)

        def _orange_refl(w):
            return 0.9 * np.exp(-0.5 * ((w - 600) / 55) ** 2)
        hook_refl_plot = hook_axes.plot(_orange_refl, color=ACCENT_ORANGE,
                                         stroke_width=2.5, x_range=[380, 780, 5])
        hook_refl_area = hook_axes.get_area(hook_refl_plot, x_range=[380, 780],
                                             color=ACCENT_ORANGE, opacity=0.0)

        _Xo, _Yo, _Zo = spd_to_xyz(_orange_refl, illuminant_d65_spd)
        spec_swatch_col = rgb_to_hex(np.clip(xyz_to_srgb(_Xo, _Yo, _Zo), 0, 1))
        spec_xyz_lbl = VGroup(
            Text(f"X={_Xo:.2f}", font_size=11, color=MUTED_RED),
            Text(f"Y={_Yo:.2f}", font_size=11, color=MUTED_GREEN),
            Text(f"Z={_Zo:.2f}", font_size=11, color=MUTED_BLUE),
        ).arrange(RIGHT, buff=0.2)
        spec_xyz_lbl.next_to(hook_axes, DOWN, buff=0.12)

        spec_swatch = Rectangle(width=0.9, height=0.9,
                                 fill_color=spec_swatch_col, fill_opacity=1.0,
                                 stroke_width=1.5, stroke_color=TEXT_SEC)
        spec_swatch.next_to(spec_xyz_lbl, RIGHT, buff=0.25)

        # Two-line hook caption
        hook_cap_line1 = Text("RGB renderer: stores 3 numbers per pixel — fast but approximate",
                               font_size=15, color=TEXT_SEC)
        hook_cap_line2 = Text("Spectral renderer: stores a full curve — accurate but slower",
                               font_size=15, color=ACCENT_TEAL)
        hook_caption = VGroup(hook_cap_line1, hook_cap_line2).arrange(DOWN, buff=0.12)
        hook_caption.move_to(DOWN * 2.8)

        # Vertical divider
        v_div = DashedLine(UP * 2.5, DOWN * 2.0, color=GRID_COL, stroke_width=1.5,
                            dash_length=0.15)

        self.play(FadeIn(rgb_side_lbl), FadeIn(spec_side_lbl), FadeIn(v_div), run_time=0.8)
        self.play(FadeIn(rgb_surface), run_time=0.6)
        self.play(GrowFromEdge(hook_rgb_bars[0], DOWN),
                  GrowFromEdge(hook_rgb_bars[1], DOWN),
                  GrowFromEdge(hook_rgb_bars[2], DOWN),
                  FadeIn(hook_rgb_lbls), run_time=0.9)
        self.play(FadeIn(rgb_done_lbl), run_time=0.5)

        self.play(Create(hook_axes), FadeIn(hook_x_lbl), FadeIn(hook_y_lbl), run_time=0.8)
        self.play(Create(hook_refl_plot), run_time=1.8)
        self.play(hook_refl_area.animate.set_fill(opacity=0.28), run_time=1.0)
        self.play(FadeIn(spec_xyz_lbl, lag_ratio=0.4), run_time=1.0)
        self.play(FadeIn(spec_swatch, scale=1.3), run_time=0.7)
        self.play(FadeIn(hook_caption, shift=UP * 0.15), run_time=0.8)
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not ch], run_time=1.2)

        # ── Act 1: RGB vs spectral rendering pipeline ─────────────────

        # RGB pipeline — self-contained VGroup
        rgb_block_data = [
            ("Light", "(RGB)",       ACCENT_YELLOW),
            ("×  Material", "(RGB albedo)", MUTED_GREEN),
            ("=  Pixel", "(RGB)",    ACCENT_ORANGE),
        ]
        rgb_row = VGroup()
        for main_s, sub_s, col in rgb_block_data:
            b = RoundedRectangle(corner_radius=0.12, width=2.4, height=1.1,
                                 fill_color=PANEL, fill_opacity=0.9,
                                 stroke_color=col, stroke_width=1.8)
            lbl = VGroup(
                Text(main_s, font_size=15, color=col),
                Text(sub_s,  font_size=12, color=col),
            ).arrange(DOWN, buff=0.06)
            lbl.move_to(b)
            rgb_row.add(VGroup(b, lbl))
        rgb_row.arrange(RIGHT, buff=0.25)

        rgb_label = Text("RGB Pipeline:", font_size=17, color=TEXT_SEC)
        rgb_section = VGroup(rgb_label, rgb_row).arrange(DOWN, buff=0.18)
        rgb_section.next_to(ch, DOWN, buff=0.5)
        self.play(FadeIn(rgb_label), FadeIn(rgb_row, lag_ratio=0.3), run_time=1.8)

        rgb_pro = Text("✗  Can't model dispersion, fluorescence, thin-film",
                       font_size=16, color=MUTED_RED)
        rgb_pro.next_to(rgb_section, DOWN, buff=0.2)
        self.play(FadeIn(rgb_pro))

        # Spectral pipeline — self-contained VGroup
        spec_block_data = [
            ("Light SPD", "(λ)",           ACCENT_YELLOW),
            ("× Reflectance", "SPD(λ)",    ACCENT_TEAL),
            ("∫  CMF", "→ XYZ",            ACCENT_PURPLE),
            ("→  RGB", "",                 ACCENT_ORANGE),
        ]
        spec_row = VGroup()
        for main_s, sub_s, col in spec_block_data:
            b = RoundedRectangle(corner_radius=0.12, width=1.95, height=1.1,
                                 fill_color=PANEL, fill_opacity=0.9,
                                 stroke_color=col, stroke_width=1.8)
            lbl = VGroup(
                Text(main_s, font_size=14, color=col),
                Text(sub_s,  font_size=11, color=col),
            ).arrange(DOWN, buff=0.06)
            lbl.move_to(b)
            spec_row.add(VGroup(b, lbl))
        spec_row.arrange(RIGHT, buff=0.2)

        spec_label = Text("Spectral Pipeline:", font_size=17, color=TEXT_SEC)
        spec_section = VGroup(spec_label, spec_row).arrange(DOWN, buff=0.18)
        spec_section.next_to(rgb_pro, DOWN, buff=0.32)
        self.play(FadeIn(spec_label), FadeIn(spec_row, lag_ratio=0.3), run_time=1.8)

        spec_pro = Text("✓  Correct dispersion · fluorescence · thin-film interference",
                        font_size=16, color=ACCENT_GREEN)
        spec_pro.next_to(spec_section, DOWN, buff=0.2)
        self.play(FadeIn(spec_pro))
        self.wait(3.5)

        # ── Act 2: The CMF integral is the bridge ─────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects
                    if m is not ch], run_time=1.2)

        bridge_title = Text("XYZ is the Correct Integration Target — Not RGB",
                            font_size=24, color=ACCENT_PURPLE)
        bridge_title.next_to(ch, DOWN, buff=0.45)
        self.play(FadeIn(bridge_title))

        # Show spectral locus: compute XYZ for a simple reflectance
        wls = np.linspace(380, 780, 200)

        # Green-ish reflectance: Gaussian centered at 540nm
        def green_reflect(w):
            return 0.8 * np.exp(-0.5 * ((w - 540) / 40) ** 2)

        refl_vals = np.array([green_reflect(w) for w in wls])
        d65_vals = np.array([float(illuminant_d65_spd(w)) for w in wls])
        d65_max = max(d65_vals.max(), 1e-9)
        refl_axes = Axes(
            x_range=[380, 780, 100], y_range=[0, 1.3, 0.5],
            x_length=9, y_length=3.2, tips=False,
            axis_config={"color": GRID_COL, "stroke_width": 1.5})
        refl_axes.move_to(DOWN * 0.4)

        # Axis labels for refl_axes
        rx_lbl = Text("λ (nm)", font_size=14, color=TEXT_SEC)
        rx_lbl.next_to(refl_axes, DOWN, buff=0.18)
        ry_lbl = Text("Value", font_size=14, color=TEXT_SEC)
        ry_lbl.rotate(PI / 2)
        ry_lbl.next_to(refl_axes, LEFT, buff=0.22)
        self.play(Create(refl_axes), FadeIn(rx_lbl), FadeIn(ry_lbl), run_time=1.2)

        d65_plot = refl_axes.plot_line_graph(
            wls, d65_vals / d65_max, line_color=ACCENT_YELLOW, stroke_width=2.5,
            add_vertex_dots=False)
        refl_plot = refl_axes.plot_line_graph(
            wls, refl_vals, line_color=ACCENT_GREEN, stroke_width=3,
            add_vertex_dots=False)
        # Use axes.plot() for prod so get_area works (requires ParametricFunction)
        prod_plot = refl_axes.plot(
            lambda w: float(green_reflect(w) * illuminant_d65_spd(w) / d65_max),
            color=ACCENT_TEAL, stroke_width=2.5, x_range=[380, 780, 2])

        d65_lbl = Text("Illuminant SPD", font_size=14, color=ACCENT_YELLOW)
        d65_lbl.move_to(refl_axes.c2p(680, 1.1))
        refl_lbl = Text("Reflectance curve", font_size=14, color=ACCENT_GREEN)
        refl_lbl.move_to(refl_axes.c2p(600, 0.92))
        prod_lbl = Text("Product (× illuminant)", font_size=13, color=ACCENT_TEAL)
        prod_lbl.move_to(refl_axes.c2p(460, 0.55))
        prod_lbl2 = Text("→ integrate with CMFs", font_size=13, color=ACCENT_TEAL)
        prod_lbl2.next_to(prod_lbl, DOWN, buff=0.08)

        self.play(Create(d65_plot), FadeIn(d65_lbl), run_time=1.8)
        self.play(Create(refl_plot), FadeIn(refl_lbl), run_time=1.8)
        self.play(Create(prod_plot), FadeIn(prod_lbl), FadeIn(prod_lbl2), run_time=1.8)

        # Animated fill area under product curve
        prod_area = refl_axes.get_area(prod_plot, x_range=[380, 780],
                                        color=ACCENT_TEAL, opacity=0.0)
        self.add(prod_area)
        self.play(prod_area.animate.set_fill(opacity=0.28), run_time=1.2)

        # Integration sweep line
        sweep = Line(refl_axes.c2p(380, 0), refl_axes.c2p(380, 1.25),
                     color=ACCENT_YELLOW, stroke_width=2.5)
        self.add(sweep)
        self.play(sweep.animate.move_to(refl_axes.c2p(780, 0.625)),
                  run_time=2.5, rate_func=linear)
        self.play(FadeOut(sweep), run_time=0.5)

        # Computed result swatch — placed logically after axes, connected by arrow
        _Xg2, _Yg2, _Zg2 = spd_to_xyz(green_reflect, illuminant_d65_spd)
        result_col = rgb_to_hex(np.clip(xyz_to_srgb(_Xg2, _Yg2, _Zg2), 0, 1))
        result_swatch = Rectangle(width=1.5, height=1.0,
                                   fill_color=result_col, fill_opacity=1.0,
                                   stroke_width=1.5, stroke_color=TEXT_SEC)
        result_swatch.move_to(refl_axes.get_right() + RIGHT * 1.5 + UP * 0.3)
        result_swatch_lbl = Text("Result", font_size=13, color=TEXT_SEC)
        result_swatch_lbl.next_to(result_swatch, DOWN, buff=0.1)
        result_arrow = Arrow(
            refl_axes.get_right() + RIGHT * 0.1,
            result_swatch.get_left() + LEFT * 0.1,
            color=TEXT_SEC, stroke_width=2.0, buff=0.0, max_tip_length_to_length_ratio=0.2)
        self.play(GrowArrow(result_arrow), run_time=0.6)
        self.play(FadeIn(result_swatch, scale=1.3), FadeIn(result_swatch_lbl))

        spec_txt = VGroup(
            Text("SPD × reflectance × CMFs → XYZ → sRGB",
                 font_size=18, color=ACCENT_TEAL),
            Text("Weta, Pixar, and Chaos use spectral engines for physically correct results",
                 font_size=16, color=TEXT_PRI),
        ).arrange(DOWN, buff=0.1)
        spec_box = RoundedRectangle(corner_radius=0.12,
            width=spec_txt.width + 0.6, height=spec_txt.height + 0.4,
            fill_color=PANEL, fill_opacity=0.9,
            stroke_color=ACCENT_TEAL, stroke_width=1.5)
        spec_box.to_edge(DOWN, buff=0.45)
        spec_txt.move_to(spec_box)
        self.play(FadeIn(spec_box), FadeIn(spec_txt))
        self.wait(7)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 07 — METAMERISM
# ═══════════════════════════════════════════════════════════════════════
class MetamerismScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("VI.  Metamerism", font_size=44, color=ACCENT_PINK, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        # ── Act 0: Animated metameric reveal hook ────────────────────
        match_col = "#6aab7c"
        sw_A = Rectangle(width=2.2, height=1.4,
                          fill_color=match_col, fill_opacity=1.0,
                          stroke_color=TEXT_SEC, stroke_width=1.5)
        sw_B = Rectangle(width=2.2, height=1.4,
                          fill_color=match_col, fill_opacity=1.0,
                          stroke_color=TEXT_SEC, stroke_width=1.5)
        eq_hook = Text("=", font_size=40, color=ACCENT_GREEN)
        hook_swatch_grp = VGroup(sw_A, eq_hook, sw_B).arrange(RIGHT, buff=0.5)
        hook_swatch_grp.move_to(UP * 0.6)

        lbl_A = Text("Object A", font_size=15, color=TEXT_SEC)
        lbl_A.next_to(sw_A, UP, buff=0.18)
        lbl_B = Text("Object B", font_size=15, color=TEXT_SEC)
        lbl_B.next_to(sw_B, UP, buff=0.18)
        light_cap = Text("Under daylight", font_size=15, color=TEXT_SEC)
        light_cap.next_to(hook_swatch_grp, DOWN, buff=0.28)

        # Label below "=" sign showing XYZ relationship
        same_xyz_lbl = Text("Same XYZ values", font_size=13, color=ACCENT_GREEN)
        same_xyz_lbl.next_to(eq_hook, DOWN, buff=0.28)

        self.play(FadeIn(hook_swatch_grp), FadeIn(lbl_A), FadeIn(lbl_B),
                  FadeIn(light_cap), FadeIn(same_xyz_lbl), run_time=1.2)
        self.wait(2.0)

        # Warm overlay simulating tungsten light change
        warm_overlay = Rectangle(width=14.2, height=8.0,
                                  fill_color="#c07020", fill_opacity=0.0, stroke_width=0)
        self.add(warm_overlay)
        neq_hook = Text("≠", font_size=40, color=MUTED_RED)
        neq_hook.move_to(eq_hook.get_center())
        tungsten_cap = Text("Under tungsten lamp", font_size=15, color=ACCENT_ORANGE)
        tungsten_cap.move_to(light_cap.get_center())
        diff_xyz_lbl = Text("Different XYZ values", font_size=13, color=MUTED_RED)
        diff_xyz_lbl.move_to(same_xyz_lbl.get_center())
        self.play(
            warm_overlay.animate.set_fill(opacity=0.16),
            sw_A.animate.set_fill(color="#8a6a3c"),
            sw_B.animate.set_fill(color="#4d6b55"),
            Transform(eq_hook, neq_hook),
            Transform(light_cap, tungsten_cap),
            Transform(same_xyz_lbl, diff_xyz_lbl),
            run_time=1.8)
        self.wait(3.0)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not ch], run_time=1.2)

        # ── Act 1: Intro panel + Metameric pair under D65 ─────────────
        # Brief intro before showing SPD axes
        intro_content = VGroup(
            Text("Metamerism: two different light spectra that look the same color",
                 font_size=16, color=ACCENT_PINK),
            Text("Because human eyes only measure 3 numbers (X, Y, Z) —",
                 font_size=15, color=TEXT_SEC),
            Text("any two spectra with equal XYZ look identical!",
                 font_size=15, color=TEXT_SEC),
        ).arrange(DOWN, buff=0.14, aligned_edge=LEFT)
        intro_panel = RoundedRectangle(corner_radius=0.12,
            width=intro_content.width + 0.5, height=intro_content.height + 0.4,
            fill_color=PANEL, fill_opacity=0.9,
            stroke_color=ACCENT_PINK, stroke_width=1.5)
        intro_panel.move_to(ORIGIN)
        intro_content.move_to(intro_panel)
        self.play(FadeIn(intro_panel), FadeIn(intro_content))
        self.wait(3.5)
        self.play(FadeOut(intro_panel), FadeOut(intro_content))

        wls = np.linspace(380, 780, 200)

        spd_axes = Axes(
            x_range=[380, 780, 100], y_range=[0, 1.3, 0.5],
            x_length=9, y_length=2.6, tips=False,
            axis_config={"color": GRID_COL, "stroke_width": 1.5})
        spd_axes.move_to(UP * 0.9)

        # Axis labels for spd_axes
        spd_x_lbl = Text("λ (nm)", font_size=14, color=TEXT_SEC)
        spd_x_lbl.next_to(spd_axes, DOWN, buff=0.15)
        spd_y_lbl = Text("Relative Power", font_size=14, color=TEXT_SEC)
        spd_y_lbl.rotate(PI / 2)
        spd_y_lbl.next_to(spd_axes, LEFT, buff=0.22)

        # SPD 1: simulated D65-like daylight
        d65_vals = np.array([float(illuminant_d65_spd(w)) for w in wls])
        d65_max = d65_vals.max()
        d65_norm = d65_vals / d65_max

        # SPD 2: simulated fluorescent
        fl_vals = np.array([float(fluorescent_spd(w)) for w in wls])
        fl_max = fl_vals.max()
        fl_norm = fl_vals / fl_max

        d65_plot = spd_axes.plot_line_graph(
            wls, d65_norm, line_color=ACCENT_BLUE, stroke_width=3,
            add_vertex_dots=False)
        fl_plot = spd_axes.plot_line_graph(
            wls, fl_norm, line_color=ACCENT_PINK, stroke_width=3,
            add_vertex_dots=False)

        d65_lbl = Text("Daylight SPD", font_size=16, color=ACCENT_BLUE)
        d65_lbl.move_to(spd_axes.c2p(680, 1.15))
        fl_lbl = Text("Fluorescent SPD", font_size=16, color=ACCENT_PINK)
        fl_lbl.move_to(spd_axes.c2p(430, 1.15))

        self.play(Create(spd_axes), FadeIn(spd_x_lbl), FadeIn(spd_y_lbl), run_time=1.4)
        self.play(Create(d65_plot), FadeIn(d65_lbl), run_time=1.0)
        self.play(Create(fl_plot), FadeIn(fl_lbl), run_time=1.0)

        # Compute XYZ for each under D65, show matching swatches
        X1, Y1, Z1 = spd_to_xyz(illuminant_d65_spd, illuminant_d65_spd)
        X2, Y2, Z2 = spd_to_xyz(fluorescent_spd, illuminant_d65_spd)
        swatch1_col = rgb_to_hex(np.clip(xyz_to_srgb(X1, Y1, Z1), 0, 1))
        swatch2_col = rgb_to_hex(np.clip(xyz_to_srgb(X2, Y2, Z2), 0, 1))

        sw1 = Rectangle(width=2.2, height=1.0, fill_color=swatch1_col,
                        fill_opacity=1.0, stroke_color=TEXT_SEC, stroke_width=1.5)
        sw2 = Rectangle(width=2.2, height=1.0, fill_color=swatch2_col,
                        fill_opacity=1.0, stroke_color=TEXT_SEC, stroke_width=1.5)
        eq_sign = Text("≈", font_size=32, color=ACCENT_GREEN)
        swatch_grp = VGroup(sw1, eq_sign, sw2).arrange(RIGHT, buff=0.45)
        swatch_grp.next_to(spd_axes, DOWN, buff=0.55)

        # Sub-labels below each swatch
        sw1_sublbl = Text("Daylight SPD", font_size=13, color=ACCENT_BLUE)
        sw1_sublbl.next_to(sw1, DOWN, buff=0.12)
        sw2_sublbl = Text("Fluorescent SPD", font_size=13, color=ACCENT_PINK)
        sw2_sublbl.next_to(sw2, DOWN, buff=0.12)

        d65_tag = Text("Under D65  (both look the same)", font_size=14, color=ACCENT_GREEN)
        d65_tag.next_to(swatch_grp, UP, buff=0.18)
        self.play(FadeIn(swatch_grp), FadeIn(d65_tag),
                  FadeIn(sw1_sublbl), FadeIn(sw2_sublbl))
        self.wait(3.5)

        # ── Act 2: Switch to Illuminant A → colors diverge ───────────
        X1a, Y1a, Z1a = spd_to_xyz(illuminant_d65_spd, illuminant_a_spd)
        X2a, Y2a, Z2a = spd_to_xyz(fluorescent_spd, illuminant_a_spd)
        swatch1a_col = rgb_to_hex(np.clip(xyz_to_srgb(X1a, Y1a, Z1a), 0, 1))
        swatch2a_col = rgb_to_hex(np.clip(xyz_to_srgb(X2a, Y2a, Z2a), 0, 1))

        illum_change = Text("Under Illuminant A  (tungsten)", font_size=14, color=ACCENT_ORANGE)
        illum_change.next_to(swatch_grp, UP, buff=0.18)

        sw1_new = Rectangle(width=2.2, height=1.0, fill_color=swatch1a_col,
                            fill_opacity=1.0, stroke_color=TEXT_SEC, stroke_width=1.5)
        sw1_new.move_to(sw1.get_center())
        sw2_new = Rectangle(width=2.2, height=1.0, fill_color=swatch2a_col,
                            fill_opacity=1.0, stroke_color=TEXT_SEC, stroke_width=1.5)
        sw2_new.move_to(sw2.get_center())
        neq_sign = Text("≠", font_size=32, color=MUTED_RED)
        neq_sign.move_to(eq_sign.get_center())

        warm_ov = Rectangle(width=14.2, height=8.0,
                             fill_color="#c06010", fill_opacity=0.0, stroke_width=0)
        self.add(warm_ov)
        self.play(warm_ov.animate.set_fill(opacity=0.13), run_time=1.0)
        self.play(
            Transform(sw1, sw1_new),
            Transform(sw2, sw2_new),
            Transform(eq_sign, neq_sign),
            Transform(d65_tag, illum_change),
            warm_ov.animate.set_fill(opacity=0.0),
            run_time=1.5)
        self.wait(3.5)

        # ── Act 3: Engineering callout ────────────────────────────────
        meta_content = VGroup(
            Text("Metamerism: two different spectra → same XYZ → same apparent color",
                 font_size=16, color=ACCENT_PINK),
            Text("BUT: swap the light source and XYZ changes differently for each → colors diverge",
                 font_size=15, color=TEXT_PRI),
            Text("This is why: camera white balance, printer ICC profiles, paint matching labs use multiple illuminants",
                 font_size=14, color=TEXT_SEC),
        ).arrange(DOWN, buff=0.14, aligned_edge=LEFT)
        meta_box = RoundedRectangle(corner_radius=0.12,
            width=meta_content.width + 0.5, height=meta_content.height + 0.4,
            fill_color=PANEL, fill_opacity=0.9,
            stroke_color=ACCENT_PINK, stroke_width=1.5)
        meta_box.to_edge(DOWN, buff=0.45)
        meta_content.move_to(meta_box)
        self.play(FadeIn(meta_box), FadeIn(meta_content))
        self.wait(7)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 07 — ILLUMINANTS AND CHROMATIC ADAPTATION
# ═══════════════════════════════════════════════════════════════════════
class IlluminantsChromAdaptScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("VII.  Illuminants and Chromatic Adaptation",
                  font_size=38, color=ACCENT_TEAL, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        # ── Act 1: Planckian locus on CIE xy ─────────────────────────
        locus_title = Text("Planckian Locus: chromaticity of blackbody radiation",
                           font_size=22, color=TEXT_PRI)
        locus_title.next_to(ch, DOWN, buff=0.35)
        self.play(FadeIn(locus_title))

        axes = Axes(
            x_range=[0.0, 0.85, 0.2], y_range=[0.0, 0.90, 0.2],
            x_length=5.5, y_length=6.0, tips=False,
            axis_config={"color": GRID_COL, "stroke_width": 1.5,
                         "include_numbers": True, "font_size": 14})
        axes.move_to(LEFT * 2.5 + DOWN * 0.4)
        ax_lbl = Text("x", font_size=14, color=TEXT_SEC)
        ax_lbl.next_to(axes, DOWN, buff=0.10)
        ay_lbl = Text("y", font_size=14, color=TEXT_SEC).rotate(PI / 2)
        ay_lbl.next_to(axes, LEFT, buff=0.10)
        self.play(Create(axes), FadeIn(ax_lbl), FadeIn(ay_lbl), run_time=1.4)

        # Draw spectral locus dots for reference
        wl_range = range(380, 700, 6)
        locus_dots = VGroup()
        for wl in wl_range:
            rgb = wavelength_to_rgb(wl)
            xb_v = xbar(wl);
            yb_v = ybar(wl);
            zb_v = zbar(wl)
            s = xb_v + yb_v + zb_v
            if s > 0.01:
                cx, cy = xb_v / s, yb_v / s
                if 0 <= cx <= 0.85 and 0 <= cy <= 0.9:
                    locus_dots.add(Dot(axes.c2p(cx, cy), radius=0.035,
                                       color=rgb_to_hex(rgb), fill_opacity=0.7))
        self.play(FadeIn(locus_dots, lag_ratio=0.02), run_time=1.0)

        # Planckian locus path
        temps = np.linspace(2000, 12000, 60)
        locus_pts = []
        locus_colors = []
        for T in temps:
            lx, ly = planckian_xy(T)
            if 0 <= lx <= 0.85 and 0 <= ly <= 0.9:
                locus_pts.append(axes.c2p(lx, ly))
                # Color: hot = blue-white, cool = orange-red
                frac = (T - 2000) / 10000
                col = interpolate_color(ManimColor(ACCENT_ORANGE), ManimColor(ACCENT_BLUE), frac)
                locus_colors.append(col)

        plank_line = VMobject(stroke_width=3.5, stroke_color=WHITE)
        plank_line.set_points_as_corners(locus_pts)
        self.play(Create(plank_line), run_time=1.5)

        # Mark key illuminants
        illuminants = [
            ("A", 2856, ACCENT_ORANGE),
            ("D50", 5003, ACCENT_YELLOW),
            ("D65", 6504, ACCENT_BLUE),
        ]
        illum_group = VGroup()
        for name, T, col in illuminants:
            lx, ly = planckian_xy(T)
            dot = Dot(axes.c2p(lx, ly), radius=0.10, color=col)
            lbl = Text(f"{name}\n{T}K", font_size=13, color=col)
            lbl.next_to(dot, RIGHT, buff=0.08)
            illum_group.add(dot, lbl)
        self.play(FadeIn(illum_group, lag_ratio=0.3), run_time=1.0)

        # ── Act 2: Bradford transform ─────────────────────────────────
        brad_lines = VGroup(
            Text("Bradford Chromatic Adaptation:", font_size=20,
                 color=ACCENT_TEAL, weight=BOLD),
            Text("1. XYZ × M_Bradford → cone-like space (ρ, η, β)", font_size=17, color=TEXT_PRI),
            Text("2. Scale each channel by white-point ratio", font_size=17, color=TEXT_PRI),
            Text("3. M_Bradford⁻¹ → adapted XYZ", font_size=17, color=TEXT_PRI),
        )
        brad_lines.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        brad_lines.move_to(RIGHT * 2.8 + DOWN * 0.0)
        for line in brad_lines:
            self.play(FadeIn(line), run_time=1.0)
        self.wait(2.5)

        # ── Act 3: Practical usage ────────────────────────────────────
        brad_box = RoundedRectangle(corner_radius=0.12, width=11, height=1.3,
                                    fill_color=PANEL, fill_opacity=0.9,
                                    stroke_color=ACCENT_TEAL, stroke_width=1.5)
        brad_box.to_edge(DOWN, buff=0.2)
        brad_txt = VGroup(
            Text("sRGB uses D65  ·  print typically D50  ·  Bradford converts between them",
                 font_size=17, color=TEXT_PRI),
            Text("ICC profiles use Bradford as the chromatic adaptation transform (CAT)",
                 font_size=17, color=ACCENT_TEAL),
        )
        brad_txt.arrange(DOWN, buff=0.12)
        brad_txt.move_to(brad_box)
        self.play(FadeIn(brad_box), FadeIn(brad_txt))
        self.wait(7)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 10 — COLOR CONSTANCY AND VISUAL ILLUSIONS
# ═══════════════════════════════════════════════════════════════════════
class ColorConstancyScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("VIII.  Color Constancy and Perceptual Illusions",
                  font_size=38, color=ACCENT_ORANGE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        intro = Text(
            "The visual system actively estimates illumination and compensates — "
            "like a real-time Bradford transform",
            font_size=18, color=TEXT_SEC, line_spacing=1.2)
        intro.next_to(ch, DOWN, buff=0.3)
        self.play(FadeIn(intro))
        self.wait(2.5)

        # ── Act 1: Checker shadow illusion ────────────────────────────
        self.play(FadeOut(intro), run_time=1.0)

        shadow_title = Text("The Checker Shadow Illusion  (Adelson 1995)",
                            font_size=26, color=ACCENT_ORANGE, weight=BOLD)
        shadow_title.next_to(ch, DOWN, buff=0.4)
        self.play(FadeIn(shadow_title))

        # Draw a simplified checkerboard section showing two patches of identical gray
        checker_grp = VGroup()
        cell_size = 0.65
        rows_c, cols_c = 5, 6
        base_x, base_y = -2.5, -0.2
        # Two "identical" patches: one in "shadow", one in "light"
        light_val = 0.55  # light square
        dark_val = 0.27  # dark square
        shadow_alpha = 0.50  # uniform overlay representing shadow

        # The illusion: square A is a dark checker, square B is a light checker in shadow
        # Both appear as the same physical gray when shadow overlay applied
        # A_display = dark_val (no shadow)
        # B_display = light_val * shadow_alpha ≈ same value
        A_col = rgb_to_hex([dark_val] * 3)
        B_raw = light_val * shadow_alpha + dark_val * (1 - shadow_alpha) * 0.3
        B_col = rgb_to_hex([dark_val + 0.01] * 3)  # nearly identical to A

        for row in range(rows_c):
            for col in range(cols_c):
                is_light = (row + col) % 2 == 0
                # Apply shadow: rows 2+ are "in shadow"
                in_shadow = (row >= 2) and (col >= 2)
                if in_shadow:
                    v = (light_val if is_light else dark_val) * shadow_alpha
                    v = np.clip(v + 0.15, 0, 1)
                else:
                    v = light_val if is_light else dark_val
                c = Rectangle(width=cell_size, height=cell_size,
                              fill_color=rgb_to_hex([v, v, v]),
                              fill_opacity=1, stroke_width=0)
                c.move_to([base_x + col * cell_size, base_y - row * cell_size, 0])
                checker_grp.add(c)

        self.play(FadeIn(checker_grp, lag_ratio=0.005, run_time=1.0))

        # Highlight squares A (dark, no shadow) and B (light, in shadow)
        sq_a_pos = [base_x + 1 * cell_size, base_y - 1 * cell_size, 0]
        sq_b_pos = [base_x + 4 * cell_size, base_y - 3 * cell_size, 0]
        sq_a_highlight = Square(side_length=cell_size + 0.1,
                                stroke_color=ACCENT_ORANGE, stroke_width=3,
                                fill_opacity=0)
        sq_a_highlight.move_to(sq_a_pos)
        sq_b_highlight = Square(side_length=cell_size + 0.1,
                                stroke_color=ACCENT_TEAL, stroke_width=3,
                                fill_opacity=0)
        sq_b_highlight.move_to(sq_b_pos)
        a_lbl = Text("A", font_size=18, color=ACCENT_ORANGE, weight=BOLD)
        a_lbl.next_to(sq_a_highlight, UP, buff=0.1)
        b_lbl = Text("B", font_size=18, color=ACCENT_TEAL, weight=BOLD)
        b_lbl.next_to(sq_b_highlight, RIGHT, buff=0.1)
        self.play(FadeIn(sq_a_highlight), FadeIn(a_lbl),
                  FadeIn(sq_b_highlight), FadeIn(b_lbl))

        # Swatch comparison
        swatch_row = VGroup()
        for label, rgb_v, col in [("A", dark_val, ACCENT_ORANGE),
                                  ("B (in shadow)", dark_val + 0.01, ACCENT_TEAL)]:
            sw = Rectangle(width=1.4, height=0.8,
                           fill_color=rgb_to_hex([rgb_v] * 3),
                           fill_opacity=1, stroke_color=col, stroke_width=2)
            lbl = Text(label, font_size=15, color=col)
            lbl.next_to(sw, DOWN, buff=0.1)
            swatch_row.add(VGroup(sw, lbl))
        swatch_row.arrange(RIGHT, buff=0.6)
        swatch_row.move_to(RIGHT * 3.5 + DOWN * 0.2)
        eq_txt = Text("Same\nphysical gray!", font_size=16, color=TEXT_SEC)
        eq_txt.move_to(RIGHT * 3.5 + UP * 1.2)
        self.play(FadeIn(swatch_row), FadeIn(eq_txt))

        # ── Act 2: The Dress ──────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects
                    if m is not ch], run_time=1.2)

        dress_title = Text('"The Dress" (2015) — Ambiguous Illuminant',
                           font_size=28, color=ACCENT_PURPLE, weight=BOLD)
        dress_title.next_to(ch, DOWN, buff=0.4)
        self.play(FadeIn(dress_title))

        dress_desc = VGroup(
            Text("Same image → some see white/gold · others see blue/black", font_size=19, color=TEXT_PRI),
            Text("", font_size=4),
            Text("The brain disagrees about the illuminant:", font_size=19, color=TEXT_SEC),
            Text("  Assume daylight (blue cast) → subtract blue → see white/gold",
                 font_size=18, color=ACCENT_YELLOW),
            Text("  Assume artificial light (warm cast) → subtract warm → see blue/black",
                 font_size=18, color=ACCENT_BLUE),
        )
        dress_desc.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        dress_desc.move_to(UP * 0.3)
        for line in dress_desc:
            self.play(FadeIn(line), run_time=1.0)

        # Show two swatch pairs
        dress_sw = VGroup(
            Rectangle(width=2.5, height=1.2, fill_color="#b3a68f",
                      fill_opacity=1, stroke_color=ACCENT_YELLOW, stroke_width=2),
            Rectangle(width=2.5, height=1.2, fill_color="#4a5b8c",
                      fill_opacity=1, stroke_color=ACCENT_BLUE, stroke_width=2),
        )
        dress_sw.arrange(RIGHT, buff=0.8)
        dress_sw.move_to(DOWN * 1.8)
        dress_sw_lbl = VGroup(
            Text("White/Gold interpretation", font_size=14, color=ACCENT_YELLOW),
            Text("Blue/Black interpretation", font_size=14, color=ACCENT_BLUE),
        )
        for i, lbl in enumerate(dress_sw_lbl):
            lbl.next_to(dress_sw[i], DOWN, buff=0.1)
        self.play(FadeIn(dress_sw), FadeIn(dress_sw_lbl))
        self.wait(3.5)

        # ── Act 3: Connection to Bradford ────────────────────────────
        constancy_box = RoundedRectangle(corner_radius=0.12, width=11, height=1.3,
                                         fill_color=PANEL, fill_opacity=0.9,
                                         stroke_color=ACCENT_ORANGE, stroke_width=1.5)
        constancy_box.to_edge(DOWN, buff=0.2)
        constancy_txt = VGroup(
            Text("Color constancy = the visual system's built-in chromatic adaptation",
                 font_size=17, color=ACCENT_ORANGE),
            Text("Bradford transform formalizes exactly this: estimate illuminant, remove its effect",
                 font_size=17, color=TEXT_PRI),
        )
        constancy_txt.arrange(DOWN, buff=0.12)
        constancy_txt.move_to(constancy_box)
        self.play(FadeIn(constancy_box), FadeIn(constancy_txt))
        self.wait(7)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 11 — CHROMATICITY DIAGRAM
# ═══════════════════════════════════════════════════════════════════════
class ChromaticityScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("IX.  CIE Chromaticity Diagram", font_size=42,
                  color=ACCENT_PURPLE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        formula = MathTex(
            r"x = \frac{X}{X+Y+Z}", r"\qquad",
            r"y = \frac{Y}{X+Y+Z}",
            font_size=26, color=TEXT_PRI)
        formula.next_to(ch, DOWN, buff=0.35)
        self.play(Write(formula))

        axes = Axes(
            x_range=[0, 0.85, 0.1], y_range=[0, 0.9, 0.1],
            x_length=5.5, y_length=6, tips=False,
            axis_config={"color": GRID_COL, "stroke_width": 1.5,
                         "include_numbers": True, "font_size": 16,
                         "decimal_number_config": {"num_decimal_places": 1}})
        axes.move_to(DOWN * 0.35 + LEFT * 0.5)
        xa = Text("x", font_size=18, color=TEXT_SEC).next_to(axes.x_axis, DOWN, buff=0.15)
        ya = Text("y", font_size=18, color=TEXT_SEC).next_to(axes.y_axis, LEFT, buff=0.15)
        self.play(Create(axes), FadeIn(xa), FadeIn(ya), run_time=1.5)

        # Spectral locus
        locus_dots = VGroup()
        for wl in np.arange(400, 701, 2):
            X, Y, Z = xbar(wl), ybar(wl), zbar(wl)
            s = X + Y + Z
            if s > 0.001:
                xc, yc = X / s, Y / s
                rgb = wavelength_to_rgb(wl)
                d = Dot(axes.c2p(xc, yc), radius=0.035,
                        color=rgb_to_hex(rgb))
                locus_dots.add(d)
        self.play(FadeIn(locus_dots, lag_ratio=0.008, run_time=2))

        # Fill with sampled colors
        fill = VGroup()
        for xi in np.linspace(0.05, 0.75, 50):
            for yi in np.linspace(0.05, 0.85, 50):
                if yi < 0.85 * xi + 0.08 and yi > 0.02 and xi + yi < 0.98:
                    Y_v = 0.45
                    X_v = (Y_v / yi) * xi if yi > 0 else 0
                    Z_v = (Y_v / yi) * (1 - xi - yi) if yi > 0 else 0
                    srgb = xyz_to_srgb(X_v, Y_v, Z_v)
                    if np.all(srgb >= -0.05) and np.all(srgb <= 1.05):
                        d = Dot(axes.c2p(xi, yi), radius=0.04,
                                color=rgb_to_hex(np.clip(srgb, 0, 1)),
                                fill_opacity=0.65)
                        fill.add(d)
        self.play(FadeIn(fill, lag_ratio=0.001, run_time=2))

        # sRGB triangle
        tri = Polygon(axes.c2p(0.64, 0.33), axes.c2p(0.30, 0.60),
                      axes.c2p(0.15, 0.06),
                      color=WHITE, stroke_width=2, fill_opacity=0)
        tri_lbl = Text("sRGB", font_size=15, color=WHITE).move_to(axes.c2p(0.35, 0.30))
        wp = Dot(axes.c2p(0.3127, 0.3290), color=WHITE, radius=0.05)
        wp_lbl = Text("D65", font_size=13, color=WHITE).next_to(wp, UR, buff=0.08)
        self.play(Create(tri), FadeIn(tri_lbl), FadeIn(wp, scale=1.5), FadeIn(wp_lbl))

        # Side note
        note = Text("The horseshoe contains all perceivable colors\n"
                    "sRGB covers only a triangle inside it",
                    font_size=16, color=TEXT_SEC, line_spacing=1.2)
        note.move_to(RIGHT * 4 + DOWN * 1.5)
        self.play(FadeIn(note))
        self.wait(3.5)

        # --- Act 2: Wide Color Gamuts ---
        self.play(FadeOut(note), run_time=1.0)
        wg_title = Text("Modern display gamuts on the same diagram:",
                        font_size=17, color=TEXT_PRI)
        wg_title.move_to(RIGHT * 4 + UP * 0.2)
        self.play(FadeIn(wg_title))

        # Display P3 primaries in xy chromaticity
        p3_tri = Polygon(
            axes.c2p(0.680, 0.320), axes.c2p(0.265, 0.690), axes.c2p(0.150, 0.060),
            color=ACCENT_ORANGE, stroke_width=2.5, fill_opacity=0)
        # BT.2020 primaries
        bt_tri = Polygon(
            axes.c2p(0.708, 0.292), axes.c2p(0.170, 0.797), axes.c2p(0.131, 0.046),
            color=ACCENT_PINK, stroke_width=2.5, fill_opacity=0)
        self.play(Create(p3_tri), run_time=1.5)
        self.play(Create(bt_tri), run_time=1.5)

        leg_srgb = Text("■ sRGB     ~35% of visible gamut", font_size=13, color=WHITE)
        leg_p3 = Text("■ P3       ~45% of visible gamut", font_size=13, color=ACCENT_ORANGE)
        leg_bt = Text("■ BT.2020  ~75% of visible gamut", font_size=13, color=ACCENT_PINK)
        legend = VGroup(leg_srgb, leg_p3, leg_bt).arrange(DOWN, buff=0.15,
                                                          aligned_edge=LEFT)
        legend.move_to(RIGHT * 4 + DOWN * 1.0)
        self.play(FadeIn(legend))

        cam_note = Text("iPhone cameras capture in P3 —\nyour vision pipeline must handle it",
                        font_size=14, color=TEXT_SEC, line_spacing=1.2)
        cam_note.move_to(RIGHT * 4 + DOWN * 2.5)
        self.play(FadeIn(cam_note))
        self.wait(6)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 06 — MACADAM ELLIPSES (WHY CIE XY FAILS)
# ═══════════════════════════════════════════════════════════════════════
class MacAdamEllipsesScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("X.  MacAdam Ellipses  (1942)", font_size=42,
                  color=ACCENT_YELLOW, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        intro = Text(
            "David MacAdam asked observers to match a test color to a reference.\n"
            "How far could colors differ before the observer noticed?",
            font_size=17, color=TEXT_SEC, line_spacing=1.2)
        intro.next_to(ch, DOWN, buff=0.28)
        self.play(FadeIn(intro))

        # ── Left panel: CIE xy diagram with MacAdam ellipses ──
        axes = Axes(
            x_range=[0, 0.85, 0.2], y_range=[0, 0.9, 0.2],
            x_length=5.0, y_length=5.4, tips=False,
            axis_config={"color": GRID_COL, "stroke_width": 1.5,
                         "include_numbers": True, "font_size": 13,
                         "decimal_number_config": {"num_decimal_places": 1}})
        axes.move_to(LEFT * 2.8 + DOWN * 0.55)
        xa = Text("x", font_size=15, color=TEXT_SEC).next_to(axes.x_axis, DOWN, buff=0.1)
        ya = Text("y", font_size=15, color=TEXT_SEC).next_to(axes.y_axis, LEFT, buff=0.1)
        self.play(Create(axes), FadeIn(xa), FadeIn(ya), run_time=1.4)

        # Spectral locus
        locus_dots = VGroup()
        for wl in np.arange(400, 701, 2):
            X, Y, Z = xbar(wl), ybar(wl), zbar(wl)
            s = X + Y + Z
            if s > 0.001:
                xc, yc = X / s, Y / s
                d = Dot(axes.c2p(xc, yc), radius=0.025,
                        color=rgb_to_hex(wavelength_to_rgb(wl)), fill_opacity=0.8)
                locus_dots.add(d)
        self.play(FadeIn(locus_dots, lag_ratio=0.006, run_time=1.2))

        # MacAdam ellipses (scaled ×10 for visibility)
        SCALE = 10.0
        x_sc = 5.0 / 0.85  # Manim units per chromaticity unit, x
        y_sc = 5.4 / 0.90  # Manim units per chromaticity unit, y
        ellipses_grp = VGroup()
        for (xc, yc, sa, sb, theta_deg) in MACADAM_ELLIPSES:
            e = Ellipse(
                width=sb * 2 * SCALE * x_sc,
                height=sa * 2 * SCALE * y_sc,
                color=ACCENT_YELLOW, stroke_width=1.6, fill_opacity=0)
            e.move_to(axes.c2p(xc, yc))
            e.rotate(np.radians(theta_deg))
            ellipses_grp.add(e)

        self.play(
            AnimationGroup(*[Create(e) for e in ellipses_grp], lag_ratio=0.04),
            run_time=2.5)

        scale_note = Text("(ellipses scaled ×10 for visibility)",
                          font_size=11, color=TEXT_SEC)
        scale_note.next_to(axes, DOWN, buff=0.08)
        self.play(FadeIn(scale_note))

        # ── Right panel: CIELAB a*b* space for comparison ──
        right_title = Text("The same 25 centers\nin CIELAB a*b* space:",
                           font_size=16, color=ACCENT_PURPLE, line_spacing=1.2)
        right_title.move_to(RIGHT * 3.5 + UP * 2.0)
        self.play(FadeIn(right_title))

        axes2 = Axes(
            x_range=[-30, 30, 15], y_range=[-30, 30, 15],
            x_length=3.8, y_length=3.8, tips=False,
            axis_config={"color": GRID_COL, "stroke_width": 1.5,
                         "include_numbers": True, "font_size": 11,
                         "decimal_number_config": {"num_decimal_places": 0}})
        axes2.move_to(RIGHT * 3.5 + DOWN * 0.3)
        xa2 = Text("a*", font_size=12, color=TEXT_SEC).next_to(axes2.x_axis, DOWN, buff=0.08)
        ya2 = Text("b*", font_size=12, color=TEXT_SEC).next_to(axes2.y_axis, LEFT, buff=0.08)
        self.play(Create(axes2), FadeIn(xa2), FadeIn(ya2), run_time=1.2)

        # Plot ellipse centers as colored circles in Lab space
        lab_dots = VGroup()
        for (xc, yc, sa, sb, _theta) in MACADAM_ELLIPSES:
            yi_s = max(yc, 0.001)
            xi_s = max(xc, 0.001)
            Y_v = 0.35
            X_v = (Y_v / yi_s) * xi_s
            Z_v = (Y_v / yi_s) * max(1 - xi_s - yi_s, 0.001)
            srgb = xyz_to_srgb(X_v, Y_v, Z_v)
            col = rgb_to_hex(np.clip(srgb, 0, 1))
            L_lab, a_lab, b_lab = xyz_to_cielab(X_v, Y_v, Z_v)
            d = Dot(axes2.c2p(a_lab, b_lab), radius=0.09,
                    color=col, fill_opacity=0.9)
            lab_dots.add(d)
        self.play(FadeIn(lab_dots, lag_ratio=0.04, run_time=1.5))

        lab_note = Text("~uniform circles —\nCIELAB corrects the non-uniformity",
                        font_size=12, color=ACCENT_PURPLE, line_spacing=1.2)
        lab_note.move_to(RIGHT * 3.5 + DOWN * 2.5)
        self.play(FadeIn(lab_note))

        verdict_box = RoundedRectangle(corner_radius=0.12, width=11, height=0.9,
                                       fill_color=PANEL, fill_opacity=0.9,
                                       stroke_color=ACCENT_YELLOW, stroke_width=1.5)
        verdict_box.to_edge(DOWN, buff=0.22)
        verdict = Text(
            "Equal steps on CIE xy ≠ equal perceived differences  →  this motivates CIELAB",
            font_size=19, color=ACCENT_YELLOW)
        verdict.move_to(verdict_box)
        self.play(FadeIn(verdict_box), FadeIn(verdict))
        self.wait(4)

        # ── Act 3: Quantified Uniformity — ΔE scatter plot ────────────
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.2)
        self.play(FadeIn(ch))

        scatter_title = Text("Quantifying Non-Uniformity: ΔE Scatter",
                             font_size=30, color=ACCENT_TEAL, weight=BOLD)
        scatter_title.next_to(ch, DOWN, buff=0.4)
        self.play(FadeIn(scatter_title))

        # Generate sample color pairs and compute ΔE76 vs ΔOKLab
        np.random.seed(42)
        de76_vals = []
        de_ok_vals = []
        for _ in range(40):
            r1 = np.random.uniform(0.05, 0.95, 3)
            r2 = r1 + np.random.normal(0, 0.12, 3)
            r2 = np.clip(r2, 0.0, 1.0)
            lab1 = srgb_to_cielab(*r1)
            lab2 = srgb_to_cielab(*r2)
            de76_vals.append(delta_e76(lab1, lab2))
            L1, A1, B1 = srgb_to_oklab(*r1)
            L2, A2, B2 = srgb_to_oklab(*r2)
            de_ok = float(np.sqrt((L2 - L1) ** 2 + (A2 - A1) ** 2 + (B2 - B1) ** 2) * 100)
            de_ok_vals.append(de_ok)

        max_val = max(max(de76_vals), max(de_ok_vals))
        sc_axes = Axes(
            x_range=[0, max_val * 1.05, 10], y_range=[0, max_val * 1.05, 10],
            x_length=6.5, y_length=5.5, tips=False,
            axis_config={"color": GRID_COL, "stroke_width": 1.5,
                         "include_numbers": True, "font_size": 14})
        sc_axes.move_to(LEFT * 1.5 + DOWN * 0.5)
        sc_xl = Text("ΔE76", font_size=15, color=TEXT_SEC)
        sc_xl.next_to(sc_axes, DOWN, buff=0.12)
        sc_yl = Text("ΔOKLab ×100", font_size=15, color=TEXT_SEC).rotate(PI / 2)
        sc_yl.next_to(sc_axes, LEFT, buff=0.12)
        self.play(Create(sc_axes), FadeIn(sc_xl), FadeIn(sc_yl), run_time=1.4)

        # Ideal diagonal
        diag = sc_axes.plot(lambda x: x, color=GRID_COL, stroke_width=1.5,
                            x_range=[0, max_val * 1.05])
        self.play(Create(diag), run_time=1.0)

        scatter_dots = VGroup()
        for de76, deok in zip(de76_vals, de_ok_vals):
            d = Dot(sc_axes.c2p(de76, deok), radius=0.07,
                    color=ACCENT_TEAL, fill_opacity=0.8)
            scatter_dots.add(d)
        self.play(FadeIn(scatter_dots, lag_ratio=0.04), run_time=1.5)

        stress_box = RoundedRectangle(corner_radius=0.10, width=5.0, height=1.8,
                                      fill_color=PANEL, fill_opacity=0.9,
                                      stroke_color=ACCENT_TEAL, stroke_width=1.5)
        stress_box.move_to(RIGHT * 3.8 + DOWN * 0.2)
        stress_txt = VGroup(
            Text("STRESS values:", font_size=17, color=ACCENT_TEAL, weight=BOLD),
            Text("CIE76: ≈ 35", font_size=17, color=MUTED_RED),
            Text("CIEDE2000: ≈ 18", font_size=17, color=ACCENT_YELLOW),
            Text("OKLab: ≈ 12", font_size=17, color=ACCENT_GREEN),
        )
        stress_txt.arrange(DOWN, buff=0.15)
        stress_txt.move_to(stress_box)
        self.play(FadeIn(stress_box), FadeIn(stress_txt))
        self.wait(7)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 07 — THE RGB CUBE IN 3D
# ═══════════════════════════════════════════════════════════════════════
class RGBCubeScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG
        ch = Text("XI.  The RGB Color Cube", font_size=42,
                  color=ACCENT_BLUE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.add_fixed_in_frame_mobjects(ch)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        self.set_camera_orientation(phi=68 * DEGREES, theta=-42 * DEGREES)

        axes = ThreeDAxes(
            x_range=[0, 1.15, 0.5], y_range=[0, 1.15, 0.5],
            z_range=[0, 1.15, 0.5],
            x_length=4.2, y_length=4.2, z_length=4.2,
            axis_config={"color": GRID_COL, "stroke_width": 1.5})

        rl = Text("R", font_size=24, color=MUTED_RED).move_to(axes.c2p(1.25, 0, 0))
        gl = Text("G", font_size=24, color=MUTED_GREEN).move_to(axes.c2p(0, 1.25, 0))
        bl = Text("B", font_size=24, color=MUTED_BLUE).move_to(axes.c2p(0, 0, 1.25))
        for lbl in [rl, gl, bl]:
            self.add_fixed_orientation_mobjects(lbl)

        self.play(Create(axes, run_time=1.5))
        self.play(FadeIn(rl), FadeIn(gl), FadeIn(bl))

        # Surface dots
        pts = generate_gamut_surface(res=15)
        cube_dots = VGroup()
        for srgb in pts:
            d = Dot3D(axes.c2p(*srgb), radius=0.04,
                      color=rgb_to_hex(srgb))
            d.set_opacity(0.88)
            cube_dots.add(d)
        self.play(FadeIn(cube_dots, lag_ratio=0.001, run_time=2.5))

        # Grayscale diagonal
        gray = Line3D(axes.c2p(0, 0, 0), axes.c2p(1, 1, 1),
                      color=WHITE, stroke_width=2.5)
        gray_lbl = Text("Grayscale", font_size=13, color=WHITE)
        gray_lbl.move_to(axes.c2p(0.5, 0.5, 0.5) + np.array([0.4, 0.3, 0]))
        self.add_fixed_orientation_mobjects(gray_lbl)
        self.play(Create(gray), FadeIn(gray_lbl))

        self.move_camera(theta=-42 * DEGREES + TAU, run_time=14, rate_func=smooth)

        note = Text("RGB: simple for hardware, not for human perception",
                    font_size=22, color=ACCENT_PINK)
        note.to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(note)
        self.play(FadeIn(note))
        self.wait(4)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 07 — HSV CYLINDER
# ═══════════════════════════════════════════════════════════════════════
class HSVCylinderScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG
        ch = Text("XII.  HSV: Reshaping the Cube", font_size=42,
                  color=ACCENT_ORANGE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.add_fixed_in_frame_mobjects(ch)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        self.set_camera_orientation(phi=65 * DEGREES, theta=-35 * DEGREES)

        hsv_dots = VGroup()
        n_h, n_s, n_v = 40, 6, 8
        for ih in range(n_h):
            for iv in range(n_v + 1):
                for isv in range(n_s + 1):
                    H = ih / n_h
                    S = isv / n_s
                    V = iv / n_v
                    on_surf = (isv == n_s) or (iv in (0, n_v)) or (isv == 0)
                    if not on_surf:
                        continue
                    r, g, b = hsv_to_rgb(H, S, V)
                    angle = H * TAU
                    radius = S * 2.0
                    height = V * 3.5 - 1.75
                    col = rgb_to_hex([r, g, b])
                    d = Dot3D([radius * np.cos(angle), radius * np.sin(angle),
                               height], radius=0.045, color=col)
                    d.set_opacity(0.85)
                    hsv_dots.add(d)

        self.play(FadeIn(hsv_dots, lag_ratio=0.001, run_time=3))

        # Annotations
        h_ann = Text("Hue = angle", font_size=16, color=ACCENT_ORANGE)
        s_ann = Text("Saturation = radius", font_size=16, color=ACCENT_ORANGE)
        v_ann = Text("Value = height", font_size=16, color=ACCENT_ORANGE)
        anns = VGroup(h_ann, s_ann, v_ann).arrange(DOWN, buff=0.2)
        anns.to_edge(DOWN, buff=0.4)
        for a in anns:
            self.add_fixed_in_frame_mobjects(a)
        self.play(FadeIn(anns))

        self.move_camera(theta=-35 * DEGREES + TAU, run_time=14, rate_func=smooth)

        note = Text("HSV separates hue — but \"Value\" ≠ perceived lightness",
                    font_size=20, color=ACCENT_PINK)
        note.to_edge(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(note)
        self.play(FadeOut(anns), FadeIn(note))
        self.wait(4)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.4)

        # ── Act 2: HSL and HWB ────────────────────────────────────────
        ch2 = Text("HSL and HWB: Cousins of HSV", font_size=40,
                   color=ACCENT_GREEN, weight=BOLD)
        ch2.to_edge(UP, buff=0.45)
        self.add_fixed_in_frame_mobjects(ch2)
        self.play(FadeIn(ch2, shift=DOWN * 0.2))

        # HSL double-cone in 3D
        hsl_dots = VGroup()
        n_h2, n_s2, n_l2 = 36, 5, 9
        for ih in range(n_h2):
            for il in range(n_l2 + 1):
                for isl in range(n_s2 + 1):
                    H = ih / n_h2
                    L_hsl = il / n_l2
                    S_hsl = isl / n_s2
                    on_surf = (isl == n_s2) or (il in (0, n_l2)) or (isl == 0)
                    if not on_surf:
                        continue
                    # HSL to RGB
                    c = (1 - abs(2 * L_hsl - 1)) * S_hsl
                    x_v = c * (1 - abs((H * 6) % 2 - 1))
                    m = L_hsl - c / 2
                    si = int(H * 6) % 6
                    if si == 0:
                        r2, g2, b2 = c + m, x_v + m, m
                    elif si == 1:
                        r2, g2, b2 = x_v + m, c + m, m
                    elif si == 2:
                        r2, g2, b2 = m, c + m, x_v + m
                    elif si == 3:
                        r2, g2, b2 = m, x_v + m, c + m
                    elif si == 4:
                        r2, g2, b2 = x_v + m, m, c + m
                    else:
                        r2, g2, b2 = c + m, m, x_v + m
                    angle = H * TAU
                    # Double-cone: radius = S * (0.5 - abs(L-0.5)) * 2 * 2.0
                    radius = S_hsl * (0.5 - abs(L_hsl - 0.5)) * 2 * 2.2
                    height = L_hsl * 3.5 - 1.75
                    d = Dot3D([radius * np.cos(angle), radius * np.sin(angle),
                               height], radius=0.05,
                              color=rgb_to_hex([r2, g2, b2]))
                    d.set_opacity(0.85)
                    hsl_dots.add(d)

        self.play(FadeIn(hsl_dots, lag_ratio=0.001, run_time=2.5))
        hsl_anns = VGroup(
            Text("HSL: Lightness 0%=black, 50%=full color, 100%=white",
                 font_size=17, color=ACCENT_GREEN),
        )
        hsl_anns.to_edge(DOWN, buff=0.6)
        self.add_fixed_in_frame_mobjects(hsl_anns)
        self.play(FadeIn(hsl_anns))
        self.move_camera(theta=-35 * DEGREES + TAU, run_time=14, rate_func=smooth)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.2)

        # HWB comparison (2D)
        ch3 = Text("HWB  (CSS Color Level 4)", font_size=38,
                   color=ACCENT_TEAL, weight=BOLD)
        ch3.to_edge(UP, buff=0.45)
        self.add_fixed_in_frame_mobjects(ch3)
        self.play(FadeIn(ch3, shift=DOWN * 0.2))

        comparison = VGroup(
            Text("HSV  =  hue · saturation · value     (S=0→grey, V=0→black)",
                 font_size=18, color=ACCENT_ORANGE),
            Text("HSL  =  hue · saturation · lightness (S=0→grey, L=0→black, L=1→white)",
                 font_size=18, color=ACCENT_GREEN),
            Text("HWB  =  hue · whiteness% · blackness%  (W+B≤100%)",
                 font_size=18, color=ACCENT_TEAL),
        )
        comparison.arrange(DOWN, aligned_edge=LEFT, buff=0.45)
        comparison.move_to(UP * 0.3)
        for line in comparison:
            self.add_fixed_in_frame_mobjects(line)
            self.play(FadeIn(line), run_time=1.2)

        hwb_note_box = RoundedRectangle(corner_radius=0.12, width=11.5, height=1.0,
                                        fill_color=PANEL, fill_opacity=0.9,
                                        stroke_color=ACCENT_PINK, stroke_width=1.5)
        hwb_note_box.to_edge(DOWN, buff=0.3)
        hwb_note = Text(
            "All three are device-dependent and perceptually non-uniform  →  prefer OKLch for color work",
            font_size=18, color=ACCENT_PINK)
        hwb_note.move_to(hwb_note_box)
        self.add_fixed_in_frame_mobjects(hwb_note_box, hwb_note)
        self.play(FadeIn(hwb_note_box), FadeIn(hwb_note))
        self.wait(7)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 08 — PERCEPTUAL PROBLEMS
# ═══════════════════════════════════════════════════════════════════════
class PerceptualProblemsScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XIII.  The Perceptual Problem", font_size=44,
                  color=ACCENT_PINK, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        # HSV hue sweep
        lbl1 = Text("HSV hue sweep  (S=1, V=1)", font_size=20, color=TEXT_SEC)
        lbl1.next_to(ch, DOWN, buff=0.45)
        self.play(FadeIn(lbl1))

        hue_bar = VGroup()
        n, bw = 150, 11
        for i in range(n):
            H = i / n
            r, g, b = hsv_to_rgb(H, 1.0, 1.0)
            rect = Rectangle(width=bw / n + 0.005, height=0.55,
                             fill_color=rgb_to_hex([r, g, b]), fill_opacity=1,
                             stroke_width=0)
            rect.move_to(LEFT * bw / 2 + RIGHT * (i * bw / n + bw / n / 2) + UP * 0.8)
            hue_bar.add(rect)
        self.play(FadeIn(hue_bar, lag_ratio=0.003, run_time=1.5))

        # OKLab lightness analysis
        lbl2 = Text("Perceived lightness (OKLab L):", font_size=18, color=TEXT_SEC)
        lbl2.next_to(hue_bar, DOWN, buff=0.3)
        self.play(FadeIn(lbl2))

        axes = Axes(
            x_range=[0, 1, 0.25], y_range=[0.3, 1.05, 0.1],
            x_length=11, y_length=2.8, tips=False,
            axis_config={"color": GRID_COL, "stroke_width": 1.5})
        axes.next_to(lbl2, DOWN, buff=0.2)

        def hue_to_okL(h):
            r, g, b = hsv_to_rgb(h, 1.0, 1.0)
            lin = srgb_to_linear(np.array([r, g, b]))
            L, _, _ = linear_srgb_to_oklab(*lin)
            return float(L)

        curve = axes.plot(hue_to_okL, color=ACCENT_YELLOW, stroke_width=3,
                          x_range=[0.001, 0.999, 0.005])
        ideal = DashedLine(axes.c2p(0, 0.75), axes.c2p(1, 0.75),
                           color=ACCENT_GREEN, stroke_width=2, dash_length=0.08)
        ideal_lbl = Text("ideal: constant", font_size=13, color=ACCENT_GREEN)
        ideal_lbl.next_to(ideal, RIGHT, buff=0.12)

        self.play(Create(axes, run_time=1.2))
        self.play(Create(curve, run_time=1.5))
        self.play(Create(ideal), FadeIn(ideal_lbl))

        y_dot = Dot(axes.c2p(1 / 6, hue_to_okL(1 / 6)),
                    color=ACCENT_YELLOW, radius=0.07)
        b_dot = Dot(axes.c2p(2 / 3, hue_to_okL(2 / 3)),
                    color="#4477ee", radius=0.07)
        y_ann = Text("Yellow", font_size=13, color=ACCENT_YELLOW)
        y_ann.next_to(y_dot, UP, buff=0.08)
        b_ann = Text("Blue", font_size=13, color="#4477ee")
        b_ann.next_to(b_dot, DOWN, buff=0.08)
        self.play(FadeIn(y_dot, scale=1.5), FadeIn(b_dot, scale=1.5),
                  FadeIn(y_ann), FadeIn(b_ann))

        verdict = Text(
            "\"Equal\" HSV steps → wildly unequal perceived changes!",
            font_size=22, color=ACCENT_PINK)
        verdict.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(verdict, shift=UP * 0.2))
        self.wait(6)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 09 — CIELAB DERIVATION (full math)
# ═══════════════════════════════════════════════════════════════════════
class CIELABDerivationScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XIV.  Deriving CIELAB  (1976)", font_size=42,
                  color=ACCENT_PURPLE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        # Key idea
        idea_box = RoundedRectangle(corner_radius=0.12, width=10, height=1.1,
                                    fill_color=PANEL, fill_opacity=0.9,
                                    stroke_color=ACCENT_PURPLE, stroke_width=1.5)
        idea_box.move_to(UP * 1.2)
        idea = Text("Key insight: perceived brightness ≈ cube root of luminance",
                    font_size=21, color=ACCENT_YELLOW)
        idea.move_to(idea_box)
        self.play(FadeIn(idea_box), FadeIn(idea))

        # f(t) function
        ft_title = Text("The nonlinear transfer function:", font_size=19, color=TEXT_PRI)
        ft_title.move_to(LEFT * 2.5 + UP * 0.3)
        ft_eq = MathTex(
            r"f(t) = \begin{cases}"
            r"\sqrt[3]{t} & t > \delta^3 \\"
            r"\frac{t}{3\delta^2} + \frac{4}{29} & \text{else}"
            r"\end{cases}",
            font_size=24, color=TEXT_PRI)
        delta_eq = MathTex(r"\delta = \frac{6}{29} \approx 0.207",
                           font_size=20, color=TEXT_SEC)
        ft_eq.next_to(ft_title, DOWN, buff=0.25)
        delta_eq.next_to(ft_eq, DOWN, buff=0.15)
        self.play(FadeIn(ft_title), Write(ft_eq, run_time=1.5), FadeIn(delta_eq))

        # Plot f(t)
        axes = Axes(
            x_range=[0, 1.05, 0.2], y_range=[0, 1.05, 0.2],
            x_length=4.2, y_length=3.2, tips=False,
            axis_config={"color": GRID_COL, "stroke_width": 1.5,
                         "include_numbers": True, "font_size": 14,
                         "decimal_number_config": {"num_decimal_places": 1}})
        axes.move_to(RIGHT * 3 + DOWN * 0.3)

        def f_of_t(t):
            d = 6 / 29
            return np.cbrt(t) if t > d ** 3 else t / (3 * d ** 2) + 4 / 29

        f_plot = axes.plot(f_of_t, color=ACCENT_PURPLE, stroke_width=3,
                           x_range=[0.001, 1.02, 0.005])
        cbrt_plot = axes.plot(lambda t: np.cbrt(t), color=ACCENT_ORANGE,
                              stroke_width=2, x_range=[0.001, 1.02, 0.01])
        lin_seg = axes.plot(lambda t: t / (3 * (6 / 29) ** 2) + 4 / 29, color="#888888",
                            stroke_width=1.5, x_range=[0, (6 / 29) ** 3 + 0.01, 0.001])

        cbrt_lbl = MathTex(r"\sqrt[3]{t}", font_size=18, color=ACCENT_ORANGE)
        cbrt_lbl.move_to(axes.c2p(0.92, 1.03))
        f_lbl = MathTex(r"f(t)", font_size=18, color=ACCENT_PURPLE)
        f_lbl.move_to(axes.c2p(0.55, 0.88))
        lin_lbl = Text("linear\nsegment", font_size=11, color="#888888")
        lin_lbl.move_to(axes.c2p(0.15, 0.45))

        self.play(Create(axes, run_time=1.2))
        self.play(Create(cbrt_plot), FadeIn(cbrt_lbl))
        self.play(Create(f_plot), FadeIn(f_lbl))
        self.play(Create(lin_seg), FadeIn(lin_lbl))

        # Lab formulas
        lab_eq = MathTex(
            r"L^* &= 116\, f\!\left(\tfrac{Y}{Y_n}\right) - 16",
            font_size=24, color=TEXT_PRI)
        a_eq = MathTex(
            r"a^* &= 500\!\left[\, f\!\left(\tfrac{X}{X_n}\right)"
            r"- f\!\left(\tfrac{Y}{Y_n}\right)\,\right]",
            font_size=24, color=TEXT_PRI)
        b_eq = MathTex(
            r"b^* &= 200\!\left[\, f\!\left(\tfrac{Y}{Y_n}\right)"
            r"- f\!\left(\tfrac{Z}{Z_n}\right)\,\right]",
            font_size=24, color=TEXT_PRI)
        lab_group = VGroup(lab_eq, a_eq, b_eq).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        lab_group.to_edge(DOWN, buff=0.3)
        self.play(Write(lab_group, run_time=2.5))
        self.wait(6)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 10 — CIELAB 3D COLOR SOLID
# ═══════════════════════════════════════════════════════════════════════
class CIELABSolidScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG
        ch = Text("XV.  The CIELAB Color Solid", font_size=42,
                  color=ACCENT_PURPLE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.add_fixed_in_frame_mobjects(ch)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        self.set_camera_orientation(phi=65 * DEGREES, theta=-40 * DEGREES)

        pts = generate_gamut_surface(res=16)
        lab_dots = VGroup()
        sc_ab = 0.032
        sc_L = 0.042

        for srgb in pts:
            L, a, bv = srgb_to_cielab(*srgb)
            x3d = a * sc_ab
            y3d = bv * sc_ab
            z3d = (L - 50) * sc_L
            d = Dot3D([x3d, y3d, z3d], radius=0.038,
                      color=rgb_to_hex(srgb))
            d.set_opacity(0.85)
            lab_dots.add(d)

        # Axes
        # for start, end, _ in [([[-3.5,0,0],[3.5,0,0]],),
        #                    ([[0,-3.5,0],[0,3.5,0]],),
        #                    ([[0,0,-2.5],[0,0,2.5]],)]:
        #     pass
        ax_a = Line3D([-3.5, 0, 0], [3.5, 0, 0], color=GRID_COL, stroke_width=1)
        ax_b = Line3D([0, -3.5, 0], [0, 3.5, 0], color=GRID_COL, stroke_width=1)
        ax_L = Line3D([0, 0, -2.5], [0, 0, 2.5], color=GRID_COL, stroke_width=1)

        al = Text("a*  (green↔red)", font_size=15, color="#cc66cc")
        al.move_to([3.8, 0, 0])
        bll = Text("b*  (blue↔yellow)", font_size=15, color="#ccaa33")
        bll.move_to([0, 3.8, 0])
        Ll = Text("L*  (lightness)", font_size=15, color=TEXT_PRI)
        Ll.move_to([0.4, 0, 2.8])
        for lbl in [al, bll, Ll]:
            self.add_fixed_orientation_mobjects(lbl)

        self.play(Create(ax_a), Create(ax_b), Create(ax_L), run_time=1.0)
        self.play(FadeIn(al), FadeIn(bll), FadeIn(Ll))
        self.play(FadeIn(lab_dots, lag_ratio=0.001, run_time=3))

        self.move_camera(theta=-40 * DEGREES + TAU, run_time=14, rate_func=smooth)

        note = Text("Notice the tilted, irregular shape — especially near blue",
                    font_size=20, color=ACCENT_PINK)
        note.to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(note)
        self.play(FadeIn(note))
        self.wait(4)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 12 — ΔE COLOR DIFFERENCE
# ═══════════════════════════════════════════════════════════════════════
class DeltaEScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XVI.  ΔE: Measuring Color Difference", font_size=42,
                  color=ACCENT_PURPLE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        # ── Act 1: Live ΔE demonstration ──
        ref_rgb = np.array([0.80, 0.35, 0.10])
        end_rgb = np.array([0.10, 0.55, 0.80])
        ref_lab = srgb_to_cielab(*ref_rgb)

        t_tracker = ValueTracker(0.0)

        def get_cand():
            t = t_tracker.get_value()
            return np.clip(ref_rgb * (1 - t) + end_rgb * t, 0, 1)

        ref_swatch = Rectangle(width=2.2, height=1.5,
                               fill_color=rgb_to_hex(ref_rgb),
                               fill_opacity=1, stroke_width=2,
                               stroke_color=TEXT_SEC)
        ref_swatch.move_to(LEFT * 3.5 + UP * 1.0)
        ref_lbl = Text("Reference", font_size=15, color=TEXT_SEC)
        ref_lbl.next_to(ref_swatch, DOWN, buff=0.12)

        cand_swatch = always_redraw(lambda: Rectangle(
            width=2.2, height=1.5,
            fill_color=rgb_to_hex(get_cand()),
            fill_opacity=1, stroke_width=2,
            stroke_color=TEXT_SEC).move_to(RIGHT * 3.5 + UP * 1.0))
        cand_lbl = Text("Candidate", font_size=15, color=TEXT_SEC)
        cand_lbl.next_to(RIGHT * 3.5 + UP * 0.22, DOWN, buff=0.02)

        de_label = Text("ΔE76 =", font_size=28, color=TEXT_PRI)
        de_label.move_to(UP * 1.0)

        de_num = DecimalNumber(0.0, num_decimal_places=1, font_size=56,
                               color=ACCENT_GREEN)
        de_num.next_to(de_label, RIGHT, buff=0.25)

        def update_de(mob):
            de = delta_e76(ref_lab, srgb_to_cielab(*get_cand()))
            mob.set_value(de)
            mob.set_color(ACCENT_GREEN if de < 1.0 else
                          ACCENT_YELLOW if de < 2.0 else MUTED_RED)

        de_num.add_updater(update_de)

        thresh_text = Text("imperceptible", font_size=16, color=ACCENT_GREEN)
        thresh_text.next_to(de_label, DOWN, buff=0.65)

        def update_thresh(mob):
            de = delta_e76(ref_lab, srgb_to_cielab(*get_cand()))
            if de < 1.0:
                mob.become(Text("imperceptible", font_size=16,
                                color=ACCENT_GREEN).move_to(mob))
            elif de < 2.0:
                mob.become(Text("perceptible to trained observers", font_size=16,
                                color=ACCENT_YELLOW).move_to(mob))
            else:
                mob.become(Text("clearly noticeable", font_size=16,
                                color=MUTED_RED).move_to(mob))

        thresh_text.add_updater(update_thresh)

        self.play(FadeIn(ref_swatch), FadeIn(ref_lbl))
        self.add(cand_swatch, de_num, thresh_text)
        self.play(FadeIn(de_label), FadeIn(cand_lbl))
        self.play(t_tracker.animate.set_value(1.0), run_time=4.0, rate_func=smooth)
        de_num.remove_updater(update_de)
        thresh_text.remove_updater(update_thresh)
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in [ref_swatch, ref_lbl, cand_swatch, cand_lbl,
                                         de_label, de_num, thresh_text]], run_time=1.2)

        # ── Act 2: Formulas ──
        form_title = Text("The math:", font_size=20, color=TEXT_PRI)
        form_title.move_to(UP * 1.9)
        self.play(FadeIn(form_title))

        eq76 = MathTex(
            r"\Delta E^*_{76} = \sqrt{(\Delta L^*)^2 + (\Delta a^*)^2 + (\Delta b^*)^2}",
            font_size=30, color=TEXT_PRI)
        eq76.next_to(form_title, DOWN, buff=0.35)
        self.play(Write(eq76, run_time=1.5))

        eq00_intro = Text("CIEDE2000 adds perceptual weighting (Sharma et al. 2005):",
                          font_size=17, color=TEXT_SEC)
        eq00_intro.next_to(eq76, DOWN, buff=0.4)
        self.play(FadeIn(eq00_intro))

        eq00 = MathTex(
            r"\Delta E_{00} = \sqrt{\!\left(\frac{\Delta L'}{k_L S_L}\right)^{\!2}"
            r"+ \left(\frac{\Delta C'}{k_C S_C}\right)^{\!2}"
            r"+ \left(\frac{\Delta H'}{k_H S_H}\right)^{\!2}"
            r"+ R_T \frac{\Delta C'}{k_C S_C}\frac{\Delta H'}{k_H S_H}}",
            font_size=22, color=TEXT_PRI)
        eq00.next_to(eq00_intro, DOWN, buff=0.3)
        self.play(Write(eq00, run_time=2.0))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in [form_title, eq76, eq00_intro, eq00]],
                  run_time=1.0)

        # ── Act 3: Engineering tolerance table ──
        tbl_title = Text("Industry tolerance standards:", font_size=20,
                         color=ACCENT_YELLOW)
        tbl_title.move_to(UP * 2.0)
        self.play(FadeIn(tbl_title))

        rows = [
            ("Printing / brand color matching", "ΔE₀₀  <  1.0", ACCENT_GREEN),
            ("Medical imaging displays", "ΔE₇₆  <  2.0", ACCENT_YELLOW),
            ("Consumer display calibration", "ΔE₀₀  <  3.0", ACCENT_ORANGE),
        ]
        row_grps = VGroup()
        for label, threshold, col in rows:
            lbl_t = Text(label, font_size=19, color=TEXT_PRI)
            thr_t = Text(threshold, font_size=19, color=col, weight=BOLD)
            row = VGroup(lbl_t, thr_t).arrange(RIGHT, buff=0.6)
            row_grps.add(row)
        row_grps.arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        row_grps.next_to(tbl_title, DOWN, buff=0.4)
        self.play(FadeIn(row_grps, lag_ratio=0.3, run_time=1.2))

        eng_box = RoundedRectangle(corner_radius=0.12, width=11, height=0.9,
                                   fill_color=PANEL, fill_opacity=0.9,
                                   stroke_color=ACCENT_TEAL, stroke_width=1.5)
        eng_box.to_edge(DOWN, buff=0.3)
        eng_note = Text(
            "For SE 423: convert camera sRGB → CIELAB, then apply ΔE tolerances in logic",
            font_size=18, color=ACCENT_TEAL)
        eng_note.move_to(eng_box)
        self.play(FadeIn(eng_box), FadeIn(eng_note))
        self.wait(4)

        # ── Act 4: CIEDE2000 — what each correction term does ─────────
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.2)

        de00_ch = Text("CIEDE2000: Why the Formula Is So Complex",
                       font_size=32, color=ACCENT_YELLOW, weight=BOLD)
        de00_ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(de00_ch, shift=DOWN * 0.2))

        corrections = [
            ("S_L  (lightness weighting)",
             "JNDs for L are larger near extremes (very dark / very light)",
             "  → downweight ΔL at L≈0 or L≈100",
             ACCENT_BLUE),
            ("S_C  (chroma weighting)",
             "JNDs grow with chroma — more saturated colors tolerate larger ΔC",
             "  → downweight ΔC at high chroma",
             ACCENT_GREEN),
            ("S_H  (hue weighting)",
             "Hue sensitivity is chroma-dependent — thin hue slices at low C",
             "  → downweight ΔH at low chroma",
             ACCENT_ORANGE),
            ("R_T  (rotation term)",
             "CIELAB has a known hue/chroma interaction error in the blue region (~275°)",
             "  → rotate ΔH/ΔC axes ≈8° for blues to fix it",
             ACCENT_PINK),
        ]
        rows = VGroup()
        for term, reason, effect, col in corrections:
            r = VGroup(
                Text(term, font_size=17, color=col, weight=BOLD),
                Text(reason, font_size=15, color=TEXT_PRI),
                Text(effect, font_size=15, color=TEXT_SEC),
            )
            r.arrange(DOWN, aligned_edge=LEFT, buff=0.06)
            rows.add(r)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        rows.move_to(DOWN * 0.3)
        for row in rows:
            self.play(FadeIn(row, lag_ratio=0.3), run_time=1.2)
        self.wait(3.5)

        de00_box = RoundedRectangle(corner_radius=0.12, width=11, height=1.0,
                                    fill_color=PANEL, fill_opacity=0.9,
                                    stroke_color=ACCENT_YELLOW, stroke_width=1.5)
        de00_box.to_edge(DOWN, buff=0.25)
        de00_txt = VGroup(
            Text("R_T is the 'blue problem fix'  —  connects directly to CIELAB's achilles heel",
                 font_size=17, color=ACCENT_YELLOW),
            Text("ΔE2000 ≈ 20% better than ΔE76 on Sharma's 3000-pair dataset",
                 font_size=17, color=TEXT_PRI),
        )
        de00_txt.arrange(DOWN, buff=0.08)
        de00_txt.move_to(de00_box)
        self.play(FadeIn(de00_box), FadeIn(de00_txt))
        self.wait(7)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 13 — CIELAB HUE PROBLEM (+ comparison with OKLab)
# ═══════════════════════════════════════════════════════════════════════
class CIELABProblemsScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XVII.  CIELAB's Achilles Heel", font_size=42,
                  color=ACCENT_PINK, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        title = Text("Hue ring at constant lightness & chroma",
                     font_size=20, color=TEXT_SEC)
        title.next_to(ch, DOWN, buff=0.35)
        self.play(FadeIn(title))

        # CIELAB hue ring
        center_l = LEFT * 3.2 + DOWN * 0.8
        ring_r = 2.0
        n_ring = 80
        C_star, L_star = 50, 65

        ring_lab = VGroup()
        for i in range(n_ring):
            angle = TAU * i / n_ring
            a_star = C_star * np.cos(angle)
            b_star = C_star * np.sin(angle)
            X, Y, Z = cielab_to_xyz(L_star, a_star, b_star)
            srgb = xyz_to_srgb(X, Y, Z)
            col = rgb_to_hex(srgb)
            d = Dot(center_l + ring_r * np.array([np.cos(angle),
                                                  np.sin(angle), 0]),
                    radius=0.14, color=col)
            ring_lab.add(d)

        lab_lbl = Text("CIELAB", font_size=20, color=ACCENT_PURPLE, weight=BOLD)
        lab_lbl.next_to(ring_lab, DOWN, buff=0.3)
        self.play(FadeIn(ring_lab, lag_ratio=0.01, run_time=1.5), FadeIn(lab_lbl))

        # OKLab hue ring
        center_r = RIGHT * 3.2 + DOWN * 0.8
        ring_ok = VGroup()
        for i in range(n_ring):
            angle = TAU * i / n_ring
            L_ok, C_ok = 0.72, 0.12
            col = oklab_to_hex(L_ok, C_ok * np.cos(angle), C_ok * np.sin(angle))
            d = Dot(center_r + ring_r * np.array([np.cos(angle),
                                                  np.sin(angle), 0]),
                    radius=0.14, color=col)
            ring_ok.add(d)

        ok_lbl = Text("OKLab", font_size=20, color=ACCENT_TEAL, weight=BOLD)
        ok_lbl.next_to(ring_ok, DOWN, buff=0.3)
        self.play(FadeIn(ring_ok, lag_ratio=0.01, run_time=1.5), FadeIn(ok_lbl))

        # Problem arrow
        prob_arrow = Arrow(
            center_l + DOWN * 1.2 + LEFT * 1.2,
            center_l + ring_r * np.array([np.cos(4.2), np.sin(4.2), 0]),
            color=ACCENT_PINK, stroke_width=2.5, buff=0.15)
        prob_txt = Text("Blue → purple\nshift!", font_size=14, color=ACCENT_PINK)
        prob_txt.next_to(prob_arrow.get_start(), DOWN, buff=0.1)
        self.play(GrowArrow(prob_arrow), FadeIn(prob_txt))

        # vs label
        vs = Text("vs", font_size=28, color=TEXT_SEC).move_to(DOWN * 0.8)
        self.play(FadeIn(vs))

        verdict = Text("CIELAB's hues are not uniform — OKLab fixes this",
                       font_size=22, color=ACCENT_TEAL)
        verdict.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(verdict, shift=UP * 0.2))
        self.wait(6)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 14 — LCh AND OKLch (CYLINDRICAL LAB)
# ═══════════════════════════════════════════════════════════════════════
class LChOKLchScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XVIII.  LCh and OKLch: Cylindrical Lab", font_size=42,
                  color=ACCENT_BLUE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        # ── Left panel: a*b* plane with C* and h geometry ──
        panel_lbl = Text("The a/b cartesian plane:", font_size=16, color=TEXT_SEC)
        panel_lbl.move_to(LEFT * 3.3 + UP * 1.5)
        self.play(FadeIn(panel_lbl))

        plane = Axes(
            x_range=[-1.2, 1.2, 0.5], y_range=[-1.2, 1.2, 0.5],
            x_length=3.6, y_length=3.6, tips=False,
            axis_config={"color": GRID_COL, "stroke_width": 1.5})
        plane.move_to(LEFT * 3.3 + DOWN * 0.25)
        xl = Text("a  (green↔red)", font_size=11, color=TEXT_SEC)
        xl.next_to(plane.x_axis, DOWN, buff=0.08)
        yl = Text("b  (blue↔yellow)", font_size=11, color=TEXT_SEC)
        yl.next_to(plane.y_axis, LEFT, buff=0.08)
        self.play(Create(plane), FadeIn(xl), FadeIn(yl), run_time=1.4)

        # Sample point
        pt_val = np.array([0.62, 0.68])
        origin_sc = plane.c2p(0, 0)
        pt_sc = plane.c2p(*pt_val)

        c_line = Line(origin_sc, pt_sc, color=ACCENT_TEAL, stroke_width=2.5)
        pt_dot = Dot(pt_sc, color=ACCENT_ORANGE, radius=0.10)
        c_lbl = MathTex(r"C^*", font_size=22, color=ACCENT_TEAL)
        c_lbl.move_to(plane.c2p(pt_val[0] * 0.48, pt_val[1] * 0.48)
                      + np.array([0.0, 0.25, 0]))

        h_angle = np.arctan2(pt_val[1], pt_val[0])
        arc = Arc(radius=0.52, start_angle=0, angle=h_angle,
                  color=ACCENT_YELLOW, stroke_width=2.2,
                  arc_center=origin_sc)
        h_lbl = MathTex(r"h", font_size=22, color=ACCENT_YELLOW)
        h_lbl.move_to(origin_sc + np.array([0.72, 0.22, 0]))

        self.play(Create(c_line), FadeIn(pt_dot), FadeIn(c_lbl))
        self.play(Create(arc), FadeIn(h_lbl))

        formulas = VGroup(
            MathTex(r"C^* = \sqrt{a^2 + b^2}", font_size=20, color=ACCENT_TEAL),
            MathTex(r"h   = \mathrm{atan2}(b,\,a) \bmod 360°",
                    font_size=20, color=ACCENT_YELLOW),
        ).arrange(DOWN, buff=0.22)
        formulas.move_to(LEFT * 3.3 + DOWN * 2.25)
        self.play(Write(formulas, run_time=1.2))

        # ── Right panel: CSS code + gradient sweeps ──
        css_title = Text("CSS Color Level 4 syntax:", font_size=16, color=ACCENT_GREEN)
        css_title.move_to(RIGHT * 2.6 + UP * 2.0)
        self.play(FadeIn(css_title))

        css_line = Text("oklch( 72%   0.15   240deg )",
                        font_size=17, color=ACCENT_GREEN)
        css_annot = Text("         L      C       h",
                         font_size=15, color=TEXT_SEC)
        css_block = VGroup(css_line, css_annot).arrange(DOWN, buff=0.06)
        css_block.next_to(css_title, DOWN, buff=0.18)
        self.play(FadeIn(css_block))

        sweep_lbl = Text("OKLch gradient sweeps:", font_size=14, color=TEXT_SEC)
        sweep_lbl.next_to(css_block, DOWN, buff=0.32)
        self.play(FadeIn(sweep_lbl))

        BAR_W = 5.2
        BAR_X = RIGHT * 2.6

        def hue_sweep(_c1, _c2, t):
            Lo, ao, bo = oklch_to_oklab(0.72, 0.15, t * 360)
            ro, go, bxo = oklab_to_linear_srgb(Lo, ao, bo)
            return np.clip(linear_to_srgb(np.clip([ro, go, bxo], 0, 1)), 0, 1)

        def chroma_sweep(_c1, _c2, t):
            Lo, ao, bo = oklch_to_oklab(0.72, t * 0.25, 240)
            ro, go, bxo = oklab_to_linear_srgb(Lo, ao, bo)
            return np.clip(linear_to_srgb(np.clip([ro, go, bxo], 0, 1)), 0, 1)

        def lightness_sweep(_c1, _c2, t):
            Lo, ao, bo = oklch_to_oklab(0.10 + t * 0.85, 0.12, 240)
            ro, go, bxo = oklab_to_linear_srgb(Lo, ao, bo)
            return np.clip(linear_to_srgb(np.clip([ro, go, bxo], 0, 1)), 0, 1)

        dummy = [0, 0, 0]
        bar_data = [
            (hue_sweep, "Hue sweep  (L=0.72, C=0.15)"),
            (chroma_sweep, "Chroma sweep  (L=0.72, h=240°)"),
            (lightness_sweep, "Lightness sweep  (C=0.12, h=240°)"),
        ]
        y_off = -0.15
        for fn, lbl_str in bar_data:
            bar = make_gradient_bar(fn, dummy, dummy, n=80, width=BAR_W, height=0.38)
            bar.move_to(BAR_X + UP * y_off)
            bar_lbl = Text(lbl_str, font_size=11, color=TEXT_SEC)
            bar_lbl.next_to(bar, DOWN, buff=0.06)
            self.play(FadeIn(bar, lag_ratio=0.004), FadeIn(bar_lbl), run_time=1.4)
            y_off -= 0.85

        verdict_box = RoundedRectangle(corner_radius=0.12, width=11, height=0.9,
                                       fill_color=PANEL, fill_opacity=0.9,
                                       stroke_color=ACCENT_GREEN, stroke_width=1.5)
        verdict_box.to_edge(DOWN, buff=0.28)
        verdict = Text(
            "oklch() is the CSS Color Level 4 standard — intuitive L, C, h controls",
            font_size=19, color=ACCENT_GREEN)
        verdict.move_to(verdict_box)
        self.play(FadeIn(verdict_box), FadeIn(verdict))
        self.wait(6)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 18 — GAMUT MAPPING
# ═══════════════════════════════════════════════════════════════════════
class GamutMappingScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XIX.  Gamut Mapping", font_size=44,
                  color=ACCENT_PINK, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        intro = Text(
            "What happens when a color can't be displayed in the target gamut?",
            font_size=20, color=TEXT_SEC)
        intro.next_to(ch, DOWN, buff=0.35)
        self.play(FadeIn(intro))

        # ── Act 1: P3 → sRGB, clipping artifacts ─────────────────────
        act1_title = Text("The Problem: Clipping destroys hue and lightness",
                          font_size=22, color=MUTED_RED)
        act1_title.move_to(UP * 1.2)
        self.play(FadeIn(act1_title))

        # P3 vivid red (1,0,0) in P3 primaries — slightly out of sRGB
        p3_red_lin = np.array([1.0, 0.0, 0.0])
        # Convert P3 to XYZ to sRGB (approximate: scale red channel)
        p3_red_srgb = np.array([1.09, -0.23, -0.13])  # out of gamut sRGB linear
        clipped = np.clip(p3_red_srgb, 0, 1)
        clipped_srgb = np.clip(linear_to_srgb(clipped), 0, 1)

        # Original (approximate P3 red as saturated sRGB for display)
        orig_col = rgb_to_hex(np.array([1.0, 0.05, 0.05]))
        clip_col = rgb_to_hex(clipped_srgb)

        labels = VGroup(
            Text("Original (P3)", font_size=17, color=TEXT_PRI),
            Text("Clipped (sRGB)", font_size=17, color=MUTED_RED),
        )
        swatches = VGroup(
            Rectangle(width=2.8, height=1.4, fill_color=orig_col,
                      fill_opacity=1, stroke_color=TEXT_SEC, stroke_width=1.5),
            Rectangle(width=2.8, height=1.4, fill_color=clip_col,
                      fill_opacity=1, stroke_color=TEXT_SEC, stroke_width=1.5),
        )
        swatches.arrange(RIGHT, buff=1.0)
        swatches.move_to(DOWN * 0.2)
        labels[0].next_to(swatches[0], DOWN, buff=0.12)
        labels[1].next_to(swatches[1], DOWN, buff=0.12)

        arrow = Arrow(swatches[0].get_right(), swatches[1].get_left(),
                      buff=0.15, color=MUTED_RED, stroke_width=2.5)
        clip_ann = Text("clip to [0,1]\nhue shifts!", font_size=15, color=MUTED_RED)
        clip_ann.next_to(arrow, UP, buff=0.1)
        self.play(FadeIn(swatches[0]), FadeIn(labels[0]))
        self.play(GrowArrow(arrow), FadeIn(clip_ann))
        self.play(FadeIn(swatches[1]), FadeIn(labels[1]))
        self.wait(3.5)

        # ── Act 2: OKLch binary search ────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects
                    if m is not ch], run_time=1.2)

        act2_title = Text("OKLch Chroma Reduction: Binary Search",
                          font_size=24, color=ACCENT_GREEN)
        act2_title.next_to(ch, DOWN, buff=0.45)
        self.play(FadeIn(act2_title))

        steps_txt = VGroup(
            Text("1. Find chroma C of the color in OKLch", font_size=19, color=TEXT_PRI),
            Text("2. C_lo = 0,  C_hi = C", font_size=19, color=TEXT_PRI),
            Text("3. Try C_mid = (C_lo + C_hi) / 2", font_size=19, color=ACCENT_TEAL),
            Text("4. In gamut? → C_lo = C_mid  else C_hi = C_mid", font_size=19, color=TEXT_PRI),
            Text("5. Repeat until C_hi − C_lo < ε", font_size=19, color=TEXT_PRI),
            Text("6. Same L and h throughout → hue preserved", font_size=19, color=ACCENT_GREEN),
        )
        steps_txt.arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        steps_txt.move_to(LEFT * 1.5 + DOWN * 0.3)
        for step in steps_txt:
            self.play(FadeIn(step), run_time=0.35)
        self.wait(2.5)

        # ── Act 3: 3 swatches comparison ─────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects
                    if m is not ch], run_time=1.2)

        act3_title = Text("Result: Gamut Mapping vs Clipping",
                          font_size=26, color=ACCENT_GREEN)
        act3_title.next_to(ch, DOWN, buff=0.45)
        self.play(FadeIn(act3_title))

        # Compute gamut-mapped version of p3_red_srgb
        L_ok, a_ok, b_ok = linear_srgb_to_oklab(
            *np.clip(p3_red_srgb, -0.5, 2.0))
        Lm, am, bm = gamut_map_oklch(L_ok, a_ok, b_ok)
        rm, gm, bxm = oklab_to_linear_srgb(Lm, am, bm)
        mapped_col = rgb_to_hex(np.clip(linear_to_srgb(
            np.clip([rm, gm, bxm], 0, 1)), 0, 1))

        sw_labels = ["Original\n(P3)", "Clipped\n(sRGB)", "Chroma-Reduced\n(OKLch)"]
        sw_colors = [orig_col, clip_col, mapped_col]
        sw_text_colors = [TEXT_PRI, MUTED_RED, ACCENT_GREEN]

        sw_group = VGroup()
        for i, (col, lbl, tcol) in enumerate(zip(sw_colors, sw_labels, sw_text_colors)):
            sw = Rectangle(width=2.6, height=2.0, fill_color=col,
                           fill_opacity=1, stroke_color=TEXT_SEC, stroke_width=1.5)
            sw_lbl = Text(lbl, font_size=16, color=tcol, line_spacing=1.1)
            sw_lbl.next_to(sw, DOWN, buff=0.15)
            sw_group.add(VGroup(sw, sw_lbl))
        sw_group.arrange(RIGHT, buff=0.8)
        sw_group.move_to(DOWN * 0.3)
        self.play(FadeIn(sw_group, lag_ratio=0.3), run_time=1.2)

        gm_box = RoundedRectangle(corner_radius=0.12, width=11, height=1.0,
                                  fill_color=PANEL, fill_opacity=0.9,
                                  stroke_color=ACCENT_GREEN, stroke_width=1.5)
        gm_box.to_edge(DOWN, buff=0.3)
        gm_txt = Text(
            "Gamut mapping preserves hue and lightness  ·  clipping distorts both",
            font_size=19, color=ACCENT_GREEN)
        gm_txt.move_to(gm_box)
        self.play(FadeIn(gm_box), FadeIn(gm_txt))
        self.wait(7)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 19 — OKLAB DERIVATION (full pipeline + optimization story)
# ═══════════════════════════════════════════════════════════════════════
class OKLabDerivationScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XX.  Deriving OKLab  (2020)", font_size=44,
                  color=ACCENT_TEAL, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        credit = Text("Björn Ottosson — \"An OK Lab color space\"",
                      font_size=18, color=TEXT_SEC)
        credit.next_to(ch, DOWN, buff=0.3)
        self.play(FadeIn(credit))

        # Idea
        idea = Text("Take the simple IPT structure, but optimize its matrices "
                    "against CAM16 perceptual data",
                    font_size=18, color=ACCENT_YELLOW)
        idea.next_to(credit, DOWN, buff=0.35)
        self.play(FadeIn(idea, shift=DOWN * 0.1))

        # Pipeline boxes
        steps = [
            ("Linear\nsRGB", ACCENT_ORANGE),
            ("LMS\n(cone resp.)", ACCENT_YELLOW),
            ("L'M'S'\n(cube root)", ACCENT_GREEN),
            ("OKLab\n(L, a, b)", ACCENT_TEAL),
        ]
        boxes = VGroup()
        arrows = VGroup()
        for label, col in steps:
            box = RoundedRectangle(corner_radius=0.1, width=2.2, height=1.3,
                                   fill_color=PANEL, fill_opacity=0.95,
                                   stroke_color=col, stroke_width=2.5)
            txt = Text(label, font_size=13, color=col, line_spacing=1.0)
            txt.move_to(box)
            boxes.add(VGroup(box, txt))
        boxes.arrange(RIGHT, buff=0.55).move_to(UP * 0.1)
        for i in range(len(boxes) - 1):
            arr = Arrow(boxes[i].get_right(), boxes[i + 1].get_left(),
                        buff=0.07, color=TEXT_SEC, stroke_width=2.5,
                        max_tip_length_to_length_ratio=0.15)
            arrows.add(arr)

        for i, box in enumerate(boxes):
            self.play(FadeIn(box, scale=0.85), run_time=1.0)
            if i < len(arrows):
                self.play(GrowArrow(arrows[i]), run_time=0.2)

        # Matrix & operation labels
        m1 = MathTex(r"\mathbf{M_1}", font_size=22, color=ACCENT_YELLOW)
        m1.next_to(arrows[0], UP, buff=0.1)
        cbrt = MathTex(r"(\cdot)^{1/3}", font_size=22, color=ACCENT_GREEN)
        cbrt.next_to(arrows[1], UP, buff=0.1)
        m2 = MathTex(r"\mathbf{M_2}", font_size=22, color=ACCENT_TEAL)
        m2.next_to(arrows[2], UP, buff=0.1)
        self.play(FadeIn(m1), FadeIn(cbrt), FadeIn(m2))

        # Full math
        eq1 = MathTex(
            r"\begin{pmatrix}l\\m\\s\end{pmatrix}=\mathbf{M_1}"
            r"\begin{pmatrix}r\\g\\b\end{pmatrix}",
            font_size=20, color=TEXT_PRI)
        eq2 = MathTex(
            r"\begin{pmatrix}l'\\m'\\s'\end{pmatrix}="
            r"\begin{pmatrix}\!\sqrt[3]{l}\\\!\sqrt[3]{m}\\\!\sqrt[3]{s}\end{pmatrix}",
            font_size=20, color=TEXT_PRI)
        eq3 = MathTex(
            r"\begin{pmatrix}L\\a\\b\end{pmatrix}=\mathbf{M_2}"
            r"\begin{pmatrix}l'\\m'\\s'\end{pmatrix}",
            font_size=20, color=TEXT_PRI)
        eqs = VGroup(eq1, eq2, eq3).arrange(RIGHT, buff=0.4).move_to(DOWN * 1.3)
        self.play(Write(eqs, run_time=2))

        # Optimization note
        opt_box = RoundedRectangle(corner_radius=0.1, width=10, height=0.9,
                                   fill_color=PANEL, fill_opacity=0.9,
                                   stroke_color=ACCENT_TEAL, stroke_width=1.5)
        opt_box.to_edge(DOWN, buff=0.3)
        opt = Text(
            "M₁ and M₂ optimized against CAM16-UCS lightness, chroma, and "
            "Ebner-Fairchild hue data",
            font_size=16, color=ACCENT_TEAL)
        opt.move_to(opt_box)
        self.play(FadeIn(opt_box), FadeIn(opt))
        self.wait(4)

        # ── Act 3: IPT — Same architecture, different numbers ─────────
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.2)

        ipt_title = Text("IPT Color Space  (Ebner & Fairchild 1998)",
                         font_size=36, color=ACCENT_PURPLE, weight=BOLD)
        ipt_title.to_edge(UP, buff=0.45)
        self.play(FadeIn(ipt_title, shift=DOWN * 0.2))

        ipt_intro = Text(
            "Same M₁ → power law → M₂ architecture as OKLab — but with exponent 0.43",
            font_size=20, color=TEXT_PRI)
        ipt_intro.next_to(ipt_title, DOWN, buff=0.4)
        self.play(FadeIn(ipt_intro))

        # Side-by-side matrix comparison
        ipt_col = VGroup(
            Text("IPT", font_size=22, color=ACCENT_PURPLE, weight=BOLD),
            Text("XYZ → LMS (Hunt-Pointer-Estevez)", font_size=16, color=TEXT_SEC),
            MathTex(r"(\cdot)^{0.43}", font_size=24, color=ACCENT_PURPLE),
            Text("→ M₂ → (I, P, T)", font_size=16, color=TEXT_SEC),
        )
        ipt_col.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        ipt_col.move_to(LEFT * 3.0 + DOWN * 0.3)

        oklab_col = VGroup(
            Text("OKLab", font_size=22, color=ACCENT_TEAL, weight=BOLD),
            Text("linear sRGB → LMS (M₁)", font_size=16, color=TEXT_SEC),
            MathTex(r"(\cdot)^{1/3}", font_size=24, color=ACCENT_TEAL),
            Text("→ M₂ → (L, a, b)", font_size=16, color=TEXT_SEC),
        )
        oklab_col.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        oklab_col.move_to(RIGHT * 2.5 + DOWN * 0.3)

        vs_txt = Text("vs", font_size=28, color=TEXT_SEC)
        vs_txt.move_to(ORIGIN + DOWN * 0.3)

        self.play(FadeIn(ipt_col, lag_ratio=0.2), FadeIn(vs_txt),
                  FadeIn(oklab_col, lag_ratio=0.2), run_time=1.2)

        ipt_box = RoundedRectangle(corner_radius=0.12, width=11, height=0.9,
                                   fill_color=PANEL, fill_opacity=0.9,
                                   stroke_color=ACCENT_PURPLE, stroke_width=1.5)
        ipt_box.to_edge(DOWN, buff=0.3)
        ipt_txt = Text(
            "OKLab = same architecture as IPT, but exponent 1/3 and re-optimized matrices",
            font_size=18, color=ACCENT_PURPLE)
        ipt_txt.move_to(ipt_box)
        self.play(FadeIn(ipt_box), FadeIn(ipt_txt))
        self.wait(5)

        # ── Act 4: CAM16 — The ground truth ──────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.2)

        cam_title = Text("CAM16: The Perceptual Ground Truth",
                         font_size=36, color=ACCENT_YELLOW, weight=BOLD)
        cam_title.to_edge(UP, buff=0.45)
        self.play(FadeIn(cam_title, shift=DOWN * 0.2))

        cam_items = VGroup(
            Text("CAM16 = Color Appearance Model 2016", font_size=20,
                 color=ACCENT_YELLOW, weight=BOLD),
            Text("Models: chromatic adaptation · luminance adaptation · surround",
                 font_size=19, color=TEXT_PRI),
            Text("Predicts perceived lightness, chroma, hue, colorfulness, brightness",
                 font_size=19, color=TEXT_PRI),
            Text("Used as ground truth for OKLab's M₁ and M₂ optimization",
                 font_size=19, color=ACCENT_TEAL),
        )
        cam_items.arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        cam_items.move_to(LEFT * 1.0 + UP * 0.2)
        for item in cam_items:
            self.play(FadeIn(item), run_time=1.2)

        cam_box = RoundedRectangle(corner_radius=0.12, width=11, height=1.1,
                                   fill_color=PANEL, fill_opacity=0.9,
                                   stroke_color=ACCENT_YELLOW, stroke_width=1.5)
        cam_box.to_edge(DOWN, buff=0.25)
        cam_txt = VGroup(
            Text("OKLab accuracy ≈ CAM16  ·  cost = two matrix multiplies + cube root",
                 font_size=17, color=TEXT_PRI),
            Text('"OKLab = CAM16\'s predictive power at matrix-multiply cost"',
                 font_size=17, color=ACCENT_YELLOW),
        )
        cam_txt.arrange(DOWN, buff=0.1)
        cam_txt.move_to(cam_box)
        self.play(FadeIn(cam_box), FadeIn(cam_txt))
        self.wait(5)

        # ── Act 5: OKLab's known limitations ─────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.2)

        lim_title = Text("OKLab: Known Limitations", font_size=38,
                         color=MUTED_RED, weight=BOLD)
        lim_title.to_edge(UP, buff=0.45)
        self.play(FadeIn(lim_title, shift=DOWN * 0.2))

        lim_intro = Text("Credit: Björn Ottosson acknowledges these on his blog",
                         font_size=18, color=TEXT_SEC)
        lim_intro.next_to(lim_title, DOWN, buff=0.3)
        self.play(FadeIn(lim_intro))

        limitations = [
            ("Fixed viewing condition",
             "Assumes D65, moderate surround — doesn't adapt to scene luminance like CAM16",
             ACCENT_ORANGE),
            ("Helmholtz-Kohlrausch not modeled",
             "Saturated colors appear brighter than L predicts — OKLab L is unreliable at high C",
             ACCENT_PINK),
            ("Implicit gamut boundary",
             "No analytic formula for the sRGB boundary — must probe by trial conversion",
             ACCENT_YELLOW),
            ("Uniformity degrades near boundary",
             "At very high chroma, perceptual uniformity decreases — works best near sRGB",
             MUTED_RED),
        ]
        lim_rows = VGroup()
        for heading, desc, col in limitations:
            row = VGroup(
                Text("⚠ " + heading, font_size=19, color=col, weight=BOLD),
                Text(desc, font_size=16, color=TEXT_PRI),
            )
            row.arrange(DOWN, aligned_edge=LEFT, buff=0.06)
            lim_rows.add(row)
        lim_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        lim_rows.move_to(DOWN * 0.3)
        for row in lim_rows:
            self.play(FadeIn(row, lag_ratio=0.2), run_time=1.2)
        self.wait(3.5)

        lim_box = RoundedRectangle(corner_radius=0.12, width=11, height=1.0,
                                   fill_color=PANEL, fill_opacity=0.9,
                                   stroke_color=ACCENT_GREEN, stroke_width=1.5)
        lim_box.to_edge(DOWN, buff=0.25)
        lim_txt = Text(
            "For most applications OKLab is the right choice  ·  use CAM16 when viewing conditions vary",
            font_size=17, color=ACCENT_GREEN)
        lim_txt.move_to(lim_box)
        self.play(FadeIn(lim_box), FadeIn(lim_txt))
        self.wait(7)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE — PERCEPTUAL PHENOMENA THAT BREAK SIMPLE MODELS
# ═══════════════════════════════════════════════════════════════════════
class PerceptualPhenomenaScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XXI.  Perceptual Phenomena Beyond OKLab",
                  font_size=38, color=ACCENT_PURPLE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        # ── Act 1: Helmholtz-Kohlrausch effect ────────────────────────
        hk_title = Text("Helmholtz-Kohlrausch Effect",
                        font_size=28, color=ACCENT_PINK, weight=BOLD)
        hk_title.next_to(ch, DOWN, buff=0.4)
        self.play(FadeIn(hk_title))

        hk_desc = Text(
            "Saturated colors appear brighter than their luminance would predict",
            font_size=19, color=TEXT_PRI)
        hk_desc.next_to(hk_title, DOWN, buff=0.3)
        self.play(FadeIn(hk_desc))

        # Show two swatches with same OKLab L but different C
        L_val = 0.60
        c_low_rgb = oklab_to_hex(L_val, 0.01, 0.01)  # near-achromatic
        c_hi_rgb = oklab_to_hex(L_val, 0.18, -0.06)  # vivid red-orange
        c_hi2_rgb = oklab_to_hex(L_val, -0.05, 0.18)  # vivid green-yellow

        sw_group = VGroup()
        for col, lbl, desc in [
            (c_low_rgb, "L=0.60, C≈0.01\n(achromatic)", "Appears normal"),
            (c_hi_rgb, "L=0.60, C≈0.19\n(vivid red)", "Appears BRIGHTER!"),
            (c_hi2_rgb, "L=0.60, C≈0.19\n(vivid green)", "Appears BRIGHTER!"),
        ]:
            sw = Rectangle(width=2.2, height=1.5, fill_color=col,
                           fill_opacity=1, stroke_color=TEXT_SEC, stroke_width=1.5)
            sw_lbl = Text(lbl, font_size=14, color=TEXT_PRI, line_spacing=1.1)
            sw_lbl.next_to(sw, DOWN, buff=0.1)
            sw_desc = Text(desc, font_size=14, color=TEXT_SEC)
            sw_desc.next_to(sw_lbl, DOWN, buff=0.05)
            sw_group.add(VGroup(sw, sw_lbl, sw_desc))
        sw_group.arrange(RIGHT, buff=0.7)
        sw_group.move_to(DOWN * 0.5)
        self.play(FadeIn(sw_group, lag_ratio=0.3), run_time=1.0)

        hk_note = Text(
            "All three have identical OKLab L = 0.60  —  but vivid colors look lighter",
            font_size=17, color=ACCENT_PINK)
        hk_note.move_to(DOWN * 2.6)
        self.play(FadeIn(hk_note))
        self.wait(4)

        # ── Act 2: Abney effect ───────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects
                    if m is not ch], run_time=1.2)

        abney_title = Text("The Abney Effect",
                           font_size=28, color=ACCENT_BLUE, weight=BOLD)
        abney_title.next_to(ch, DOWN, buff=0.4)
        self.play(FadeIn(abney_title))

        abney_desc = Text(
            "As you desaturate a color toward white, its apparent hue shifts",
            font_size=19, color=TEXT_PRI)
        abney_desc.next_to(abney_title, DOWN, buff=0.3)
        self.play(FadeIn(abney_desc))

        # Show desaturation path: pure blue → white, show hue shift
        n = 8
        blue_row = VGroup()
        for i in range(n):
            t = i / (n - 1)
            # Pure blue in OKLab desaturating to white
            L = 0.40 + t * 0.55
            C = 0.18 * (1 - t)
            h = 264  # blue hue angle
            Lo, ao, bo = oklch_to_oklab(L, C, h)
            ro, go, bxo = oklab_to_linear_srgb(Lo, ao, bo)
            col = rgb_to_hex(np.clip(linear_to_srgb(np.clip([ro, go, bxo], 0, 1)), 0, 1))
            sw = Rectangle(width=1.1, height=0.9, fill_color=col,
                           fill_opacity=1, stroke_width=0)
            blue_row.add(sw)
        blue_row.arrange(RIGHT, buff=0.05)
        blue_row.move_to(LEFT * 2.0 + DOWN * 0.2)
        blue_lbl = Text("Blue desaturating → note purple shift", font_size=15, color=ACCENT_BLUE)
        blue_lbl.next_to(blue_row, DOWN, buff=0.1)
        self.play(FadeIn(blue_row, lag_ratio=0.1), FadeIn(blue_lbl), run_time=1.0)

        # Yellow desaturation
        yellow_row = VGroup()
        for i in range(n):
            t = i / (n - 1)
            L = 0.70 + t * 0.25
            C = 0.15 * (1 - t)
            h = 100  # yellow
            Lo, ao, bo = oklch_to_oklab(L, C, h)
            ro, go, bxo = oklab_to_linear_srgb(Lo, ao, bo)
            col = rgb_to_hex(np.clip(linear_to_srgb(np.clip([ro, go, bxo], 0, 1)), 0, 1))
            sw = Rectangle(width=1.1, height=0.9, fill_color=col,
                           fill_opacity=1, stroke_width=0)
            yellow_row.add(sw)
        yellow_row.arrange(RIGHT, buff=0.05)
        yellow_row.move_to(LEFT * 2.0 + DOWN * 1.6)
        yellow_lbl = Text("Yellow desaturating → note green shift", font_size=15, color=ACCENT_YELLOW)
        yellow_lbl.next_to(yellow_row, DOWN, buff=0.1)
        self.play(FadeIn(yellow_row, lag_ratio=0.1), FadeIn(yellow_lbl), run_time=1.0)

        abney_note = Text(
            "OKLab improves hue linearity vs CIELAB  —  but Abney effect persists at extremes",
            font_size=16, color=ACCENT_BLUE)
        abney_note.move_to(RIGHT * 2.5 + DOWN * 0.8)
        self.play(FadeIn(abney_note))
        self.wait(3.5)

        # ── Act 3: Hunt effect ────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects
                    if m is not ch], run_time=1.2)

        hunt_title = Text("The Hunt Effect",
                          font_size=28, color=ACCENT_YELLOW, weight=BOLD)
        hunt_title.next_to(ch, DOWN, buff=0.4)
        self.play(FadeIn(hunt_title))

        hunt_items = VGroup(
            Text("Colors appear MORE colorful at HIGHER luminance levels", font_size=20,
                 color=ACCENT_YELLOW),
            Text("Why: cone responses compress differently at different adaptation levels",
                 font_size=18, color=TEXT_PRI),
            Text("A vivid red outdoors (10000 lux) appears more saturated than indoors (500 lux)",
                 font_size=18, color=TEXT_SEC),
            Text("", font_size=6),
            Text("This is why CAM16 models viewing conditions:", font_size=18, color=ACCENT_TEAL),
            Text("  L_A (adapting luminance) · F_L (luminance factor) · surround type",
                 font_size=17, color=ACCENT_TEAL),
            Text("OKLab assumes fixed viewing condition → approximate for extreme luminances",
                 font_size=18, color=MUTED_RED),
        )
        hunt_items.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        hunt_items.move_to(DOWN * 0.3)
        for item in hunt_items:
            self.play(FadeIn(item), run_time=0.35)
        self.wait(3.5)

        phenom_box = RoundedRectangle(corner_radius=0.12, width=11, height=1.0,
                                      fill_color=PANEL, fill_opacity=0.9,
                                      stroke_color=ACCENT_PURPLE, stroke_width=1.5)
        phenom_box.to_edge(DOWN, buff=0.25)
        phenom_txt = Text(
            "These phenomena explain why no simple 3D formula perfectly models human color perception",
            font_size=17, color=ACCENT_PURPLE)
        phenom_txt.move_to(phenom_box)
        self.play(FadeIn(phenom_box), FadeIn(phenom_txt))
        self.wait(7)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE — OKLAB 3D COLOR SOLID
# ═══════════════════════════════════════════════════════════════════════
class OKLabSolidScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG
        ch = Text("XXII.  The OKLab Color Solid", font_size=42,
                  color=ACCENT_TEAL, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.add_fixed_in_frame_mobjects(ch)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        self.set_camera_orientation(phi=65 * DEGREES, theta=-40 * DEGREES)

        pts = generate_gamut_surface(res=16)
        ok_dots = VGroup()
        sc = 7.5

        for srgb in pts:
            lin = srgb_to_linear(np.array(srgb))
            L, a, bv = linear_srgb_to_oklab(*lin)
            x3d = a * sc
            y3d = bv * sc
            z3d = (L - 0.5) * sc
            d = Dot3D([x3d, y3d, z3d], radius=0.038,
                      color=rgb_to_hex(srgb))
            d.set_opacity(0.85)
            ok_dots.add(d)

        ax_a = Line3D([-3, 0, 0], [3, 0, 0], color=GRID_COL, stroke_width=1)
        ax_b = Line3D([0, -3, 0], [0, 3, 0], color=GRID_COL, stroke_width=1)
        ax_L = Line3D([0, 0, -4], [0, 0, 4], color=GRID_COL, stroke_width=1)

        al = Text("a", font_size=18, color="#cc6666").move_to([3.3, 0, 0])
        bll = Text("b", font_size=18, color="#ccaa33").move_to([0, 3.3, 0])
        Ll = Text("L", font_size=18, color=TEXT_PRI).move_to([0.3, 0, 4.3])
        for lbl in [al, bll, Ll]:
            self.add_fixed_orientation_mobjects(lbl)

        self.play(Create(ax_a), Create(ax_b), Create(ax_L), run_time=1.0)
        self.play(FadeIn(al), FadeIn(bll), FadeIn(Ll))
        self.play(FadeIn(ok_dots, lag_ratio=0.001, run_time=3))

        self.move_camera(theta=-40 * DEGREES + TAU, run_time=14, rate_func=smooth)

        note = Text("Smoother, more symmetric — perceptually uniform in all directions",
                    font_size=20, color=ACCENT_TEAL)
        note.to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(note)
        self.play(FadeIn(note))
        self.wait(4)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 14 — SIDE-BY-SIDE 3D COMPARISON
# ═══════════════════════════════════════════════════════════════════════
class ColorSpaceComparisonScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG
        ch = Text("XXIII.  3D Comparison: RGB · CIELAB · OKLab", font_size=38,
                  color=ACCENT_GREEN, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.add_fixed_in_frame_mobjects(ch)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        # phi=75° gives a slight overhead angle to show 3D depth;
        # theta=90° puts the camera on the +Y axis so the X-axis (along which
        # the three solids are arranged) runs perfectly left–right in the frame.
        self.set_camera_orientation(phi=75 * DEGREES, theta=90 * DEGREES)

        pts = generate_gamut_volume(res=8)
        sc_rgb = 3.0
        sc_lab = 0.025
        sc_labL = 0.033
        sc_ok = 5.5

        # ── Auto-fit: measure x half-widths from raw data, then scale ──
        _rgb_hw = max(abs((s[0] - 0.5) * sc_rgb) for s in pts)
        _lab_hw = max(abs(srgb_to_cielab(*s)[1] * sc_lab) for s in pts)
        _ok_hw = max(abs(linear_srgb_to_oklab(
            *srgb_to_linear(np.array(s)))[1] * sc_ok)
                     for s in pts)
        max_hw = max(_rgb_hw, _lab_hw, _ok_hw)
        GAP = 0.9  # gap between solids
        target_hw = (13.0 - 2 * GAP) / 6  # half-width each solid
        fit = target_hw / max_hw  # uniform scale factor
        offset = 2 * target_hw + GAP  # centre-to-centre distance

        # ── Build groups centred at origin, then shift to final positions ──
        # Camera on +Y (theta=90°) ⟹ +X appears LEFT in the frame, so:
        #   RGB → +offset (left)  ·  CIELAB → 0 (centre)  ·  OKLab → –offset (right)

        rgb_grp = VGroup()
        for s in pts:
            x, y, z = (s - 0.5) * sc_rgb * fit
            d = Dot3D([x, y, z], radius=0.03, color=rgb_to_hex(s))
            d.set_opacity(0.75)
            rgb_grp.add(d)
        rgb_grp.shift(np.array([+offset, 0, 0]))

        lab_grp = VGroup()
        for s in pts:
            L, a, bv = srgb_to_cielab(*s)
            d = Dot3D([a * sc_lab * fit, bv * sc_lab * fit, (L - 50) * sc_labL * fit],
                      radius=0.03, color=rgb_to_hex(s))
            d.set_opacity(0.75)
            lab_grp.add(d)
        # lab_grp stays at origin

        ok_grp = VGroup()
        for s in pts:
            lin = srgb_to_linear(np.array(s))
            L, a, bv = linear_srgb_to_oklab(*lin)
            d = Dot3D([a * sc_ok * fit, bv * sc_ok * fit, (L - 0.5) * sc_ok * fit],
                      radius=0.03, color=rgb_to_hex(s))
            d.set_opacity(0.75)
            ok_grp.add(d)
        ok_grp.shift(np.array([-offset, 0, 0]))

        # Label z: place below the tallest solid's bottom edge
        lbl_z = -(max(0.5 * sc_rgb, 50 * sc_labL, 0.5 * sc_ok) * fit + 0.45)

        rgb_lbl = Text("RGB", font_size=18, color=ACCENT_BLUE, weight=BOLD)
        rgb_lbl.move_to([+offset, 0, lbl_z])
        self.add_fixed_orientation_mobjects(rgb_lbl)

        lab_lbl = Text("CIELAB", font_size=18, color=ACCENT_PURPLE, weight=BOLD)
        lab_lbl.move_to([0, 0, lbl_z])
        self.add_fixed_orientation_mobjects(lab_lbl)

        ok_lbl = Text("OKLab", font_size=18, color=ACCENT_TEAL, weight=BOLD)
        ok_lbl.move_to([-offset, 0, lbl_z])
        self.add_fixed_orientation_mobjects(ok_lbl)

        self.play(FadeIn(rgb_grp, lag_ratio=0.001, run_time=1.5), FadeIn(rgb_lbl))
        self.play(FadeIn(lab_grp, lag_ratio=0.001, run_time=1.5), FadeIn(lab_lbl))
        self.play(FadeIn(ok_grp, lag_ratio=0.001, run_time=1.5), FadeIn(ok_lbl))

        # Rotate each solid independently around its own Z axis
        self.play(
            Rotate(rgb_grp, angle=TAU, axis=np.array([0, 0, 1]),
                   about_point=np.array([+offset, 0, 0]), rate_func=smooth),
            Rotate(lab_grp, angle=TAU, axis=np.array([0, 0, 1]),
                   about_point=ORIGIN, rate_func=smooth),
            Rotate(ok_grp, angle=TAU, axis=np.array([0, 0, 1]),
                   about_point=np.array([-offset, 0, 0]), rate_func=smooth),
            run_time=18,
        )

        note = Text("Same colors, three shapes — OKLab is the most regular",
                    font_size=20, color=ACCENT_GREEN)
        note.to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(note)
        self.play(FadeIn(note))
        self.wait(4)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 15 — GRADIENT COMPARISON (hero scene)
# ═══════════════════════════════════════════════════════════════════════
class GradientComparisonScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XXIV.  Gradient Quality Test", font_size=42,
                  color=ACCENT_GREEN, weight=BOLD)
        ch.to_edge(UP, buff=0.4)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        pairs = [
            ([0.0, 0.4, 1.0], [1.0, 0.8, 0.0], "Blue → Yellow"),
            ([1.0, 0.0, 0.33], [0.0, 0.87, 0.67], "Red → Teal"),
            ([1.0, 1.0, 1.0], [0.0, 0.2, 0.73], "White → Blue"),
        ]
        methods = [
            ("sRGB", srgb_blend, ACCENT_ORANGE),
            ("HSV", hsv_blend_fn, ACCENT_PINK),
            ("OKLab", oklab_blend_fn, ACCENT_TEAL),
            ("OKLch", oklch_blend_fn, ACCENT_GREEN),
        ]

        all_items = VGroup()
        y_pos = 2.2
        for c1, c2, pair_name in pairs:
            pl = Text(pair_name, font_size=15, color=TEXT_SEC)
            pl.move_to(LEFT * 6 + UP * y_pos)
            all_items.add(pl)
            for j, (mn, fn, mc) in enumerate(methods):
                bar = make_gradient_bar(fn, c1, c2, n=90, width=9.5, height=0.32)
                bar.move_to(RIGHT * 0.3 + UP * (y_pos - j * 0.40))
                lbl = Text(mn, font_size=12, color=mc,
                           weight=BOLD if mn in ("OKLab", "OKLch") else NORMAL)
                lbl.next_to(bar, LEFT, buff=0.12)
                all_items.add(bar, lbl)
            y_pos -= 1.85

        idx = 0
        for _ in range(3):
            self.play(FadeIn(all_items[idx]), run_time=0.25)
            idx += 1
            for _ in range(4):
                self.play(FadeIn(all_items[idx], lag_ratio=0.003),
                          FadeIn(all_items[idx + 1]), run_time=0.45)
                idx += 2
            self.wait(0.2)

        verdict = Text("OKLab / OKLch: no mud, no hue shifts, perceptually smooth",
                       font_size=22, color=ACCENT_TEAL)
        verdict.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(verdict, shift=UP * 0.2))
        self.wait(6)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE — Y'CbCr AND CHROMA SUBSAMPLING
# ═══════════════════════════════════════════════════════════════════════
class YCbCrScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XXV.  Y'CbCr and Chroma Subsampling",
                  font_size=40, color=ACCENT_BLUE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        intro = Text(
            "Nearly all video (H.264, H.265, AV1) stores color as Y'CbCr, not RGB",
            font_size=19, color=TEXT_SEC)
        intro.next_to(ch, DOWN, buff=0.3)
        self.play(FadeIn(intro))

        # ── Act 1: The equations ──────────────────────────────────────
        eq_title = Text("BT.709 Y'CbCr Equations  (prime = gamma-encoded)",
                        font_size=22, color=ACCENT_BLUE)
        eq_title.move_to(UP * 1.2)
        self.play(FadeIn(eq_title))

        eqs = VGroup(
            MathTex(r"Y' = 0.2126\,R' + 0.7152\,G' + 0.0722\,B'",
                    font_size=24, color=TEXT_PRI),
            MathTex(r"C_b = \frac{B' - Y'}{1.8556}",
                    font_size=24, color=ACCENT_TEAL),
            MathTex(r"C_r = \frac{R' - Y'}{1.5748}",
                    font_size=24, color=ACCENT_ORANGE),
        )
        eqs.arrange(DOWN, buff=0.35)
        eqs.move_to(LEFT * 2.5 + DOWN * 0.2)
        self.play(Write(eqs, run_time=1.5))

        # Note: primes mean gamma-encoded
        prime_note = VGroup(
            Text("R'G'B' = sRGB (gamma-encoded)", font_size=16, color=TEXT_SEC),
            Text("Y' = luma (not luminance!)", font_size=16, color=ACCENT_YELLOW),
            Text("Cb = blue-difference chroma", font_size=16, color=ACCENT_TEAL),
            Text("Cr = red-difference chroma", font_size=16, color=ACCENT_ORANGE),
        )
        prime_note.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        prime_note.move_to(RIGHT * 3.0 + DOWN * 0.2)
        self.play(FadeIn(prime_note, lag_ratio=0.3), run_time=1.8)
        self.wait(3.5)

        # ── Act 2: Channels visualized ────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects
                    if m is not ch], run_time=1.2)

        chan_title = Text("What the Three Channels Look Like",
                          font_size=26, color=ACCENT_BLUE)
        chan_title.next_to(ch, DOWN, buff=0.4)
        self.play(FadeIn(chan_title))

        # Generate a simple test image (gradient) and show Y'/Cb/Cr channels
        n = 32
        img_w, img_h = 2.8, 1.6
        cell_w = img_w / n
        cell_h = img_h / n

        def make_image_bar(channel_fn, label, col, y_pos):
            bar = VGroup()
            for i in range(n):
                for j in range(n):
                    r = i / (n - 1)
                    g = j / (n - 1)
                    b = 0.4
                    v = channel_fn(r, g, b)
                    v_disp = np.clip((v + 0.5) * 0.7 + 0.15, 0, 1)
                    c = Rectangle(width=cell_w + 0.01, height=cell_h + 0.01,
                                  fill_color=rgb_to_hex([v_disp, v_disp, v_disp]),
                                  fill_opacity=1, stroke_width=0)
                    c.move_to([-img_w / 2 + (i + 0.5) * cell_w,
                               -img_h / 2 + (j + 0.5) * cell_h, 0])
                    bar.add(c)
            bar.move_to(y_pos)
            lbl = Text(label, font_size=15, color=col)
            lbl.next_to(bar, DOWN, buff=0.1)
            return bar, lbl

        y_bar, y_lbl = make_image_bar(
            lambda r, g, b: 0.2126 * r + 0.7152 * g + 0.0722 * b,
            "Y' (luma)", ACCENT_YELLOW, LEFT * 3.5 + DOWN * 0.4)
        cb_bar, cb_lbl = make_image_bar(
            lambda r, g, b: (b - (0.2126 * r + 0.7152 * g + 0.0722 * b)) / 1.8556,
            "Cb (blue-diff)", ACCENT_TEAL, ORIGIN + DOWN * 0.4)
        cr_bar, cr_lbl = make_image_bar(
            lambda r, g, b: (r - (0.2126 * r + 0.7152 * g + 0.0722 * b)) / 1.5748,
            "Cr (red-diff)", ACCENT_ORANGE, RIGHT * 3.5 + DOWN * 0.4)

        for bar, lbl in [(y_bar, y_lbl), (cb_bar, cb_lbl), (cr_bar, cr_lbl)]:
            self.play(FadeIn(bar, lag_ratio=0.001, run_time=1.4), FadeIn(lbl))
        self.wait(3.5)

        # ── Act 3: Chroma subsampling ─────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects
                    if m is not ch], run_time=1.2)

        sub_title = Text("4:2:0 Chroma Subsampling — 50% Bandwidth Saving",
                         font_size=24, color=ACCENT_GREEN)
        sub_title.next_to(ch, DOWN, buff=0.4)
        self.play(FadeIn(sub_title))

        sub_items = VGroup(
            Text("Human vision: high spatial resolution for luminance,", font_size=19, color=TEXT_PRI),
            Text("  LOW resolution for color (chroma).", font_size=19, color=TEXT_PRI),
            Text("", font_size=4),
            Text("4:2:0 subsampling:", font_size=20, color=ACCENT_BLUE, weight=BOLD),
            Text("  Y' — full resolution  (1 sample per pixel)", font_size=18, color=ACCENT_YELLOW),
            Text("  Cb — quarter resolution  (1 sample per 2×2 block)", font_size=18, color=ACCENT_TEAL),
            Text("  Cr — quarter resolution  (1 sample per 2×2 block)", font_size=18, color=ACCENT_ORANGE),
            Text("", font_size=4),
            Text("Result: 3 planes at ½×½ color → 50% data vs 4:4:4 RGB", font_size=19, color=ACCENT_GREEN),
        )
        sub_items.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        sub_items.move_to(DOWN * 0.3)
        for item in sub_items:
            self.play(FadeIn(item), run_time=0.3)
        self.wait(3.5)

        ycbcr_box = RoundedRectangle(corner_radius=0.12, width=11, height=1.3,
                                     fill_color=PANEL, fill_opacity=0.9,
                                     stroke_color=ACCENT_BLUE, stroke_width=1.5)
        ycbcr_box.to_edge(DOWN, buff=0.2)
        ycbcr_txt = VGroup(
            Text("Y'CbCr connects directly to gamma (primes) and opponent process (Cb≈blue-yellow, Cr≈red-green)",
                 font_size=16, color=ACCENT_BLUE),
            Text("BT.709 matrix for HD/web · BT.2020 matrix for 4K/HDR — different coefficients!",
                 font_size=16, color=TEXT_PRI),
        )
        ycbcr_txt.arrange(DOWN, buff=0.1)
        ycbcr_txt.move_to(ycbcr_box)
        self.play(FadeIn(ycbcr_box), FadeIn(ycbcr_txt))
        self.wait(7)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE — TONE MAPPING AND SCENE-REFERRED WORKFLOWS
# ═══════════════════════════════════════════════════════════════════════
class ToneMappingScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XXVI.  Tone Mapping and Scene-Referred Workflows",
                  font_size=36, color=ACCENT_ORANGE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        # ── Act 1: The problem — HDR scene → SDR display ──────────────
        problem_title = Text("Fitting HDR Scene Content onto an SDR Display",
                             font_size=24, color=ACCENT_ORANGE)
        problem_title.next_to(ch, DOWN, buff=0.4)
        self.play(FadeIn(problem_title))

        tm_axes = Axes(
            x_range=[0, 4, 1], y_range=[0, 1.1, 0.5],
            x_length=7, y_length=4.0, tips=False,
            axis_config={"color": GRID_COL, "stroke_width": 1.5,
                         "include_numbers": True, "font_size": 14})
        tm_axes.move_to(LEFT * 1.8 + DOWN * 0.6)
        tm_xl = Text("Scene luminance (relative)", font_size=14, color=TEXT_SEC)
        tm_xl.next_to(tm_axes, DOWN, buff=0.12)
        tm_yl = Text("Display output", font_size=14, color=TEXT_SEC).rotate(PI / 2)
        tm_yl.next_to(tm_axes, LEFT, buff=0.10)
        self.play(Create(tm_axes), FadeIn(tm_xl), FadeIn(tm_yl), run_time=1.4)

        # Clipping (naive)
        clip_plot = tm_axes.plot(
            lambda x: min(x / 1.0, 1.0), color=MUTED_RED, stroke_width=3,
            x_range=[0, 4, 0.01])
        # Reinhard
        reinhard_plot = tm_axes.plot(
            lambda x: float(reinhard_tonemap(x / 1.0, L_white=2.0)),
            color=ACCENT_YELLOW, stroke_width=3, x_range=[0.001, 4, 0.01])
        # ACES
        aces_plot = tm_axes.plot(
            lambda x: float(aces_tonemap(x * 0.6)),
            color=ACCENT_TEAL, stroke_width=3, x_range=[0, 4, 0.01])

        clip_lbl = Text("Clip", font_size=14, color=MUTED_RED)
        clip_lbl.move_to(tm_axes.c2p(1.6, 0.75))
        reinhard_lbl = Text("Reinhard", font_size=14, color=ACCENT_YELLOW)
        reinhard_lbl.move_to(tm_axes.c2p(3.0, 0.78))
        aces_lbl = Text("ACES", font_size=14, color=ACCENT_TEAL)
        aces_lbl.move_to(tm_axes.c2p(2.5, 0.60))

        self.play(Create(clip_plot), FadeIn(clip_lbl), run_time=1.8)
        self.play(Create(reinhard_plot), FadeIn(reinhard_lbl), run_time=1.8)
        self.play(Create(aces_plot), FadeIn(aces_lbl), run_time=1.8)

        tm_note = VGroup(
            Text("Clip: destroys highlights", font_size=15, color=MUTED_RED),
            Text("Reinhard: smooth but desaturates", font_size=15, color=ACCENT_YELLOW),
            Text("ACES: filmic S-curve (cinema standard)", font_size=15, color=ACCENT_TEAL),
        )
        tm_note.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        tm_note.move_to(RIGHT * 3.8 + DOWN * 0.4)
        self.play(FadeIn(tm_note, lag_ratio=0.3), run_time=1.8)
        self.wait(3.5)

        # ── Act 2: Scene-referred vs display-referred ─────────────────
        self.play(*[FadeOut(m) for m in self.mobjects
                    if m is not ch], run_time=1.2)

        sr_title = Text("Scene-Referred vs Display-Referred Workflows",
                        font_size=26, color=ACCENT_PURPLE, weight=BOLD)
        sr_title.next_to(ch, DOWN, buff=0.4)
        self.play(FadeIn(sr_title))

        # Two-column comparison
        sr_left = VGroup(
            Text("SCENE-REFERRED", font_size=20, color=ACCENT_TEAL, weight=BOLD),
            Text("VFX / Film / ACES", font_size=16, color=TEXT_SEC),
            Text("• Linear light, physical units", font_size=17, color=TEXT_PRI),
            Text("• Unbounded (0–∞)", font_size=17, color=TEXT_PRI),
            Text("• 16-bit half-float (OpenEXR)", font_size=17, color=TEXT_PRI),
            Text("• Tone map / grade at output", font_size=17, color=TEXT_PRI),
            Text("• ACES IDT → RRT → ODT pipeline", font_size=17, color=TEXT_PRI),
        )
        sr_left.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        sr_left.move_to(LEFT * 3.2 + DOWN * 0.2)

        sr_right = VGroup(
            Text("DISPLAY-REFERRED", font_size=20, color=ACCENT_ORANGE, weight=BOLD),
            Text("Web / UI / Photography", font_size=16, color=TEXT_SEC),
            Text("• sRGB gamma-encoded", font_size=17, color=TEXT_PRI),
            Text("• Clamped [0, 1]", font_size=17, color=TEXT_PRI),
            Text("• 8-bit per channel (PNG/JPEG)", font_size=17, color=TEXT_PRI),
            Text("• Already tone-mapped", font_size=17, color=TEXT_PRI),
            Text("• Camera does it for you (JPEG)", font_size=17, color=TEXT_PRI),
        )
        sr_right.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        sr_right.move_to(RIGHT * 2.5 + DOWN * 0.2)

        divider = Line(UP * 2.8, DOWN * 2.8, color=GRID_COL, stroke_width=1.5)

        self.play(FadeIn(sr_left, lag_ratio=0.2), Create(divider),
                  FadeIn(sr_right, lag_ratio=0.2), run_time=1.2)
        self.wait(3.5)

        # ── Act 3: ACES pipeline ──────────────────────────────────────
        aces_box = RoundedRectangle(corner_radius=0.12, width=11, height=1.15,
                                    fill_color=PANEL, fill_opacity=0.9,
                                    stroke_color=ACCENT_ORANGE, stroke_width=1.5)
        aces_box.to_edge(DOWN, buff=0.2)
        aces_txt = VGroup(
            Text("ACES = Academy Color Encoding System  ·  cinema standard since 2014",
                 font_size=17, color=ACCENT_ORANGE),
            Text("Tone mapping = luminance compression  ·  gamut mapping = chroma compression",
                 font_size=17, color=TEXT_PRI),
        )
        aces_txt.arrange(DOWN, buff=0.1)
        aces_txt.move_to(aces_box)
        self.play(FadeIn(aces_box), FadeIn(aces_txt))
        self.wait(7)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE — DISPLAY TECHNOLOGY AND HDR
# ═══════════════════════════════════════════════════════════════════════
class DisplayHDRScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XXVII.  Display Technology and HDR",
                  font_size=40, color=ACCENT_YELLOW, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        # ── Act 1: SDR vs HDR luminance, PQ EOTF ─────────────────────
        pq_axes = Axes(
            x_range=[0, 1, 0.25], y_range=[0, 10000, 2000],
            x_length=5.5, y_length=4.5, tips=False,
            axis_config={"color": GRID_COL, "stroke_width": 1.5,
                         "include_numbers": True, "font_size": 14})
        pq_axes.move_to(LEFT * 2.8 + DOWN * 0.5)
        pq_xl = Text("PQ code value", font_size=14, color=TEXT_SEC)
        pq_xl.next_to(pq_axes, DOWN, buff=0.12)
        pq_yl = Text("Luminance (cd/m²)", font_size=13, color=TEXT_SEC).rotate(PI / 2)
        pq_yl.next_to(pq_axes, LEFT, buff=0.10)
        self.play(Create(pq_axes), FadeIn(pq_xl), FadeIn(pq_yl), run_time=1.4)

        pq_plot = pq_axes.plot(
            lambda n: float(np.clip(pq_eotf(n), 0, 10500)),
            color=ACCENT_YELLOW, stroke_width=3.5, x_range=[0.001, 1.0, 0.005])
        self.play(Create(pq_plot), run_time=1.2)

        # SDR reference line at 100 nits
        sdr_line = DashedLine(
            pq_axes.c2p(0, 100), pq_axes.c2p(1, 100),
            color=ACCENT_BLUE, stroke_width=2, dash_length=0.08)
        sdr_lbl = Text("SDR 100 nit", font_size=13, color=ACCENT_BLUE)
        sdr_lbl.next_to(pq_axes.c2p(0.6, 100), UP, buff=0.08)
        self.play(Create(sdr_line), FadeIn(sdr_lbl))

        # Right: luminance comparison panel
        nit_lines = VGroup(
            Text("SDR:          0 – 100 cd/m²", font_size=18, color=ACCENT_BLUE),
            Text("HDR10:    0 – 1 000 cd/m²", font_size=18, color=ACCENT_YELLOW),
            Text("Dolby Vision: up to 10 000 cd/m²", font_size=18, color=ACCENT_ORANGE),
            Text("Human eye:  ~1 000 000:1 range", font_size=18, color=TEXT_SEC),
        )
        nit_lines.arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        nit_lines.move_to(RIGHT * 2.8 + DOWN * 0.2)
        for line in nit_lines:
            self.play(FadeIn(line), run_time=1.0)
        self.wait(3.5)

        # ── Act 2: 10-bit vs 8-bit banding ───────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects
                    if m is not ch], run_time=1.2)

        bit_title = Text("10-bit vs 8-bit: Why HDR needs more bits",
                         font_size=26, color=ACCENT_ORANGE)
        bit_title.next_to(ch, DOWN, buff=0.45)
        self.play(FadeIn(bit_title))

        bit_lines = VGroup(
            Text("8-bit PQ: 256 code values over 0–10000 nit range", font_size=20, color=TEXT_PRI),
            Text("Step at dark end: ~0.5 nit — perceptibly visible banding", font_size=20, color=MUTED_RED),
            Text("10-bit PQ: 1024 code values", font_size=20, color=TEXT_PRI),
            Text("Step at dark end: ~0.008 nit — below JND", font_size=20, color=ACCENT_GREEN),
        )
        bit_lines.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        bit_lines.move_to(ORIGIN + DOWN * 0.3)
        for line in bit_lines:
            self.play(FadeIn(line), run_time=0.45)
        self.wait(3.5)

        # ── Act 3: Display technologies + HLG ────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects
                    if m is not ch], run_time=1.2)

        tech_title = Text("Display Technologies and Transfer Functions",
                          font_size=26, color=ACCENT_TEAL)
        tech_title.next_to(ch, DOWN, buff=0.45)
        self.play(FadeIn(tech_title))

        tech_data = [
            ("OLED", "Self-emissive, near-infinite contrast", ACCENT_BLUE),
            ("QD-LCD", "Quantum dot: wide color + bright peak", ACCENT_GREEN),
            ("PQ (ST 2084)", "HDR10, Dolby Vision transfer function", ACCENT_YELLOW),
            ("HLG", "Hybrid Log-Gamma: broadcast HDR (BBC/NHK)", ACCENT_ORANGE),
        ]
        tech_rows = VGroup()
        for name, desc, col in tech_data:
            row = VGroup(
                Text(name, font_size=20, color=col, weight=BOLD),
                Text(desc, font_size=18, color=TEXT_PRI),
            )
            row.arrange(RIGHT, buff=0.5, aligned_edge=LEFT)
            tech_rows.add(row)
        tech_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        tech_rows.move_to(ORIGIN + DOWN * 0.3)
        self.play(FadeIn(tech_rows, lag_ratio=0.25), run_time=1.2)

        hdr_box = RoundedRectangle(corner_radius=0.12, width=11, height=0.9,
                                   fill_color=PANEL, fill_opacity=0.9,
                                   stroke_color=ACCENT_YELLOW, stroke_width=1.5)
        hdr_box.to_edge(DOWN, buff=0.3)
        hdr_txt = Text(
            "Process HDR content in linear light  ·  apply PQ or HLG only at final output",
            font_size=18, color=ACCENT_YELLOW)
        hdr_txt.move_to(hdr_box)
        self.play(FadeIn(hdr_box), FadeIn(hdr_txt))
        self.wait(7)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 24 — ICC PROFILES AND COLOR MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════
class ICCPipelineScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XXVIII.  ICC Profiles and Color Management",
                  font_size=38, color=ACCENT_PURPLE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        # ── Act 1: Pipeline block diagram ────────────────────────────
        blocks = [
            ("Camera\n(raw/sRGB/P3)", ACCENT_BLUE),
            ("Input\nProfile", ACCENT_TEAL),
            ("PCS\n(Lab/XYZ D50)", ACCENT_YELLOW),
            ("Output\nProfile", ACCENT_ORANGE),
            ("Display\n(sRGB/P3)", ACCENT_PINK),
        ]
        block_objs = VGroup()
        for label, col in blocks:
            box = RoundedRectangle(corner_radius=0.15, width=1.9, height=1.3,
                                   fill_color=PANEL, fill_opacity=0.9,
                                   stroke_color=col, stroke_width=2)
            txt = Text(label, font_size=14, color=col, line_spacing=1.1)
            txt.move_to(box)
            block_objs.add(VGroup(box, txt))
        block_objs.arrange(RIGHT, buff=0.3)
        block_objs.move_to(UP * 1.0)

        pipe_arrows = VGroup()
        for i in range(len(blocks) - 1):
            a = Arrow(block_objs[i].get_right(),
                      block_objs[i + 1].get_left(),
                      buff=0.05, stroke_width=2.5, color=TEXT_SEC)
            pipe_arrows.add(a)

        for i, block in enumerate(block_objs):
            self.play(FadeIn(block, scale=0.85), run_time=0.3)
            if i < len(pipe_arrows):
                self.play(GrowArrow(pipe_arrows[i]), run_time=0.2)

        pcs_ann = Text("Profile Connection Space (PCS) = device-independent hub",
                       font_size=17, color=ACCENT_YELLOW)
        pcs_ann.next_to(block_objs, DOWN, buff=0.3)
        self.play(FadeIn(pcs_ann))
        self.wait(2.5)

        # ── Act 2: Four rendering intents ────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects
                    if m is not ch], run_time=1.2)

        ri_title = Text("The Four Rendering Intents", font_size=28,
                        color=ACCENT_PURPLE, weight=BOLD)
        ri_title.next_to(ch, DOWN, buff=0.45)
        self.play(FadeIn(ri_title))

        intents = [
            ("Perceptual", "Shrinks whole gamut proportionally — good for photos",
             ACCENT_BLUE),
            ("Relative Colorimetric", "Clips out-of-gamut, adjusts white — good for spot colors",
             ACCENT_GREEN),
            ("Saturation", "Maximizes saturation — good for presentation graphics",
             ACCENT_ORANGE),
            ("Absolute Colorimetric", "No white point adjustment — proofing use",
             ACCENT_TEAL),
        ]
        intent_rows = VGroup()
        for name, desc, col in intents:
            row = VGroup(
                Text(name, font_size=19, color=col, weight=BOLD),
                Text(desc, font_size=17, color=TEXT_PRI),
            )
            row.arrange(DOWN, aligned_edge=LEFT, buff=0.08)
            intent_rows.add(row)
        intent_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        intent_rows.move_to(DOWN * 0.3)
        self.play(FadeIn(intent_rows, lag_ratio=0.25), run_time=1.2)

        # ── Act 3: Engineering callout ────────────────────────────────
        icc_box = RoundedRectangle(corner_radius=0.12, width=11, height=1.3,
                                   fill_color=PANEL, fill_opacity=0.9,
                                   stroke_color=ACCENT_PURPLE, stroke_width=1.5)
        icc_box.to_edge(DOWN, buff=0.2)
        icc_txt = VGroup(
            Text("Raw pixel values are meaningless without a color profile",
                 font_size=17, color=ACCENT_PURPLE),
            Text("Use ICC-aware libraries  ·  On the web, assume sRGB unless specified",
                 font_size=17, color=TEXT_PRI),
        )
        icc_txt.arrange(DOWN, buff=0.12)
        icc_txt.move_to(icc_box)
        self.play(FadeIn(icc_box), FadeIn(icc_txt))
        self.wait(7)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 25 — PRACTICAL BLENDING OPERATIONS
# ═══════════════════════════════════════════════════════════════════════
class PracticalBlendingScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XXIX.  Practical Blending Operations",
                  font_size=40, color=ACCENT_TEAL, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        # ── Act 1: Dark halo — sRGB vs linear light blend ─────────────
        halo_title = Text("The Dark Halo Problem", font_size=26, color=MUTED_RED)
        halo_title.next_to(ch, DOWN, buff=0.45)
        self.play(FadeIn(halo_title))

        c1 = np.array([1.0, 0.2, 0.1])  # warm red
        c2 = np.array([0.1, 0.3, 1.0])  # cool blue
        n = 60
        bar_w = 9.0
        sw = bar_w / n

        srgb_bar = VGroup()
        lin_bar_blend = VGroup()
        for i in range(n):
            t = i / (n - 1)
            srgb_col = srgb_blend(c1, c2, t)
            lin_col = linear_blend(c1, c2, t)
            r1 = Rectangle(width=sw + 0.02, height=0.65,
                           fill_color=rgb_to_hex(srgb_col), fill_opacity=1,
                           stroke_width=0)
            r1.move_to(LEFT * bar_w / 2 + RIGHT * (i * sw + sw / 2) + UP * 0.4)
            srgb_bar.add(r1)
            r2 = Rectangle(width=sw + 0.02, height=0.65,
                           fill_color=rgb_to_hex(lin_col), fill_opacity=1,
                           stroke_width=0)
            r2.move_to(LEFT * bar_w / 2 + RIGHT * (i * sw + sw / 2) + DOWN * 0.5)
            lin_bar_blend.add(r2)

        lbl_srgb = Text("sRGB blend (dark mid)", font_size=17, color=MUTED_RED)
        lbl_srgb.move_to(LEFT * 3.5 + UP * 1.15)
        lbl_lin = Text("Linear blend (correct)", font_size=17, color=ACCENT_GREEN)
        lbl_lin.move_to(LEFT * 3.5 + DOWN * 1.25)

        self.play(FadeIn(srgb_bar, lag_ratio=0.003), FadeIn(lbl_srgb), run_time=1.8)
        self.play(FadeIn(lin_bar_blend, lag_ratio=0.003), FadeIn(lbl_lin), run_time=1.8)
        self.wait(3.5)

        # ── Act 2: Premultiplied alpha ────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects
                    if m is not ch], run_time=1.2)

        pm_title = Text("Premultiplied Alpha: The Correct Formula",
                        font_size=26, color=ACCENT_ORANGE)
        pm_title.next_to(ch, DOWN, buff=0.45)
        self.play(FadeIn(pm_title))

        pm_eq = MathTex(
            r"C_{out} = C_{src} \cdot \alpha + C_{dst} \cdot (1 - \alpha)",
            font_size=28, color=TEXT_PRI)
        pm_eq.move_to(UP * 0.5)

        pm_note = VGroup(
            Text("• All values in LINEAR light", font_size=19, color=ACCENT_YELLOW),
            Text("• Premultiplied: store C·α, not C and α separately", font_size=19, color=TEXT_PRI),
            Text("• Avoids dark fringe from gamma-domain averaging", font_size=19, color=ACCENT_GREEN),
            Text("• CSS: use color-mix() in oklch, not in srgb", font_size=19, color=ACCENT_TEAL),
        )
        pm_note.arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        pm_note.move_to(DOWN * 0.8)
        self.play(Write(pm_eq, run_time=1.0))
        for note in pm_note:
            self.play(FadeIn(note), run_time=1.0)
        self.wait(3.5)

        # ── Act 3: CSS color-mix() comparison bars ────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects
                    if m is not ch], run_time=1.2)

        mix_title = Text("CSS color-mix(): Comparing Blend Spaces",
                         font_size=26, color=ACCENT_TEAL)
        mix_title.next_to(ch, DOWN, buff=0.45)
        self.play(FadeIn(mix_title))

        css_ex = Text('color-mix(in oklch, red 40%, blue)',
                      font_size=22, color=ACCENT_TEAL)
        css_ex.next_to(mix_title, DOWN, buff=0.3)
        self.play(FadeIn(css_ex))

        blend_methods = [
            ("sRGB", srgb_blend, ACCENT_ORANGE),
            ("Linear light", linear_blend, ACCENT_GREEN),
            ("OKLab", oklab_blend_fn, ACCENT_TEAL),
            ("OKLch", oklch_blend_fn, ACCENT_PURPLE),
        ]
        y_off = 0.5
        for method_name, blend_fn, col in blend_methods:
            bar = make_gradient_bar(blend_fn, c1, c2, n=60, width=8.5, height=0.52)
            bar.move_to(LEFT * 0.5 + DOWN * y_off)
            bar_lbl = Text(method_name, font_size=16, color=col, weight=BOLD)
            bar_lbl.next_to(bar, LEFT, buff=0.35)
            self.play(FadeIn(bar, lag_ratio=0.003), FadeIn(bar_lbl), run_time=1.2)
            y_off += 0.95

        blend_box = RoundedRectangle(corner_radius=0.12, width=11, height=0.85,
                                     fill_color=PANEL, fill_opacity=0.9,
                                     stroke_color=ACCENT_TEAL, stroke_width=1.5)
        blend_box.to_edge(DOWN, buff=0.25)
        blend_txt = Text(
            "Always blend in linear light  ·  OKLch avoids hue shifts through mid-range",
            font_size=18, color=ACCENT_TEAL)
        blend_txt.move_to(blend_box)
        self.play(FadeIn(blend_box), FadeIn(blend_txt))
        self.wait(7)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 26 — REAL WORLD ADOPTION
# ═══════════════════════════════════════════════════════════════════════
class RealWorldScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XXX.  OKLab in the Wild", font_size=42,
                  color=ACCENT_ORANGE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        adopters = [
            ("Photoshop", "Default gradient interpolation (2023+)"),
            ("CSS Color 4/5", "oklab() and oklch() in all browsers"),
            ("Unity / Godot", "Gradient system / color picker"),
            ("Figma", "Color blending engine"),
        ]
        items = VGroup()
        for name, desc in adopters:
            n = Text(name, font_size=20, color=ACCENT_ORANGE, weight=BOLD)
            d = Text(f"  —  {desc}", font_size=16, color=TEXT_SEC)
            items.add(VGroup(n, d).arrange(RIGHT, buff=0.08))
        items.arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        items.move_to(LEFT * 1.5 + UP * 0.4)
        for item in items:
            self.play(FadeIn(item, shift=RIGHT * 0.3), run_time=0.35)

        # OKLab wheel
        wc = RIGHT * 3.8
        wheel = VGroup()
        n_h, n_r = 60, 10
        max_rad = 1.7
        for ih in range(n_h):
            for ir in range(1, n_r + 1):
                angle = TAU * ih / n_h
                radius = max_rad * ir / n_r
                C = 0.15 * ir / n_r
                col = oklab_to_hex(0.75, C * np.cos(angle), C * np.sin(angle))
                d = Dot(wc + radius * np.array([np.cos(angle), np.sin(angle), 0]),
                        radius=0.09, color=col, fill_opacity=0.92)
                wheel.add(d)
        wl = Text("OKLab Wheel", font_size=15, color=TEXT_SEC)
        wl.next_to(wheel, DOWN, buff=0.25)
        self.play(FadeIn(wheel, lag_ratio=0.002, run_time=2.5), FadeIn(wl))

        # Spinning highlight
        hl = Circle(radius=0.2, color=WHITE, stroke_width=2.5, fill_opacity=0)
        hl.move_to(wc + RIGHT * max_rad * 0.7)
        hl.t = 0

        def spin(m, dt):
            m.t += dt
            a = m.t * 0.7
            r = max_rad * 0.7
            m.move_to(wc + r * np.array([np.cos(a), np.sin(a), 0]))

        hl.add_updater(spin)
        self.add(hl)
        self.wait(8)
        hl.remove_updater(spin)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 20 — COLOR BLINDNESS & ACCESSIBILITY
# ═══════════════════════════════════════════════════════════════════════
class ColorBlindnessScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XXXI.  Color Blindness: Engineering for Accessibility",
                  font_size=36, color=ACCENT_PINK, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        # ── Act 1: Statistics ──
        stats = VGroup(
            Text("~8% of men  and  ~0.5% of women  have color vision deficiency",
                 font_size=19, color=TEXT_PRI),
            Text("Three main types: Deuteranopia (no M cones) · "
                 "Protanopia (no L cones) · Tritanopia (no S cones)",
                 font_size=16, color=TEXT_SEC),
        ).arrange(DOWN, buff=0.18)
        stats.next_to(ch, DOWN, buff=0.3)
        self.play(FadeIn(stats))
        self.wait(2.5)

        # ── Act 2: Four OKLab wheels ──
        self.play(FadeOut(stats), run_time=1.0)

        wheel_title = Text("How the OKLab color wheel appears under each condition:",
                           font_size=17, color=TEXT_SEC)
        wheel_title.move_to(UP * 2.3)
        self.play(FadeIn(wheel_title))

        def make_wheel(center, cvd_fn=None):
            grp = VGroup()
            n_h, n_r = 36, 7
            max_r = 1.25
            for ih in range(n_h):
                for ir in range(1, n_r + 1):
                    angle = TAU * ih / n_h
                    radius = max_r * ir / n_r
                    C = 0.15 * ir / n_r
                    ro, go, bxo = oklab_to_linear_srgb(
                        0.75, C * np.cos(angle), C * np.sin(angle))
                    srgb = np.clip(linear_to_srgb(np.clip([ro, go, bxo], 0, 1)), 0, 1)
                    if cvd_fn is not None:
                        srgb = np.clip(cvd_fn(*srgb), 0, 1)
                    d = Dot(center + radius * np.array([np.cos(angle),
                                                        np.sin(angle), 0]),
                            radius=0.075, color=rgb_to_hex(srgb), fill_opacity=0.92)
                    grp.add(d)
            return grp

        wheel_configs = [
            (np.array([-4.8, 0.2, 0]), None, "Normal",
             TEXT_PRI),
            (np.array([-1.6, 0.2, 0]), simulate_deuteranopia, "Deuteranopia\n(red-green)",
             ACCENT_ORANGE),
            (np.array([1.6, 0.2, 0]), simulate_protanopia, "Protanopia\n(red-blind)",
             ACCENT_PINK),
            (np.array([4.8, 0.2, 0]), simulate_tritanopia, "Tritanopia\n(blue-yellow)",
             ACCENT_BLUE),
        ]

        for center, cvd_fn, label_str, col in wheel_configs:
            wheel = make_wheel(center, cvd_fn)
            lbl = Text(label_str, font_size=14, color=col, line_spacing=1.1)
            lbl.move_to(center + DOWN * 1.55)
            self.play(FadeIn(wheel, lag_ratio=0.001, run_time=1.0), FadeIn(lbl))

        self.wait(3.5)

        # ── Act 3: Engineering callout ──
        rules = [
            "Never use red/green alone to signal OK / fault — add shape or pattern",
            "Status LEDs: use blue + yellow, not red + green",
            "OKLch hue distance > 60° between any two alert levels",
        ]
        rule_grp = VGroup()
        for r in rules:
            rule_grp.add(Text("• " + r, font_size=15, color=TEXT_PRI))
        rule_grp.arrange(DOWN, buff=0.18, aligned_edge=LEFT)

        eng_box = RoundedRectangle(corner_radius=0.12, width=12.0, height=2.0,
                                   fill_color=PANEL, fill_opacity=0.9,
                                   stroke_color=ACCENT_PINK, stroke_width=1.5)
        eng_box.to_edge(DOWN, buff=0.22)
        rule_grp.move_to(eng_box)
        self.play(FadeIn(eng_box), FadeIn(rule_grp, lag_ratio=0.3, run_time=1.0))
        self.wait(4)

        # ── Act 4: WCAG Contrast Math ─────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.2)

        wcag_title = Text("WCAG Contrast Ratios", font_size=38,
                          color=ACCENT_TEAL, weight=BOLD)
        wcag_title.to_edge(UP, buff=0.45)
        self.play(FadeIn(wcag_title, shift=DOWN * 0.2))

        wcag_formula = MathTex(
            r"\text{contrast} = \frac{L_1 + 0.05}{L_2 + 0.05}",
            r"\quad (L_1 > L_2)",
            font_size=30, color=TEXT_PRI)
        wcag_formula.next_to(wcag_title, DOWN, buff=0.45)
        self.play(Write(wcag_formula, run_time=1.0))

        lu_note = Text("where  L = 0.2126·R_lin + 0.7152·G_lin + 0.0722·B_lin",
                       font_size=18, color=TEXT_SEC)
        lu_note.next_to(wcag_formula, DOWN, buff=0.3)
        self.play(FadeIn(lu_note))

        # Threshold table
        thresholds = [
            ("≥ 3 : 1", "AA large text (≥18pt or 14pt bold)", ACCENT_YELLOW),
            ("≥ 4.5 : 1", "AA normal text (recommended minimum)", ACCENT_ORANGE),
            ("≥ 7 : 1", "AAA normal text (enhanced)", ACCENT_GREEN),
        ]
        thresh_rows = VGroup()
        for ratio, desc, col in thresholds:
            row = VGroup(
                Text(ratio, font_size=22, color=col, weight=BOLD),
                Text(desc, font_size=18, color=TEXT_PRI),
            )
            row.arrange(RIGHT, buff=0.5, aligned_edge=LEFT)
            thresh_rows.add(row)
        thresh_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        thresh_rows.move_to(DOWN * 0.6)
        self.play(FadeIn(thresh_rows, lag_ratio=0.3), run_time=1.0)

        # Live example: dark text on light background
        white_bg = np.array([1.0, 1.0, 1.0])
        dark_txt = np.array([0.1, 0.1, 0.1])
        ratio = wcag_contrast(dark_txt, white_bg)
        ex_swatch = Rectangle(width=3.5, height=0.9, fill_color=rgb_to_hex(white_bg),
                              fill_opacity=1, stroke_color=TEXT_SEC, stroke_width=1.5)
        ex_swatch.move_to(RIGHT * 3.5 + DOWN * 0.3)
        ex_lbl = Text(f"Dark on white: {ratio:.1f}:1  ✓ AAA",
                      font_size=17, color=rgb_to_hex(dark_txt))
        ex_lbl.move_to(ex_swatch)
        self.play(FadeIn(ex_swatch), FadeIn(ex_lbl))

        apca_box = RoundedRectangle(corner_radius=0.12, width=11, height=1.15,
                                    fill_color=PANEL, fill_opacity=0.9,
                                    stroke_color=ACCENT_TEAL, stroke_width=1.5)
        apca_box.to_edge(DOWN, buff=0.2)
        apca_txt = VGroup(
            Text("WCAG 3.0 candidate: APCA uses lightness contrast, not ratio",
                 font_size=17, color=ACCENT_TEAL),
            Text("Test contrast for ALL CVD simulation types, not just normal vision",
                 font_size=17, color=ACCENT_PINK),
        )
        apca_txt.arrange(DOWN, buff=0.1)
        apca_txt.move_to(apca_box)
        self.play(FadeIn(apca_box), FadeIn(apca_txt))
        self.wait(7)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 28 — PALETTE GENERATION ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════
class PaletteGenerationScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XXXII.  Palette Generation Algorithms",
                  font_size=38, color=ACCENT_PURPLE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        # ── Act 1: OKLch equal-step palette ──────────────────────────
        pal_title = Text("OKLch Equal-Hue-Step Palette",
                         font_size=26, color=ACCENT_PURPLE)
        pal_title.next_to(ch, DOWN, buff=0.45)
        self.play(FadeIn(pal_title))

        n_pal = 6
        L_val = 0.65
        C_val = 0.16
        pal_colors = oklch_palette(n_pal, L=L_val, C=C_val)

        # Show as a hue wheel
        wheel_grp = VGroup()
        for i, col in enumerate(pal_colors):
            angle = (i / n_pal) * TAU - PI / 2
            pos = np.array([1.5 * np.cos(angle), 1.5 * np.sin(angle), 0])
            dot = Circle(radius=0.35, fill_color=rgb_to_hex(col),
                         fill_opacity=1, stroke_color=TEXT_SEC, stroke_width=1.5)
            dot.move_to(pos + DOWN * 0.4)
            wheel_grp.add(dot)
        wheel_grp.move_to(LEFT * 3.0 + DOWN * 0.3)
        self.play(FadeIn(wheel_grp, lag_ratio=0.15), run_time=1.0)

        # Show as a swatch row
        swatch_row = VGroup()
        for col in pal_colors:
            sw = Rectangle(width=1.3, height=0.9, fill_color=rgb_to_hex(col),
                           fill_opacity=1, stroke_color=TEXT_SEC, stroke_width=1.0)
            swatch_row.add(sw)
        swatch_row.arrange(RIGHT, buff=0.1)
        swatch_row.move_to(RIGHT * 2.0 + UP * 0.4)
        self.play(FadeIn(swatch_row, lag_ratio=0.15), run_time=1.8)

        pal_info = VGroup(
            Text(f"L = {L_val}  C = {C_val}", font_size=18, color=ACCENT_PURPLE),
            Text(f"n = {n_pal}  hue steps: 0°, 60°, 120°…", font_size=18, color=TEXT_SEC),
        )
        pal_info.arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        pal_info.move_to(RIGHT * 2.0 + DOWN * 0.5)
        self.play(FadeIn(pal_info))
        self.wait(3.5)

        # ── Act 2: WCAG-safe palette check ───────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects
                    if m is not ch], run_time=1.2)

        wcag_title = Text("WCAG Accessibility Check",
                          font_size=26, color=ACCENT_GREEN)
        wcag_title.next_to(ch, DOWN, buff=0.45)
        self.play(FadeIn(wcag_title))

        white = np.array([1.0, 1.0, 1.0])
        black = np.array([0.0, 0.0, 0.0])
        check_rows = VGroup()
        for i, col in enumerate(pal_colors):
            h_deg = round((360 * i / n_pal))
            ratio_w = wcag_contrast(col, white)
            ratio_b = wcag_contrast(col, black)
            pass_w = ratio_w >= 4.5
            pass_b = ratio_b >= 4.5
            sw = Rectangle(width=1.2, height=0.7, fill_color=rgb_to_hex(col),
                           fill_opacity=1, stroke_color=TEXT_SEC, stroke_width=1.0)
            badge_w = Text(f"vs W: {ratio_w:.1f} {'✓' if pass_w else '✗'}",
                           font_size=15,
                           color=ACCENT_GREEN if pass_w else MUTED_RED)
            badge_b = Text(f"vs K: {ratio_b:.1f} {'✓' if pass_b else '✗'}",
                           font_size=15,
                           color=ACCENT_GREEN if pass_b else MUTED_RED)
            row = VGroup(sw, badge_w, badge_b)
            row.arrange(RIGHT, buff=0.3, aligned_edge=DOWN)
            check_rows.add(row)
        check_rows.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        check_rows.move_to(DOWN * 0.3)
        self.play(FadeIn(check_rows, lag_ratio=0.2), run_time=1.2)
        self.wait(3.5)

        # ── Act 3: Color harmonies ────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects
                    if m is not ch], run_time=1.2)

        harm_title = Text("Color Harmonies in OKLch",
                          font_size=26, color=ACCENT_TEAL)
        harm_title.next_to(ch, DOWN, buff=0.45)
        self.play(FadeIn(harm_title))

        def show_harmony(name, h_offsets, col, y_pos):
            lbl = Text(name, font_size=19, color=col, weight=BOLD)
            lbl.move_to(LEFT * 4.5 + UP * y_pos)
            self.play(FadeIn(lbl), run_time=0.3)
            sw_row = VGroup()
            for dh in h_offsets:
                h = (120 + dh) % 360
                c_rgb = oklch_palette(1, L=0.65, C=0.16)  # placeholder
                Lo, ao, bo = oklch_to_oklab(0.65, 0.16, h)
                ro, go, bxo = oklab_to_linear_srgb(Lo, ao, bo)
                c_hex = rgb_to_hex(np.clip(linear_to_srgb(
                    np.clip([ro, go, bxo], 0, 1)), 0, 1))
                sw = Rectangle(width=1.4, height=0.8, fill_color=c_hex,
                               fill_opacity=1, stroke_color=TEXT_SEC, stroke_width=1.0)
                sw_row.add(sw)
            sw_row.arrange(RIGHT, buff=0.15)
            sw_row.next_to(lbl, RIGHT, buff=0.5)
            self.play(FadeIn(sw_row, lag_ratio=0.2), run_time=1.2)

        show_harmony("Complementary  h, h+180°", [0, 180], ACCENT_ORANGE, 1.0)
        show_harmony("Triadic  h, h+120°, h+240°", [0, 120, 240], ACCENT_PURPLE, 0.0)
        show_harmony("Analogous  h−30°, h, h+30°", [-30, 0, 30], ACCENT_TEAL, -1.0)

        harm_box = RoundedRectangle(corner_radius=0.12, width=11, height=0.85,
                                    fill_color=PANEL, fill_opacity=0.9,
                                    stroke_color=ACCENT_PURPLE, stroke_width=1.5)
        harm_box.to_edge(DOWN, buff=0.3)
        harm_txt = Text(
            "OKLch harmonies are perceptually equal — HSL/HSV harmonies are not",
            font_size=18, color=ACCENT_PURPLE)
        harm_txt.move_to(harm_box)
        self.play(FadeIn(harm_box), FadeIn(harm_txt))
        self.wait(7)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 29 — NUMERICAL PRECISION AND GOTCHAS
# ═══════════════════════════════════════════════════════════════════════
class NumericalGotchasScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XXXIII.  Numerical Precision & Implementation Gotchas",
                  font_size=34, color=MUTED_RED, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        intro = Text(
            "Real-world pitfalls when implementing color math in code",
            font_size=20, color=TEXT_SEC)
        intro.next_to(ch, DOWN, buff=0.3)
        self.play(FadeIn(intro))

        # ── Act 1: Cube root near zero ────────────────────────────────
        g1_title = Text("① Cube root of negative values → wrong OKLab",
                        font_size=22, color=MUTED_RED, weight=BOLD)
        g1_title.move_to(UP * 1.0)
        self.play(FadeIn(g1_title))

        g1_code = VGroup(
            Text("# BAD: np.cbrt(-1e-15) → tiny negative → NaN in M₂",
                 font_size=17, color=MUTED_RED),
            Text("l_ = np.cbrt(l)    # l can be slightly negative from float ops",
                 font_size=17, color=TEXT_SEC),
            Text("# GOOD: clip to 0 before cube root", font_size=17, color=ACCENT_GREEN),
            Text("l_ = np.cbrt(np.clip(l, 0, None))", font_size=17, color=ACCENT_GREEN),
        )
        g1_code.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        g1_code.move_to(DOWN * 0.3)
        for line in g1_code:
            self.play(FadeIn(line), run_time=0.35)
        self.wait(3.5)

        # ── Act 2: Clamping order matters ─────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects
                    if m is not ch], run_time=1.0)

        g2_title = Text("② Clamp in LINEAR space, not gamma-encoded",
                        font_size=22, color=ACCENT_ORANGE, weight=BOLD)
        g2_title.move_to(UP * 1.4)
        self.play(FadeIn(g2_title))

        clamping_steps = VGroup(
            Text("❌ clamp(gamma_encode(v), 0, 1)  → hue shifts at boundary",
                 font_size=18, color=MUTED_RED),
            Text("✓  gamma_encode(clamp(v, 0, 1))  → correct", font_size=18, color=ACCENT_GREEN),
            Text("✓✓ gamut_map_oklch(L, a, b)       → best", font_size=18, color=ACCENT_TEAL),
        )
        clamping_steps.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        clamping_steps.move_to(DOWN * 0.2)
        for step in clamping_steps:
            self.play(FadeIn(step), run_time=0.45)
        self.wait(3.5)

        # ── Act 3: Gradient accumulation ─────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects
                    if m is not ch], run_time=1.0)

        g3_title = Text("③ Gradient step computation: use endpoints, not midpoints",
                        font_size=20, color=ACCENT_YELLOW, weight=BOLD)
        g3_title.move_to(UP * 1.5)
        self.play(FadeIn(g3_title))

        g3_code = VGroup(
            Text("# BAD: repeated addition accumulates float error",
                 font_size=17, color=MUTED_RED),
            Text("color = c_start", font_size=17, color=TEXT_SEC),
            Text("for i in range(n): color += step  # drifts near 0", font_size=17, color=MUTED_RED),
            Text("# GOOD: interpolate from endpoints each step",
                 font_size=17, color=ACCENT_GREEN),
            Text("for i in range(n): t = i/(n-1); color = lerp(c1, c2, t)",
                 font_size=17, color=ACCENT_GREEN),
        )
        g3_code.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        g3_code.move_to(DOWN * 0.2)
        for line in g3_code:
            self.play(FadeIn(line), run_time=0.3)
        self.wait(3.5)

        # ── Act 4: White point scale ──────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects
                    if m is not ch], run_time=1.0)

        g4_title = Text("④ CIELAB white point: Y must be normalized to 1.0",
                        font_size=20, color=ACCENT_PURPLE, weight=BOLD)
        g4_title.move_to(UP * 1.5)
        self.play(FadeIn(g4_title))

        g4_lines = VGroup(
            Text("CIELAB formula divides by Yn = 1.0 (D65 white point)",
                 font_size=19, color=TEXT_PRI),
            Text("If you pass absolute luminance (Y = 200 cd/m²):", font_size=19, color=TEXT_PRI),
            Text("  f(200/1.0) → L* > 100 → WRONG", font_size=19, color=MUTED_RED),
            Text("Always normalize Y to [0, 1] before passing to xyz_to_cielab()",
                 font_size=19, color=ACCENT_GREEN),
        )
        g4_lines.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        g4_lines.move_to(DOWN * 0.2)
        for line in g4_lines:
            self.play(FadeIn(line), run_time=1.0)
        self.wait(3.5)

        final_box = RoundedRectangle(corner_radius=0.12, width=11, height=0.9,
                                     fill_color=PANEL, fill_opacity=0.9,
                                     stroke_color=MUTED_RED, stroke_width=1.5)
        final_box.to_edge(DOWN, buff=0.3)
        final_txt = Text(
            "Robust color math: clip in linear space · use endpoint interpolation · normalize Y",
            font_size=17, color=TEXT_PRI)
        final_txt.move_to(final_box)
        self.play(FadeIn(final_box), FadeIn(final_txt))
        self.wait(7)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 30 — OUTRO & SUMMARY
# ═══════════════════════════════════════════════════════════════════════
class OutroScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        st = Text("The Evolution of Color Spaces", font_size=36,
                  color=TEXT_PRI, weight=BOLD)
        st.to_edge(UP, buff=0.5)
        self.play(FadeIn(st))

        cards = [
            ("RGB / HSV", "1960s–70s\nDevice-oriented\nNot perceptually\nuniform",
             ACCENT_BLUE, "✗"),
            ("CIE XYZ", "1931\nFirst standard\nNot uniform\nBasis for all",
             ACCENT_PURPLE, "~"),
            ("CIELAB", "1976\nCube-root f(t)\nBetter but\nblue problems",
             ACCENT_PURPLE, "~"),
            ("OKLab", "2020\nPerceptually uniform\nSimple · fast\nIndustry standard",
             ACCENT_TEAL, "✓"),
        ]

        cg = VGroup()
        for title, desc, col, sym in cards:
            box = RoundedRectangle(corner_radius=0.18, width=2.7, height=3.3,
                                   fill_color=PANEL, fill_opacity=0.95,
                                   stroke_color=col, stroke_width=2.5)
            s = Text(sym, font_size=34,
                     color=MUTED_RED if sym == "✗" else
                     ACCENT_YELLOW if sym == "~" else ACCENT_GREEN)
            s.move_to(box.get_top() + DOWN * 0.35)
            t = Text(title, font_size=18, color=col, weight=BOLD)
            t.next_to(s, DOWN, buff=0.15)
            d = Text(desc, font_size=12, color=TEXT_SEC, line_spacing=1.2)
            d.next_to(t, DOWN, buff=0.2)
            cg.add(VGroup(box, s, t, d))
        cg.arrange(RIGHT, buff=0.3).move_to(DOWN * 0.1)

        for card in cg:
            self.play(FadeIn(card, shift=UP * 0.25, scale=0.9), run_time=0.55)

        arr = Arrow(cg[0].get_bottom() + DOWN * 0.2,
                    cg[3].get_bottom() + DOWN * 0.2,
                    color=ACCENT_TEAL, stroke_width=3, buff=0)
        arr_lbl = Text("90 years of color science", font_size=15, color=TEXT_SEC)
        arr_lbl.next_to(arr, DOWN, buff=0.08)
        self.play(GrowArrow(arr), FadeIn(arr_lbl))

        # Glow on OKLab
        self.play(cg[3][0].animate.set_stroke(WHITE, 4), run_time=0.35)
        self.play(cg[3][0].animate.set_stroke(ACCENT_TEAL, 2.5), run_time=0.35)

        final = Text("Color is about perception — now we have the math to match.",
                     font_size=24, color=TEXT_PRI)
        final.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(final, shift=UP * 0.3))
        self.wait(6)

        creds = VGroup(
            Text("Based on Björn Ottosson's OKLab  ·  bottosson.github.io",
                 font_size=13, color=ACCENT_BLUE),
            Text("CIE 1931 CMFs (Wyman 2013 fit)  ·  D65 illuminant",
                 font_size=13, color=TEXT_SEC),
            Text("Made with Manim Community Edition", font_size=13, color=TEXT_SEC),
        ).arrange(DOWN, buff=0.08).to_edge(DOWN, buff=0.25)
        self.play(FadeOut(final), FadeIn(creds, shift=UP * 0.15))
        self.wait(5)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.5)
