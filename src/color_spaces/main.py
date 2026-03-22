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
BG             = "#0b0e17"
PANEL          = "#111827"
ACCENT_BLUE    = "#58c4dd"
ACCENT_TEAL    = "#5ce1e6"
ACCENT_PINK    = "#ff6b9d"
ACCENT_PURPLE  = "#b388ff"
ACCENT_ORANGE  = "#ffab40"
ACCENT_YELLOW  = "#ffee58"
ACCENT_GREEN   = "#69f0ae"
TEXT_PRI       = "#e8eaf6"
TEXT_SEC       = "#9fa8da"
GRID_COL       = "#1e2640"
MUTED_RED      = "#ef5350"
MUTED_GREEN    = "#66bb6a"
MUTED_BLUE     = "#42a5f5"

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
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
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
        return t**3 if t > delta else (t - 4.0/29.0) * 3 * delta**2
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
    L  =  0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    A  =  1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    B  =  0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return L, A, B

def oklab_to_linear_srgb(L, a, b):
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l = l_ ** 3;  m = m_ ** 3;  s = s_ ** 3
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
    if   380 <= wl < 440: R, G, B = -(wl-440)/(440-380), 0, 1
    elif 440 <= wl < 490: R, G, B = 0, (wl-440)/(490-440), 1
    elif 490 <= wl < 510: R, G, B = 0, 1, -(wl-510)/(510-490)
    elif 510 <= wl < 580: R, G, B = (wl-510)/(580-510), 1, 0
    elif 580 <= wl < 645: R, G, B = 1, -(wl-645)/(645-580), 0
    elif 645 <= wl <= 780: R, G, B = 1, 0, 0
    else: R, G, B = 0, 0, 0
    if 380 <= wl < 420:   f = 0.3 + 0.7 * (wl - 380) / 40
    elif 700 < wl <= 780: f = 0.3 + 0.7 * (780 - wl) / 80
    else:                 f = 1.0
    return np.array([f * abs(R)**gamma, f * abs(G)**gamma, f * abs(B)**gamma])

# ── CIE 1931 CMF approximation (Wyman et al. 2013 Gaussian fit) ───
def xbar(w):
    return (1.056 * np.exp(-0.5 * ((w - 599.8) / 37.9)**2)
          + 0.362 * np.exp(-0.5 * ((w - 442.0) / 16.0)**2)
          - 0.065 * np.exp(-0.5 * ((w - 501.1) / 20.4)**2))

def ybar(w):
    return (0.821 * np.exp(-0.5 * ((w - 568.8) / 46.9)**2)
          + 0.286 * np.exp(-0.5 * ((w - 530.9) / 16.3)**2))

def zbar(w):
    return (1.217 * np.exp(-0.5 * ((w - 437.0) / 11.8)**2)
          + 0.681 * np.exp(-0.5 * ((w - 459.0) / 26.0)**2))

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
        if dh > 0: h1 += 1
        else:      h2 += 1
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
    C = np.sqrt(a**2 + b**2)
    h = np.degrees(np.arctan2(b, a)) % 360
    return L, C, h

def lch_to_cielab(L, C, h_deg):
    """LCh → CIELAB."""
    h = np.radians(h_deg)
    return L, C * np.cos(h), C * np.sin(h)

def oklab_to_oklch(L, a, b):
    """OKLab → OKLch (L, C, h°).  h in [0, 360)."""
    C = np.sqrt(a**2 + b**2)
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
    return float(np.sqrt(sum((a - b)**2 for a, b in zip(lab1, lab2))))

def delta_e2000(lab1, lab2, kL=1, kC=1, kH=1):
    """CIEDE2000 color difference (Sharma et al. 2005)."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2.0
    C_avg7 = C_avg**7
    G = 0.5 * (1 - np.sqrt(C_avg7 / (C_avg7 + 25**7)))
    a1p = a1 * (1 + G);  a2p = a2 * (1 + G)
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)
    def hp(ap, bv):
        if ap == 0 and bv == 0: return 0.0
        return np.degrees(np.arctan2(bv, ap)) % 360
    h1p = hp(a1p, b1);  h2p = hp(a2p, b2)
    dLp = L2 - L1;  dCp = C2p - C1p
    if C1p * C2p == 0:          dhp = 0.0
    elif abs(h2p - h1p) <= 180: dhp = h2p - h1p
    elif h2p - h1p > 180:       dhp = h2p - h1p - 360
    else:                        dhp = h2p - h1p + 360
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))
    Lbar = (L1 + L2) / 2
    Cbar_p = (C1p + C2p) / 2
    if C1p * C2p == 0:          hbar_p = h1p + h2p
    elif abs(h1p - h2p) <= 180: hbar_p = (h1p + h2p) / 2
    elif h1p + h2p < 360:       hbar_p = (h1p + h2p + 360) / 2
    else:                        hbar_p = (h1p + h2p - 360) / 2
    T = (1 - 0.17 * np.cos(np.radians(hbar_p - 30))
           + 0.24 * np.cos(np.radians(2 * hbar_p))
           + 0.32 * np.cos(np.radians(3 * hbar_p + 6))
           - 0.20 * np.cos(np.radians(4 * hbar_p - 63)))
    SL = 1 + 0.015 * (Lbar - 50)**2 / np.sqrt(20 + (Lbar - 50)**2)
    SC = 1 + 0.045 * Cbar_p
    SH = 1 + 0.015 * Cbar_p * T
    Cbar7 = Cbar_p**7
    RC = 2 * np.sqrt(Cbar7 / (Cbar7 + 25**7))
    d_theta = 30 * np.exp(-((hbar_p - 275) / 25)**2)
    RT = -np.sin(np.radians(2 * d_theta)) * RC
    return float(np.sqrt(
        (dLp / (kL * SL))**2 + (dCp / (kC * SC))**2 + (dHp / (kH * SH))**2
        + RT * (dCp / (kC * SC)) * (dHp / (kH * SH))))

# ── MacAdam 1942 ellipses (25 JND loci, Brown & MacAdam 1949) ────────
# Format: (x_chrom, y_chrom, semi_minor, semi_major, angle_deg_from_x_axis)
MACADAM_ELLIPSES = [
    (0.160, 0.057, 0.0030, 0.0085,  62.5),
    (0.187, 0.118, 0.0027, 0.0102,  77.0),
    (0.253, 0.125, 0.0021, 0.0076,  55.5),
    (0.150, 0.680, 0.0048, 0.0198,   8.0),
    (0.131, 0.521, 0.0039, 0.0158,  11.0),
    (0.212, 0.550, 0.0068, 0.0295,  26.0),
    (0.258, 0.450, 0.0062, 0.0201,  30.0),
    (0.152, 0.365, 0.0032, 0.0196,   4.0),
    (0.280, 0.385, 0.0048, 0.0184,  30.0),
    (0.380, 0.498, 0.0080, 0.0307,  41.0),
    (0.160, 0.200, 0.0023, 0.0100,  34.5),
    (0.228, 0.250, 0.0032, 0.0143,  44.0),
    (0.305, 0.323, 0.0040, 0.0136,  50.0),
    (0.385, 0.393, 0.0043, 0.0151,  36.0),
    (0.472, 0.399, 0.0058, 0.0159,  38.0),
    (0.527, 0.350, 0.0079, 0.0183,  68.0),
    (0.475, 0.300, 0.0058, 0.0141,  47.0),
    (0.510, 0.236, 0.0040, 0.0104,  60.0),
    (0.596, 0.283, 0.0063, 0.0134,  68.0),
    (0.344, 0.284, 0.0037, 0.0118,  48.0),
    (0.390, 0.237, 0.0039, 0.0101,  53.0),
    (0.441, 0.198, 0.0044, 0.0103,  60.0),
    (0.278, 0.223, 0.0027, 0.0082,  37.0),
    (0.240, 0.290, 0.0028, 0.0092,  44.0),
    (0.300, 0.255, 0.0032, 0.0097,  49.0),
]

# ── Color vision deficiency simulation (Viénot et al. 1999) ─────────
# Hunt-Pointer-Estevez matrix (D65 adapted)
M_RGB_TO_LMS = np.array([
    [ 0.4002,  0.7076, -0.0808],
    [-0.2263,  1.1653,  0.0457],
    [ 0.0000,  0.0000,  0.9182],
])
M_LMS_TO_RGB = np.linalg.inv(M_RGB_TO_LMS)

# LMS projection matrices for each deficiency type
M_DEUTERANOPIA = np.array([   # M-cones absent (red-green)
    [1.0,    0.0,    0.0],
    [0.4942, 0.0,    1.2483],
    [0.0,    0.0,    1.0],
])
M_PROTANOPIA = np.array([     # L-cones absent (red-blind)
    [0.0,    2.0234, -2.5258],
    [0.0,    1.0,     0.0],
    [0.0,    0.0,     1.0],
])
M_TRITANOPIA = np.array([     # S-cones absent (blue-yellow)
    [1.0,    0.0,     0.0],
    [0.0,    1.0,     0.0],
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


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 01 — TITLE
# ═══════════════════════════════════════════════════════════════════════
class TitleScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        # Particle field using OKLab hue circle
        dots = VGroup()
        np.random.seed(7)
        for _ in range(150):
            h = np.random.random() * TAU
            L = 0.65 + 0.25 * np.random.random()
            C = 0.08 + 0.10 * np.random.random()
            col = oklab_to_hex(L, C * np.cos(h), C * np.sin(h))
            d = Dot(point=[np.random.uniform(-7.5, 7.5),
                           np.random.uniform(-4.5, 4.5), 0],
                    radius=np.random.uniform(0.02, 0.10),
                    color=col, fill_opacity=np.random.uniform(0.15, 0.55))
            dots.add(d)
        self.play(FadeIn(dots, lag_ratio=0.015), run_time=1.5)

        title = Text("Color Spaces", font_size=84, weight=BOLD, color=TEXT_PRI)
        sub   = Text("From Light to OKLab", font_size=38, color=ACCENT_TEAL)
        sub.next_to(title, DOWN, buff=0.4)
        VGroup(title, sub).move_to(ORIGIN)

        rainbow = Line(LEFT * 4.5, RIGHT * 4.5, stroke_width=4)
        rainbow.set_color(color=[MUTED_RED, ACCENT_ORANGE, ACCENT_YELLOW,
                                  ACCENT_GREEN, ACCENT_BLUE, ACCENT_PURPLE, ACCENT_PINK])
        rainbow.next_to(sub, DOWN, buff=0.35)

        self.play(Write(title, run_time=1.6),
                  FadeIn(sub, shift=UP * 0.3, run_time=1.4))
        self.play(Create(rainbow, run_time=0.8))
        self.play(dots.animate.scale(1.12).set_opacity(0.18), run_time=1.8,
                  rate_func=smooth)
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 02 — ELECTROMAGNETIC SPECTRUM → VISIBLE LIGHT
# ═══════════════════════════════════════════════════════════════════════
class ElectromagneticSpectrumScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("I.  What Is Light?", font_size=44, color=ACCENT_BLUE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        # Full EM spectrum bar
        em_data = [
            ("γ-rays",  "#7b1fa2", 0.8),
            ("X-rays",  "#6a1b9a", 1.0),
            ("UV",      "#4a148c", 0.8),
            ("VISIBLE", None,      2.0),
            ("IR",      "#b71c1c", 1.2),
            ("Micro",   "#880e4f", 1.0),
            ("Radio",   "#4a0033", 2.0),
        ]
        em_bar = VGroup()
        x = -5.4
        vis_start_x = None
        vis_end_x = None
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
                lbl = Text(name, font_size=12, color="#ffffff88")
                lbl.move_to(r)
                em_bar.add(VGroup(r, lbl))
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
        self.wait(1.5)

        # Zoom into visible spectrum
        self.play(*[FadeOut(m) for m in [em_bar, vis_brace, vis_lbl]], run_time=0.6)

        zoom_title = Text("The Visible Spectrum", font_size=28, color=TEXT_PRI)
        zoom_title.move_to(UP * 1.8)
        self.play(FadeIn(zoom_title))

        spectrum = VGroup()
        n = 250
        bw = 11
        for i in range(n):
            wl = 380 + (780 - 380) * i / (n - 1)
            rgb = wavelength_to_rgb(wl)
            rect = Rectangle(width=bw/n + 0.005, height=1.0,
                             fill_color=rgb_to_hex(rgb), fill_opacity=1,
                             stroke_width=0)
            rect.move_to(LEFT * bw/2 + RIGHT * (i * bw/n + bw/n/2) + UP * 0.4)
            spectrum.add(rect)
        self.play(FadeIn(spectrum, lag_ratio=0.003, run_time=2))

        # Wavelength ticks
        ticks = VGroup()
        for wl in [400, 450, 500, 550, 600, 650, 700, 750]:
            frac = (wl - 380) / (780 - 380)
            xp = -bw/2 + frac * bw
            tick = Line([xp, -0.15, 0], [xp, -0.35, 0],
                        color=TEXT_SEC, stroke_width=1.5)
            lbl = Text(str(wl), font_size=14, color=TEXT_SEC)
            lbl.next_to(tick, DOWN, buff=0.05)
            ticks.add(VGroup(tick, lbl))
        nm = Text("wavelength (nm)", font_size=16, color=TEXT_SEC).move_to(DOWN * 0.9)
        self.play(FadeIn(ticks), FadeIn(nm))

        # Wave animation below spectrum
        wave_note = Text(
            "Shorter wavelength = higher energy (violet) · Longer = lower energy (red)",
            font_size=18, color=TEXT_SEC)
        wave_note.move_to(DOWN * 1.6)
        self.play(FadeIn(wave_note, shift=UP * 0.15))

        insight = Text(
            "Each wavelength triggers a unique response in our eyes",
            font_size=22, color=ACCENT_TEAL)
        insight.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(insight, shift=UP * 0.2))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 03 — HUMAN VISION: CONE CELLS
# ═══════════════════════════════════════════════════════════════════════
class HumanVisionScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("II.  How We See Color", font_size=44,
                   color=ACCENT_BLUE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        # Simplified eye cross-section
        eye = Ellipse(width=2.6, height=2.2, color=TEXT_SEC, stroke_width=2)
        lens = Arc(radius=0.6, start_angle=-PI/3, angle=2*PI/3,
                   color=ACCENT_BLUE, stroke_width=2.5)
        lens.move_to(eye.get_left() + RIGHT * 0.6)
        pupil = Dot(eye.get_left() + RIGHT * 0.35, radius=0.18,
                    color="#1a1a2e", fill_opacity=1)
        retina = Arc(radius=1.2, start_angle=-PI/4, angle=PI/2,
                     color=ACCENT_ORANGE, stroke_width=3.5)
        retina.move_to(eye.get_right() + LEFT * 0.2)
        ret_lbl = Text("Retina", font_size=14, color=ACCENT_ORANGE)
        ret_lbl.next_to(retina, RIGHT, buff=0.15)
        eye_group = VGroup(eye, lens, pupil, retina, ret_lbl)
        eye_group.move_to(LEFT * 4 + UP * 0.5)
        self.play(Create(eye), FadeIn(lens), FadeIn(pupil),
                  Create(retina), FadeIn(ret_lbl))

        # Arrow
        arrow = Arrow(eye.get_right() + RIGHT * 0.4, RIGHT * 0.2 + UP * 0.5,
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
        y_lbl = Text("response", font_size=15, color=TEXT_SEC).rotate(PI/2)
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

        self.play(Create(axes), FadeIn(x_lbl), FadeIn(y_lbl), run_time=0.7)
        self.play(Create(s_plot), FadeIn(s_area), FadeIn(s_lbl), run_time=0.9)
        self.play(Create(m_plot), FadeIn(m_area), FadeIn(m_lbl), run_time=0.9)
        self.play(Create(l_plot), FadeIn(l_area), FadeIn(l_lbl), run_time=0.9)

        box = RoundedRectangle(corner_radius=0.12, width=11, height=1.0,
                                fill_color=PANEL, fill_opacity=0.9,
                                stroke_color=ACCENT_YELLOW, stroke_width=1.5)
        box.to_edge(DOWN, buff=0.3)
        insight = Text(
            "3 cone types → every color we see is described by just 3 numbers  (trichromacy)",
            font_size=20, color=ACCENT_YELLOW)
        insight.move_to(box)
        self.play(FadeIn(box), FadeIn(insight))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 04 — CIE 1931: COLOR MATCHING FUNCTIONS & XYZ
# ═══════════════════════════════════════════════════════════════════════
class CIE1931Scene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("III.  The CIE 1931 Standard", font_size=44,
                   color=ACCENT_PURPLE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        # Context
        ctx = Text("International Commission on Illumination — "
                    "first mathematical model of human color vision",
                    font_size=18, color=TEXT_SEC)
        ctx.next_to(ch, DOWN, buff=0.35)
        self.play(FadeIn(ctx))

        # Experiment diagram
        exp = RoundedRectangle(corner_radius=0.15, width=10, height=2.0,
                                fill_color=PANEL, fill_opacity=0.9,
                                stroke_color=ACCENT_PURPLE, stroke_width=1.5)
        exp.move_to(UP * 0.3)
        self.play(FadeIn(exp))

        obs = Text("👁", font_size=36).move_to(exp.get_left() + RIGHT * 1.0)
        test = Rectangle(width=1.5, height=1.2, fill_color="#553311",
                          fill_opacity=0.5, stroke_color=TEXT_SEC, stroke_width=1)
        test.move_to(exp.get_center() + LEFT * 1.5)
        test_lbl = Text("Test λ", font_size=14, color=TEXT_PRI).move_to(test)
        match = Rectangle(width=1.5, height=1.2, fill_color="#334455",
                           fill_opacity=0.5, stroke_color=TEXT_SEC, stroke_width=1)
        match.move_to(exp.get_center() + RIGHT * 0.2)
        match_lbl = Text("r·R+g·G+b·B", font_size=12, color=TEXT_PRI).move_to(match)
        result_txt = Text("Record r,g,b\nfor every λ", font_size=15, color=ACCENT_ORANGE)
        result_txt.move_to(exp.get_right() + LEFT * 1.3)

        self.play(FadeIn(obs), FadeIn(VGroup(test, test_lbl)),
                  FadeIn(VGroup(match, match_lbl)), FadeIn(result_txt))
        self.wait(1.5)

        # Transition to CMFs
        self.play(FadeOut(VGroup(exp, obs, test, test_lbl, match,
                                  match_lbl, result_txt, ctx)), run_time=0.5)

        cmf_title = Text("Color Matching Functions  x̄(λ), ȳ(λ), z̄(λ)", font_size=24,
                          color=TEXT_PRI)
        cmf_title.next_to(ch, DOWN, buff=0.4)
        self.play(FadeIn(cmf_title))

        axes = Axes(
            x_range=[380, 780, 50], y_range=[-0.1, 1.85, 0.5],
            x_length=10, y_length=3.8, tips=False,
            axis_config={"color": GRID_COL, "stroke_width": 1.5})
        axes.move_to(DOWN * 0.6)

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

        self.play(Create(axes, run_time=0.6))
        self.play(Create(x_plot, run_time=1.2), FadeIn(xl))
        self.play(Create(y_plot, run_time=1.2), FadeIn(yl))
        self.play(Create(z_plot, run_time=1.2), FadeIn(zl))

        formula = MathTex(
            r"X = \int S(\lambda)\,\bar{x}(\lambda)\,d\lambda",
            r"\quad Y = \int S(\lambda)\,\bar{y}(\lambda)\,d\lambda",
            r"\quad Z = \int S(\lambda)\,\bar{z}(\lambda)\,d\lambda",
            font_size=22, color=TEXT_PRI)
        formula.to_edge(DOWN, buff=0.3)
        self.play(Write(formula, run_time=1.5))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 05 — CHROMATICITY DIAGRAM
# ═══════════════════════════════════════════════════════════════════════
class ChromaticityScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("IV.  CIE Chromaticity Diagram", font_size=42,
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
        self.play(Create(axes), FadeIn(xa), FadeIn(ya), run_time=0.7)

        # Spectral locus
        locus_dots = VGroup()
        for wl in np.arange(400, 701, 2):
            X, Y, Z = xbar(wl), ybar(wl), zbar(wl)
            s = X + Y + Z
            if s > 0.001:
                xc, yc = X/s, Y/s
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
        self.wait(1.5)

        # --- Act 2: Wide Color Gamuts ---
        self.play(FadeOut(note), run_time=0.4)
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
        self.play(Create(p3_tri), run_time=0.7)
        self.play(Create(bt_tri), run_time=0.7)

        leg_srgb = Text("■ sRGB     ~35% of visible gamut", font_size=13, color=WHITE)
        leg_p3   = Text("■ P3       ~45% of visible gamut", font_size=13, color=ACCENT_ORANGE)
        leg_bt   = Text("■ BT.2020  ~75% of visible gamut", font_size=13, color=ACCENT_PINK)
        legend = VGroup(leg_srgb, leg_p3, leg_bt).arrange(DOWN, buff=0.15,
                                                            aligned_edge=LEFT)
        legend.move_to(RIGHT * 4 + DOWN * 1.0)
        self.play(FadeIn(legend))

        cam_note = Text("iPhone cameras capture in P3 —\nyour vision pipeline must handle it",
                         font_size=14, color=TEXT_SEC, line_spacing=1.2)
        cam_note.move_to(RIGHT * 4 + DOWN * 2.5)
        self.play(FadeIn(cam_note))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 06 — MACADAM ELLIPSES (WHY CIE XY FAILS)
# ═══════════════════════════════════════════════════════════════════════
class MacAdamEllipsesScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("V.  MacAdam Ellipses  (1942)", font_size=42,
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
        self.play(Create(axes), FadeIn(xa), FadeIn(ya), run_time=0.6)

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
        x_sc = 5.0 / 0.85   # Manim units per chromaticity unit, x
        y_sc = 5.4 / 0.90   # Manim units per chromaticity unit, y
        ellipses_grp = VGroup()
        for (xc, yc, sa, sb, theta_deg) in MACADAM_ELLIPSES:
            e = Ellipse(
                width  = sb * 2 * SCALE * x_sc,
                height = sa * 2 * SCALE * y_sc,
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
        self.play(Create(axes2), FadeIn(xa2), FadeIn(ya2), run_time=0.5)

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
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 07 — THE RGB CUBE IN 3D
# ═══════════════════════════════════════════════════════════════════════
class RGBCubeScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG
        ch = Text("VI.  The RGB Color Cube", font_size=42,
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

        self.play(Create(axes, run_time=0.7))
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
        gray = Line3D(axes.c2p(0,0,0), axes.c2p(1,1,1),
                      color=WHITE, stroke_width=2.5)
        gray_lbl = Text("Grayscale", font_size=13, color=WHITE)
        gray_lbl.move_to(axes.c2p(0.5, 0.5, 0.5) + np.array([0.4, 0.3, 0]))
        self.add_fixed_orientation_mobjects(gray_lbl)
        self.play(Create(gray), FadeIn(gray_lbl))

        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(5)
        self.stop_ambient_camera_rotation()

        note = Text("RGB: simple for hardware, not for human perception",
                     font_size=22, color=ACCENT_PINK)
        note.to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(note)
        self.play(FadeIn(note))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 07 — HSV CYLINDER
# ═══════════════════════════════════════════════════════════════════════
class HSVCylinderScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG
        ch = Text("VII.  HSV: Reshaping the Cube", font_size=42,
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

        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(4)
        self.stop_ambient_camera_rotation()

        note = Text("HSV separates hue — but \"Value\" ≠ perceived lightness",
                     font_size=20, color=ACCENT_PINK)
        note.to_edge(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(note)
        self.play(FadeOut(anns), FadeIn(note))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 08 — PERCEPTUAL PROBLEMS
# ═══════════════════════════════════════════════════════════════════════
class PerceptualProblemsScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("VIII.  The Perceptual Problem", font_size=44,
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
            rect = Rectangle(width=bw/n+0.005, height=0.55,
                             fill_color=rgb_to_hex([r,g,b]), fill_opacity=1,
                             stroke_width=0)
            rect.move_to(LEFT*bw/2 + RIGHT*(i*bw/n + bw/n/2) + UP*0.8)
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

        self.play(Create(axes, run_time=0.5))
        self.play(Create(curve, run_time=1.5))
        self.play(Create(ideal), FadeIn(ideal_lbl))

        y_dot = Dot(axes.c2p(1/6, hue_to_okL(1/6)),
                    color=ACCENT_YELLOW, radius=0.07)
        b_dot = Dot(axes.c2p(2/3, hue_to_okL(2/3)),
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
        self.play(FadeIn(verdict, shift=UP*0.2))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 09 — CIELAB DERIVATION (full math)
# ═══════════════════════════════════════════════════════════════════════
class CIELABDerivationScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("IX.  Deriving CIELAB  (1976)", font_size=42,
                   color=ACCENT_PURPLE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN*0.2))

        # Key idea
        idea_box = RoundedRectangle(corner_radius=0.12, width=10, height=1.1,
                                     fill_color=PANEL, fill_opacity=0.9,
                                     stroke_color=ACCENT_PURPLE, stroke_width=1.5)
        idea_box.move_to(UP*1.2)
        idea = Text("Key insight: perceived brightness ≈ cube root of luminance",
                     font_size=21, color=ACCENT_YELLOW)
        idea.move_to(idea_box)
        self.play(FadeIn(idea_box), FadeIn(idea))

        # f(t) function
        ft_title = Text("The nonlinear transfer function:", font_size=19, color=TEXT_PRI)
        ft_title.move_to(LEFT*2.5 + UP*0.3)
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
        axes.move_to(RIGHT*3 + DOWN*0.3)

        def f_of_t(t):
            d = 6/29
            return np.cbrt(t) if t > d**3 else t/(3*d**2) + 4/29

        f_plot = axes.plot(f_of_t, color=ACCENT_PURPLE, stroke_width=3,
                           x_range=[0.001, 1.02, 0.005])
        cbrt_plot = axes.plot(lambda t: np.cbrt(t), color=ACCENT_ORANGE,
                              stroke_width=2, x_range=[0.001, 1.02, 0.01])
        lin_seg = axes.plot(lambda t: t/(3*(6/29)**2) + 4/29, color="#888888",
                            stroke_width=1.5, x_range=[0, (6/29)**3 + 0.01, 0.001])

        cbrt_lbl = MathTex(r"\sqrt[3]{t}", font_size=18, color=ACCENT_ORANGE)
        cbrt_lbl.move_to(axes.c2p(0.92, 1.03))
        f_lbl = MathTex(r"f(t)", font_size=18, color=ACCENT_PURPLE)
        f_lbl.move_to(axes.c2p(0.55, 0.88))
        lin_lbl = Text("linear\nsegment", font_size=11, color="#888888")
        lin_lbl.move_to(axes.c2p(0.15, 0.45))

        self.play(Create(axes, run_time=0.5))
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
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 10 — CIELAB 3D COLOR SOLID
# ═══════════════════════════════════════════════════════════════════════
class CIELABSolidScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG
        ch = Text("X.  The CIELAB Color Solid", font_size=42,
                   color=ACCENT_PURPLE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.add_fixed_in_frame_mobjects(ch)
        self.play(FadeIn(ch, shift=DOWN*0.2))

        self.set_camera_orientation(phi=65*DEGREES, theta=-40*DEGREES)

        pts = generate_gamut_surface(res=16)
        lab_dots = VGroup()
        sc_ab = 0.032
        sc_L  = 0.042

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
        ax_a = Line3D([-3.5,0,0], [3.5,0,0], color=GRID_COL, stroke_width=1)
        ax_b = Line3D([0,-3.5,0], [0,3.5,0], color=GRID_COL, stroke_width=1)
        ax_L = Line3D([0,0,-2.5], [0,0,2.5], color=GRID_COL, stroke_width=1)

        al = Text("a*  (green↔red)", font_size=15, color="#cc66cc")
        al.move_to([3.8, 0, 0])
        bll = Text("b*  (blue↔yellow)", font_size=15, color="#ccaa33")
        bll.move_to([0, 3.8, 0])
        Ll = Text("L*  (lightness)", font_size=15, color=TEXT_PRI)
        Ll.move_to([0.4, 0, 2.8])
        for lbl in [al, bll, Ll]:
            self.add_fixed_orientation_mobjects(lbl)

        self.play(Create(ax_a), Create(ax_b), Create(ax_L), run_time=0.4)
        self.play(FadeIn(al), FadeIn(bll), FadeIn(Ll))
        self.play(FadeIn(lab_dots, lag_ratio=0.001, run_time=3))

        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(5)
        self.stop_ambient_camera_rotation()

        note = Text("Notice the tilted, irregular shape — especially near blue",
                     font_size=20, color=ACCENT_PINK)
        note.to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(note)
        self.play(FadeIn(note))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 12 — ΔE COLOR DIFFERENCE
# ═══════════════════════════════════════════════════════════════════════
class DeltaEScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XI.  ΔE: Measuring Color Difference", font_size=42,
                   color=ACCENT_PURPLE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN * 0.2))

        # ── Act 1: Live ΔE demonstration ──
        ref_rgb  = np.array([0.80, 0.35, 0.10])
        end_rgb  = np.array([0.10, 0.55, 0.80])
        ref_lab  = srgb_to_cielab(*ref_rgb)

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
        self.wait(0.5)
        self.play(*[FadeOut(m) for m in [ref_swatch, ref_lbl, cand_swatch, cand_lbl,
                                          de_label, de_num, thresh_text]], run_time=0.5)

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
        self.wait(1.0)
        self.play(*[FadeOut(m) for m in [form_title, eq76, eq00_intro, eq00]],
                  run_time=0.4)

        # ── Act 3: Engineering tolerance table ──
        tbl_title = Text("Industry tolerance standards:", font_size=20,
                          color=ACCENT_YELLOW)
        tbl_title.move_to(UP * 2.0)
        self.play(FadeIn(tbl_title))

        rows = [
            ("Printing / brand color matching", "ΔE₀₀  <  1.0", ACCENT_GREEN),
            ("Medical imaging displays",          "ΔE₇₆  <  2.0", ACCENT_YELLOW),
            ("Consumer display calibration",      "ΔE₀₀  <  3.0", ACCENT_ORANGE),
        ]
        row_grps = VGroup()
        for label, threshold, col in rows:
            lbl_t = Text(label,     font_size=19, color=TEXT_PRI)
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
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 13 — CIELAB HUE PROBLEM (+ comparison with OKLab)
# ═══════════════════════════════════════════════════════════════════════
class CIELABProblemsScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XII.  CIELAB's Achilles Heel", font_size=42,
                   color=ACCENT_PINK, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN*0.2))

        title = Text("Hue ring at constant lightness & chroma",
                      font_size=20, color=TEXT_SEC)
        title.next_to(ch, DOWN, buff=0.35)
        self.play(FadeIn(title))

        # CIELAB hue ring
        center_l = LEFT*3.2 + DOWN*0.8
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
        center_r = RIGHT*3.2 + DOWN*0.8
        ring_ok = VGroup()
        for i in range(n_ring):
            angle = TAU * i / n_ring
            L_ok, C_ok = 0.72, 0.12
            col = oklab_to_hex(L_ok, C_ok*np.cos(angle), C_ok*np.sin(angle))
            d = Dot(center_r + ring_r * np.array([np.cos(angle),
                     np.sin(angle), 0]),
                    radius=0.14, color=col)
            ring_ok.add(d)

        ok_lbl = Text("OKLab", font_size=20, color=ACCENT_TEAL, weight=BOLD)
        ok_lbl.next_to(ring_ok, DOWN, buff=0.3)
        self.play(FadeIn(ring_ok, lag_ratio=0.01, run_time=1.5), FadeIn(ok_lbl))

        # Problem arrow
        prob_arrow = Arrow(
            center_l + DOWN*1.2 + LEFT*1.2,
            center_l + ring_r * np.array([np.cos(4.2), np.sin(4.2), 0]),
            color=ACCENT_PINK, stroke_width=2.5, buff=0.15)
        prob_txt = Text("Blue → purple\nshift!", font_size=14, color=ACCENT_PINK)
        prob_txt.next_to(prob_arrow.get_start(), DOWN, buff=0.1)
        self.play(GrowArrow(prob_arrow), FadeIn(prob_txt))

        # vs label
        vs = Text("vs", font_size=28, color=TEXT_SEC).move_to(DOWN*0.8)
        self.play(FadeIn(vs))

        verdict = Text("CIELAB's hues are not uniform — OKLab fixes this",
                        font_size=22, color=ACCENT_TEAL)
        verdict.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(verdict, shift=UP*0.2))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 14 — LCh AND OKLch (CYLINDRICAL LAB)
# ═══════════════════════════════════════════════════════════════════════
class LChOKLchScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XIII.  LCh and OKLch: Cylindrical Lab", font_size=42,
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
        xl = Text("a  (green↔red)",   font_size=11, color=TEXT_SEC)
        xl.next_to(plane.x_axis, DOWN, buff=0.08)
        yl = Text("b  (blue↔yellow)", font_size=11, color=TEXT_SEC)
        yl.next_to(plane.y_axis, LEFT, buff=0.08)
        self.play(Create(plane), FadeIn(xl), FadeIn(yl), run_time=0.6)

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
            (hue_sweep,       "Hue sweep  (L=0.72, C=0.15)"),
            (chroma_sweep,    "Chroma sweep  (L=0.72, h=240°)"),
            (lightness_sweep, "Lightness sweep  (C=0.12, h=240°)"),
        ]
        y_off = -0.15
        for fn, lbl_str in bar_data:
            bar = make_gradient_bar(fn, dummy, dummy, n=80, width=BAR_W, height=0.38)
            bar.move_to(BAR_X + UP * y_off)
            bar_lbl = Text(lbl_str, font_size=11, color=TEXT_SEC)
            bar_lbl.next_to(bar, DOWN, buff=0.06)
            self.play(FadeIn(bar, lag_ratio=0.004), FadeIn(bar_lbl), run_time=0.6)
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
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 15 — OKLAB DERIVATION (full pipeline + optimization story)
# ═══════════════════════════════════════════════════════════════════════
class OKLabDerivationScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XIV.  Deriving OKLab  (2020)", font_size=44,
                   color=ACCENT_TEAL, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN*0.2))

        credit = Text("Björn Ottosson — \"An OK Lab color space\"",
                       font_size=18, color=TEXT_SEC)
        credit.next_to(ch, DOWN, buff=0.3)
        self.play(FadeIn(credit))

        # Idea
        idea = Text("Take the simple IPT structure, but optimize its matrices "
                     "against CAM16 perceptual data",
                     font_size=18, color=ACCENT_YELLOW)
        idea.next_to(credit, DOWN, buff=0.35)
        self.play(FadeIn(idea, shift=DOWN*0.1))

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
        boxes.arrange(RIGHT, buff=0.55).move_to(UP*0.1)
        for i in range(len(boxes)-1):
            arr = Arrow(boxes[i].get_right(), boxes[i+1].get_left(),
                        buff=0.07, color=TEXT_SEC, stroke_width=2.5,
                        max_tip_length_to_length_ratio=0.15)
            arrows.add(arr)

        for i, box in enumerate(boxes):
            self.play(FadeIn(box, scale=0.85), run_time=0.4)
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
        eqs = VGroup(eq1, eq2, eq3).arrange(RIGHT, buff=0.4).move_to(DOWN*1.3)
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
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 13 — OKLAB 3D COLOR SOLID
# ═══════════════════════════════════════════════════════════════════════
class OKLabSolidScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG
        ch = Text("XV.  The OKLab Color Solid", font_size=42,
                   color=ACCENT_TEAL, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.add_fixed_in_frame_mobjects(ch)
        self.play(FadeIn(ch, shift=DOWN*0.2))

        self.set_camera_orientation(phi=65*DEGREES, theta=-40*DEGREES)

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

        ax_a = Line3D([-3,0,0], [3,0,0], color=GRID_COL, stroke_width=1)
        ax_b = Line3D([0,-3,0], [0,3,0], color=GRID_COL, stroke_width=1)
        ax_L = Line3D([0,0,-4], [0,0,4], color=GRID_COL, stroke_width=1)

        al = Text("a", font_size=18, color="#cc6666").move_to([3.3, 0, 0])
        bll = Text("b", font_size=18, color="#ccaa33").move_to([0, 3.3, 0])
        Ll = Text("L", font_size=18, color=TEXT_PRI).move_to([0.3, 0, 4.3])
        for lbl in [al, bll, Ll]:
            self.add_fixed_orientation_mobjects(lbl)

        self.play(Create(ax_a), Create(ax_b), Create(ax_L), run_time=0.4)
        self.play(FadeIn(al), FadeIn(bll), FadeIn(Ll))
        self.play(FadeIn(ok_dots, lag_ratio=0.001, run_time=3))

        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(5)
        self.stop_ambient_camera_rotation()

        note = Text("Smoother, more symmetric — perceptually uniform in all directions",
                     font_size=20, color=ACCENT_TEAL)
        note.to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(note)
        self.play(FadeIn(note))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 14 — SIDE-BY-SIDE 3D COMPARISON
# ═══════════════════════════════════════════════════════════════════════
class ColorSpaceComparisonScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG
        ch = Text("XVI.  3D Comparison: RGB · CIELAB · OKLab", font_size=38,
                   color=ACCENT_GREEN, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.add_fixed_in_frame_mobjects(ch)
        self.play(FadeIn(ch, shift=DOWN*0.2))

        self.set_camera_orientation(phi=60*DEGREES, theta=-45*DEGREES)

        pts = generate_gamut_volume(res=8)
        sc_rgb  = 3.0
        sc_lab  = 0.025
        sc_labL = 0.033
        sc_ok   = 5.5
        offset  = 4.8

        # RGB
        rgb_grp = VGroup()
        for s in pts:
            x, y, z = (s - 0.5) * sc_rgb
            d = Dot3D([x - offset, y, z], radius=0.03, color=rgb_to_hex(s))
            d.set_opacity(0.75)
            rgb_grp.add(d)
        rgb_lbl = Text("RGB", font_size=18, color=ACCENT_BLUE, weight=BOLD)
        rgb_lbl.move_to([-offset, 0, -2.3])
        self.add_fixed_orientation_mobjects(rgb_lbl)

        # CIELAB
        lab_grp = VGroup()
        for s in pts:
            L, a, bv = srgb_to_cielab(*s)
            d = Dot3D([a*sc_lab, bv*sc_lab, (L-50)*sc_labL], radius=0.03,
                      color=rgb_to_hex(s))
            d.set_opacity(0.75)
            lab_grp.add(d)
        lab_lbl = Text("CIELAB", font_size=18, color=ACCENT_PURPLE, weight=BOLD)
        lab_lbl.move_to([0, 0, -2.3])
        self.add_fixed_orientation_mobjects(lab_lbl)

        # OKLab
        ok_grp = VGroup()
        for s in pts:
            lin = srgb_to_linear(np.array(s))
            L, a, bv = linear_srgb_to_oklab(*lin)
            d = Dot3D([a*sc_ok + offset, bv*sc_ok, (L-0.5)*sc_ok],
                      radius=0.03, color=rgb_to_hex(s))
            d.set_opacity(0.75)
            ok_grp.add(d)
        ok_lbl = Text("OKLab", font_size=18, color=ACCENT_TEAL, weight=BOLD)
        ok_lbl.move_to([offset, 0, -2.3])
        self.add_fixed_orientation_mobjects(ok_lbl)

        self.play(FadeIn(rgb_grp, lag_ratio=0.001, run_time=1.5), FadeIn(rgb_lbl))
        self.play(FadeIn(lab_grp, lag_ratio=0.001, run_time=1.5), FadeIn(lab_lbl))
        self.play(FadeIn(ok_grp, lag_ratio=0.001, run_time=1.5), FadeIn(ok_lbl))

        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(6)
        self.stop_ambient_camera_rotation()

        note = Text("Same colors, three shapes — OKLab is the most regular",
                     font_size=20, color=ACCENT_GREEN)
        note.to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(note)
        self.play(FadeIn(note))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 15 — GRADIENT COMPARISON (hero scene)
# ═══════════════════════════════════════════════════════════════════════
class GradientComparisonScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XVII.  Gradient Quality Test", font_size=42,
                   color=ACCENT_GREEN, weight=BOLD)
        ch.to_edge(UP, buff=0.4)
        self.play(FadeIn(ch, shift=DOWN*0.2))

        pairs = [
            ([0.0, 0.4, 1.0], [1.0, 0.8, 0.0], "Blue → Yellow"),
            ([1.0, 0.0, 0.33], [0.0, 0.87, 0.67], "Red → Teal"),
            ([1.0, 1.0, 1.0], [0.0, 0.2, 0.73], "White → Blue"),
        ]
        methods = [
            ("sRGB",  srgb_blend,      ACCENT_ORANGE),
            ("HSV",   hsv_blend_fn,    ACCENT_PINK),
            ("OKLab", oklab_blend_fn,  ACCENT_TEAL),
            ("OKLch", oklch_blend_fn,  ACCENT_GREEN),
        ]

        all_items = VGroup()
        y_pos = 2.2
        for c1, c2, pair_name in pairs:
            pl = Text(pair_name, font_size=15, color=TEXT_SEC)
            pl.move_to(LEFT*6 + UP*y_pos)
            all_items.add(pl)
            for j, (mn, fn, mc) in enumerate(methods):
                bar = make_gradient_bar(fn, c1, c2, n=90, width=9.5, height=0.32)
                bar.move_to(RIGHT*0.3 + UP*(y_pos - j*0.40))
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
                          FadeIn(all_items[idx+1]), run_time=0.45)
                idx += 2
            self.wait(0.2)

        verdict = Text("OKLab / OKLch: no mud, no hue shifts, perceptually smooth",
                        font_size=22, color=ACCENT_TEAL)
        verdict.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(verdict, shift=UP*0.2))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 16 — REAL WORLD ADOPTION
# ═══════════════════════════════════════════════════════════════════════
class RealWorldScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XVIII.  OKLab in the Wild", font_size=42,
                   color=ACCENT_ORANGE, weight=BOLD)
        ch.to_edge(UP, buff=0.45)
        self.play(FadeIn(ch, shift=DOWN*0.2))

        adopters = [
            ("Photoshop",     "Default gradient interpolation (2023+)"),
            ("CSS Color 4/5", "oklab() and oklch() in all browsers"),
            ("Unity / Godot", "Gradient system / color picker"),
            ("Figma",         "Color blending engine"),
        ]
        items = VGroup()
        for name, desc in adopters:
            n = Text(name, font_size=20, color=ACCENT_ORANGE, weight=BOLD)
            d = Text(f"  —  {desc}", font_size=16, color=TEXT_SEC)
            items.add(VGroup(n, d).arrange(RIGHT, buff=0.08))
        items.arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        items.move_to(LEFT*1.5 + UP*0.4)
        for item in items:
            self.play(FadeIn(item, shift=RIGHT*0.3), run_time=0.35)

        # OKLab wheel
        wc = RIGHT*3.8
        wheel = VGroup()
        n_h, n_r = 60, 10
        max_rad = 1.7
        for ih in range(n_h):
            for ir in range(1, n_r+1):
                angle = TAU * ih / n_h
                radius = max_rad * ir / n_r
                C = 0.15 * ir / n_r
                col = oklab_to_hex(0.75, C*np.cos(angle), C*np.sin(angle))
                d = Dot(wc + radius*np.array([np.cos(angle), np.sin(angle), 0]),
                        radius=0.09, color=col, fill_opacity=0.92)
                wheel.add(d)
        wl = Text("OKLab Wheel", font_size=15, color=TEXT_SEC)
        wl.next_to(wheel, DOWN, buff=0.25)
        self.play(FadeIn(wheel, lag_ratio=0.002, run_time=2.5), FadeIn(wl))

        # Spinning highlight
        hl = Circle(radius=0.2, color=WHITE, stroke_width=2.5, fill_opacity=0)
        hl.move_to(wc + RIGHT*max_rad*0.7)
        hl.t = 0
        def spin(m, dt):
            m.t += dt
            a = m.t * 0.7
            r = max_rad * 0.7
            m.move_to(wc + r*np.array([np.cos(a), np.sin(a), 0]))
        hl.add_updater(spin)
        self.add(hl)
        self.wait(4)
        hl.remove_updater(spin)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 20 — COLOR BLINDNESS & ACCESSIBILITY
# ═══════════════════════════════════════════════════════════════════════
class ColorBlindnessScene(Scene):
    def construct(self):
        self.camera.background_color = BG

        ch = Text("XIX.  Color Blindness: Engineering for Accessibility",
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
        self.wait(1.0)

        # ── Act 2: Four OKLab wheels ──
        self.play(FadeOut(stats), run_time=0.4)

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
            (np.array([-4.8, 0.2, 0]), None,                  "Normal",
             TEXT_PRI),
            (np.array([-1.6, 0.2, 0]), simulate_deuteranopia, "Deuteranopia\n(red-green)",
             ACCENT_ORANGE),
            (np.array([ 1.6, 0.2, 0]), simulate_protanopia,   "Protanopia\n(red-blind)",
             ACCENT_PINK),
            (np.array([ 4.8, 0.2, 0]), simulate_tritanopia,   "Tritanopia\n(blue-yellow)",
             ACCENT_BLUE),
        ]

        for center, cvd_fn, label_str, col in wheel_configs:
            wheel = make_wheel(center, cvd_fn)
            lbl = Text(label_str, font_size=14, color=col, line_spacing=1.1)
            lbl.move_to(center + DOWN * 1.55)
            self.play(FadeIn(wheel, lag_ratio=0.001, run_time=1.0), FadeIn(lbl))

        self.wait(1.5)

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
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ═══════════════════════════════════════════════════════════════════════
#  SCENE 21 — OUTRO & SUMMARY
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
            ("CIE XYZ",   "1931\nFirst standard\nNot uniform\nBasis for all",
             ACCENT_PURPLE, "~"),
            ("CIELAB",    "1976\nCube-root f(t)\nBetter but\nblue problems",
             ACCENT_PURPLE, "~"),
            ("OKLab",     "2020\nPerceptually uniform\nSimple · fast\nIndustry standard",
             ACCENT_TEAL, "✓"),
        ]

        cg = VGroup()
        for title, desc, col, sym in cards:
            box = RoundedRectangle(corner_radius=0.18, width=2.7, height=3.3,
                                   fill_color=PANEL, fill_opacity=0.95,
                                   stroke_color=col, stroke_width=2.5)
            s = Text(sym, font_size=34,
                     color=MUTED_RED if sym=="✗" else
                           ACCENT_YELLOW if sym=="~" else ACCENT_GREEN)
            s.move_to(box.get_top() + DOWN*0.35)
            t = Text(title, font_size=18, color=col, weight=BOLD)
            t.next_to(s, DOWN, buff=0.15)
            d = Text(desc, font_size=12, color=TEXT_SEC, line_spacing=1.2)
            d.next_to(t, DOWN, buff=0.2)
            cg.add(VGroup(box, s, t, d))
        cg.arrange(RIGHT, buff=0.3).move_to(DOWN*0.1)

        for card in cg:
            self.play(FadeIn(card, shift=UP*0.25, scale=0.9), run_time=0.55)

        arr = Arrow(cg[0].get_bottom() + DOWN*0.2,
                    cg[3].get_bottom() + DOWN*0.2,
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
        self.play(FadeIn(final, shift=UP*0.3))
        self.wait(3)

        creds = VGroup(
            Text("Based on Björn Ottosson's OKLab  ·  bottosson.github.io",
                 font_size=13, color=ACCENT_BLUE),
            Text("CIE 1931 CMFs (Wyman 2013 fit)  ·  D65 illuminant",
                 font_size=13, color=TEXT_SEC),
            Text("Made with Manim Community Edition", font_size=13, color=TEXT_SEC),
        ).arrange(DOWN, buff=0.08).to_edge(DOWN, buff=0.25)
        self.play(FadeOut(final), FadeIn(creds, shift=UP*0.15))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.5)