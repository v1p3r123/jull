"""
Tilpasser en vektormodel af et snefnug til fotoet.

Model (6 ens arme, hver spejlsymmetrisk):
  - kerne: sekskant med radius rc
  - arm: stang fra centrum til L, bredde w, afrundet spids
  - n gren-par: udgår fra armen i afstand b_i, vinkel phi, længde l_i, bredde wb
Parametrene findes ved at minimere forskellen mellem den (slørede) model og fotoets
hvidhed – dvs. modellen tilpasses fotoet matematisk.
"""
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.optimize import minimize

FIT_UPS = 5


def render(params, n, shape, scale=1.0, ss=2):
    """Tegn modellen som maske (0/1) i et billede af størrelse shape (h, w), pixel-koordinater."""
    cx, cy, th, rc, L, w, phi, wb, psf = params[:9]
    br = params[9:]
    H, W = shape
    im = Image.new("L", (W * ss, H * ss), 0)
    d = ImageDraw.Draw(im)
    S = ss * scale

    def P(x, y):
        return (x * S, y * S)

    def bar(x0, y0, x1, y1, width):
        width = max(width, 0.3)
        dx, dy = x1 - x0, y1 - y0
        ln = math.hypot(dx, dy) or 1e-6
        nx, ny = -dy / ln * width / 2, dx / ln * width / 2
        d.polygon([P(x0 + nx, y0 + ny), P(x1 + nx, y1 + ny), P(x1 - nx, y1 - ny), P(x0 - nx, y0 - ny)], fill=255)
        r = width / 2
        d.ellipse([P(x1 - r, y1 - r), P(x1 + r, y1 + r)], fill=255)

    hexa = [P(cx + rc * math.cos(th + k * math.pi / 3), cy + rc * math.sin(th + k * math.pi / 3)) for k in range(6)]
    d.polygon(hexa, fill=255)
    for k in range(6):
        a = th + k * math.pi / 3
        ca, sa = math.cos(a), math.sin(a)
        bar(cx, cy, cx + L * ca, cy + L * sa, w)
        for i in range(n):
            b, l = br[2 * i], br[2 * i + 1]
            bx, by = cx + b * ca, cy + b * sa
            for sgn in (1, -1):
                aa = a + sgn * phi
                bar(bx, by, bx + l * math.cos(aa), by + l * math.sin(aa), wb)
    m = np.asarray(im, float) / 255
    if ss > 1:
        m = m.reshape(H, ss, W, ss).mean((1, 3))
    return m


def arm_angle(score, cy, cx, R):
    angs = np.radians(np.arange(0, 60, 0.5))
    vals = []
    for a in angs:
        tot = 0
        for k in range(6):
            rr = np.linspace(R * 0.25, R * 0.95, 30)
            aa = a + k * math.pi / 3
            tot += ndi.map_coordinates(score, [cy + rr * np.sin(aa), cx + rr * np.cos(aa)], order=1).sum()
        vals.append(tot)
    return float(angs[int(np.argmax(vals))])


def fit(score):
    """score: hvidhed 0..1 (h, w) i FIT-opløsning. Returnerer (params, n, loss)."""
    H, W = score.shape
    m = score > 0.5
    lab, k = ndi.label(m)
    sizes = ndi.sum(m, lab, range(1, k + 1))
    main = lab == (np.argmax(sizes) + 1)
    cy, cx = ndi.center_of_mass(main)
    ys, xs = np.nonzero(main)
    R = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2).max()
    th = arm_angle(score, cy, cx, R)
    best = None
    for n in (1, 2, 3):
        for phi0 in (math.radians(50), math.radians(62)):
            x0 = [cx, cy, th, R * 0.18, R * 0.97, R * 0.13, phi0, R * 0.1, 1.2]
            for i in range(n):
                b = R * (0.3 + 0.55 * (i + 0.5) / n)
                x0 += [b, R * 0.32 * (1 - 0.35 * i / max(n, 1))]
            x0 = np.array(x0)

            def loss(p):
                if p[3] < 0 or p[5] < 0.3 or p[7] < 0.3 or p[8] < 0.3 or not (0.35 < p[6] < 1.4):
                    return 9.0
                mdl = ndi.gaussian_filter(render(p, n, (H, W)), p[8])
                return float(((mdl - score) ** 2).mean())

            r = minimize(loss, x0, method="Nelder-Mead", options={"maxiter": 2500, "xatol": 1e-3, "fatol": 1e-7})
            r = minimize(loss, r.x, method="Nelder-Mead", options={"maxiter": 1500, "xatol": 1e-3, "fatol": 1e-7})
            val = r.fun * (1 + 0.015 * n)          # lille straf for flere grene
            if best is None or val < best[2]:
                best = (r.x, n, val)
    return best
