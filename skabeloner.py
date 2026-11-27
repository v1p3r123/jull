"""
Tegner skabeloner direkte efter fotoet (doer.png).

For hver del: udsnit af fotoet forstørres, pixel klassificeres efter farve,
konturen findes med sub-pixel-præcision og gemmes som polygoner i cm.
Skala: motivet fylder dørens bredde (80 cm) – samme skala vandret og lodret,
så formerne ikke bliver forvrænget.

Kør:  python skabeloner.py   (eller tools/skabeloner.py i den gamle mappestruktur)
Output: skabeloner/skabeloner.json + skabeloner/skabeloner.js  +  kontrol.png (overlay til kontrol)
"""
import json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
import contourpy
from matplotlib.path import Path as MPath

TOOLS = Path(__file__).resolve().parent
# Virker både i den oprindelige mappestruktur (tools/ + skabeloner/) og når alle filer ligger i samme mappe
if TOOLS.name == "tools" and (TOOLS.parent / "doer.png").exists():
    ROOT, OUT = TOOLS.parent, TOOLS.parent / "skabeloner"
else:
    ROOT, OUT = TOOLS, TOOLS / "skabeloner"
IMG = Image.open(ROOT / "doer.png").convert("RGB")
W, H = IMG.size
ARR = np.asarray(IMG).astype(float)

DOOR_X1, DOOR_X2 = 14.5, 91.5          # dørbladets kanter i % af fotoets bredde
DOOR_W_CM = 80.0
CM_PR_PX = DOOR_W_CM / ((DOOR_X2 - DOOR_X1) / 100 * W)
UPS = 10                                 # forstørrelse ved sporing

# ---------------------------------------------------------------- farver
WHITE = (242, 218, 206)
BLACK = (35, 27, 27)
PINK = (232, 108, 168)
ORANGE_B = (200, 85, 35)
YELLOW = (250, 205, 110)
RED = (175, 30, 35)
GREEN_TREE = (45, 75, 55)
GREEN_WREATH = (60, 120, 55)
GLASS = (180, 110, 200)
GOLD = (200, 150, 60)


def px(xp, yp):
    return xp / 100 * W, yp / 100 * H


def crop(b, pad=1.5):
    x1, y1 = px(b[0] - pad, b[1] - pad * W / H)
    x2, y2 = px(b[2] + pad, b[3] + pad * W / H)
    x1, y1 = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
    x2, y2 = min(W, int(math.ceil(x2))), min(H, int(math.ceil(y2)))
    sub = IMG.crop((x1, y1, x2, y2)).resize(((x2 - x1) * UPS, (y2 - y1) * UPS), Image.BICUBIC)
    return np.asarray(sub).astype(float), (x1, y1)


def kmeans(pix, k=6, it=12, seed=1):
    rng = np.random.default_rng(seed)
    c = pix[rng.choice(len(pix), k, replace=False)]
    for _ in range(it):
        d = ((pix[:, None, :] - c[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                c[j] = pix[lab == j].mean(0)
    return c


def membership(a, targets, k=6, tau=900.0, tol=60):
    """Blød sandsynlighed (0..1) for at en pixel tilhører en af målfarverne."""
    flat = a.reshape(-1, 3)
    sample = flat[:: max(1, len(flat) // 6000)]
    cents = kmeans(sample, k)
    t = np.array(targets, float)
    # klynger der ligner målfarven tælles som "mål"
    dct = np.sqrt(((cents[:, None] - t[None]) ** 2).sum(-1)).min(1)
    is_t = dct < max(tol, dct.min() + 25 * tol / 60)
    centers = np.vstack([t, cents[~is_t]])
    lab_t = np.zeros(len(centers), bool); lab_t[: len(t)] = True
    d = ((flat[:, None, :] - centers[None]) ** 2).sum(-1)
    e = np.exp(-(d - d.min(1, keepdims=True)) / tau)
    p = e[:, lab_t].sum(1) / e.sum(1)
    return p.reshape(a.shape[:2])


def contours(score, level=0.5):
    P = 3
    z = np.pad(score, P, constant_values=0.0)          # lukker konturer ved kanten
    gen = contourpy.contour_generator(z=z, line_type="Separate")
    return [l - P for l in gen.lines(level) if len(l) > 8]


def otsu(v):
    h, e = np.histogram(v, 128)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1); m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def sig(x):
    return 1 / (1 + np.exp(-x))


def score_mode(a, mode, bias=0):
    mn, mx = a.min(2), a.max(2)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    if mode == "white":                  # hvidt papir: alle kanaler høje
        t = otsu(mn); return sig((mn - t) / 10)
    if mode == "dark":                   # sort karton
        t = otsu(mx) - bias; return sig((t - mx) / 8)
    if mode == "tree":                   # mørkegrøn + hvid sne
        sky = ((R - G) > 35) | ((B - G) > 25)   # himlen er altid rødlig/blålig
        return ndi.gaussian_filter((~sky).astype(float), 1)
    if mode == "logo":                   # hvide bogstaver + sort kant + VI
        dark = mx < 80
        white = mn > 170
        return ndi.gaussian_filter((dark | white).astype(float), 1)
    raise ValueError(mode)


def poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def simplify(p, tol):
    """Douglas–Peucker."""
    if len(p) < 4:
        return p
    def rdp(pts):
        a, b = pts[0], pts[-1]
        ab = b - a; n = np.hypot(*ab) or 1e-9
        d = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / n
        i = d.argmax()
        if d[i] > tol:
            return np.vstack([rdp(pts[: i + 1])[:-1], rdp(pts[i:])])
        return np.vstack([a, b])
    # lukket kontur: del ved punktet længst fra start
    j = int(np.hypot(*(p - p[0]).T).argmax())
    if j < 2 or j > len(p) - 3:
        return p
    return np.vstack([rdp(p[: j + 1])[:-1], rdp(np.vstack([p[j:], p[:1]]))[:-1]])


def select(paths, seed=None, holes=True, keep_all=False, min_frac=0.02):
    """Vælg den ydre kontur (indeholder frøpunktet / størst) og dens huller."""
    if not paths:
        return []
    paths = sorted(paths, key=poly_area, reverse=True)
    if keep_all:
        big = poly_area(paths[0])
        return [p for p in paths if poly_area(p) > big * min_frac]
    outer = None
    if seed is not None:
        cands = [p for p in paths if MPath(p).contains_point(seed)]
        # mindste kontur der indeholder frøet og stadig er stor nok = selve delen
        if cands:
            outer = cands[0]
    if outer is None:
        outer = paths[0]
    res = [outer]
    if holes:
        mp = MPath(outer); A = poly_area(outer)
        for p in paths:
            if p is outer:
                continue
            if poly_area(p) < A * 0.004:
                continue
            if mp.contains_point(p[len(p) // 2]) and poly_area(p) < A:
                res.append(p)
    return res


def to_cm(polys, origin):
    ox, oy = origin
    out = []
    for p in polys:
        q = np.empty_like(p)
        q[:, 0] = (p[:, 0] / UPS + ox) * CM_PR_PX
        q[:, 1] = (p[:, 1] / UPS + oy) * CM_PR_PX
        out.append(q)
    return out


def symmetrize6(score):
    """6-dobbelt rotations-symmetri om tyngdepunktet (snefnug)."""
    m = score > 0.5
    lab, n = ndi.label(m)
    if n == 0:
        return score
    sizes = ndi.sum(m, lab, range(1, n + 1))
    main = lab == (np.argmax(sizes) + 1)
    cy, cx = ndi.center_of_mass(main)
    acc = np.zeros_like(score)
    for k in range(6):
        ang = math.radians(60 * k)
        c, s = math.cos(ang), math.sin(ang)
        mat = np.array([[c, -s], [s, c]])
        off = np.array([cy, cx]) - mat @ np.array([cy, cx])
        acc += ndi.affine_transform(score, mat, offset=off, order=1, mode="constant")
    return acc / 6


def fill_holes_score(score):
    m = ndi.binary_fill_holes(score > 0.5)
    return np.maximum(score, m * 0.9)


# ---------------------------------------------------------------- dele
def bb(x1, y1, x2, y2):
    return (x1, y1, x2, y2)


SNE = [(24.5, 6.9, 4.6), (44.4, 5.8, 3.9), (85.7, 4.4, 3.6), (35.1, 14.2, 3.5), (67.9, 26.8, 3.4),
       (77.1, 32.1, 2.7), (26.3, 44.3, 2.8), (27, 56.5, 2.8), (85.7, 54.8, 2.8), (76.6, 61.2, 3.2),
       (29, 90, 3.3), (42, 93.8, 3.3), (88.4, 91.7, 2.4)]
K = H / W

SPECS = []
for i, (cx, cy, r) in enumerate(SNE):
    SPECS.append(dict(id=f"sf{i+1}", box=bb(cx - r, cy - r / K, cx + r, cy + r / K), col=[WHITE], mode="white", flake=True))

words = [("grand", 55.7, 76.9, 5, 12), ("theft", 59.6, 82.6, 10.9, 15.9), ("auto", 56.4, 75.2, 16.4, 20.6)]
for w, x1, x2, y1, y2 in words:
    bw = (x2 - x1) / len(w)
    for i, ch in enumerate(w):
        SPECS.append(dict(id=f"lg_{w}_{i}", box=bb(x1 + bw * i, y1, x1 + bw * (i + 1), y2), col=[WHITE], mode="white",
                          seedc=True, holes=True, pad=0.4, blur=0.5))
SPECS += [
    dict(id="lg_V", box=bb(74.1, 15.9, 84.4, 26.2), col=[(150, 80, 175), (225, 80, 150), (250, 165, 75), (200, 90, 150)],
         seedc=(79.3, 18), holes=False, pad=0.6, blur=0.7, k=6),
    dict(id="lg_I", box=bb(84.4, 15.9, 89, 25.5), col=[(150, 80, 175), (225, 80, 150), (250, 165, 75), (200, 90, 150)],
         seedc=(86.7, 20), holes=False, pad=0.6, blur=0.7, k=6),
    dict(id="logobund", box=bb(55, 4, 90.5, 27), col=[], mode="logo", close=UPS * 2,
         seedc=(66, 13), holes=False, fill=True, pad=1.0, blur=0.6),
]
for i, b in enumerate([(36.9, 24.1, 39, 27.8), (40.6, 22.5, 42.9, 27), (44.3, 21.2, 46.7, 26.5), (48.2, 23.1, 50.5, 27), (51.8, 24.4, 53.9, 28.1)]):
    SPECS.append(dict(id=f"lys{i+1}", box=b, col=[WHITE], mode="white", seedc=True, holes=False, pad=0.5, blur=0.5))
for i, (cx, cy, rx, ry) in enumerate([(37.9, 22.7, 1.2, 1.45), (41.7, 21, 1.3, 1.55), (45.5, 19.6, 1.3, 1.6), (49.3, 21.6, 1.25, 1.45), (52.8, 23, 1.15, 1.45)]):
    SPECS.append(dict(id=f"flamme{i+1}", box=bb(cx - rx, cy - ry, cx + rx, cy + ry), col=[YELLOW, (245, 160, 60), (255, 235, 170)],
                      seedc=True, holes=False, pad=0.6, blur=0.6))
for i, (cx, cy, r) in enumerate([(22.6, 77.5, 4.1), (31.8, 77.5, 3.8), (80.3, 78.2, 3), (87, 78.2, 3)]):
    SPECS.append(dict(id=f"lygte{i+1}", box=bb(cx - r, cy - r / K, cx + r, cy + r / K), col=[GLASS, (200, 140, 220), (160, 90, 180)],
                      seedc=True, holes=False, circle=True, ring_cm=0.8, pad=0.8, blur=0.8))
SPECS += [
    dict(id="blinkV", box=bb(21, 83.3, 29.1, 85.4), col=[ORANGE_B], seedc=True, holes=False, pad=0.8, blur=0.6),
    dict(id="blinkH", box=bb(82.2, 84.1, 89, 86.2), col=[ORANGE_B], seedc=True, holes=False, pad=0.8, blur=0.6),
    dict(id="indtag", box=bb(35.7, 84.6, 46.6, 87.0), col=[], mode="dark", seedc=True, holes=False, pad=0.0, blur=0.4, bias=25),
    dict(id="stribeV", box=bb(17, 66, 35, 72.5), col=[PINK, (240, 150, 200)], keep_all=True, holes=False, pad=0.5, blur=0.6, min_frac=0.03),
    dict(id="stribeH", box=bb(72, 68, 91.5, 74.5), col=[PINK, (240, 150, 200)], keep_all=True, holes=False, pad=0.3, blur=0.6, min_frac=0.03),
    dict(id="listV", box=bb(17.8, 80.9, 48.2, 82), col=[PINK, (240, 150, 200)], keep_all=True, holes=False, pad=0.4, blur=0.5, min_frac=0.05),
    dict(id="listH", box=bb(80.2, 80.9, 91.5, 82), col=[PINK, (240, 150, 200)], keep_all=True, holes=False, pad=0.4, blur=0.5, min_frac=0.3),
    dict(id="pistol", box=bb(55.2, 33.2, 63.6, 44), col=[BLACK, (55, 45, 45)], seedc=(61.5, 35.5), holes=False, fill=True, pad=0.6, blur=0.6),
    dict(id="krans", box=bb(34.3, 26, 55.4, 32), col=[GREEN_WREATH, (40, 90, 45), (90, 150, 80)], seedc=None, keep_all=True, merge=True, holes=False, fill=True, pad=0.8, blur=0.7, min_frac=0.05),
    dict(id="sneV", box=bb(15.3, 16.5, 32.5, 40.0), col=[], mode="white", keep_all=True, holes=False, pad=0.0, blur=0.5, min_frac=0.02),
    dict(id="sneH", box=bb(79.8, 26.5, 91.2, 45.5), col=[], mode="white", keep_all=True, holes=False, pad=0.0, blur=0.5, min_frac=0.06),
]


def trace(spec):
    pad = spec.get("pad", 1.0)
    a, origin = crop(spec["box"], pad)
    if spec.get("mode"):
        score = score_mode(a, spec["mode"], spec.get("bias", 0))
    else:
        score = membership(a, spec["col"], k=spec.get("k", 6))
    if spec.get("close"):
        m = ndi.binary_closing(score > 0.5, iterations=spec["close"])
        score = ndi.gaussian_filter(np.maximum(score, m.astype(float)), 2)
    score = ndi.gaussian_filter(score, spec.get("blur", 0.7) * UPS)
    if spec.get("fill"):
        score = fill_holes_score(score)
    if spec.get("sym6"):
        score = symmetrize6(score)
    paths = contours(score)
    seed = None
    sc = spec.get("seedc")
    if sc is True:
        seed = (a.shape[1] / 2, a.shape[0] / 2)
    elif isinstance(sc, tuple):
        sx, sy = px(*sc)
        seed = ((sx - origin[0]) * UPS, (sy - origin[1]) * UPS)
    elif spec.get("sym6"):
        seed = (a.shape[1] / 2, a.shape[0] / 2)
    if spec.get("circle"):
        big = select(paths, seed, holes=False)
        if not big:
            return None
        p = big[0]
        c = p.mean(0)
        r = np.sqrt(((p - c) ** 2).sum(1)).mean()
        polys_px = [c + r * np.c_[np.cos(t), np.sin(t)] for t in [np.linspace(0, 2 * np.pi, 120)]]
        res = to_cm(polys_px, origin)
        extra = {"circle_cm": [float(res[0][:, 0].mean()), float(res[0][:, 1].mean()), float(r / UPS * CM_PR_PX)]}
        if spec.get("ring_cm"):
            extra["ring_cm"] = spec["ring_cm"]
        return res, extra
    polys = select(paths, seed, holes=spec.get("holes", True), keep_all=spec.get("keep_all", False), min_frac=spec.get("min_frac", 0.02))
    polys = [simplify(p, 0.12 * UPS) for p in polys]
    return to_cm(polys, origin), {}


def trace_region(spec):
    """Del afgrænset af farve + snitlinjer (se tools/regioner.py)."""
    import regioner as RG
    xs = [x for x, _ in spec["ext"]]; ys = [y for _, y in spec["ext"]]
    x1, y1 = max(0, int(min(xs)) - 2), max(0, int(min(ys)) - 2)
    x2, y2 = min(W, int(max(xs)) + 3), min(H, int(max(ys)) + 3)
    a = np.asarray(IMG.crop((x1, y1, x2, y2)).resize(((x2 - x1) * UPS, (y2 - y1) * UPS), Image.BICUBIC)).astype(float)
    origin = (x1, y1)
    if spec["mode"] == "white":
        score = RG.score_white(a, spec.get("sat", 80))
    elif spec["mode"] == "full":
        score = np.ones(a.shape[:2])
    elif spec["mode"] == "tree":
        score = score_mode(a, "tree")
    elif spec["mode"] == "notwhite":
        score = 1 - RG.score_white(a, spec.get("sat", 80))
    else:
        score = membership(a, RG.sample(ARR, spec["inc"]), k=spec["k"], tol=spec.get("tol") or 60)
    if spec["occ"]:
        score = np.maximum(score, membership(a, RG.sample(ARR, spec["occ"]), k=spec["k"], tol=spec.get("tol") or 60))
    if spec.get("occ_mode") == "white":
        score = np.maximum(score, RG.score_white(a))
    m = Image.new("L", (a.shape[1], a.shape[0]), 0)
    ImageDraw.Draw(m).polygon([((x - x1) * UPS, (y - y1) * UPS) for x, y in spec["ext"]], fill=255)
    mask = np.asarray(m) > 0
    for fp in spec.get("force", []):
        fm = Image.new("L", (a.shape[1], a.shape[0]), 0)
        ImageDraw.Draw(fm).polygon([((x - x1) * UPS, (y - y1) * UPS) for x, y in fp], fill=255)
        score = np.maximum(score, np.asarray(fm) > 0)
    score = score * mask
    if spec["close"]:
        cl = ndi.binary_closing(score > 0.5, iterations=int(spec["close"] * UPS))
        score = np.maximum(score, cl * 1.0)
    score = ndi.gaussian_filter(score, spec["blur"] * UPS) * mask
    score = fill_holes_score(score) * mask
    seed = ((spec["seed"][0] - x1) * UPS, (spec["seed"][1] - y1) * UPS)
    polys = select(contours(score), None if spec["keep_all"] else seed, holes=False,
                   keep_all=spec["keep_all"], min_frac=spec["min_frac"])
    polys = [simplify(p, spec["smooth"] * UPS) for p in polys]
    if spec.get("rect_fit"):            # vinduer er rektangler – brug sporets yderkanter (median af kanterne)
        p = polys[0]; x0, x1_ = np.percentile(p[:, 0], [8, 92]); y0, y1_ = np.percentile(p[:, 1], [8, 92])
        polys = [np.array([[x0, y0], [x1_, y0], [x1_, y1_], [x0, y1_]])]
    return to_cm(polys, origin), {}


def flake_template(spec):
    """Snefnug: vektormodel tilpasset fotoet + fuld skabelon + 1/12-foldekile."""
    import snefnug_fit as F
    global UPS
    keep = UPS; UPS = F.FIT_UPS
    a, origin = crop(spec["box"], 0.6)
    UPS = keep
    sc = score_mode(a, "white")
    p, n, loss = F.fit(sc)
    f = CM_PR_PX / F.FIT_UPS
    cx = (p[0] / F.FIT_UPS + origin[0]) * CM_PR_PX
    cy = (p[1] / F.FIT_UPS + origin[1]) * CM_PR_PX
    th, phi = float(p[2]), float(p[6])
    rc, L, w, wb = (float(v) * f for v in (p[3], p[4], p[5], p[7]))
    br = [float(v) * f for v in p[9:]]
    RES = 40                                       # pixel pr. cm ved udtegning
    Rmax = L + max(w, wb) + 0.5
    size = int(2 * Rmax * RES) + 8
    c = size / 2

    def draw(theta):
        pp = [c, c, theta, rc * RES, L * RES, w * RES, phi, wb * RES, 1] + [v * RES for v in br]
        return F.render(pp, n, (size, size), ss=2)

    full = draw(th)
    full_polys = [np.c_[q[:, 0] / RES - c / RES + cx, q[:, 1] / RES - c / RES + cy]
                  for q in select(contours(full), (c, c), holes=True)]
    # foldekile: arm peger lige op, kile fra 0° til 30° til højre
    up = draw(-math.pi / 2)
    yy, xx = np.mgrid[0:size, 0:size]
    ang = np.degrees(np.arctan2(xx - c, c - yy))   # 0 = op, positiv = mod højre
    wedge = up * ((ang >= -0.3) & (ang <= 30.3))
    wpolys = [np.c_[(q[:, 0] - c) / RES, (q[:, 1] - c) / RES] for q in contours(wedge) if poly_area(q) > RES * RES * 0.02]
    model = {"arm_cm": round(L, 1), "arm_bredde_cm": round(w, 2), "kerne_cm": round(rc, 2), "gren_vinkel": round(math.degrees(phi)),
             "gren_bredde_cm": round(wb, 2), "grene": [[round(br[2 * i], 1), round(br[2 * i + 1], 1)] for i in range(n)],
             "fejl": round(float(loss), 4)}
    print(f"   {spec['id']}: arm {L:.1f} cm, {n} gren-par, tilpasningsfejl {loss:.4f}")
    return full_polys, {"wedge": [[[round(float(x), 3), round(float(y), 3)] for x, y in q] for q in wpolys],
                        "wedge_R": round(Rmax, 2), "model": model}


def snap_profil(pts, tags, lukket=True, grow=2.0, glat_px=1.0, skridt=0.35):
    """Aktiv kontur direkte på fotoet (kubisk interpolation, fotopixel).
    tags pr. støttepunkt: "k" = farvekant (50 % mellem farven indenfor og udenfor),
                          "l" = mørk streg (kanten lægges midt i stregen, fx kæbelinjen).
    Normalen peger udad (mod venstre i kurvens retning for en kurve med uret i billedkoordinater)."""
    P0 = np.array(pts, float)
    seg = np.vstack([P0, P0[:1]]) if lukket else P0
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(seg, axis=0).T))]
    t = np.arange(0, d[-1] + 1e-9, skridt)
    P = np.c_[np.interp(t, d, seg[:, 0]), np.interp(t, d, seg[:, 1])]
    # tag for hvert punkt: fra nærmeste støttepunkt-segment
    tg = np.array([tags[min(int(np.searchsorted(d, v, "right")) - 1, len(tags) - 1)] for v in t])
    n = len(P)
    if lukket:
        tan = np.roll(P, -2, 0) - np.roll(P, 2, 0)
    else:
        tan = np.gradient(P, axis=0)
    nor = np.c_[tan[:, 1], -tan[:, 0]] / np.maximum(np.hypot(*tan.T), 1e-9)[:, None]
    if lukket and 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]) < 0:
        nor = -nor
    ss = np.arange(-grow, grow + 1e-9, 0.05)
    lum = ndi.gaussian_filter(ARR.mean(2), 0.6)
    flyt = np.zeros(n); ok = np.zeros(n, bool)
    for i in range(n):
        q = P[i] + ss[:, None] * nor[i]
        if tg[i] == "l":
            v = ndi.map_coordinates(lum, [q[:, 1], q[:, 0]], order=3)
            j = int(np.argmin(v))
            if 0 < j < len(ss) - 1:
                a, b, c = v[j - 1], v[j], v[j + 1]
                den = a - 2 * b + c
                flyt[i] = ss[j] + (0.5 * (a - c) / den * 0.05 if den > 1e-9 else 0); ok[i] = True
            continue
        col = np.stack([ndi.map_coordinates(ARR[..., ch], [q[:, 1], q[:, 0]], order=3) for ch in range(3)], 1)
        ind, ud = col[:8].mean(0), col[-8:].mean(0)                 # farven inderst og yderst i profilen
        D = np.linalg.norm(ud - ind)
        if D < 25:                                                  # ingen tydelig kant – punktet bliver
            continue
        f = np.linalg.norm(col - ind, axis=1) / D                   # 0 = indenfor, 1 = udenfor
        kr = np.nonzero((f[:-1] < 0.5) & (f[1:] >= 0.5))[0]
        if len(kr):
            j = kr[np.argmin(np.abs(ss[kr]))]
            flyt[i] = ss[j] + (0.5 - f[j]) / (f[j + 1] - f[j]) * 0.05; ok[i] = True
    if ok.any():
        idx = np.arange(n)
        flyt = np.interp(idx, idx[ok], flyt[ok], period=n if lukket else None)
        flyt = ndi.median_filter(flyt, size=5, mode="wrap" if lukket else "nearest")
        sg = glat_px / skridt
        flyt = ndi.gaussian_filter1d(flyt, sg, mode="wrap" if lukket else "nearest")
    return P + flyt[:, None] * nor


def spor_ansigt():
    """Ansigtsdetaljer sporet i fotoet (fotopixel). Returnerer {id: [polygoner]} – iris og næse beregnes ud fra sporet."""
    import regioner as RG
    U = 20                                                       # forstørrelse (20 delpixel pr. fotopixel)
    X1, Y1, X2, Y2 = 128, 145, 190, 188
    A = np.asarray(IMG.crop((X1, Y1, X2, Y2)).resize(((X2 - X1) * U, (Y2 - Y1) * U), Image.BICUBIC)).astype(float)
    HS, WS = A.shape[:2]

    def raster(poly):
        m = Image.new("L", (WS, HS), 0)
        ImageDraw.Draw(m).polygon([((x - X1) * U, (y - Y1) * U) for x, y in poly], fill=255)
        return np.asarray(m) > 0

    hand = {k: raster(v[0]) for k, v in RG.ANSIGT_DELE.items()}
    hand["mund"] = hand["overlaebe"] | hand["underlaebe"]
    dist = {k: ndi.distance_transform_edt(~m) / U for k, m in hand.items()}     # afstand (fotopx) til håndkonturen
    inde = {k: ndi.distance_transform_edt(m) / U for k, m in hand.items()}      # dybde inde i håndkonturen
    to_px = lambda q: np.c_[q[:, 0] / U + X1, q[:, 1] / U + Y1]
    out, masker = {}, {}
    L = ndi.gaussian_filter(A.mean(2), 0.3 * U)
    rows = np.arange(HS)[:, None]; xs = np.arange(WS)[None, :]

    def snap(kurve, felt, grow):
        """Aktiv kontur: hvert punkt (hver ca. 0,4 fotopx) flyttes vinkelret ud/ind til 50 %-kanten i feltet;
        flytningerne glattes langs kurven, så formen følger fotoet uden at tage hvert slør-artefakt med."""
        # jævn fordeling af punkter langs kurven
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(np.vstack([kurve, kurve[:1]]), axis=0).T))]
        t = np.arange(0, d[-1], 0.4)
        lukket = np.vstack([kurve, kurve[:1]])
        P = np.c_[np.interp(t, d, lukket[:, 0]), np.interp(t, d, lukket[:, 1])]
        n = len(P)
        tan = np.roll(P, -1, 0) - np.roll(P, 1, 0)
        nor = np.c_[tan[:, 1], -tan[:, 0]] / np.maximum(np.hypot(*tan.T), 1e-9)[:, None]
        # normalen skal pege udad
        areal = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
        if areal < 0:
            nor = -nor
        ss = np.arange(-grow - 1.0, grow + 0.001, 0.05)
        flyt = np.zeros(n); ok = np.zeros(n, bool)
        for i in range(n):
            q = P[i] + ss[:, None] * nor[i]
            v = ndi.map_coordinates(felt, [(q[:, 1] - Y1) * U, (q[:, 0] - X1) * U], order=1)
            kr = np.nonzero((v[:-1] >= 0.5) & (v[1:] < 0.5))[0]   # indefra og ud: fra detalje til omgivelse
            if len(kr):
                j = kr[np.argmin(np.abs(ss[kr]))]                  # krydsningen nærmest håndkurven
                a, b = v[j], v[j + 1]
                flyt[i] = ss[j] + (a - 0.5) / (a - b) * 0.05; ok[i] = True
        if not ok.any():
            return P
        idx = np.arange(n)
        flyt = np.interp(idx, idx[ok], flyt[ok], period=n)       # huller udfyldes fra naboerne
        flyt = ndi.median_filter(flyt, size=5, mode="wrap")       # enkelte afvigere (spidser) fjernes
        w = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2); w /= w.sum()   # glatning over ca. ±1 fotopx
        flyt = np.convolve(np.r_[flyt[-6:], flyt, flyt[:6]], w, "same")[6:-6]
        flyt = np.clip(flyt, -grow, grow)
        return P + flyt[:, None] * nor

    def skillelinje(b, o):
        """Lyseste linje mellem bryn b og øje o (rækkeindeks pr. kolonne)."""
        cols, ys = [], []
        for x in range(WS):
            hb, ho = np.nonzero(hand[b][:, x])[0], np.nonzero(hand[o][:, x])[0]
            if not len(hb) or not len(ho):
                continue
            ya, yb = int(hb.mean()), int(ho.min() + 1.0 * U)       # fra bryn-midten til lidt inde i øjet
            if yb <= ya + 2:
                continue
            cols.append(x); ys.append(ya + int(np.argmax(L[ya:yb, x])))
        cf = np.polyfit(cols, ys, 2)
        return np.polyval(cf, np.arange(WS))[None, :]

    for k, c in RG.ANSIGT_SPOR.items():
        region = dist[k] <= c["grow"]
        if "skil" in c:
            lin = skillelinje(*c["skil"])
            region &= (rows <= lin) if k == c["skil"][0] else (rows >= lin)
        if "maxx" in c:
            region &= xs <= (c["maxx"] - X1) * U
        andre = [o for o in hand if o != k and not (k == "mund" and o in ("overlaebe", "underlaebe"))]
        if c.get("ref") == "haar":
            # øret: kanten mod det mørke hår; mod ansigtet stopper det ved håndkonturens højre kant (skyggestregen)
            ref = np.median(A[(dist[k] > 0.3) & (dist[k] < 3) & (A.mean(2) < 70)], 0)
            hojre = np.where(hand[k].any(1), (hand[k] * np.arange(WS)[None, :]).max(1), -1)[:, None]
            region &= xs <= hojre
        else:
            ring = (dist[k] > c["grow"] + 1) & (dist[k] < c["grow"] + 3.5)
            for o in andre:
                ring &= dist[o] > 1
            ref = np.median(A[ring], 0)                          # den omgivende hud
        sc = ndi.gaussian_filter(np.sqrt(((A - ref) ** 2).sum(-1)), 0.25 * U)
        lvl = 0.5 * np.percentile(sc[hand[k] & (inde[k] > 0.6)], 60)   # halvvejs mellem omgivelse og detalje
        f = sc / (2 * lvl)                                       # 0,5 = kanten
        if c.get("hvid"):
            B = ndi.gaussian_filter(A[..., 2], 0.25 * U)
            top = np.percentile(B[hand[k]], 99)
            f = np.maximum(f, (B - ref[2]) / (top - ref[2]))     # det hvide: halvvejs mellem hudens og det hvides blå
        felt = ndi.gaussian_filter(np.where(region, f, 0.0), 0.1 * U)
        start = RG.ANSIGT_DELE[k][0] if k != "mund" else RG.MUND
        p = (snap(np.array(start, float), felt, c["grow"]) - [X1, Y1]) * U
        m = Image.new("L", (WS, HS), 0); ImageDraw.Draw(m).polygon([tuple(v) for v in p], fill=255)
        # rens: små løkker ved spidse ender (hvor normalerne vender) fjernes med en åbning på 0,25 fotopx
        mk = ndi.binary_opening(np.asarray(m) > 0, structure=np.ones((3, 3)), iterations=int(0.25 * U / 2))
        p = simplify(select(contours(ndi.gaussian_filter(mk.astype(float), 0.08 * U)), None, holes=False)[0], 0.02 * U)
        masker[k] = mk
        out[k] = [to_px(p)]
        print(f"   spor {k:9s} {len(p)} punkter")

    # munden deles ved mundlinjen (den mørkeste linje mellem læberne); underlæben går 1 fotopx op under overlæben
    m = masker.pop("mund"); out.pop("mund")
    cols = np.nonzero(m.sum(0) > 0.8 * U)[0]
    yy = np.array([np.argmin(np.where(m[:, x], L[:, x], 1e9)) for x in cols])
    kant = (m.sum(0)[cols] > 1.6 * U)                            # kun hvor læberne er tykke nok til at have en mundlinje
    cf = np.polyfit(cols[kant], yy[kant], 3)
    linje = np.polyval(cf, np.arange(WS))[None, :]
    for k, sel in (("overlaebe", rows < linje), ("underlaebe", rows > linje - 1.0 * U)):
        mm = ndi.gaussian_filter((m & sel).astype(float), 0.1 * U)
        p = simplify(select(contours(mm), None, holes=False)[0], 0.03 * U)
        out[k] = [to_px(p)]

    # iris: cirkel mellem det hvide på hver side af øjet (50 %-kanten mod det hvide), bunden ved nederste øjenlåg
    B = ndi.gaussian_filter(A[..., 2], 0.2 * U)
    for side in "VH":
        eye = masker["oeje" + side]
        lo, hi = np.percentile(B[eye], 5), np.percentile(B[eye], 99.5)
        hvid = (B > (lo + hi) / 2) & eye                         # det hvide: halvvejs mellem iris og det hvideste
        lab, n = ndi.label(hvid)
        st = sorted(range(1, n + 1), key=lambda i: -(lab == i).sum())[:2]
        if len(st) < 2 or min((lab == i).sum() for i in st) < 0.3 * U * U:
            print(f"   iris{side}: kun ét hvidt felt – håndmålt cirkel bruges"); continue
        v, h = sorted(st, key=lambda i: np.nonzero(lab == i)[1].mean())
        # iris' kant = de to hvide felters inderkanter, målt i rækken hvor de står tættest på hinanden
        best = None
        for y in range(HS):
            a, b = np.nonzero(lab[y] == v)[0], np.nonzero(lab[y] == h)[0]
            if len(a) and len(b) and (best is None or b.min() - a.max() < best[2] - best[1]):
                best = (y, a.max(), b.min())
        y0, vx, hx = best
        cx, r = (vx + hx) / 2, (hx - vx) / 2
        cy = y0                                                   # iris' vandrette diameter ligger hvor det hvide er tættest
        out["iris" + side] = [np.array([((cx + r * math.cos(t)) / U + X1, (cy + r * math.sin(t)) / U + Y1)
                                        for t in np.linspace(0, 2 * math.pi, 48, endpoint=False)])]
        print(f"   iris{side}: centrum ({cx / U + X1:.1f}, {cy / U + Y1:.1f}), radius {r / U:.2f} px")
    # iris beskæres af øjenlågene: kun den del, der ligger inde i øjet, kommer med
    for side in "VH":
        k = "iris" + side
        cirk = out.get(k, RG.ANSIGT_DELE[k])[0]
        m = Image.new("L", (WS, HS), 0)
        ImageDraw.Draw(m).polygon([((x - X1) * U, (y - Y1) * U) for x, y in cirk], fill=255)
        mm = ndi.gaussian_filter(((np.asarray(m) > 0) & masker["oeje" + side]).astype(float), 0.05 * U)
        out[k] = [to_px(simplify(select(contours(mm), None, holes=False)[0], 0.02 * U))]
    return out


BIL_SIDER = {
    # id: (dele der smeltes sammen til omridset, dele der tegnes som stiplede placeringslinjer indeni)
    "bilV": (["hjelmV", "frontV", "rudeV", "stribeV", "gitter", "lygte1", "lygte2", "listV", "blinkV", "indtag"],
             ["frontV", "rudeV", "stribeV", "gitter", "lygte1", "lygte2", "listV", "blinkV", "indtag"]),
    "bilH": (["hjelmH", "frontH", "stribeH", "lygte3", "lygte4", "listH", "blinkH"],
             ["frontH", "stribeH", "lygte3", "lygte4", "listH", "blinkH"]),
}
BIL_ERSTAT = ["hjelmV", "frontV", "hjelmH", "frontH"]      # karosseriet findes kun i de samlede skabeloner


def bil_samlet(out, ABS, door_x0, draw):
    """Bilen i to store skabeloner (venstre/højre): ét omrids med lygter, rude, striber m.m. som stiplede linjer.
    Forneden går bilen 3 px ind under sneen (limkant)."""
    import regioner as RG
    RES = 30
    for bid, (dele, marker) in BIL_SIDER.items():
        if not all(k in ABS for k in dele):
            continue
        ekstra = [np.array(q, float) * CM_PR_PX for q in RG.BIL_UNDER_KJOLE.get(bid, [])]
        allp = np.vstack([p for k in dele for p in ABS[k]] + ekstra)
        mn = allp.min(0) - 1.0; mx = allp.max(0) + 1.5
        size = (int((mx[0] - mn[0]) * RES) + 1, int((mx[1] - mn[1]) * RES) + 1)
        im = Image.new("L", size, 0); dr = ImageDraw.Draw(im)
        tr = lambda pts: [((x - mn[0]) * RES, (y - mn[1]) * RES) for x, y in pts]
        for k in dele:
            for p in ABS[k]:
                dr.polygon(tr(p), fill=255)
        # limkant under sneen: bånd fra snekanten 3 px ned, i bilens bredde
        for k in [d for d in dele if d.startswith("front")]:
            xa = min(p[:, 0].min() for p in ABS[k]) / CM_PR_PX; xb = max(p[:, 0].max() for p in ABS[k]) / CM_PR_PX
            xs = np.arange(xa, xb + 1e-9, 0.5)
            top = [(x * CM_PR_PX, (RG.sne_top(x) - 1.0) * CM_PR_PX) for x in xs]
            bund = [(x * CM_PR_PX, (RG.sne_top(x) + 3.0) * CM_PR_PX) for x in xs[::-1]]
            dr.polygon(tr(top + bund), fill=255)
        for poly in RG.BIL_UNDER_KJOLE.get(bid, []):          # skjult under kjolen – lukker hullet mellem kjole og sne
            dr.polygon(tr([(x * CM_PR_PX, y * CM_PR_PX) for x, y in poly]), fill=255)
        m = np.asarray(im) > 0
        # alt under snekanten + 3 px skæres væk (sneen ligger ovenpå)
        xs_px = (np.arange(size[0]) / RES + mn[0]) / CM_PR_PX
        ys_cm = np.arange(size[1])[:, None] / RES + mn[1]
        graense = np.array([RG.sne_top(x) + 3.0 for x in xs_px])[None, :] * CM_PR_PX
        m = m & (ys_cm <= graense)
        m = ndi.binary_closing(np.pad(m, 10), iterations=int(0.25 * RES))[10:-10, 10:-10]
        m = ndi.binary_fill_holes(m)
        sc = ndi.gaussian_filter(m.astype(float), 0.03 * RES)
        polys = select(contours(sc), None, holes=False)
        omrids = [np.c_[q[:, 0] / RES + mn[0], q[:, 1] / RES + mn[1]] for q in [simplify(polys[0], 0.02 * RES)]]
        mk = [p for k in marker for p in ABS[k]]
        A = np.vstack(omrids); a0, a1 = A.min(0), A.max(0)
        rel = lambda ps: [[[round(float(x - a0[0]), 3), round(float(y - a0[1]), 3)] for x, y in p] for p in ps]
        out["dele"][bid] = {"w": round(float(a1[0] - a0[0]), 2), "h": round(float(a1[1] - a0[1]), 2),
                            "polys": rel(omrids), "marks": rel(mk),
                            "pos_cm": [round(float(a0[0] - door_x0), 1), round(float(a0[1]), 1)]}
        for p in omrids:
            pts = [(x / CM_PR_PX * 4, y / CM_PR_PX * 4) for x, y in p]
            draw.line(pts + [pts[0]], fill=(255, 128, 0), width=3)
        print(f"{bid:12s} {out['dele'][bid]['w']:6.1f} x {out['dele'][bid]['h']:6.1f} cm  (samlet af {len(dele)} dele, {len(mk)} stiplede linjer)")
    for k in BIL_ERSTAT:
        out["dele"].pop(k, None)


def lys_samlet(out, ABS, door_x0, draw):
    """Lucia-lysene: stage og flamme smeltes sammen til ét omrids pr. lys, og alle 5 lys samles i én skabelon."""
    RES = 40                                              # pixel pr. cm
    alle = []
    for i in range(1, 6):
        st = np.vstack(ABS[f"lys{i}"])               # stagen er et rektangel – kanten forneden er dækket af kransens blade,
        xa, xb = np.percentile(st[:, 0], [4, 96])     # så den rettes op og forlænges 0,5 cm ned bag kransen (limkant)
        ya, yb = st[:, 1].min(), st[:, 1].max() + 0.5
        ps = [np.array([[xa, ya], [xb, ya], [xb, yb], [xa, yb]])] + ABS[f"flamme{i}"]
        allp = np.vstack(ps); mn = allp.min(0) - 0.5; mx = allp.max(0) + 0.5
        size = (int((mx[0] - mn[0]) * RES) + 1, int((mx[1] - mn[1]) * RES) + 1)
        im = Image.new("L", size, 0); dr = ImageDraw.Draw(im)
        for p in ps:
            dr.polygon([((x - mn[0]) * RES, (y - mn[1]) * RES) for x, y in p], fill=255)
        m = np.asarray(im) > 0
        # vægen: flammen og stagen forbindes med en lukning (lukker smalle spalter mellem dem)
        m = ndi.binary_closing(np.pad(m, 20), iterations=int(0.15 * RES))[20:-20, 20:-20]
        sc = ndi.gaussian_filter(m.astype(float), 0.03 * RES)
        polys = select(contours(sc), None, holes=False)
        alle += [np.c_[q[:, 0] / RES + mn[0], q[:, 1] / RES + mn[1]] for q in [simplify(polys[0], 0.03 * RES)]]
    allp = np.vstack(alle); mn, mx = allp.min(0), allp.max(0)
    out["dele"]["lys"] = {"w": round(float(mx[0] - mn[0]), 2), "h": round(float(mx[1] - mn[1]), 2),
                          "polys": [[[round(float(x - mn[0]), 3), round(float(y - mn[1]), 3)] for x, y in p] for p in alle],
                          "pos_cm": [round(float(mn[0] - door_x0), 1), round(float(mn[1]), 1)]}
    for i in range(1, 6):
        out["dele"].pop(f"lys{i}", None); out["dele"].pop(f"flamme{i}", None)
    for p in alle:
        pts = [(x / CM_PR_PX * 4, y / CM_PR_PX * 4) for x, y in p]
        draw.line(pts + [pts[0]], fill=(255, 0, 255), width=2)
    print(f"lys          {out['dele']['lys']['w']:6.1f} x {out['dele']['lys']['h']:6.1f} cm  (5 lys med flamme)")


def all_specs():
    import regioner as RG
    return SPECS + RG.R


def main(only=None):
    meta = json.loads((TOOLS / "dele.json").read_text(encoding="utf-8"))
    out = {"cm_pr_px": CM_PR_PX, "motiv_cm": [DOOR_W_CM, (99 / 100 * H) * CM_PR_PX], "dele": {},
           "meta": {k: {"navn": v["navn"], "g": v["g"], "lag": v["lag"]} for k, v in meta.items()}}
    door_x0 = DOOR_X1 / 100 * W * CM_PR_PX
    over = IMG.resize((W * 4, H * 4), Image.LANCZOS)
    d = ImageDraw.Draw(over)
    specs = all_specs()
    ABS = {}                                  # absolutte konturer i cm (til sammensatte skabeloner)
    if only:
        specs = [sp for sp in specs if sp["id"] in only]
    for spec in specs:
        r = flake_template(spec) if spec.get("flake") else trace_region(spec) if spec.get("ext") else trace(spec)
        if not r:
            print("!! ingen kontur:", spec["id"]); continue
        polys, extra = r
        allp = np.vstack(polys)
        mn, mx = allp.min(0), allp.max(0)
        rel = [[[round(float(x - mn[0]), 3), round(float(y - mn[1]), 3)] for x, y in p] for p in polys]
        entry = {"w": round(float(mx[0] - mn[0]), 2), "h": round(float(mx[1] - mn[1]), 2), "polys": rel,
                 "pos_cm": [round(float(mn[0] - door_x0), 1), round(float(mn[1]), 1)]}
        for key in ("wedge", "wedge_R", "model"):
            if key in extra:
                entry[key] = extra[key]
        if "circle_cm" in extra:
            cx, cy, rr = extra["circle_cm"]
            entry["circle"] = {"cx": round(cx - mn[0], 3), "cy": round(cy - mn[1], 3), "r": round(rr, 3)}
            if spec.get("ring"):
                entry["ring"] = True
            if "ring_cm" in extra:
                entry["ring_cm"] = extra["ring_cm"]
        out["dele"][spec["id"]] = entry
        ABS[spec["id"]] = polys
        for p in polys:
            pts = [(x / CM_PR_PX * 4, y / CM_PR_PX * 4) for x, y in p]
            d.line(pts + [pts[0]], fill=(0, 255, 255), width=2)
        print(f"{spec['id']:12s} {entry['w']:6.1f} x {entry['h']:6.1f} cm  ({len(polys)} kontur(er))")
    bil_samlet(out, ABS, door_x0, d)
    if all(f"{a}{i}" in ABS for a in ("lys", "flamme") for i in range(1, 6)):
        lys_samlet(out, ABS, door_x0, d)
    import regioner as RG
    ANSIGT = dict(RG.ANSIGT_DELE)
    if not only or any(k in only for k in ANSIGT):
        ANSIGT.update(spor_ansigt())                          # sporede kanter erstatter håndkonturerne
    # ansigtet uden detaljer: én ren kontur trukket ud på hud-kanten (og midt i kæbelinjen)
    ANSIGT["ansigt"] = [snap_profil([p for p, _ in RG.ANSIGT_REN], [t for _, t in RG.ANSIGT_REN])]
    for rid, rp in {**RG.RINGE, **ANSIGT}.items():
        if only and rid not in only:
            continue
        polys = [np.array([(x * CM_PR_PX, y * CM_PR_PX) for x, y in q]) for q in rp]
        allp = np.vstack(polys); mn, mx = allp.min(0), allp.max(0)
        out["dele"][rid] = {"w": round(float(mx[0] - mn[0]), 2), "h": round(float(mx[1] - mn[1]), 2),
                            "polys": [[[round(float(x - mn[0]), 3), round(float(y - mn[1]), 3)] for x, y in p] for p in polys],
                            "pos_cm": [round(float(mn[0] - door_x0), 1), round(float(mn[1]), 1)]}
        for p in polys:
            pts = [(x / CM_PR_PX * 4, y / CM_PR_PX * 4) for x, y in p]
            d.line(pts + [pts[0]], fill=(0, 255, 255), width=2)
    if only:
        return out
    OUT.mkdir(exist_ok=True)
    txt = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
    (OUT / "skabeloner.json").write_text(txt, encoding="utf-8")
    # samme data som script, så siderne også virker ved dobbeltklik (file://)
    (OUT / "skabeloner.js").write_text("window.SKAB=" + txt + ";", encoding="utf-8")
    over.save(TOOLS / "kontrol.png")
    print("motiv:", out["motiv_cm"])


if __name__ == "__main__":
    main(sys.argv[1:] or None)
