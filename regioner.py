"""
Skabeloner for dele, der ikke kan spores alene på farve (kjole, hud, hår, bygninger, bil ...).

Metode pr. del:
  1. Farven/farverne for delen samples direkte i fotoet (inc = punkter i fotoets pixel).
  2. Fotoet klassificeres (k-means i udsnittet) – pixel der ligner delens farve = delen.
  3. occ = farver på dele, der ligger OVENPÅ (fx bæltet over kjolen). De tælles med,
     så skabelonen også dækker det skjulte område under dem (= limkant/underlap).
  4. ext = afgrænsning (fotopixel). Hvor to dele har samme farve (fx ærme og overdel),
     er det her snitlinjen ligger. Synlige kanter kommer fra fotoet, ikke fra ext.
  5. Huller fyldes, delen der indeholder frøpunktet vælges.
Alle koordinater er fotopixel (doer.png er 353 x 472).
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

W, H = 353, 472


def sample(arr, pts, r=1):
    out = []
    for x, y in pts:
        x, y = int(round(x)), int(round(y))
        out.append(arr[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].reshape(-1, 3).mean(0))
    return out


R = []          # liste af specs


def reg(id, inc, ext, occ=(), seed=None, k=8, blur=0.5, close=0, tol=None, keep_all=False, min_frac=0.02, smooth=0.12, mode=None, occ_mode=None, sat=80, force=(), rect_fit=False):
    R.append(dict(id=id, inc=inc, ext=ext, occ=list(occ), seed=seed or inc[0], k=k, blur=blur, close=close,
                  keep_all=keep_all, min_frac=min_frac, smooth=smooth, mode=mode, occ_mode=occ_mode, sat=sat, tol=tol, force=list(force), rect_fit=rect_fit))


# ------------------------------------------------------------------ kjolen
DRESS_W = [(160, 225), (230, 380), (120, 300), (250, 420), (200, 230)]
RED = [(165, 265), (170, 320)]
reg("aermeV", inc=[(115, 260), (125, 215), (112, 300)], mode="white",
    ext=[(90, 185), (141, 188), (141, 242), (136, 262), (133, 300), (133, 345), (80, 345), (80, 185)], seed=(115, 260))
reg("overdel", inc=[(160, 225)], mode="white", occ=RED,
    ext=[(141, 185), (184, 185), (189, 248), (194, 266), (138, 266), (141, 242)], seed=(160, 225))
reg("aermeH", inc=[(200, 240)], mode="white",
    ext=[(184, 185), (215, 195), (228, 214), (222, 230), (206, 268), (194, 266), (189, 248)], seed=(200, 240))
reg("skoed", inc=[(145, 300), (210, 280)], mode="white", occ=RED,
    ext=[(133, 262), (196, 262), (215, 270), (245, 282), (250, 286), (238, 291), (185, 301), (172, 316), (130, 336)], seed=(145, 300))
reg("slaeb", inc=[(230, 380)], mode="white", occ=RED,
    ext=[(172, 316), (185, 301), (238, 291), (250, 286), (258, 292), (264, 305), (259, 316), (270, 333), (296, 380), (306, 420), (311, 452),
         (301, 460), (286, 466), (262, 464), (245, 461), (230, 459), (200, 452), (150, 420), (150, 330)],
    seed=(230, 380), sat=65)


def score_white(a, sat=80, lo=165):
    mn = a.min(2); mx = a.max(2)
    return 1 / (1 + np.exp(-((mn - lo) / 10))) * (1 / (1 + np.exp(((mx - mn) - sat) / 8)))


# ------------------------------------------------------------------ hoved, hud og hår
SKIN = [(160, 170), (150, 158), (175, 150), (165, 195)]
HAIR = [(115, 200), (125, 170), (140, 148)]
GREEN = [(150, 135), (175, 138)]
reg("haar", inc=HAIR, occ=GREEN, seed=(115, 200),
    ext=[(126, 128), (190, 128), (196, 150), (192, 180), (178, 189), (152, 186), (142, 192), (134, 196), (132, 258), (90, 258), (90, 215), (104, 200), (112, 175), (119, 150)], tol=45)
# "ansigt" spores nu som ren kontur med aktiv kontur – se ANSIGT_REN nederst
reg("hals", inc=SKIN, seed=(165, 195), occ_mode="white", tol=40,
    ext=[(142.5, 171), (144, 170.5), (149, 175.5), (153, 179.5), (157, 182.5), (161, 184), (166, 182.5), (169.5, 181),
         (171, 186), (192, 190), (192, 203), (138, 203), (140, 185)])     # går ca. 2,5 px op under kæbelinjen (limkant)
reg("haandH", inc=[(214, 210), (208, 200), (220, 195)], occ=[(205, 185), (215, 165)], seed=(212, 210), occ_mode="white",
    ext=[(203, 180), (222, 178), (226, 190), (225, 205), (219, 224), (205, 226), (202, 205), (205, 190)])
reg("haandV", inc=[(95, 325), (80, 327)], seed=(95, 325), occ_mode="white",
    ext=[(68, 318), (106, 318), (111, 327), (108, 336), (68, 336)])


# ------------------------------------------------------------------ rød sløjfe (del for del)
REDP = [(155, 262), (165, 320), (182, 290)]
# Hele sløjfen (bælte, knude og begge bånd) i én skabelon
reg("sloejfe", inc=REDP, seed=(155, 262),
    ext=[(144, 252), (196, 252), (196, 273), (193, 273), (193, 306), (179, 306), (179, 356), (150, 356), (150, 273), (144, 273)])


# ------------------------------------------------------------------ bilen (flader bag lygter, gitter m.m.)
# mode "full": hele afgrænsningen er delen (alt indenfor er bil – også det, der ligger under kjolen).
# mode "notwhite": alt der ikke er hvidt (sne, snefnug, kjole); force = områder under kjolen, der tages med.
reg("gitter", inc=[(140, 360)], mode="full", seed=(140, 360),
    ext=[(126, 344), (186, 344), (186, 383), (126, 383)])
reg("frontV", inc=[(80, 420)], mode="notwhite", seed=(80, 420),
    ext=[(55, 340), (196, 340), (196, 458), (55, 458)], force=[[(150, 340), (196, 340), (196, 452), (158, 412)]])
reg("frontH", inc=[(310, 420)], mode="notwhite", seed=(310, 420),
    ext=[(250, 349), (323, 349), (323, 456), (250, 456)], force=[[(262, 349), (272, 349), (309, 453), (297, 453)]])
# "sne" er flyttet ned under bilens overkant (bruger kant_lodret) – se SNE_TOP


# ------------------------------------------------------------------ baggrund (vandrette bånd over hele døren)
DX1, DX2 = 0.145 * W, 0.915 * W
for bid, ya, yb in [("bg1", 0, 22), ("bg2", 20, 42), ("bg3", 40, 53), ("bg4", 52, 94)]:
    y1, y2 = ya / 100 * H, yb / 100 * H
    reg(bid, inc=[(200, (y1 + y2) / 2)], mode="full", seed=(200, (y1 + y2) / 2), smooth=0.02,
        ext=[(DX1, y1), (DX2, y1), (DX2, y2), (DX1, y2)])

# ------------------------------------------------------------------ solen og striberne (målt i fotoet)
import math as _m
SUN_C, SUN_R, SUN_BUND = (231.3, 216.8), 51.8, 240.0


def sun_band(ya, yb, n=90):
    """Del af solskiven mellem y = ya og y = yb (fotopixel)."""
    cx, cy = SUN_C
    pts = []
    for i in range(n + 1):                                    # højre bue nedad
        y = ya + (yb - ya) * i / n
        pts.append((cx + _m.sqrt(max(SUN_R ** 2 - (y - cy) ** 2, 0)), y))
    for i in range(n + 1):                                    # venstre bue opad
        y = yb - (yb - ya) * i / n
        pts.append((cx - _m.sqrt(max(SUN_R ** 2 - (y - cy) ** 2, 0)), y))
    return pts


# Hele solen i én skabelon: skiven ned til horisonten, de 3 orange striber er udskæringer (huller) i skiven
SOL_POLYS = [sun_band(SUN_C[1] - SUN_R + 0.01, SUN_BUND, 160)] +             [sun_band(ya, yb) for ya, yb in [(201.5, 205.5), (212.0, 218.5), (226.5, 233.5)]]

# ------------------------------------------------------------------ bilens overkant (hvor bilen begynder og byen slutter)
_FOTO = np.asarray(Image.open(__import__("pathlib").Path(__file__).with_name("doer.png")).convert("RGB")).astype(float)


def kant_lodret(xs, y_gaet, win=4.0, bred=1.0, modus="kontrast"):
    """Måler pr. kolonne x (sub-pixel) hvor farven skifter fra 'over' til 'under' – 50 % mellem de to farver.
    y_gaet(x) = startgæt. Hver kolonne er gennemsnittet af en strimmel på ±bred/2 px."""
    ys = []
    for x in xs:
        yg = y_gaet(x)
        t = np.arange(yg - win, yg + win + 1e-9, 0.05)
        cols = []
        for dx in np.linspace(-bred / 2, bred / 2, 3):
            cols.append(np.stack([ndi.map_coordinates(_FOTO[..., c], [t, np.full_like(t, x + dx)], order=3) for c in range(3)], 1))
        col = np.mean(cols, 0)
        over = col[:12].mean(0)
        dd = np.linalg.norm(col - over, axis=1)
        if modus == "kontrast":                                     # 50 % mellem 'over' og den stærkeste kontrast lige under
            f = dd / max(dd[: len(dd) * 3 // 4].max(), 1e-9)
        else:                                                       # 50 % mellem 'over' og farven i bunden af vinduet
            f = dd / max(np.linalg.norm(col[-12:].mean(0) - over), 1e-9)
        kr = np.nonzero((f[:-1] < 0.5) & (f[1:] >= 0.5))[0]
        if not len(kr):
            ys.append(np.nan); continue
        j = kr[np.argmin(np.abs(t[kr] - yg))]
        ys.append(t[j] + (0.5 - f[j]) / (f[j + 1] - f[j]) * 0.05)
    ys = np.array(ys); ok = ~np.isnan(ys)
    ys = np.interp(np.arange(len(ys)), np.nonzero(ok)[0], ys[ok])
    return ndi.gaussian_filter1d(ndi.median_filter(ys, 5, mode="nearest"), 2, mode="nearest")


# venstre side: taglinjens lyse kant (fra dørkanten til ærmet); højre side: den mørke kontur over den pink stribe
_XV = np.arange(63.0, 107.01, 0.5)          # x < 63: dørhængslet dækker – linjen forlænges lige ud til dørkanten
_YV = kant_lodret(_XV, lambda x: 280.3 + (x - 75) * 0.448, win=3.0)
_a = np.polyfit(_XV[:20], _YV[:20], 1)
BIL_TOP_V = [(x, float(np.polyval(_a, x))) for x in np.arange(55.0, 63.0, 0.5)] + list(zip(_XV, _YV))
_XH = np.arange(254.0, 318.01, 0.5)          # x > 318: dørkarmen forstyrrer – linjen forlænges lige ud
_YH = kant_lodret(_XH, lambda x: 320.5 + (x - 255) * 0.245, modus="fjern")
_b = np.polyfit(_XH[-30:], _YH[-30:], 1)
BIL_TOP_H = list(zip(_XH, _YH)) + [(x, float(np.polyval(_b, x))) for x in np.arange(318.5, 324.01, 0.5)]


def bil_top(x):
    """Bilens overkant i fotopixel; under figuren (skjult) en ret linje mellem de målte stykker."""
    xv, yv = np.array(BIL_TOP_V).T; xh, yh = np.array(BIL_TOP_H).T
    if x <= xv[-1]:
        return float(np.interp(x, xv, yv))
    if x >= xh[0]:
        return float(np.interp(x, xh, yh))
    return float(np.interp(x, [xv[-1], xh[0]], [yv[-1], yh[0]]))


# ------------------------------------------------------------------ sneen forneden: ét sammenhængende stykke
# Snekanten måles hvor den ses (mørk lilla over, hvid sne under). Under kjolens slæb er den skjult –
# der går sneen 6–8 px op under slæbet (limkant), så der ikke bliver et hul.
_g = [(56, 446), (64, 445), (84, 440), (100, 440.5), (116, 449), (136, 458), (150, 458.5), (164, 456), (188, 449),
      (196, 451), (208, 457), (220, 460), (240, 464), (244, 462.5)]
_XS = np.arange(56.0, 244.01, 0.5)
_YS = kant_lodret(_XS, lambda x: float(np.interp(x, *zip(*_g))), win=3.0, modus="fjern")
for _a, _b in [(138, 160)]:                    # snefnug 12 ligger hen over kanten – lineær bro
    _sel = (_XS >= _a) & (_XS <= _b)
    _YS[_sel] = np.interp(_XS[_sel], [_a, _b], [_YS[_XS < _a][-1], _YS[_XS > _b][0]])
_XR = np.arange(302.0, 322.01, 0.5)
_YR = kant_lodret(_XR, lambda x: 450.5, win=3.0, modus="fjern")
SNE_TOP = list(zip(_XS, _YS)) + [(250, 457.0), (265, 458.0), (285, 457.5), (298, 453.0)] + list(zip(_XR, _YR)) + [(323, float(_YR[-1]))]
reg("sne", inc=[(100, 462)], mode="full", seed=(100, 464), smooth=0.02,
    ext=[(55, float(_YS[0]))] + [(float(x), float(y)) for x, y in SNE_TOP] + [(323, 467), (55, 467)])


# bilen under kjolen (skjult): venstre skabelon fortsætter ind under slæbet, så den mørke stribe
# mellem kjolens kant og sneen (x ≈ 196–245) er dækket
BIL_UNDER_KJOLE = {"bilV": [[(186, 341), (196, 341), (254, 446), (254, 470), (190, 470)]]}


def sne_top(x):
    xs, ys = np.array(SNE_TOP).T
    return float(np.interp(x, xs, ys))


LIM = 3.0                                     # limkant: bygningerne går 3 px (ca. 9 mm) ind under bilen
_top = lambda xa, xb, st=1.0: [(float(x), bil_top(x)) for x in np.arange(xa, xb + 1e-9, st)]
# kølerhjelmen: overkanten er den målte tagkant (forruden ligger ovenpå som egen del); under kjolen fortsætter den lige
reg("hjelmV", inc=[(70, 320)], mode="full", seed=(70, 320), smooth=0.02, ext=_top(55, 190) + [(190, 341), (55, 341)])
reg("hjelmH", inc=[(300, 340)], mode="full", seed=(300, 340), smooth=0.02, ext=_top(244, 323) + [(323, 350), (244, 350)])
# forruden (mørk med brune striber) mellem tagkanten og motorhjelmen – går ind under ærmet
reg("rudeV", inc=[(60, 293), (60, 287), (75, 294), (90, 293)], seed=(75, 294), tol=50, blur=0.3, smooth=0.06, k=6,
    ext=_top(55, 118, 0.5) + [(118, 304), (100, 306), (80, 309), (55, 311)], force=[[(98, bil_top(98) + 1.8), (118, bil_top(118) + 1.8), (118, 304), (98, 305.5)]])

# ------------------------------------------------------------------ bygninger (målt kant for kant i fotoet; bunden går ind under bilen)
B = 305
BYG = {   # kanterne er målt i fotoet (50 %-kanten, sub-pixel); hvor to bygninger har næsten samme farve, overlapper de lidt
    "byV1": [(56.6, 228), (70.3, 228), (70.3, 242.9), (84, 242.9), (84, B), (56.6, B)],
    "byV2": [(80.3, 225.2), (100.6, 225.2), (100.6, B), (80.3, B)],
    "byV3": [(98, 242.7), (110.9, 242.7), (110.9, 256.8), (121, 256.8), (121, B), (98, B)],
    "byH1": [(198.8, 268.6), (218, 268.6), (218, B), (198.8, B)],
    "byH2": [(216.9, 254.2), (242, 254.2), (242, B), (216.9, B)],
    "byH3": [(241, 226.7), (269.2, 226.7), (269.2, B), (241, B)],
    "byH4": [(268.6, 255.6), (284.2, 255.6), (284.2, B), (268.6, B)],
    "byH5": [(283.7, 229.8), (322.8, 229.8), (322.8, B), (283.7, B)],
}
for bid, pts in BYG.items():
    # bunden følger bilens målte overkant + limkant (i stedet for en vandret linje)
    xa, xb = pts[0][0], max(x for x, _ in pts)
    pts = [p for p in pts if p[1] != B] + [(x, y + LIM) for x, y in _top(xb, xa, -0.5)] + [(xa, bil_top(xa) + LIM)]
    xm = (pts[0][0] + pts[1][0]) / 2
    reg(bid, inc=[pts[0]], mode="full", seed=(xm, bil_top(xm) - 3), smooth=0.02, ext=pts)

# vinduer: sporet på farven inde i en lille ramme
for vid, (cx, cy) in {"vindue1": (86.3, 235.5), "vindue2": (94.3, 235.5), "vindue3": (94.3, 249.3),
                      "vindue4": (226.8, 267.3), "vindue5": (262, 250), "vindue6": (262, 265)}.items():
    reg(vid, inc=[(cx, cy)], seed=(cx, cy), blur=0.3, tol=35, smooth=0.08, rect_fit=True,
        ext=[(cx - 6, cy - 6), (cx + 6, cy - 6), (cx + 6, cy + 6), (cx - 6, cy + 6)])

# ------------------------------------------------------------------ juletræer: hele træet med stamme i én skabelon
# Kronen spores på fotoet (alt der ikke er himmel), stammen er målt i fotoet og tages med til der, hvor den forsvinder bag byen.
TREES = {
    "granV": dict(ext=[(62, 76), (122, 76), (122, 186), (74, 186), (74, 246), (60, 246), (60, 186), (55, 186), (55, 113), (61, 108)],
                  stamme=[(60, 178), (74, 178), (74, 246), (60, 246)], seed=(80, 150)),
    "granH": dict(ext=[(262, 126), (321, 126), (321, 213), (319, 213), (319, 248), (309, 248), (309, 213), (262, 213)],
                  stamme=[(309, 205), (319, 205), (319, 248), (309, 248)], seed=(300, 190)),
}
for tid, t in TREES.items():
    reg(tid, inc=[t["seed"]], mode="tree", seed=t["seed"], ext=t["ext"], force=[t["stamme"]], smooth=0.1)

# ------------------------------------------------------------------ dørhåndtag (hullet der skal skæres ud)
reg("haandtag", inc=[(312, 280)], mode="full", seed=(312, 290), smooth=0.02,
    ext=[(305, 270), (321, 270), (321, 318), (305, 318), (305, 298), (283, 298), (283, 286), (305, 286)])


def ring(cx, cy, rx, ry, t, n=72):
    out = [(cx + rx * _m.cos(2 * _m.pi * i / n), cy + ry * _m.sin(2 * _m.pi * i / n)) for i in range(n)]
    inn = [(cx + (rx - t) * _m.cos(2 * _m.pi * i / n), cy + (ry - t) * _m.sin(2 * _m.pi * i / n)) for i in range(n)][::-1]
    return [out, inn]


# øreringe: guldringe målt i fotoet (ydre ellipse + hul)
RINGE = {"oereV": ring(140.5, 173.0, 4.3, 5.8, 1.2), "oereH": ring(178.5, 177.5, 3.6, 5.0, 1.1), "sol": SOL_POLYS}


# ------------------------------------------------------------------ ansigtsdetaljer (målt punkt for punkt i fotoet)
# Ansigtet er kun ca. 50 px bredt i fotoet, og øjne/bryn flyder sammen i farve – derfor er hver detalje
# målt som støttepunkter langs kanten og glattet med en lukket Catmull-Rom-kurve (samme metode som solen og bygningerne).
def glat(pts, n=10):
    """Lukket, blød kurve gennem støttepunkterne (Catmull-Rom)."""
    P = np.array(pts, float); k = len(P); out = []
    for i in range(k):
        p0, p1, p2, p3 = P[i - 1], P[i], P[(i + 1) % k], P[(i + 2) % k]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return [tuple(map(float, q)) for q in out]


def cirkel(cx, cy, r, n=48):
    return [(cx + r * _m.cos(2 * _m.pi * i / n), cy + r * _m.sin(2 * _m.pi * i / n)) for i in range(n)]


MUNDLINJE = [(170.8, 178.2), (168.0, 178.4), (165.0, 178.7), (162.0, 178.4), (158.2, 177.7)]   # højre → venstre
ANSIGT_DELE = {
    # øjenbryn: tykke og skrå (vred mine) – den indre ende ved næsen er lavest
    "brynV": [glat([(150.2, 155.7), (152.2, 154.4), (155.0, 154.0), (158.2, 154.7), (161.3, 155.8), (164.2, 157.2), (166.6, 159.1),
                    (165.8, 160.3), (163.2, 159.6), (160.2, 158.4), (157.2, 157.3), (154.2, 156.5), (151.6, 156.4)])],
    "brynH": [glat([(169.4, 160.6), (171.8, 159.3), (174.8, 158.1), (177.8, 157.3), (180.4, 157.4), (181.7, 158.3),
                    (180.2, 159.0), (177.2, 159.4), (174.3, 160.0), (171.8, 160.8), (170.0, 161.4)])],
    # øjne: hele mandelformen inkl. den tykke vippe-streg foroven
    "oejeV": [glat([(150.4, 159.0), (152.4, 157.9), (155.0, 157.7), (158.0, 158.5), (160.6, 159.6), (162.6, 161.2),
                    (161.6, 162.5), (159.0, 162.9), (156.0, 162.6), (153.0, 161.6), (151.0, 160.2)])],
    "oejeH": [glat([(169.4, 161.4), (172.0, 160.6), (175.0, 160.8), (178.0, 161.4), (180.4, 162.6),
                    (179.2, 164.0), (176.2, 164.9), (173.0, 164.8), (170.8, 163.8), (169.6, 162.6)])],
    # iris (mørkebrun cirkel) – sidder mellem det hvide på begge sider
    "irisV": [cirkel(156.9, 160.6, 1.75)],
    "irisH": [cirkel(174.0, 162.6, 1.65)],
    # næse: omrids fra næseryg til næsetip (skygge-/tegneskabelon) og den mørke underside med næsebor
    "naese": [glat([(164.3, 162.8), (166.9, 162.8), (168.0, 166.5), (169.2, 170.0), (170.4, 172.4), (169.6, 174.2), (167.2, 174.9),
                    (164.2, 174.7), (162.2, 173.6), (161.6, 171.6), (162.8, 169.4), (163.6, 166.0)])],
    "naesebor": [glat([(161.6, 171.4), (163.2, 170.9), (165.0, 171.9), (167.2, 172.2), (169.2, 171.7), (170.3, 172.6),
                       (169.6, 174.1), (167.2, 174.8), (164.2, 174.6), (162.4, 173.5)])],
    # læber: overlæbe (mørkest) og underlæbe; underlæben går 1 px op under overlæben (limkant)
    "overlaebe": [glat([(158.2, 177.7), (159.6, 176.5), (162.0, 175.6), (164.2, 175.9), (165.4, 175.5), (167.5, 175.7),
                        (169.5, 176.8), (170.8, 178.2)] + MUNDLINJE[1:-1], n=8)],
    "underlaebe": [glat([(158.2, 177.2), (161.0, 177.5), (165.0, 177.8), (168.0, 177.5), (170.8, 177.4),
                         (170.6, 178.4), (169.8, 179.8), (168.0, 181.0), (165.5, 181.6), (163.0, 181.1), (160.5, 179.8), (158.8, 178.4)], n=8)],
    # øret (venstre side af billedet) – sidder i ansigtsskabelonen, men er her som separat tegne-/skyggeskabelon
    "oere": [glat([(138.5, 154.6), (140.5, 153.7), (142.5, 154.2), (143.8, 156.0), (144.2, 159.0), (143.8, 162.5), (142.6, 165.2),
                   (141.0, 166.8), (139.5, 166.4), (138.8, 164.5), (137.8, 161.5), (136.8, 158.0), (137.2, 155.8)])],
}

# munden samlet (udgangspunkt for sporingen; deles bagefter ved mundlinjen)
MUND = glat([(158.2, 177.7), (159.6, 176.5), (162.0, 175.6), (164.2, 175.9), (165.4, 175.5), (167.5, 175.7), (169.5, 176.8),
             (170.8, 178.2), (169.8, 179.8), (168.0, 181.0), (165.5, 181.6), (163.0, 181.1), (160.5, 179.8)], n=8)

# Sporing af ansigtsdetaljerne direkte i fotoet (skabeloner.py: spor_ansigt).
# Håndkonturerne ovenfor bruges kun som søgeområde: kanten lægges der, hvor farven er præcis halvvejs
# mellem den omgivende hud og detaljens egen farve (50 %-kanten i det slørede foto).
#   grow = hvor langt (fotopixel) uden for håndkonturen kanten må ligge
#   riv  = nabodetaljer – en pixel hører til den detalje, hvis håndkontur den ligger nærmest (snitlinje bryn/øje)
#   ref  = "hud" (detalje mod hud) eller "haar" (øret mod det mørke hår)
ANSIGT_SPOR = {
    "brynV": dict(grow=1.2, skil=("brynV", "oejeV")),
    "brynH": dict(grow=0.8, skil=("brynH", "oejeH"), maxx=182.0),
    "oejeV": dict(grow=1.2, skil=("brynV", "oejeV"), hvid=True),
    "oejeH": dict(grow=0.8, skil=("brynH", "oejeH"), hvid=True, maxx=181.0),
    "naesebor": dict(grow=1.0),
    "mund": dict(grow=1.2),                                  # spores samlet og deles ved mundlinjen
    "oere": dict(grow=1.2, ref="haar"),
}
# skil = (bryn, øje): de to flyder sammen i fotoet – snitlinjen lægges i den lyseste linje mellem dem
# hvid = øjets hvide tælles med (det slørede hvide spores i blå-kanalen, som huden næsten ikke har)
# maxx = højre grænse (ansigtskanten/håret ved højre øje)

# ------------------------------------------------------------------ ansigtet uden detaljer (ren skabelon)
# Støttepunkter med uret; k = farvekant (hud mod hår/baggrund), l = mørk streg (kæbelinjen – hud mod hud).
# Kurven trækkes ud på kanterne i fotoet af skabeloner.snap_profil(). Ørerne er med, øreringene ikke.
ANSIGT_REN = [
    ((143.6, 149.5), "k"), ((147.5, 146.0), "k"), ((152.0, 144.3), "k"), ((157.0, 143.6), "k"), ((162.0, 144.0), "k"),
    ((166.5, 145.4), "k"), ((171.0, 145.9), "k"), ((175.5, 146.0), "k"), ((179.0, 147.4), "k"), ((181.3, 150.5), "k"),
    ((182.2, 155.0), "k"), ((182.0, 159.5), "k"), ((181.0, 164.0), "k"), ((179.5, 168.5), "k"), ((177.3, 173.0), "k"),
    ((174.3, 177.5), "k"), ((171.0, 181.5), "k"), ((167.5, 184.8), "k"), ((163.5, 186.6), "k"), ((159.5, 186.4), "l"),
    ((156.0, 184.6), "l"), ((152.5, 182.0), "l"), ((149.0, 179.0), "l"), ((146.0, 176.0), "l"), ((143.8, 172.8), "l"),
    ((142.2, 169.5), "k"), ((140.2, 167.6), "k"), ((138.4, 165.0), "k"), ((137.3, 161.0), "k"), ((136.9, 157.0), "k"),
    ((137.8, 154.3), "k"), ((140.0, 153.4), "k"), ((142.0, 152.0), "k"),
]
