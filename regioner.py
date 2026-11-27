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
reg("ansigt", inc=SKIN[:3], seed=(160, 170), close=1.0, tol=40,
    ext=[(128, 138), (186, 138), (184.5, 150), (184, 165), (181, 174), (175, 183), (170, 188), (152, 186), (140, 184), (128, 168)])
reg("hals", inc=SKIN, seed=(165, 195), occ_mode="white", tol=40,
    ext=[(140, 184), (152, 186), (170, 188), (175, 183), (181, 174), (192, 181), (192, 203), (138, 203)])
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
reg("hjelmV", inc=[(70, 320)], mode="full", seed=(70, 320),
    ext=[(55, 309), (70, 307), (85, 305), (100, 303), (112, 300), (140, 300), (188, 312), (190, 341), (55, 341)])
reg("hjelmH", inc=[(300, 340)], mode="full", seed=(300, 340),
    ext=[(244, 316), (255, 318), (285, 325), (323, 336), (323, 350), (244, 350)])
reg("gitter", inc=[(140, 360)], mode="full", seed=(140, 360),
    ext=[(126, 344), (186, 344), (186, 383), (126, 383)])
reg("frontV", inc=[(80, 420)], mode="notwhite", seed=(80, 420),
    ext=[(55, 340), (196, 340), (196, 458), (55, 458)], force=[[(150, 340), (196, 340), (196, 452), (158, 412)]])
reg("frontH", inc=[(310, 420)], mode="notwhite", seed=(310, 420),
    ext=[(250, 349), (323, 349), (323, 456), (250, 456)], force=[[(262, 349), (272, 349), (309, 453), (297, 453)]])
reg("sne", inc=[(100, 460)], mode="white", seed=(100, 462), sat=80,
    ext=[(55, 436), (134, 436), (134, 446), (136, 456), (162, 456), (164, 446), (196, 446), (198, 455), (230, 462), (262, 466),
         (300, 463), (311, 455), (323, 450), (323, 467), (55, 467)])


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

# ------------------------------------------------------------------ bygninger (målt kant for kant i fotoet; bunden går ind under bilen)
B = 305
BYG = {
    "byV1": [(57, 228), (70, 228), (70, 243), (84, 243), (84, B), (57, B)],
    "byV2": [(80, 225), (100, 225), (100, B), (80, B)],
    "byV3": [(98, 243), (108, 243), (108, 258), (121, 258), (121, B), (98, B)],
    "byH1": [(198, 268), (218, 268), (218, B), (198, B)],
    "byH2": [(216, 255), (242, 255), (242, B), (216, B)],
    "byH3": [(241, 227), (270, 227), (270, B), (241, B)],
    "byH4": [(269, 253), (286, 253), (286, B), (269, B)],
    "byH5": [(285, 232), (323, 232), (323, B), (285, B)],
}
for bid, pts in BYG.items():
    reg(bid, inc=[pts[0]], mode="full", seed=((pts[0][0] + pts[1][0]) / 2, B - 5), smooth=0.02, ext=pts)

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
