"""Extra real geometry: strawberry calyx (sepals + stem), banana stalk. Rows: pos, nrm, rest, flag."""
import numpy as np, math
from geom import warp, RM

def _tri_rows(P, Rst, flag, quads):
    rows = []
    for a, b, c, d in quads:
        for t in ((a, b, c), (a, c, d)):
            p = P[list(t)]; n = np.cross(p[1] - p[0], p[2] - p[0]); n /= np.linalg.norm(n) + 1e-12
            for k in t: rows.append(np.concatenate([P[k], n, Rst[k], [flag]]))
            for k in t[::-1]: rows.append(np.concatenate([P[k], -n, Rst[k], [flag]]))   # double sided
    return rows

def _surf(dirs):                      # warped surface point for unit sphere directions
    return warp(np.asarray(dirs) * RM)

def strawberry_calyx(n_leaf=8, seed=4):
    rng = np.random.default_rng(seed); rows = []
    ax = np.array([1.0, 0, 0]); e1 = np.array([0, 1.0, 0]); e2 = np.array([0, 0, 1.0])
    for k in range(n_leaf):
        phi = 2 * math.pi * k / n_leaf + rng.uniform(-0.15, 0.15); L = rng.uniform(0.62, 0.85)
        dirr = math.cos(phi) * e1 + math.sin(phi) * e2; S = 9; P = []; R_ = []
        for i in range(S + 1):
            t = i / S; ang = 0.05 + L * t
            u = math.cos(ang) * ax + math.sin(ang) * dirr
            side = np.cross(u, dirr); side /= np.linalg.norm(side)
            base = _surf(u[None])[0]; nrm = base / (np.linalg.norm(base) + 1e-9)
            lift = 0.004 + 0.014 * t ** 2.0                       # tip curls up
            w = 0.026 * math.sin(math.pi * min(1.0, t * 1.08)) ** 0.8 + 0.001
            c = base + nrm * lift
            P += [c - side * w, c + side * w]; R_ += [[t, -1.0, 0.0], [t, 1.0, 0.0]]
        P = np.array(P); R_ = np.array(R_)
        rows += _tri_rows(P, R_, 4.0, [(2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2) for i in range(S)])
    # stem
    top = _surf(ax[None])[0]; seg = 10; h = 0.035; r0 = 0.0055; P = []; R_ = []
    for j in range(2):
        for i in range(seg):
            a = 2 * math.pi * i / seg; off = np.array([0, math.cos(a), math.sin(a)]) * r0 * (1 - 0.25 * j)
            P.append(top + np.array([h * j, 0, 0.012 * j]) + off); R_.append([0, 0, 0])
    P = np.array(P); R_ = np.array(R_)
    rows += _tri_rows(P, R_, 5.0, [(i, (i + 1) % seg, seg + (i + 1) % seg, seg + i) for i in range(seg)])
    return np.array(rows, np.float32)

def banana_stalk():
    rows = []; seg = 12
    p0 = _surf(np.array([[0.985, 0, 0]]))[0]; p1 = _surf(np.array([[0.93, 0, 0]]))[0]
    d = p0 - p1; d /= np.linalg.norm(d); s1 = np.cross(d, [0, 1.0, 0]); s1 /= np.linalg.norm(s1); s2 = np.cross(d, s1)
    P = []; R_ = []
    for j, (ln, r) in enumerate(((-0.006, 0.011), (0.03, 0.008), (0.05, 0.0095))):
        for i in range(seg):
            a = 2 * math.pi * i / seg; rr = r * (1 + 0.12 * math.cos(5 * a))
            P.append(p0 + d * ln + (s1 * math.cos(a) + s2 * math.sin(a)) * rr); R_.append([0, 0, 0])
    P = np.array(P); R_ = np.array(R_)
    quads = [(j * seg + i, j * seg + (i + 1) % seg, (j + 1) * seg + (i + 1) % seg, (j + 1) * seg + i) for j in range(2) for i in range(seg)]
    rows += _tri_rows(P, R_, 5.0, quads)
    cap = p0 + d * 0.05; rows += [np.concatenate([cap, d, [0, 0, 0], [5.0]])] * 0
    return np.array(rows, np.float32)

def lychee_stem():
    rows = []; seg = 10
    p0 = _surf(np.array([[1.0, 0, 0]]))[0]; d = np.array([1.0, 0, 0.35]); d /= np.linalg.norm(d)
    s1 = np.cross(d, [0, 1.0, 0]); s1 /= np.linalg.norm(s1); s2 = np.cross(d, s1); P = []; R_ = []
    for j, (ln, r) in enumerate(((-0.004, 0.009), (0.012, 0.0045), (0.028, 0.0035))):
        for i in range(seg):
            a = 2 * math.pi * i / seg; P.append(p0 + d * ln + (s1 * math.cos(a) + s2 * math.sin(a)) * r); R_.append([0, 0, 0])
    P = np.array(P); R_ = np.array(R_)
    rows += _tri_rows(P, R_, 5.0, [(j * seg + i, j * seg + (i + 1) % seg, (j + 1) * seg + (i + 1) % seg, (j + 1) * seg + i) for j in range(2) for i in range(seg)])
    return np.array(rows, np.float32)

def cherry_stem():
    rows = []; seg = 10; P = []; R_ = []
    base = _surf(np.array([[0, 0, 1.0]]))[0]
    pts = [base + np.array([0.0, 0.0, -0.006]), base + np.array([0.004, 0.0, 0.05]), base + np.array([0.018, 0.0, 0.11]),
           base + np.array([0.040, 0.0, 0.16]), base + np.array([0.068, 0.0, 0.19])]
    for j, c in enumerate(pts):
        d = (pts[min(j + 1, len(pts) - 1)] - pts[max(j - 1, 0)]); d /= np.linalg.norm(d)
        s1 = np.cross(d, [0, 1.0, 0]); s1 /= np.linalg.norm(s1); s2 = np.cross(d, s1); r = 0.0042 if j else 0.006
        for i in range(seg):
            a = 2 * math.pi * i / seg; P.append(c + (s1 * math.cos(a) + s2 * math.sin(a)) * r); R_.append([0, 0, 0])
    P = np.array(P); R_ = np.array(R_); n = len(pts)
    rows += _tri_rows(P, R_, 4.6, [(j * seg + i, j * seg + (i + 1) % seg, (j + 1) * seg + (i + 1) % seg, (j + 1) * seg + i) for j in range(n - 1) for i in range(seg)])
    return np.array(rows, np.float32)

def tomato_calyx(n_leaf=5, seed=7):
    rng = np.random.default_rng(seed); rows = []
    ax = np.array([0, 0, 1.0])
    for k in range(n_leaf):
        phi = 2 * math.pi * k / n_leaf + rng.uniform(-0.2, 0.2); L = rng.uniform(0.40, 0.55)
        dirr = np.array([math.cos(phi), math.sin(phi), 0.0]); S = 9; P = []; R_ = []
        for i in range(S + 1):
            t = i / S; ang = 0.04 + L * t
            u = math.cos(ang) * ax + math.sin(ang) * dirr
            side = np.cross(u, dirr); side /= np.linalg.norm(side)
            base = _surf(u[None])[0]; nrm = base / (np.linalg.norm(base) + 1e-9)
            lift = 0.003 + 0.012 * math.sin(math.pi * t) * 0.6 + 0.008 * t ** 3
            w = 0.010 * math.sin(math.pi * min(1.0, t * 1.05)) ** 0.7 + 0.0008
            c = base + nrm * lift; P += [c - side * w, c + side * w]; R_ += [[t, -1.0, 0.0], [t, 1.0, 0.0]]
        rows += _tri_rows(np.array(P), np.array(R_), 4.0, [(2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2) for i in range(S)])
    top = _surf(ax[None])[0]; seg = 10; P = []; R_ = []
    for j, (h, dx, r) in enumerate(((-0.004, 0.0, 0.006), (0.020, 0.003, 0.0045), (0.034, 0.012, 0.004))):
        for i in range(seg):
            a = 2 * math.pi * i / seg; P.append(top + np.array([dx + math.cos(a) * r, math.sin(a) * r, h])); R_.append([0, 0, 0])
    rows += _tri_rows(np.array(P), np.array(R_), 4.0, [(j * seg + i, j * seg + (i + 1) % seg, (j + 1) * seg + (i + 1) % seg, (j + 1) * seg + i) for j in range(2) for i in range(seg)])
    return np.array(rows, np.float32)


def pepper_stem():
    rows = []; seg = 12; ax = np.array([0, 0, 1.0])
    top = _surf(ax[None])[0]
    for k in range(5):                                   # flat star shaped cap around the stem
        phi = 2 * math.pi * k / 5; dirr = np.array([math.cos(phi), math.sin(phi), 0.0]); S = 6; P = []; R_ = []
        for i in range(S + 1):
            t = i / S; ang = 0.02 + 0.22 * t
            u = math.cos(ang) * ax + math.sin(ang) * dirr; side = np.cross(u, dirr); side /= np.linalg.norm(side)
            base = _surf(u[None])[0]; nrm = base / (np.linalg.norm(base) + 1e-9)
            w = 0.022 * (1 - t) ** 0.8 + 0.001; c = base + nrm * 0.002
            P += [c - side * w, c + side * w]; R_ += [[t, -1.0, 0.0], [t, 1.0, 0.0]]
        rows += _tri_rows(np.array(P), np.array(R_), 4.0, [(2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2) for i in range(S)])
    P = []; R_ = []
    for j, (h, dx, r) in enumerate(((-0.006, 0.0, 0.013), (0.030, 0.004, 0.011), (0.056, 0.020, 0.0095), (0.066, 0.038, 0.009))):
        for i in range(seg):
            a = 2 * math.pi * i / seg; P.append(top + np.array([dx + math.cos(a) * r, math.sin(a) * r, h])); R_.append([0, 0, 0])
    rows += _tri_rows(np.array(P), np.array(R_), 4.0, [(j * seg + i, j * seg + (i + 1) % seg, (j + 1) * seg + (i + 1) % seg, (j + 1) * seg + i) for j in range(3) for i in range(seg)])
    return np.array(rows, np.float32)
