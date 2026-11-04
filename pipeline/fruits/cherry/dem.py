"""Fast granular sim for many small pieces (sphere proxies + free tumbling)."""
import numpy as np, time
from scipy.spatial import cKDTree
from phys import BOARD_HALF, FPS

def rotmat(axis_angle):
    th = np.linalg.norm(axis_angle, axis=1, keepdims=True) + 1e-12
    k = axis_angle / th
    K = np.zeros((len(k), 3, 3), np.float32)
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s = np.sin(th)[:, :, None]; c = np.cos(th)[:, :, None]
    I = np.eye(3, dtype=np.float32)[None]
    return I + s * K + (1 - c) * (K @ K)

def simulate_dem(pieces, center, yaw, secs, seed=0, burst=0.18, sub=20):
    rng = np.random.default_rng(seed)
    N = len(pieces)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]], np.float32)
    c = np.array([p['c'] for p in pieces], np.float32)
    x = (c @ Rz.T + center).astype(np.float64)
    half = np.array([p['half'] for p in pieces], np.float64)
    r = np.cbrt(3 * np.array([p['vol'] for p in pieces]) / (4 * np.pi)) * 0.85   # volume-equivalent contact radius (shards)
    dnn = cKDTree(x).query(x, k=2)[0][:, 1]
    r = np.maximum(np.minimum(r, 0.49 * dnn), 0.0025)                         # no initial overlap -> no explosion
    mass = np.array([p['vol'] for p in pieces]) * 960
    Mv = max(len(p['hull']) for p in pieces)
    Hv = np.array([np.concatenate([p['hull'], np.repeat(p['hull'][:1], Mv - len(p['hull']), 0)]) for p in pieces], np.float64)
    Lext = np.linalg.norm(Hv, axis=2).max(1)
    Iner = 0.6 * mass * Lext ** 2
    dist = np.linalg.norm(c, axis=1, keepdims=True) + 1e-9
    v = (c @ Rz.T) / dist * burst * np.sqrt(dist / 0.155) + rng.normal(0, 0.03, (N, 3))
    v[:, 2] = np.abs(v[:, 2]) * 0.3
    w = rng.normal(0, 3.0, (N, 3))
    R = np.repeat(Rz[None], N, 0).astype(np.float32)
    dt = 1 / FPS / sub
    om = 260.0; zeta = 0.7; mu = 0.9
    bx, by, bt = BOARD_HALF[0], BOARD_HALF[1], 2 * BOARD_HALF[2]
    F = int(secs * FPS)
    P = np.zeros((F, N, 3), np.float32); RR = np.zeros((F, N, 3, 3), np.float32)
    rmax = r.max(); pairs = None
    t0 = time.time()
    sleep = np.zeros(N, bool); slow_cnt = np.zeros(N, int)
    for f in range(F):
        P[f] = x; RR[f] = R
        for s in range(sub):
            if s % 5 == 0:
                tree = cKDTree(x)
                pairs = tree.query_pairs(2 * rmax + 0.004, output_type='ndarray')
            a = np.zeros((N, 3)); a[:, 2] = -9.81
            if len(pairs):
                i, j = pairs[:, 0], pairs[:, 1]
                d = x[j] - x[i]; L = np.linalg.norm(d, axis=1) + 1e-12
                ov = r[i] + r[j] - L
                m = ov > 0
                if m.any():
                    i, j, d, L, ov = i[m], j[m], d[m], L[m], ov[m]
                    n = d / L[:, None]
                    meff = mass[i] * mass[j] / (mass[i] + mass[j])
                    cnt = np.bincount(i, minlength=N) + np.bincount(j, minlength=N)
                    sc = 1.0 / np.sqrt(np.maximum(1.0, np.maximum(cnt[i], cnt[j]) / 2.0))   # many contacts -> softer each (stable piles)
                    meff = meff * np.maximum(sc, 0.75)                 # stay stiff: cubes must not sink into each other
                    vr = v[j] - v[i]; vn = (vr * n).sum(1)
                    fn = meff * (om ** 2 * ov - 2 * zeta * om * vn)
                    fn = np.maximum(fn, 0)
                    vt = vr - vn[:, None] * n
                    vtn = np.linalg.norm(vt, axis=1) + 1e-9
                    ft = np.minimum(mu * fn, meff * 2 * om * vtn)[:, None] * vt / vtn[:, None]
                    Fv = -fn[:, None] * n + ft          # force on i
                    np.add.at(a, i, Fv / mass[i][:, None])
                    np.add.at(a, j, -Fv / mass[j][:, None])
                    pc = np.zeros(N, bool); pc[i] = True; pc[j] = True
                    w[pc] *= 0.88                                    # cubes do not roll like balls
                    v[pc, :2] *= 0.94                                # piled cubes hold a heap instead of flowing
                    sl = pc & (np.linalg.norm(v, axis=1) < 0.03)
                    v[sl] *= 0.95; w[sl] *= 0.9
            # ground: rigid contact at the piece's hull vertices (board top or floor), with torque, so shards tip onto a face and rest
            pw = np.einsum('nij,nmj->nmi', R, Hv)
            wz = x[:, None, 2] + pw[..., 2]
            wx = x[:, None, 0] + pw[..., 0]; wy = x[:, None, 1] + pw[..., 1]
            above = (x[:, 2] > bt - 0.002)[:, None]          # only pieces that are on top of the board can stand on it
            gh = np.where((np.abs(wx) < bx) & (np.abs(wy) < by) & above, bt, 0.0)
            pen = gh - wz
            cm = pen > 0
            if cm.any():
                ncont = np.maximum(cm.sum(1), 1)
                vp = v[:, None, :] + np.cross(w[:, None, :], pw)
                ms = (mass / ncont)[:, None]
                fn = np.where(cm, ms * (om ** 2 * pen - 2 * zeta * om * vp[..., 2]), 0.0); fn = np.maximum(fn, 0)
                vt = vp[..., :2]; vtn = np.linalg.norm(vt, axis=2) + 1e-9
                ftm = np.minimum(mu * fn, ms * 0.5 * om * vtn)
                Fc = np.zeros(pw.shape); Fc[..., 2] = fn; Fc[..., :2] = -ftm[..., None] * vt / vtn[..., None]
                Ft = Fc.sum(1); Tq = np.cross(pw, Fc).sum(1)
                a += Ft / mass[:, None]
                w += Tq / Iner[:, None] * dt
                inc = cm.any(1)
                slow = inc & (np.linalg.norm(v, axis=1) < 0.03) & (np.linalg.norm(w, axis=1) < 1.0)
                v[slow] *= 0.9; w[slow] *= 0.85
                w[inc] *= 0.998
            # board side walls (pieces hitting the board edge from outside)
            onb = (np.abs(x[:, 0]) < bx) & (np.abs(x[:, 1]) < by)
            side = (~onb) & (x[:, 2] < bt) & (np.abs(x[:, 0]) < bx + r) & (np.abs(x[:, 1]) < by + r)
            if side.any():
                ox = bx + r[side] - np.abs(x[side, 0]); oy = by + r[side] - np.abs(x[side, 1])
                usex = ox < oy
                idx = np.where(side)[0]
                a[idx[usex], 0] += np.sign(x[idx[usex], 0]) * om ** 2 * ox[usex]
                a[idx[~usex], 1] += np.sign(x[idx[~usex], 1]) * om ** 2 * oy[~usex]
            a[sleep] = 0
            v += a * dt
            v[sleep] = 0; w[sleep] = 0
            v *= 0.9995
            x += v * dt
            R = rotmat(w * dt) @ R
        w *= 0.97
        sp = np.linalg.norm(v, axis=1)
        slow_cnt = np.where(sp < 0.05, slow_cnt + 1, 0)
        sleep |= (slow_cnt >= 3) & (f > 6)                  # settled cubes rest and stop yielding (a heap holds its shape)
    return P, RR, time.time() - t0

if __name__ == "__main__":
    import sys
    from geom import build_level
    cfg = tuple(int(q) for q in sys.argv[1:4])
    pcs, _ = build_level(*cfg)
    P, R, el = simulate_dem(pcs, np.array([0, 0, 0.30]), 0.3, 1.0)
    print(cfg, len(pcs), "1s took", round(el, 1), "z", P[-1, :, 2].min(), P[-1, :, 2].mean(), "spread", np.abs(P[-1, :, :2]).max(0))
