"""Watermelon pieces: sphere cut by axis aligned grid planes. Every piece is convex."""
import numpy as np
from scipy.spatial import ConvexHull

RM = 0.155   # melon radius (m)

def fib_sphere(n, r):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1) * r

SPH = fib_sphere(9000, RM)

def cut_positions(n):
    return [RM * (-1 + 2 * (i + 1) / (n + 1)) for i in range(n)]

def cell_points(lo, hi):
    """points spanning convex piece sphere ∩ box[lo,hi]"""
    pts = []
    # sphere surface samples in box
    m = np.all((SPH >= lo - 1e-9) & (SPH <= hi + 1e-9), 1)
    pts.append(SPH[m])
    # box corners inside sphere
    cs = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    pts.append(cs[np.linalg.norm(cs, axis=1) <= RM])
    # face circles (plane ∩ sphere) inside the face rect, plus box edges ∩ sphere
    ang = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    for ax in range(3):
        a1, a2 = [k for k in range(3) if k != ax]
        for c in (lo[ax], hi[ax]):
            rr2 = RM ** 2 - c ** 2
            if rr2 <= 0: continue
            rr = np.sqrt(rr2)
            p = np.zeros((len(ang), 3)); p[:, ax] = c
            p[:, a1] = rr * np.cos(ang); p[:, a2] = rr * np.sin(ang)
            mm = (p[:, a1] >= lo[a1] - 1e-9) & (p[:, a1] <= hi[a1] + 1e-9) & (p[:, a2] >= lo[a2] - 1e-9) & (p[:, a2] <= hi[a2] + 1e-9)
            pts.append(p[mm])
            # edges of this face rect along a1 at a2 = lo/hi
            for e2 in (lo[a2], hi[a2]):
                s2 = rr2 - e2 ** 2
                if s2 < 0: continue
                for sgn in (-1, 1):
                    v = sgn * np.sqrt(s2)
                    if lo[a1] - 1e-9 <= v <= hi[a1] + 1e-9:
                        q = np.zeros(3); q[ax] = c; q[a2] = e2; q[a1] = v; pts.append(q[None])
    P = np.concatenate([p for p in pts if len(p)], 0) if pts else np.zeros((0, 3))
    return P

def build_piece(lo, hi):
    """returns dict with centroid, tri vertex arrays (local), rest pos, normals, cut flag, half extents"""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    inside = np.all(np.abs(np.where(np.abs(lo) > np.abs(hi), lo, hi)) >= 0)  # placeholder
    far = np.where(np.abs(lo) > np.abs(hi), lo, hi)
    full = np.linalg.norm(far) <= RM
    if full:
        P = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    else:
        P = cell_points(lo, hi)
        if len(P) < 4: return None
    P = np.unique(np.round(P, 7), axis=0)
    if len(P) < 4: return None
    ext = P.max(0) - P.min(0)
    if ext.min() < 1.5e-3: return None
    try:
        hull = ConvexHull(P)
    except Exception:
        return None
    if hull.volume < 2e-8: return None
    tris = P[hull.simplices]                      # (F,3,3)
    eq = hull.equations                          # outward normals
    # fix winding to match outward normal
    n_geo = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    flip = (n_geo * eq[:, :3]).sum(1) < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    fn = eq[:, :3]
    # cut faces: normal axis aligned & lying on a box face
    cut = np.zeros(len(tris), bool)
    for ax in range(3):
        on_lo = (fn[:, ax] < -0.999) & np.all(np.abs(tris[:, :, ax] - lo[ax]) < 1e-6, 1)
        on_hi = (fn[:, ax] > 0.999) & np.all(np.abs(tris[:, :, ax] - hi[ax]) < 1e-6, 1)
        cut |= on_lo | on_hi
    rest = tris.reshape(-1, 3)
    nrm = np.where(np.repeat(cut, 3)[:, None], np.repeat(fn, 3, 0), rest / np.linalg.norm(rest, axis=1, keepdims=True))
    c = P.mean(0) if full else hull.points[hull.vertices].mean(0)
    # use true centroid
    vol_c = np.zeros(3); V = 0
    o = P.mean(0)
    for t in tris:
        v = np.dot(np.cross(t[0] - o, t[1] - o), t[2] - o) / 6
        vol_c += v * (t[0] + t[1] + t[2] + o) / 4; V += v
    c = vol_c / V
    return dict(c=c, rest=rest.astype(np.float32), nrm=nrm.astype(np.float32),
                cut=np.repeat(cut, 3).astype(np.float32), hull=(P - c).astype(np.float32),
                half=(ext / 2).astype(np.float32), bmid=((P.min(0) + P.max(0)) / 2 - c).astype(np.float32), vol=V)

def _build_sphere(nx, ny, nz):
    xs = [-RM - 1e-4] + cut_positions(nx) + [RM + 1e-4]
    ys = [-RM - 1e-4] + cut_positions(ny) + [RM + 1e-4]
    zs = [-RM - 1e-4] + cut_positions(nz) + [RM + 1e-4]
    pieces = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            for k in range(len(zs) - 1):
                lo = np.array([xs[i], ys[j], zs[k]]); hi = np.array([xs[i + 1], ys[j + 1], zs[k + 1]])
                near = np.clip(0, lo, hi)
                if np.linalg.norm(near) >= RM: continue
                p = build_piece(lo, hi)
                if p: pieces.append(p)
    planes = [(1, 0, 0, x) for x in cut_positions(nx)] + [(0, 1, 0, y) for y in cut_positions(ny)] + [(0, 0, 1, z) for z in cut_positions(nz)]
    return pieces, planes

SZ = 0.95
def build_level(nx, ny, nz):
    pcs, planes = _build_sphere(nx, ny, nz)
    s = np.array([1, 1, SZ], np.float32)
    for p in pcs:
        skin = p['cut'] < 0.5
        n = p['rest'].copy(); n[:, 2] /= SZ        # skin normal of the squashed sphere (rest is still unscaled here)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        p['nrm'] = np.where(skin[:, None], n, p['nrm']).astype(np.float32)
        for k in ('rest', 'hull', 'half', 'bmid'): p[k] = (p[k] * s).astype(np.float32)
        p['c'] = p['c'] * s; p['vol'] *= SZ
    planes = [(a, b, c, w * (SZ if c else 1)) for (a, b, c, w) in planes]
    return pcs, planes

def chord(plane):
    nx, ny, nz, w = plane
    if abs(nz) > 0.5: return np.sqrt(max(RM ** 2 - (w / SZ) ** 2, 0))
    return np.sqrt(max(RM ** 2 - w ** 2, 0)) * SZ

if __name__ == "__main__":
    import time
    for cfg in [(1, 0, 0), (1, 1, 1), (3, 2, 2), (5, 4, 4), (9, 8, 8), (14, 13, 13), (20, 20, 20), (34, 33, 33)]:
        t = time.time(); p, pl = build_level(*cfg)
        print(cfg, sum(cfg), len(p), sum(len(x['rest']) for x in p) // 3, "tris", round(time.time() - t, 1), "s",
              "vol", round(sum(x['vol'] for x in p) / (4 / 3 * np.pi * RM ** 3), 3))
