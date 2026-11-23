"""Chef's knife mesh + pose for a cut through the melon (melon-local frame)."""
import numpy as np
from geom import RM, chord

def prism(poly, z0, z1, flag):
    """convex polygon (x,y) extruded along z. returns (V,10) rows pos nrm rest cut"""
    P = np.array(poly, float)
    rows = []
    def tri(a, b, c, n):
        for p in (a, b, c): rows.append(np.concatenate([p, n, [0, 0, 0], [flag]]))
    top = [np.array([x, y, z1]) for x, y in P]; bot = [np.array([x, y, z0]) for x, y in P]
    # ensure CCW polygon
    area = sum(P[i, 0] * P[(i + 1) % len(P), 1] - P[(i + 1) % len(P), 0] * P[i, 1] for i in range(len(P)))
    if area < 0: top, bot, P = top[::-1], bot[::-1], P[::-1]
    for i in range(1, len(P) - 1):
        tri(top[0], top[i], top[i + 1], [0, 0, 1])
        tri(bot[0], bot[i + 1], bot[i], [0, 0, -1])
    for i in range(len(P)):
        j = (i + 1) % len(P)
        e = P[j] - P[i]; n = np.array([e[1], -e[0], 0]); n /= np.linalg.norm(n) + 1e-12
        tri(bot[i], bot[j], top[j], n); tri(bot[i], top[j], top[i], n)
    return np.array(rows, np.float32)

def knife_mesh():
    # blade: edge at y=0 (curving up to tip at x=-0.21), spine y=0.058, heel at x=0.11
    xs = np.linspace(-0.21, 0.11, 24)
    edge = [(x, 0.062 * max(0, (-0.06 - x) / 0.15) ** 1.8) for x in xs]
    poly = edge + [(0.11, 0.075), (-0.06, 0.075), (-0.21, 0.062)]
    # make convex-safe: use edge polyline + spine
    blade = prism(poly, -0.0011, 0.0011, 2.0)
    bolster = prism([(0.11, 0.004), (0.125, 0.010), (0.125, 0.068), (0.11, 0.075)], -0.006, 0.006, 2.0)
    handle = prism([(0.125, 0.016), (0.26, 0.020), (0.275, 0.038), (0.26, 0.058), (0.125, 0.064)], -0.010, 0.010, 3.0)
    return np.concatenate([blade, bolster, handle])

KNIFE = knife_mesh()
LEN_OFFSET = 0.05   # shift so the melon sits mid-blade

def ease_io(u): u = np.clip(u, 0, 1); return u * u * (3 - 2 * u)

def knife_pose(plane, u):
    """plane=(nx,ny,nz,w) in melon-local, u in [0,1] progress. returns M (3x3), o (3)"""
    nx, ny, nz, w = plane
    r = chord(plane) + 0.025
    s = r - 2 * r * ease_io(u)
    if abs(nx) > 0.5:
        M = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], np.float32)     # X->y, Y->z, Z->x  (columns)
        o = np.array([w, LEN_OFFSET, s])
    elif abs(ny) > 0.5:
        M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float32)    # X->x, Y->z, Z->-y
        o = np.array([LEN_OFFSET, w, s])
    else:
        M = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], np.float32)    # X->y, Y->x, Z->-z
        o = np.array([s, LEN_OFFSET, w])
    return M, o.astype(np.float32)

def knife_verts(plane, u, Rm, Hpos):
    M, o = knife_pose(plane, u)
    A = Rm @ M
    pos = KNIFE[:, :3] @ A.T + (Rm @ o + Hpos)
    nrm = KNIFE[:, 3:6] @ A.T
    out = KNIFE.copy(); out[:, :3] = pos; out[:, 3:6] = nrm
    return out
