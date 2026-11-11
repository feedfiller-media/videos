import numpy as np, mujoco, time

BOARD_HALF = (0.30, 0.17, 0.016)
FPS = 30

def quat_z(a):
    return np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)])

def simulate(pieces, center, yaw, secs, seed=0, use_mesh=None, burst=0.18):
    """returns per frame (F, N, 3) positions and (F, N, 3, 3) rotation matrices of the piece frames"""
    rng = np.random.default_rng(seed)
    N = len(pieces)
    if use_mesh is None: use_mesh = N <= 200
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    q = quat_z(yaw)
    assets, bodies = [], []
    for i, p in enumerate(pieces):
        pos = center + Rz @ p['c']
        if use_mesh:
            v = " ".join(f"{a:.6f}" for a in (p['hull'] * 0.96).ravel())
            assets.append(f'<mesh name="m{i}" vertex="{v}"/>')
            g = f'<geom type="mesh" mesh="m{i}"/>'
        else:
            h = np.maximum(p['half'], 0.0012); b = p['bmid']
            g = f'<geom type="box" size="{h[0]:.5f} {h[1]:.5f} {h[2]:.5f}" pos="{b[0]:.5f} {b[1]:.5f} {b[2]:.5f}"/>'
        bodies.append(f'<body name="b{i}" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" quat="{q[0]:.6f} 0 0 {q[3]:.6f}"><joint type="free" damping="0.0004"/>{g}</body>')
    dt = 1 / FPS / 16
    xml = f"""<mujoco><option timestep="{dt:.6f}" gravity="0 0 -9.81" integrator="implicitfast" iterations="{30 if N < 500 else 8}" ls_iterations="{10 if N < 500 else 4}"/>
<default><geom density="960" friction="1.0 0.05 0.05" solref="0.004 1" condim="{6 if N <= 40 else 3}"/></default>
<asset>{''.join(assets)}</asset>
<worldbody><geom type="plane" size="5 5 0.1" friction="0.8 0.01 0.001"/>
<geom type="box" size="{BOARD_HALF[0]} {BOARD_HALF[1]} {BOARD_HALF[2]}" pos="0 0 {BOARD_HALF[2]}"/>
{''.join(bodies)}</worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    # initial velocities: gentle burst outward + spin
    for i, p in enumerate(pieces):
        dirv = Rz @ p['c']; nrm = np.linalg.norm(dirv) + 1e-9
        v = dirv / nrm * burst * (nrm / 0.155) ** 0.5 + rng.normal(0, 0.03, 3)
        v[2] = abs(v[2]) * 0.3
        d.qvel[6 * i:6 * i + 3] = v
        out = dirv.copy(); out[2] = 0; on = np.linalg.norm(out) + 1e-9
        tip = np.cross([0, 0, 1], out / on) * (0.7 if N <= 8 else 0.5) * min(1, on / 0.05)
        d.qvel[6 * i + 3:6 * i + 6] = tip + rng.normal(0, 0.4, 3)
    F = int(secs * FPS)
    P = np.zeros((F, N, 3), np.float32); R = np.zeros((F, N, 3, 3), np.float32)
    t0 = time.time()
    for f in range(F):
        P[f] = d.xpos[1:]; R[f] = d.xmat[1:].reshape(-1, 3, 3)
        for _ in range(16): mujoco.mj_step(m, d)
    return P, R, time.time() - t0

if __name__ == "__main__":
    import sys
    from geom import build_level
    cfg = tuple(int(x) for x in sys.argv[1:4])
    pcs, _ = build_level(*cfg)
    P, R, el = simulate(pcs, np.array([0, 0, 0.30]), 0.3, 1.0)
    print(cfg, len(pcs), "1s sim took", round(el, 1), "final z mean", P[-1, :, 2].mean())
