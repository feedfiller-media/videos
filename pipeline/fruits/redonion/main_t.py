"""Lime vs 1000 cuts: one continuous build-up. Slices, then sticks, then cubes, then it collapses."""
import numpy as np, os, sys, json, subprocess, time, math
from PIL import Image, ImageDraw, ImageFont
from geom import build_level, RM, SZ
from phys import FPS, BOARD_HALF
from dem import simulate_dem
from render import Renderer, W, H
from knife import knife_verts

FONT = "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf"
CAM = dict(cam_eye=(0, -1.65, 0.86), cam_tgt=(0, 0.0, 0.16), fov=27)
YAW = float(os.environ.get("YAW_OVERRIDE", "nan")) if os.environ.get("YAW_OVERRIDE") else 0.45
ZC = 2 * BOARD_HALF[2] + RM * SZ + 0.001           # lime resting on the board
C = np.array([0, 0, ZC], np.float32)
def Rz(a): return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], np.float32)
RMAT = Rz(YAW)

NA, NB, NC = 333, 333, 334                           # 1000 cuts in total
PH = [(-0.05, 7.4, 0.8), (7.7, 13.7, 0.8), (14.0, 20.2, 0.85)]   # hook: several cuts in the very first second   # (start, end, ease) per direction
HOLD0, REL = 20.2, 20.7
RW0 = 27.6                                           # loop: pile rewinds into the whole fruit
TOTAL = 28.8
GRID = (32, 32, 27)                                  # real pieces for the collapse
EXT = np.array([RM, RM, RM * SZ])

def times_for(t0, t1, p, n):
    # slow start, fast finish: k-th cut at t0 + (t1-t0) * (k/n)^p
    k = np.arange(n); return t0 + (t1 - t0) * (k / n) ** p

TA = times_for(*PH[0], NA); TB = times_for(*PH[1], NB); TC = times_for(*PH[2], NC)

def plane_pos(axis, k, n):
    return -EXT[axis] + 2 * EXT[axis] * (k + 1) / (n + 1)

def done(tt, t): return int(np.searchsorted(tt, t, side='right'))

# ---------- geometry ----------
WHOLE = build_level(0, 0, 0)[0][0]
def piece_arrays(pcs):
    return dict(Lv=np.concatenate([p['rest'] - p['c'] for p in pcs]).astype(np.float32),
                Nv=np.concatenate([p['nrm'] for p in pcs]).astype(np.float32),
                Rv=np.concatenate([p['rest'] for p in pcs]).astype(np.float32),
                Cv=np.concatenate([p['cut'] for p in pcs]).astype(np.float32),
                pid=np.concatenate([np.full(len(p['rest']), k) for k, p in enumerate(pcs)]),
                C=np.array([p['c'] for p in pcs], np.float32))

def whole_verts(wob=0.0, t=0.0):
    p = WHOLE; loc = p['rest'].copy()
    if wob:
        zr = np.clip((loc[:, 2] + EXT[2]) / (2 * EXT[2]), 0, 1)
        loc[:, 0] += wob * 0.010 * math.sin(t * 19) * zr; loc[:, 2] *= 1 - wob * 0.03 * (0.5 + 0.5 * math.sin(t * 23))
    pos = loc @ RMAT.T + C; nrm = p['nrm'] @ RMAT.T
    return np.concatenate([pos, nrm, p['rest'], p['cut'][:, None]], 1).astype(np.float32)

def piece_verts(A, P, R, tr=None):
    Rg = R[A['pid']]; pos = np.einsum('nij,nj->ni', Rg, A['Lv']) + P[A['pid']]
    nrm = np.einsum('nij,nj->ni', Rg, A['Nv'])
    if tr is not None:                       # juicy pieces wobble a little when they land
        ph = (A['pid'] * 2.399) % 6.283
        amp = 0.18 * math.exp(-3.0 * tr) * np.sin(24 * tr + ph); dz = pos[:, 2] - P[A['pid'], 2]
        pos[:, 0] += amp * dz * np.cos(ph); pos[:, 1] += amp * dz * np.sin(ph)
    return np.concatenate([pos, nrm, A['Rv'], A['Cv'][:, None]], 1).astype(np.float32)

# ---------- overlays ----------
FC = {}
def font(s): return FC.setdefault(s, ImageFont.truetype(FONT, s))
def ctext(d, y, s, size, col, stroke=6):
    f = font(size); b = d.textbbox((0, 0), s, font=f, stroke_width=stroke)
    d.text(((W - (b[2] - b[0])) / 2 - b[0], y), s, font=f, fill=col, stroke_width=stroke, stroke_fill=(0, 0, 0, 230))
ORANGE = (200, 90, 200, 255); WHITE = (255, 255, 255, 255)
VOL = 4 / 3 * math.pi * RM ** 3 * SZ

def cubes_count(na, nb, nc):
    if nc == 0: return 0
    cell = (2 * EXT[0] / (na + 1)) * (2 * EXT[1] / (nb + 1)) * (2 * EXT[2] / (nc + 1))
    return int(VOL / cell)

def fmt_cubes(n):
    if n < 10000: return f"{n:,}"
    return "≈ " + f"{int(round(n, -3)):,}"

def overlay(cuts, cubes, t):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    ctext(d, 270, f"{cuts:,}", 150, WHITE, 8); ctext(d, 455, "CUTS", 54, ORANGE, 5)
    ctext(d, 1215, fmt_cubes(cubes), 104 if cubes < 10000 else 92, WHITE, 7); ctext(d, 1345, "CUBES", 50, ORANGE, 5)
    f = font(40); s = "@thefeedfiller"; b = d.textbbox((0, 0), s, font=f)
    d.text(((W - (b[2] - b[0])) / 2 - b[0], 535), s, font=f, fill=(255, 255, 255, 150), stroke_width=2, stroke_fill=(0, 0, 0, 90))
    if t < 2.5 or t > TOTAL - 0.5: ctext(d, 600, "1 cut vs 1000 cuts", 58, WHITE, 6)
    elif REL + 1.5 < t < RW0: ctext(d, 600, os.environ.get("END_TEXT", "Would you still eat it?"), 50, WHITE, 6)
    a = np.array(ov).astype(np.float32); return a[..., :3], a[..., 3:4] / 255.0

# ---------- main ----------
def main(mode="full", test_times=None):
    rend = Renderer(**CAM); rend.set_planes([]); rend.p_melon['lineA'].value = 0.0
    lo = -EXT; step = 2 * EXT / (np.array([NA, NB, NC]) + 1)
    os.makedirs("cache", exist_ok=True)
    A = P = R = None
    def load_pieces():
        pcs, _ = build_level(*GRID)
        arr = piece_arrays(pcs); fn = "cache/collapse.npz"
        if os.path.exists(fn): z = np.load(fn); Pp, Rr = z['P'], z['R']
        else:
            t = time.time(); Pp, Rr, _ = simulate_dem(pcs, C, YAW, RW0 - REL + 0.2, seed=int(os.environ.get("SIM_SEED", "3")), burst=0.02)
            np.savez(fn, P=Pp, R=Rr); print("collapse sim", len(pcs), round(time.time() - t), "s", flush=True)
        return arr, Pp, Rr
    frames = range(int(TOTAL * FPS)) if mode == "full" else [int(t * FPS) for t in test_times]
    out = None
    if mode == "full":
        out = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                                "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "video_silent.mp4"], stdin=subprocess.PIPE)
    events = []
    t0w = time.time()
    for fi in frames:
        t = fi / FPS
        na, nb, nc = done(TA, t), done(TB, t), done(TC, t)
        kv = None
        if t < REL:
            front = (plane_pos(0, na - 1, NA) + 1e-4 if na else -1.0,
                     plane_pos(1, nb - 1, NB) + 1e-4 if nb else -1.0,
                     plane_pos(2, NC - nc, NC) - 1e-4 if nc else 9.0)
            rend.set_grid(step, lo, front, (1.0 if na else 0.0, 1.0 if nb else 0.0, 1.0 if nc else 0.0))
            wob = 0.0
            if t >= HOLD0: wob = 1.0 - (t - HOLD0) / (REL - HOLD0)
            mv = whole_verts(wob, t)
            # the knife of the most recent cut (one blade, fast chop)
            for tt, ax, n, nd in ((TA, 0, NA, na), (TB, 1, NB, nb), (TC, 2, NC, nc)):
                if nd == 0 or nd > n: continue
                tk = tt[nd - 1]; rate = 1.0 / max(1e-3, (tt[min(nd, n - 1)] - tk) if nd < n else 0.2)
                sw = float(np.clip(0.7 / rate, 0.05, 0.2))
                u = (t - tk) / sw
                if 0 <= u <= 1:
                    k = nd - 1 if ax < 2 else (NC - nd)
                    pl = [0, 0, 0, plane_pos(ax, k, n)]; pl[ax] = 1
                    kv = knife_verts(tuple(pl), u, RMAT, C)
        else:
            if A is None: A, P, R = load_pieces()
            fmax = min(int((RW0 - REL) * FPS), len(P) - 1)
            if t < RW0:
                f = min(int((t - REL) * FPS), fmax); mv = piece_verts(A, P[f], R[f], t - REL); na, nb, nc = NA, NB, NC
            else:                                   # rewind with ease-in, ends exactly on the uncut fruit
                s = min(1.0, (t - RW0) / (TOTAL - 1.0 / FPS - RW0)); e = s * s * (3 - 2 * s)
                if e >= 0.999: mv = whole_verts(); A_done = True
                else:
                    f = int(round((1 - e) * fmax)); mv = piece_verts(A, P[f], R[f])
                k = 1 - e; na, nb, nc = int(NA * k), int(NB * k), int(NC * k)
                rend.set_grid(step, lo, (-1.0, -1.0, 9.0), (0.0, 0.0, 0.0))
        if kv is not None: mv = np.concatenate([mv, kv])
        img = rend.render(mv, 0).astype(np.float32)
        rgb, a = overlay(na + nb + nc, cubes_count(na, nb, nc), t)
        img = (img * (1 - a) + rgb * a).clip(0, 255).astype(np.uint8)
        if out: out.stdin.write(img.tobytes())
        else: Image.fromarray(img).save(f"test_{t:05.2f}.png")
        if fi % 150 == 0: print("frame", fi, round(time.time() - t0w), "s", flush=True)
    if out: out.stdin.close(); out.wait()
    ev = [dict(t=float(x), ax=0) for x in TA] + [dict(t=float(x), ax=1) for x in TB] + [dict(t=float(x), ax=2) for x in TC]
    json.dump(dict(cuts=[e for e in ev if e['t'] >= 0], hold=HOLD0, release=REL, rewind=RW0, total=TOTAL), open("events.json", "w"))
    print("done", flush=True)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test": main("test", [float(x) for x in sys.argv[2:]])
    else: main()
