"""Tomato audio from real Pixabay recordings (Pixabay Content License): cut hits from a tomato cutting recording,
collapse from wet squish recordings in px/ if present."""
import numpy as np, json, wave, glob, os
from scipy.signal import find_peaks, butter, sosfilt
SR = 44100; rng = np.random.default_rng(5)
def rd(fn):
    w = wave.open(fn); x = np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768.0; return x.astype(np.float32)
x = rd("px/tom.wav")
x = sosfilt(butter(2, 60 / (SR / 2), 'high', output='sos'), x).astype(np.float32)
h = 441; e = np.sqrt(np.convolve(x ** 2, np.ones(h) / h, 'same'))[::h]
pk, _ = find_peaks(e, height=e.max() * 0.15, distance=15)
hits = []
for p in pk:
    c = p * h; w0 = x[max(0, c - 4410): c + 4410]
    c = max(0, c - 4410) + int(np.argmax(np.abs(w0)))      # exact knock transient
    a, b = c - int(0.004 * SR), c + int(0.16 * SR)
    s = x[a:b].copy(); tt = np.arange(len(s)) / SR
    env = np.exp(-np.clip(tt - 0.004, 0, None) * 28); env[:int(0.004 * SR)] = np.linspace(0, 1, int(0.004 * SR))
    s *= env; s /= np.abs(s).max() + 1e-9; hits.append((s, c - a))
print(len(hits), "hits")

E = json.load(open("events.json")); total = E['total']
mix = np.zeros((int((total + 2) * SR), 2), np.float32)
def add(sig, t, vol, pan=0.0):
    i = max(0, int(t * SR)); s = sig[: len(mix) - i]
    mix[i:i + len(s), 0] += s * vol * (1 - max(0, pan)); mix[i:i + len(s), 1] += s * vol * (1 + min(0, pan))
last = -1
for ev in E['cuts']:
    t = ev['t']
    if t - last < 0.05: continue
    gap = t - last; last = t
    s, off = hits[rng.integers(len(hits))]
    if gap < 0.25:                                   # fast chopping: shorter grain so hits stay crisp
        s = s[: off + int(min(0.16, max(0.06, gap * 1.4)) * SR)].copy(); fl = int(0.02 * SR); s[-fl:] *= np.linspace(1, 0, fl)
    v = float(np.clip(0.55 + gap, 0.55, 1.0))
    add(s, t - off / SR, v, float(rng.uniform(-0.3, 0.3)))
# collapse
sq = sorted(glob.glob("px/sq_*.wav"))
if sq:
    t0 = E['release'] + 0.05
    for k, fn in enumerate(sq):
        y = rd(fn); y /= np.abs(y).max() + 1e-9; y = y[: int(4.5 * SR)]
        fo = int(0.5 * SR); y[-fo:] *= np.linspace(1, 0, fo)
        add(y, t0 + 0.12 * k, 0.8 if k == 0 else 0.5)
else:                                                # no more knife sounds once it falls: one soft, dull thud only
    s, off = hits[0]; th = sosfilt(butter(2, 300 / (SR / 2), 'low', output='sos'), s).astype(np.float32)
    add(th / (np.abs(th).max() + 1e-9), E['release'] + 0.15 - off / SR, 0.6)
if 'rewind' in E:                                   # soft reverse swoosh into the loop
    n = int((E['total'] - E['rewind']) * SR); tt = np.arange(n) / SR
    sw = sosfilt(butter(2, [300 / (SR / 2), 3000 / (SR / 2)], 'band', output='sos'), rng.normal(0, 1, n)).astype(np.float32)
    sw *= (tt / tt[-1]) ** 2 * 0.25; add(sw, E['rewind'], 1.0)
mix /= np.abs(mix).max() / 0.9; mix = mix[: int(total * SR)]
with wave.open("audio_px.wav", "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())
print("ok")
