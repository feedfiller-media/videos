"""Daily video job: renders two '1 cut vs 1000 cuts' videos, places them in the repo root and writes plan.json.
usage: python3 pipeline/run.py YYYY-MM-DD
Rendering both fruits in parallel takes about 45 to 60 minutes on 2 CPU cores."""
import sys, os, json, random, subprocess, shutil, datetime, concurrent.futures as cf

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
# only bold, instantly recognisable fruits (red ones performed best); tag = TikTok hashtag, emoji for the caption
FRUITS = [
    ("strawberry", "strawberry", "\U0001F353"), ("cherry", "cherry", "\U0001F352"), ("pomegranate", "pomegranate", ""),
    ("tomato2", "tomato", "\U0001F345"), ("apple", "apple", "\U0001F34E"), ("lychee", "lychee", ""),
    ("strawberry", "strawberry", "\U0001F353"), ("orange", "orange", "\U0001F34A"),
    ("plum", "plum", ""), ("pepper", "pepper", "\U0001FAD1"),
    ("bloodorange", "bloodorange", "\U0001F534"), ("grapefruit", "grapefruit", ""),
    ("redonion", "redonion", "\U0001F9C5"), ("cheese", "cheese", "\U0001F9C0"),
    ("raspberry", "raspberry", "\u2764\ufe0f"), ("jawbreaker", "jawbreaker", "\U0001F36C"),
    ("radish", "radish", "\u2764\ufe0f"), ("cakepop", "cakepop", "\U0001F382"),
    ("beet", "beetroot", "\u2764\ufe0f"), ("scotchegg", "scotchegg", "\U0001F95A"),
]
NAMES = {"bloodorange": "blood orange", "redonion": "red onion", "cakepop": "cake pop", "scotchegg": "scotch egg"}   # caption wording when it differs from the hashtag
END_TEXTS = ["Rate this from 1 to 10", "Would you still eat it?", "Which fruit should be next?", "Could you eat all of it?",
             "Be honest: how many cuts was that?", "Satisfying or not?", "Comment your score", "Would you try this?", "What should I cut next?"]

def render(slot, day_idx, date, rng):
    proj, tag, emo = FRUITS[(2 * day_idx + slot) % len(FRUITS)]
    want = os.environ.get(f"FRUIT_{slot + 1}")          # set by the daily analysis, e.g. FRUIT_1=apple
    if want: proj, tag, emo = next(f for f in FRUITS if f[0] == want or f[1] == want)
    work = os.path.join("/tmp", f"ff_{date}_{slot}"); shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(os.path.join(HERE, "fruits", proj), work)
    end = os.environ.get(f"END_{slot + 1}") or rng.choice(END_TEXTS)
    env = dict(os.environ, END_TEXT=end, SIM_SEED=str(rng.randint(1, 9999)))
    base_yaw = {"cherry": 0.0, "pomegranate": 0.0, "tomato2": 0.0, "strawberry": -0.55, "lychee": -0.35, "apple": 0.45, "orange": 0.45, "plum": 0.3, "pepper": 0.3, "bloodorange": 0.45, "grapefruit": 0.45, "redonion": 0.45, "cheese": 0.45, "raspberry": 0.45, "jawbreaker": 0.45, "radish": 0.45, "cakepop": 0.45, "beet": 0.45, "scotchegg": 0.45}[proj]
    env["YAW_OVERRIDE"] = str(round(base_yaw + rng.uniform(-0.25, 0.25), 3))   # every video looks a bit different
    os.makedirs(os.path.join(work, "cache"), exist_ok=True)
    subprocess.run([sys.executable, "main_t.py"], cwd=work, env=env, check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, "audio_px.py"], cwd=work, env=env, check=True, stdout=subprocess.DEVNULL)
    out = os.path.join(REPO, f"{date}_{slot + 1}_{tag}.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "video_silent.mp4", "-i", "audio_px.wav", "-map", "0:v", "-map", "1:a",
                    "-c:v", "libx264", "-preset", "medium", "-b:v", "2400k", "-maxrate", "2800k", "-bufsize", "5600k",
                    "-af", "loudnorm=I=-15:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "160k", "-shortest", out], cwd=work, check=True)
    caption = f"1 cut vs 1000 cuts {NAMES.get(tag, tag)} {emo} {end} #satisfying #oddlysatisfying #{tag} #asmr #fyp".replace("  ", " ")
    return dict(slot=slot, time="12:45" if slot == 0 else "19:45", file=os.path.basename(out), fruit=tag, caption=caption)

if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().isoformat()
    day_idx = (datetime.date.fromisoformat(date) - datetime.date(2026, 11, 5)).days
    rng = random.Random(date)
    seeds = [random.Random(rng.random()) for _ in range(2)]
    with cf.ThreadPoolExecutor(2) as ex:
        plan = sorted(ex.map(lambda s: render(s, day_idx, date, seeds[s]), [0, 1]), key=lambda p: p["slot"])
    json.dump(dict(date=date, posts=plan), open(os.path.join(REPO, "plan.json"), "w"), indent=1)
    print(json.dumps(plan, indent=1))
