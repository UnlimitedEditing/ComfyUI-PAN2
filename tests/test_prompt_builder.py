"""Pure-Python tests for prompt_builder (no ComfyUI, no network)."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prompt_builder as pb

ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra else ""))


KEYS = ["video_style", "lighting", "perspective_view", "subjects_lifecycle", "init_frame_visual_description", "events", "quality"]

pos, neg, frames = pb.build("A red fishing boat on calm water at sunrise, low sun on the left.", 13)
p = json.loads(pos)
check("positive has the 7 schema keys in order", list(p) == KEYS)
check("frames kept", frames == 13)
check("event keys in order", list(p["events"][0]) == ["event_index", "time_range_sec", "event_description", "camera_motion", "contained_actions"])
check("0.5 s span for 13 frames", p["events"][0]["time_range_sec"] == "0 to 0.5")
check("static camera by default", p["events"][0]["camera_motion"][0]["camera_motion"] == ["static"])
check("no subject => empty lifecycle + no actions", p["subjects_lifecycle"] == [] and p["events"][0]["contained_actions"] == [])
check("compact JSON", '": ' not in pos and '", "' not in pos)

n = json.loads(neg)
spans = set(re.findall(r'time_range_sec":"([^"]+)"', neg))
check("negative time ranges all stamped to 0.5 s", spans == {"0 to 0.5"}, str(spans))
check("negative keeps its 7 sections", list(n) == KEYS)
check("no leftover template duration", "0 to 10" not in neg)

pos2, neg2, f2 = pb.build("Calm sea at dusk. ;; ;; Fishing boat: small red wooden boat with a white cabin ;; Warm low sun ;; Low static view ;; Documentary film look.", 241, 6)
q = json.loads(pos2)
check("241 frames = 10 s", q["events"][0]["time_range_sec"] == "0 to 10" and set(re.findall(r'time_range_sec":"([^"]+)"', neg2)) == {"0 to 10"})
check("subject parsed", q["subjects_lifecycle"][0]["display_name"] == "Fishing boat" and q["subjects_lifecycle"][0]["subject_id"] == "<|Subject1|>")
check("camera preset 6 = track_right", q["events"][0]["camera_motion"][0]["camera_motion"] == ["track_right"])
check("overrides applied", q["lighting"] == "Warm low sun" and q["perspective_view"] == "Low static view" and q["video_style"].startswith("Documentary"))
check("subject without event => generic event, no actions", q["events"][0]["contained_actions"] == [])

pos3, _, _ = pb.build("Harbour at dawn ;; rocks gently on the swell ;; Red boat: wooden hull", 9)
r = json.loads(pos3)
check("subject + event => contained action with ID", r["events"][0]["contained_actions"][0]["action"] == "<|Subject1|> rocks gently on the swell")
check("event prose uses subject name", r["events"][0]["event_description"] == "The red boat rocks gently on the swell")

check("snap 12 -> 9, 14 -> 13, 1 -> 1", (pb.snap_frames(12), pb.snap_frames(14), pb.snap_frames(1)) == (9, 13, 1))
check("1 frame still has a positive time range", pb.duration_span(1) == "0 to 0.1")
check("camera preset clamps", json.loads(pb.build("x", 9, 99)[0])["events"][0]["camera_motion"][0]["camera_motion"] == ["unstable"])
try:
    pb.build("  ", 9)
    check("empty description rejected", False)
except ValueError:
    check("empty description rejected", True)
print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
