"""
Build PAN 2's structured prompt + the official structured negative from a short description.

PAN 2 was trained on long, time-grounded JSON captions (video_style / lighting / perspective_view /
subjects_lifecycle / init_frame_visual_description / events / quality) and its default negative is a JSON
template whose time_range_sec fields are rewritten to the clip duration. A plain one-line prompt with an
empty negative produces blown-out multi-frame clips (measured). This module is pure Python (no ComfyUI),
so it can be tested on its own.

Description syntax (parts separated by ';;', only the first is required):
    scene ;; event ;; subject ;; lighting ;; viewpoint ;; style
  scene     what the first frame looks like (a still description)
  event     what happens during the clip. Without a subject this is plain event prose.
  subject   "Name: appearance" (or just an appearance). With a subject, write the event as what it DOES
            ("rocks gently on the swell"): it becomes the subject's contained action.
  lighting / viewpoint / style: override the defaults
"""
import json
import os
import re

FPS = 24  # PAN 2's native rate; the official code derives durations as (frames - 1) / 24
CAMERA_TAGS = [
    "static", "pan_left", "pan_right", "tilt_up", "tilt_down", "track_left", "track_right",
    "dolly_forward", "dolly_backward", "dolly_up", "dolly_down", "zoom_in", "zoom_out",
    "roll_clockwise", "roll_counterclockwise", "unstable",
]
DEFAULT_STYLE = "Realistic cinematography in one uninterrupted continuous shot."
DEFAULT_LIGHTING = "Natural, scene-appropriate light with soft shadows and true-to-life color."
DEFAULT_VIEW = "Medium-wide framing at eye level that keeps the main subject and its surroundings readable."
DEFAULT_QUALITY = "Clean natural detail with smooth, continuous motion."
_NEG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "negative_template.json")


def snap_frames(n: int) -> int:
    """Wan VAE frame counts are 4k+1."""
    return max(1, (int(n) - 1) // 4 * 4 + 1)


def duration_span(frames: int) -> str:
    seconds = max((int(frames) - 1) / FPS, 0.1)  # a 1-frame image still needs a positive range
    return f"0 to {seconds:g}"


def parse_parts(text: str):
    parts = [p.strip() for p in re.split(r";;", text or "")]
    parts += [""] * (6 - len(parts))
    return parts[:6]


def build_negative(frames: int) -> str:
    span = duration_span(frames)
    with open(_NEG_PATH, encoding="utf-8") as f:
        template = json.load(f)

    def visit(node):
        if isinstance(node, dict):
            return {k: (span if k == "time_range_sec" else visit(v)) for k, v in node.items()}
        if isinstance(node, list):
            return [visit(i) for i in node]
        return node

    return json.dumps(visit(template), ensure_ascii=False, separators=(",", ":"))


def build_positive(description: str, frames: int, camera_preset: int = 0) -> str:
    scene, event, subject, lighting, view, style = parse_parts(description)
    if not scene:
        raise ValueError("description is empty: describe at least the first frame")
    span = duration_span(frames)
    cam = CAMERA_TAGS[max(0, min(int(camera_preset), len(CAMERA_TAGS) - 1))]

    subjects, actions = [], []
    if subject:
        name, _, attr = subject.partition(":")
        name, attr = (name.strip(), attr.strip()) if attr.strip() else ("Main subject", subject.strip())
        subjects = [{"subject_id": "<|Subject1|>", "display_name": name, "subject_attribute": attr}]
        if event:
            actions = [{"time_range_sec": span, "action": f"<|Subject1|> {event}"}]
            event_text = f"The {name.lower()} {event}"
        else:
            event_text = "The scene unfolds with natural, continuous motion."
    else:
        event_text = event or "The scene unfolds with subtle, natural, continuous motion."

    prompt = {
        "video_style": style or DEFAULT_STYLE,
        "lighting": lighting or DEFAULT_LIGHTING,
        "perspective_view": view or DEFAULT_VIEW,
        "subjects_lifecycle": subjects,
        "init_frame_visual_description": scene,
        "events": [{
            "event_index": 0,
            "time_range_sec": span,
            "event_description": event_text,
            "camera_motion": [{"time_range_sec": span, "camera_motion": [cam]}],
            "contained_actions": actions,
        }],
        "quality": {"video_quality": DEFAULT_QUALITY, "text_logos_overlays": None, "artifacts": None},
    }
    return json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))


def build(description: str, frames: int, camera_preset: int = 0):
    frames = snap_frames(frames)
    return build_positive(description, frames, camera_preset), build_negative(frames), frames
