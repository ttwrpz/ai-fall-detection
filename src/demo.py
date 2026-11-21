import argparse
import csv
import platform
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import yaml

from src import baseline, features, windows
from src.evaluate import feature_columns, fit_ml, load_training_windows, write_csv_atomic
from src.models import fall_scores, score_method
from src.pose import create_landmarker, detect_pose

ROOT = Path(__file__).resolve().parents[1]
WINDOW_NAME = "Fall detection: RULE (left) and AI (right), q to quit"
CONNECTIONS = [(link.start, link.end) for link in mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS]
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, BLACK, CYAN = (255, 255, 255), (0, 0, 0), (230, 200, 0)
GREEN, RED, AMBER = (60, 200, 60), (40, 40, 230), (0, 190, 255)
SAMPLE_RATE = 44100
READ_RETRY_S = 0.01
OFFLINE_POLL_MS = 200
QUIT_KEYS = {ord("q"), ord("Q"), 27}
ALERT_COLUMNS = ["wall_time", "source", "stream_t", "panel", "detector", "score", "threshold", "fallback"]
FPS_COLUMNS = ["date", "machine", "pose_task_file", "source", "frames", "pose_only_fps", "pose_plus_model_fps",
               "detector", "window_scoring_ms"]
BENCHMARK_CHOICES = ("best", "rule", "nb", "svm")
TIMING_ONLY = ("nb", "svm")
LATENCY_COLUMNS = ("latency_from_start_s", "latency_from_end_s")
P90 = 0.9
RULE_CHECKS = (
    ("width/height", baseline.RATIO_COLUMN, "ratio_"),
    ("trunk angle", baseline.ANGLE_COLUMN, "angle_deg_"),
    ("hip speed", baseline.SPEED_COLUMN, "speed_"),
)


def load_config():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["pose"]["task_file"] = str(ROOT / cfg["pose"]["task_file"])
    return cfg


def stamp():
    return datetime.now().isoformat(timespec="seconds")


def log_file(cfg, name):
    folder = ROOT / cfg["paths"]["logs"]
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def log_error(cfg, message):
    print(f"{stamp()} ERROR {message}", flush=True)
    with open(log_file(cfg, "errors.log"), "a", encoding="utf-8") as handle:
        handle.write(f"{stamp()} {message}\n")


def append_csv(path, columns, row):
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def ready(detector):
    detector["model"].set_params(**{key: 1 for key in detector["model"].get_params() if key.endswith("n_jobs")})
    fall_scores(detector["model"], pd.DataFrame([dict.fromkeys(detector["features"], np.nan)]))
    return detector


def load_detector(path):
    return ready(joblib.load(path))


def load_setup(cfg):
    demo = cfg["demo"]
    rule = load_detector(ROOT / demo["rule_file"])
    try:
        ai, fallback = load_detector(ROOT / demo["model_file"]), False
    except Exception as error:
        log_error(cfg, f"cannot load {demo['model_file']}, running RULE FALLBACK: {error!r}")
        ai, fallback = rule, True
    seconds = np.arange(round(demo["beep_s"] * SAMPLE_RATE)) / SAMPLE_RATE
    tone = (demo["beep_volume"] * np.sin(2 * np.pi * demo["beep_hz"] * seconds)).astype(np.float32)
    return {"panels": {"rule": rule, "ai": ai}, "fallback": fallback, "tone": tone}


def beep(cfg, tone):
    try:
        import sounddevice
        sounddevice.play(tone, SAMPLE_RATE)
    except Exception as error:
        log_error(cfg, f"beep failed: {error!r}")


def features_at(t, kp, t0, cfg):
    settings = cfg["windows"]
    xy = windows.resample(t, kp, windows.window_grid(t0, settings), settings["max_gap_fill_s"])
    if np.mean(~features.usable_frames(xy)) > settings["max_missing_fraction"] + windows.EPS:
        return None
    return features.window_features(xy, settings["rate_hz"], cfg["baseline"]["last_s"])


def window_span(settings):
    return windows.window_grid(0.0, settings)[-1]


def new_track():
    return {"score": np.nan, "positive": False, "streak": 0, "alert_t": -np.inf, "scored": 0, "score_s": 0.0}


def new_state():
    return {
        "t": deque(), "kp": deque(), "kp_now": None, "last_ms": -1, "next_eval": -np.inf,
        "status": "BUFFERING", "features": None, "rule": new_track(), "ai": new_track(),
        "walls": deque(), "fps": 0.0, "frames": 0, "busy_s": 0.0, "beat": time.perf_counter(),
    }


def set_idle(state, status):
    state["status"], state["features"] = status, None
    for panel in ("rule", "ai"):
        state[panel].update(score=np.nan, positive=False, streak=0)
    return []


def evaluate(state, setup, cfg):
    t, kp = np.array(state["t"]), np.stack(state["kp"])
    span = window_span(cfg["windows"])
    if t[-1] - t[0] < span:
        return set_idle(state, "BUFFERING")
    values = features_at(t, kp, t[-1] - span - windows.EPS, cfg)
    if values is None:
        return set_idle(state, "NO PERSON")
    state["status"], state["features"] = "OK", values
    row = pd.DataFrame([values])
    alerts = []
    for panel, detector in setup["panels"].items():
        track = state[panel]
        began = time.perf_counter()
        track["score"] = float(fall_scores(detector["model"], row[detector["features"]])[0])
        track["score_s"] += time.perf_counter() - began
        track["scored"] += 1
        track["positive"] = track["score"] >= detector["threshold"]
        track["streak"] = track["streak"] + 1 if track["positive"] else 0
        rested = t[-1] - track["alert_t"] >= cfg["demo"]["cooldown_s"]
        if track["streak"] >= cfg["demo"]["consecutive_positives"] and rested:
            track["alert_t"] = t[-1]
            alerts.append(panel)
    return alerts


def process(state, landmarker, frame, t, setup, cfg):
    state["last_ms"] = max(round(t * 1000), state["last_ms"] + 1)
    kp = detect_pose(landmarker, frame, state["last_ms"], cfg["pose"]["min_visibility"])
    state["kp_now"] = kp
    state["t"].append(t)
    state["kp"].append(kp)
    keep_s = cfg["windows"]["length_s"] + cfg["windows"]["max_gap_fill_s"]
    while state["t"][0] < t - keep_s:
        state["t"].popleft()
        state["kp"].popleft()
    if t < state["next_eval"]:
        return []
    state["next_eval"] = t + cfg["demo"]["eval_interval_s"]
    return evaluate(state, setup, cfg)


def raise_alerts(alerts, state, setup, cfg, label):
    for panel in alerts:
        detector = setup["panels"][panel]
        row = {
            "wall_time": stamp(), "source": label, "stream_t": state["t"][-1], "panel": panel,
            "detector": detector["name"], "score": state[panel]["score"], "threshold": detector["threshold"],
            "fallback": setup["fallback"],
        }
        append_csv(log_file(cfg, "alerts.csv"), ALERT_COLUMNS, row)
        print(f"{stamp()} FALL DETECTED on {panel} panel at stream t {row['stream_t']:.2f} s", flush=True)
        if panel == "ai":
            beep(cfg, setup["tone"])


def put_text(image, text, origin, color=WHITE, scale=0.6):
    thickness = 2 if scale > 0.7 else 1
    (width, height), baseline_px = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = origin
    cv2.rectangle(image, (x - 4, y - height - 4), (x + width + 4, y + baseline_px + 2), BLACK, -1)
    cv2.putText(image, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_skeleton(image, xy):
    points = {joint: (int(x), int(y)) for joint, (x, y) in enumerate(xy) if np.isfinite(x) and np.isfinite(y)}
    for start, end in CONNECTIONS:
        if start in points and end in points:
            cv2.line(image, points[start], points[end], CYAN, 2, cv2.LINE_AA)
    for point in points.values():
        cv2.circle(image, point, 3, WHITE, -1, cv2.LINE_AA)


def base_panel(frame, kp, privacy, height):
    scale = height / frame.shape[0]
    width = round(frame.shape[1] * scale)
    image = np.zeros((height, width, 3), np.uint8) if privacy else cv2.resize(frame, (width, height))
    draw_skeleton(image, kp[:, :2] * scale)
    return image


def bar_fraction(value, detector, cfg):
    if detector["score_method"] == "predict_proba":
        low, high = 0.0, 1.0
    else:
        half = cfg["demo"]["score_bar_half_width"]
        low, high = detector["threshold"] - half, detector["threshold"] + half
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def draw_bar(image, detector, track, cfg):
    height, width = image.shape[:2]
    left, right, top, bottom = 10, width - 10, height - 50, height - 25
    color = RED if track["positive"] else GREEN
    fill = left + round(bar_fraction(track["score"], detector, cfg) * (right - left))
    mark = left + round(bar_fraction(detector["threshold"], detector, cfg) * (right - left))
    cv2.rectangle(image, (left, top), (fill, bottom), color, -1)
    cv2.rectangle(image, (left, top), (right, bottom), WHITE, 1)
    cv2.line(image, (mark, top - 8), (mark, bottom + 8), AMBER, 3)
    label = "P(fall)" if detector["score_method"] == "predict_proba" else "SVM score (not a probability)"
    put_text(image, f"{label} {track['score']:.2f}   threshold {detector['threshold']:.2f}", (left, top - 12), color)


def draw_rule_checks(image, detector, values):
    for row, (label, column, attribute) in enumerate(RULE_CHECKS):
        value, limit = values[column], getattr(detector["model"], attribute)
        put_text(image, f"{label} {value:.2f} (alarm above {limit:.2f})", (10, 110 + 28 * row),
                 RED if value > limit else WHITE)


def decorate(image, panel, detector, state, setup, cfg):
    track = state[panel]
    fallback = panel == "ai" and setup["fallback"]
    if panel == "rule":
        title = "RULE (no AI)"
    else:
        title = "RULE FALLBACK (AI model not loaded)" if fallback else f"AI {detector['name']}"
    put_text(image, title, (10, 30), AMBER if fallback else WHITE, 0.8)
    if state["status"] != "OK":
        put_text(image, state["status"], (10, 75), AMBER, 1.2)
    else:
        put_text(image, "FALL-LIKE" if track["positive"] else "calm", (10, 75), RED if track["positive"] else GREEN, 1.0)
        if isinstance(detector["model"], baseline.RuleDetector):
            draw_rule_checks(image, detector, state["features"])
        else:
            draw_bar(image, detector, track, cfg)
    if state["t"][-1] - track["alert_t"] < cfg["demo"]["cooldown_s"]:
        middle = image.shape[0] // 2
        cv2.rectangle(image, (0, middle - 40), (image.shape[1], middle + 40), RED, -1)
        cv2.putText(image, "FALL DETECTED", (20, middle + 18), FONT, 1.6, WHITE, 3, cv2.LINE_AA)
    return image


def render(frame, state, setup, cfg, privacy):
    base = base_panel(frame, state["kp_now"], privacy, cfg["demo"]["panel_height"])
    canvas = np.hstack([decorate(base.copy(), panel, detector, state, setup, cfg)
                        for panel, detector in setup["panels"].items()])
    cv2.line(canvas, (base.shape[1], 0), (base.shape[1], canvas.shape[0]), WHITE, 2)
    put_text(canvas, f"{state['fps']:.1f} fps", (canvas.shape[1] - 140, 30), WHITE, 0.8)
    return canvas


def heartbeat_line(state, label):
    processing_fps = state["frames"] / state["busy_s"] if state["busy_s"] else 0.0
    return (f"{stamp()} {label}: frames {state['frames']}, fps {state['fps']:.1f}, "
            f"processing fps {processing_fps:.1f}, status {state['status']}, "
            f"rule {state['rule']['score']:.2f}, ai {state['ai']['score']:.2f}")


def tick(state, began, cfg, label):
    now = time.perf_counter()
    state["frames"] += 1
    state["busy_s"] += now - began
    walls = state["walls"]
    walls.append(began)
    while walls[0] < began - cfg["demo"]["fps_window_s"]:
        walls.popleft()
    state["fps"] = (len(walls) - 1) / (walls[-1] - walls[0]) if walls[-1] > walls[0] else 0.0
    if now - state["beat"] >= cfg["demo"]["heartbeat_s"]:
        state["beat"] = now
        print(heartbeat_line(state, label), flush=True)


def open_source(source):
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source {source}")
    return capture


def read_frames(cfg, source):
    capture = open_source(source)
    is_file = isinstance(source, str)
    file_fps = capture.get(cv2.CAP_PROP_FPS)
    try:
        if is_file and not file_fps > 0:
            raise RuntimeError(f"no readable fps in {source}")
        start = last_frame = time.perf_counter()
        index = 0
        while True:
            ok, frame = capture.read()
            now = time.perf_counter()
            if ok:
                last_frame = now
                yield (index / file_fps if is_file else now - start), frame
                index += 1
            elif is_file:
                return
            elif now - last_frame > cfg["demo"]["camera_offline_s"]:
                raise RuntimeError(f"camera {source} gave no frame for {now - last_frame:.1f} s")
            else:
                time.sleep(READ_RETRY_S)
    finally:
        capture.release()


def stop_requested(key):
    return (key & 0xFF) in QUIT_KEYS or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1


def run(cfg, source, privacy, setup):
    state, label, clock = new_state(), str(source), None
    with create_landmarker(cfg) as landmarker:
        for t, frame in read_frames(cfg, source):
            began = time.perf_counter()
            clock = began - t if clock is None else clock
            raise_alerts(process(state, landmarker, frame, t, setup, cfg), state, setup, cfg, label)
            canvas = render(frame, state, setup, cfg, privacy)
            tick(state, began, cfg, label)
            cv2.imshow(WINDOW_NAME, canvas)
            if stop_requested(cv2.waitKey(max(1, round(1000 * (clock + t - time.perf_counter()))))):
                break
    print(heartbeat_line(state, label), flush=True)


def wait_offline(cfg, seconds, reason):
    height = cfg["demo"]["panel_height"]
    until = time.perf_counter() + seconds
    while time.perf_counter() < until:
        canvas = np.zeros((height, height * 8 // 3, 3), np.uint8)
        put_text(canvas, "CAMERA OFFLINE", (20, 80), RED, 1.6)
        put_text(canvas, reason[:90], (20, 130), WHITE)
        put_text(canvas, f"retrying in {until - time.perf_counter():.0f} s, q to quit", (20, 170), WHITE)
        cv2.imshow(WINDOW_NAME, canvas)
        if stop_requested(cv2.waitKey(OFFLINE_POLL_MS)):
            return True
    return False


def supervise(cfg, source, privacy, setup):
    demo = cfg["demo"]
    backoff = demo["restart_backoff_start_s"]
    while True:
        started = time.perf_counter()
        try:
            run(cfg, source, privacy, setup)
            return
        except Exception as error:
            reason = f"{error}"
            log_error(cfg, f"source {source}: {error!r}\n{traceback.format_exc()}")
        if time.perf_counter() - started > demo["restart_backoff_max_s"]:
            backoff = demo["restart_backoff_start_s"]
        if wait_offline(cfg, backoff, reason):
            return
        backoff = min(2 * backoff, demo["restart_backoff_max_s"])


def measure(cfg, source, setup, with_model):
    state, busy, count = new_state(), 0.0, 0
    with create_landmarker(cfg) as landmarker:
        for t, frame in read_frames(cfg, source):
            began = time.perf_counter()
            if with_model:
                process(state, landmarker, frame, t, setup, cfg)
                render(frame, state, setup, cfg, False)
            else:
                detect_pose(landmarker, frame, round(t * 1000), cfg["pose"]["min_visibility"])
            busy += time.perf_counter() - began
            count += 1
            if count >= cfg["demo"]["benchmark_frames"]:
                break
    if not count:
        raise RuntimeError(f"no frames read from {source}")
    return count, count / busy, state["ai"]


def mean_scoring_ms(track):
    return 1000 * track["score_s"] / track["scored"] if track["scored"] else np.nan


def benchmark(cfg, source, setup):
    frames, pose_fps, _ = measure(cfg, source, setup, False)
    _, full_fps, ai_track = measure(cfg, source, setup, True)
    scoring_ms = mean_scoring_ms(ai_track)
    row = {
        "date": stamp(), "machine": f"{platform.system()} {platform.processor()}",
        "pose_task_file": Path(cfg["pose"]["task_file"]).name, "source": str(source), "frames": frames,
        "pose_only_fps": pose_fps, "pose_plus_model_fps": full_fps, "detector": setup["panels"]["ai"]["name"],
        "window_scoring_ms": scoring_ms,
    }
    path = ROOT / cfg["paths"]["tables"] / "fps.csv"
    table = pd.DataFrame([row])
    if path.exists():
        table = pd.concat([pd.read_csv(path), table], ignore_index=True)
    write_csv_atomic(table.reindex(columns=FPS_COLUMNS), path)
    print(f"{row['detector']}: frames {frames}, pose only fps {pose_fps:.1f}, pose plus model fps {full_fps:.1f}, "
          f"window scoring {scoring_ms:.3f} ms mean over {ai_track['scored']} windows, appended to {path}")


def timing_only_detector(name, cfg, training):
    columns = feature_columns(training)
    model, threshold, _ = fit_ml(name, "default", training, columns, cfg)
    print(f"{name}_default fitted in memory for timing only on {len(training)} training windows, not saved")
    return ready({"model": model, "threshold": threshold, "name": f"{name}_default", "features": columns,
                  "score_method": score_method(model)})


def benchmark_setups(cfg, setup, choices):
    training = load_training_windows(cfg) if set(choices) & set(TIMING_ONLY) else None
    rule = setup["panels"]["rule"]
    for choice in choices:
        if choice == "best":
            ai = setup["panels"]["ai"]
        elif choice == "rule":
            ai = rule
        else:
            ai = timing_only_detector(choice, cfg, training)
        yield {**setup, "panels": {"rule": rule, "ai": ai}}


def replay_alerts(cfg, source, setup):
    state, fired, t = new_state(), {panel: [] for panel in setup["panels"]}, np.nan
    with create_landmarker(cfg) as landmarker:
        for t, frame in read_frames(cfg, source):
            for panel in process(state, landmarker, frame, t, setup, cfg):
                fired[panel].append(t)
    return fired, t


def fall_clips(cfg):
    videos = pd.read_csv(ROOT / cfg["paths"]["gmdcsa24_fall_times"], usecols=["clip_id", "video"])
    labels = pd.read_csv(ROOT / cfg["paths"]["labels"], usecols=["clip_id", "dataset", "fall_start_s", "fall_end_s"])
    clips = videos.merge(labels, on="clip_id", how="left", validate="one_to_one")
    untimed = clips.loc[clips[["fall_start_s", "fall_end_s"]].isna().any(axis=1), "clip_id"].tolist()
    if untimed:
        raise ValueError(f"{len(untimed)} fall clips without times in labels.csv, e.g. {untimed[:3]}")
    return clips


def latency_rows(clip, fired, video_s, setup):
    for panel, times in fired.items():
        detector = setup["panels"][panel]
        first = min(times, default=np.nan)
        yield {
            "clip_id": clip.clip_id, "detector": detector["name"], "panel": panel,
            "fall_start_s": clip.fall_start_s, "fall_end_s": clip.fall_end_s, "video_s": video_s,
            "alerts": len(times), "first_alert_s": first, "latency_from_start_s": first - clip.fall_start_s,
            "latency_from_end_s": first - clip.fall_end_s, "no_alert": not times,
            "in_sample": clip.dataset in detector.get("train_datasets", []),
        }


def latency_summary_row(detector, rows):
    alerted = rows[~rows["no_alert"]]
    row = {
        "detector": detector, "clips": len(rows), "alerted": len(alerted), "no_alert": len(rows) - len(alerted),
        "first_alert_before_fall_start": int((alerted["latency_from_start_s"] < 0).sum()),
    }
    for column in LATENCY_COLUMNS:
        values = alerted[column]
        row.update({f"{column}_median": values.median(), f"{column}_p90": values.quantile(P90),
                    f"{column}_max": values.max()})
    row["in_sample"] = bool(rows["in_sample"].all())
    row["note"] = ("in sample: these fall clips are training data for this detector, so counts are optimistic"
                   if row["in_sample"] else "")
    return row


def alert_latency(cfg):
    setup = load_setup(cfg)
    if setup["fallback"]:
        raise RuntimeError(f"{cfg['demo']['model_file']} did not load, latency would describe RULE FALLBACK")
    clips, rows = fall_clips(cfg), []
    for number, clip in enumerate(clips.itertuples(index=False), 1):
        fired, video_s = replay_alerts(cfg, str(ROOT / clip.video), setup)
        rows += latency_rows(clip, fired, video_s, setup)
        firsts = ", ".join(f"{panel} {min(times, default=np.nan):.2f}" for panel, times in fired.items())
        print(f"{number}/{len(clips)} {clip.clip_id}: fall {clip.fall_start_s:.2f} to {clip.fall_end_s:.2f} s, "
              f"first alert {firsts}", flush=True)
    table = pd.DataFrame(rows)
    summary = pd.DataFrame([latency_summary_row(name, group) for name, group in table.groupby("detector", sort=False)])
    tables = ROOT / cfg["paths"]["tables"]
    write_csv_atomic(table, tables / "alert_latency.csv")
    write_csv_atomic(summary, tables / "alert_latency_summary.csv")
    return {"clips": clips, "table": tables / "alert_latency.csv", "summary": tables / "alert_latency_summary.csv"}


def parse_args():
    parser = argparse.ArgumentParser(description="Live fall detection: rule baseline (left) and AI model (right).")
    parser.add_argument("--source", default="0", help="webcam index or video file path")
    parser.add_argument("--privacy", action="store_true", help="draw the skeleton on a black background")
    parser.add_argument("--benchmark", action="store_true",
                        help="measure pose only and pose plus model fps, append a row to results/tables/fps.csv")
    parser.add_argument("--detectors", nargs="+", choices=BENCHMARK_CHOICES, default=["best"],
                        help="right panel detectors to benchmark; nb and svm are fitted in memory for timing only")
    args = parser.parse_args()
    args.source = int(args.source) if args.source.isdigit() else args.source
    if isinstance(args.source, str) and not Path(args.source).is_file():
        parser.error(f"video file not found: {args.source}")
    return args


def main():
    args = parse_args()
    cfg = load_config()
    setup = load_setup(cfg)
    rule, ai = setup["panels"]["rule"], setup["panels"]["ai"]
    print(f"rule: {rule['name']}, thresholds ratio {rule['model'].ratio_}, angle {rule['model'].angle_deg_}, "
          f"speed {rule['model'].speed_}")
    print(f"ai: {ai['name']}, {ai['score_method']}, threshold {ai['threshold']}, fallback {setup['fallback']}")
    print(f"pose model: {cfg['pose']['task_file']}, source: {args.source}, privacy: {args.privacy}")
    if args.benchmark:
        for candidate in benchmark_setups(cfg, setup, args.detectors):
            benchmark(cfg, args.source, candidate)
        return
    supervise(cfg, args.source, args.privacy, setup)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
