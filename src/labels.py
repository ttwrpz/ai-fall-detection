import csv
import os
import re
from pathlib import Path

import cv2
import pandas as pd

COLUMNS = ["clip_id", "dataset", "subject", "fps", "fall_start_s", "fall_end_s", "activity"]
FALL_TIME_COLUMNS = ["clip_id", "video", "fall_start_s", "fall_end_s", "description"]
URFD_FPS = 30.0


def urfd_frame_files(raw, name):
    folder = Path(raw) / "urfd" / "frames" / f"{name}-cam0-rgb" / f"{name}-cam0-rgb"
    return sorted(folder.glob(f"{name}-cam0-rgb-*.png"), key=urfd_frame_number)


def urfd_frame_number(path):
    return int(path.stem.rsplit("-", 1)[1])


def urfd_names(raw):
    frames = Path(raw) / "urfd" / "frames"
    return sorted(p.name.removesuffix("-cam0-rgb") for p in frames.iterdir() if p.is_dir())


def urfd_falling_frames(raw):
    path = Path(raw) / "urfd" / "urfall-cam0-falls.csv"
    table = pd.read_csv(path, header=None, usecols=[0, 1, 2], names=["name", "frame", "label"])
    falling = table[table.label == 0]
    return {name: group.frame.sort_values().to_numpy() for name, group in falling.groupby("name")}


def urfd_row(raw, name, falling):
    activity = name.split("-")[0]
    files = urfd_frame_files(raw, name)
    row = {"clip_id": f"urfd-{name}", "dataset": "urfd", "subject": None, "fps": URFD_FPS,
           "fall_start_s": None, "fall_end_s": None, "activity": activity}
    if activity not in ("fall", "adl"):
        return row, f"urfd {name}: name is neither fall nor adl"
    if not files:
        return row, f"urfd {name}: no frames on disk"
    if activity == "adl":
        return row, None
    frames = falling.get(name)
    if frames is None:
        return row, f"urfd {name}: no frame labelled 0 in urfall-cam0-falls.csv"
    if frames[-1] - frames[0] + 1 != len(frames):
        return row, f"urfd {name}: frames labelled 0 are not contiguous"
    if frames[-1] > urfd_frame_number(files[-1]):
        return row, f"urfd {name}: labelled frame {frames[-1]} is past the last frame on disk"
    row["fall_start_s"] = (frames[0] - 1) / URFD_FPS
    row["fall_end_s"] = frames[-1] / URFD_FPS
    return row, None


def load_urfd(cfg):
    raw = cfg["paths"]["raw"]
    falling = urfd_falling_frames(raw)
    return [urfd_row(raw, name, falling) for name in urfd_names(raw)]


def last_number(text):
    return int(re.findall(r"\d+", text)[-1])


def video_info(path):
    capture = cv2.VideoCapture(str(path))
    fps, frames = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return fps, frames


def gmdcsa24_videos(raw):
    subjects = sorted((Path(raw) / "gmdcsa24" / "files").glob("*/Subject *"), key=lambda p: last_number(p.name))
    return [(f"gmdcsa24-s{last_number(subject.name)}-{kind.lower()}-{last_number(video.stem):02d}",
             f"s{last_number(subject.name)}", kind.lower(), video)
            for subject in subjects for kind in ("ADL", "Fall") for video in sorted((subject / kind).glob("*.mp4"))]


def gmdcsa24_fall_descriptions(subject_folder):
    with open(subject_folder / "Fall.csv", newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.reader(handle) if row][1:]
    return {row[0].strip(): row[-1].strip() for row in rows}


def gmdcsa24_fall_template(raw):
    return pd.DataFrame([
        {"clip_id": clip_id, "video": video.as_posix(), "fall_start_s": "", "fall_end_s": "",
         "description": gmdcsa24_fall_descriptions(video.parent.parent).get(video.name, "")}
        for clip_id, _, activity, video in gmdcsa24_videos(raw) if activity == "fall"
    ], columns=FALL_TIME_COLUMNS)


def sync_fall_times(raw, path):
    path = Path(path)
    template = gmdcsa24_fall_template(raw)
    existing = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else template.iloc[:0]
    added = template[~template.clip_id.isin(existing.clip_id)]
    if len(added):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        pd.concat([existing, added]).to_csv(temp, index=False)
        os.replace(temp, path)
    times = pd.read_csv(path)
    return {row.clip_id: (row.fall_start_s, row.fall_end_s) for row in times.itertuples()}


def gmdcsa24_row(clip_id, subject, activity, video, times):
    fps, frames = video_info(video)
    row = {"clip_id": clip_id, "dataset": "gmdcsa24", "subject": subject, "fps": fps,
           "fall_start_s": None, "fall_end_s": None, "activity": activity}
    if not fps > 0 or frames <= 0:
        return row, f"{clip_id}: unreadable video {video}"
    start, end = times.get(clip_id, (None, None))
    if activity == "adl" or pd.isna(start) or pd.isna(end):
        return row, None
    if not 0 <= start < end <= round(frames / fps, 3):
        return row, f"{clip_id}: annotated {start} to {end} s is outside 0 to {frames / fps:.3f} s or not increasing"
    row["fall_start_s"], row["fall_end_s"] = float(start), float(end)
    return row, None


def load_gmdcsa24(cfg):
    raw = cfg["paths"]["raw"]
    times = sync_fall_times(raw, cfg["paths"]["gmdcsa24_fall_times"])
    return [gmdcsa24_row(*clip, times) for clip in gmdcsa24_videos(raw)]


def caucafall_folders(raw):
    subjects = sorted((Path(raw) / "caucafall" / "CAUCAFall").glob("Subject.*"), key=lambda p: last_number(p.name))
    return [(f"caucafall-s{last_number(subject.name)}-{folder.name.lower().replace(' ', '-')}",
             f"s{last_number(subject.name)}", folder.name.lower().replace(" ", "-"), folder)
            for subject in subjects for folder in sorted(p for p in subject.iterdir() if p.is_dir())]


def caucafall_frame_files(folder):
    return sorted(folder.glob("*.png"), key=lambda p: last_number(p.stem))


def caucafall_fps(folder):
    videos = sorted(folder.glob("*.avi"))
    return video_info(videos[0])[0] if len(videos) == 1 else 0.0


def caucafall_frame_classes(folder):
    return {last_number(txt.stem): {int(line.split()[0]) for line in txt.read_text().splitlines() if line.strip()}
            for txt in folder.glob("*.txt") if txt.name != "classes.txt"}


def caucafall_row(clip_id, subject, activity, folder, lead_s, tail_s):
    fps = caucafall_fps(folder)
    row = {"clip_id": clip_id, "dataset": "caucafall", "subject": subject, "fps": fps,
           "fall_start_s": None, "fall_end_s": None, "activity": activity, "onset_frame": None}
    pngs = caucafall_frame_files(folder)
    if not fps > 0:
        return row, f"{clip_id}: no single readable avi for fps in {folder}"
    if not pngs or not (folder / "classes.txt").exists():
        return row, f"{clip_id}: no png frames or no classes.txt in {folder}"
    names = [line.strip() for line in (folder / "classes.txt").read_text().splitlines() if line.strip()]
    frame_classes = caucafall_frame_classes(folder)
    unknown = set().union(*frame_classes.values()) - set(range(len(names)))
    if unknown:
        return row, f"{clip_id}: class index {sorted(unknown)} not in classes.txt {names}"
    fall_class = names.index("fall") if "fall" in names else None
    numbers = [last_number(png.stem) for png in pngs]
    flagged = [i for i, number in enumerate(numbers) if fall_class in frame_classes.get(number, set())]
    if not activity.startswith("fall"):
        return row, f"{clip_id}: non fall activity has {len(flagged)} fall frames" if flagged else None
    unlabelled = [png.name for png, number in zip(pngs, numbers) if number not in frame_classes]
    if unlabelled:
        return row, f"{clip_id}: png frames without a label txt: {unlabelled[:5]}"
    if not flagged:
        return row, f"{clip_id}: fall activity has no frame of class fall"
    onset_s = flagged[0] / fps
    row["onset_frame"] = pngs[flagged[0]].name
    row["fall_start_s"], row["fall_end_s"] = max(0.0, onset_s - lead_s), onset_s + tail_s
    return row, None


def load_caucafall(cfg):
    lead_s, tail_s = cfg["labels"]["caucafall_lead_s"], cfg["labels"]["caucafall_tail_s"]
    return [caucafall_row(*clip, lead_s, tail_s) for clip in caucafall_folders(cfg["paths"]["raw"])]


LOADERS = {"urfd": load_urfd, "gmdcsa24": load_gmdcsa24, "caucafall": load_caucafall}


def missing_intervals(labels):
    missing = labels[labels.activity.str.startswith("fall") & labels.fall_start_s.isna()]
    return [f"{dataset}: {count} fall clips without a falling interval" for dataset, count in missing.dataset.value_counts().items()]


def build_labels(cfg):
    results = [result for load in LOADERS.values() for result in load(cfg)]
    labels = pd.DataFrame([row for row, _ in results])
    problems = [problem for _, problem in results if problem] + missing_intervals(labels)
    problems += [f"duplicate clip_id {clip_id}" for clip_id in labels.clip_id[labels.clip_id.duplicated()]]
    if problems:
        return labels, problems
    path = Path(cfg["paths"]["labels"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    labels[COLUMNS].to_csv(temp, index=False)
    os.replace(temp, path)
    return labels, problems
