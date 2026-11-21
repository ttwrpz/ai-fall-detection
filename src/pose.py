import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from src.labels import (URFD_FPS, caucafall_folders, caucafall_fps, caucafall_frame_files, gmdcsa24_videos,
                        urfd_frame_files, urfd_frame_number, urfd_names, video_info)

NUM_LANDMARKS = 33


def create_landmarker(cfg):
    vision = mp.tasks.vision
    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=cfg["pose"]["task_file"]),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=cfg["pose"]["num_poses"],
    )
    return vision.PoseLandmarker.create_from_options(options)


def detect_pose(landmarker, frame_bgr, timestamp_ms, min_visibility):
    height, width = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
    if not result.pose_landmarks:
        return np.full((NUM_LANDMARKS, 3), np.nan, dtype=np.float32)
    kp = np.array([[lm.x * width, lm.y * height, lm.visibility] for lm in result.pose_landmarks[0]], dtype=np.float32)
    kp[kp[:, 2] < min_visibility, :2] = np.nan
    return kp


def image_frames(files, times):
    for path, t in zip(files, times):
        yield t, cv2.imread(str(path)), path


def video_frames(path, fps):
    capture = cv2.VideoCapture(str(path))
    index = 0
    ok, frame = capture.read()
    while ok:
        yield index / fps, frame, path
        index += 1
        ok, frame = capture.read()
    capture.release()


def urfd_clips(raw):
    for name in urfd_names(raw):
        files = urfd_frame_files(raw, name)
        times = [(urfd_frame_number(path) - 1) / URFD_FPS for path in files]
        yield f"urfd-{name}", name.split("-")[0], URFD_FPS, image_frames(files, times)


def gmdcsa24_clips(raw):
    for clip_id, _, activity, video in gmdcsa24_videos(raw):
        fps = video_info(video)[0]
        yield clip_id, activity, fps, video_frames(video, fps)


def caucafall_clips(raw):
    for clip_id, _, activity, folder in caucafall_folders(raw):
        fps = caucafall_fps(folder)
        files = caucafall_frame_files(folder)
        yield clip_id, activity, fps, image_frames(files, [index / fps for index in range(len(files))] if fps > 0 else [])


CLIPS = {"urfd": urfd_clips, "gmdcsa24": gmdcsa24_clips, "caucafall": caucafall_clips}


def extract_clip(cfg, clip_id, fps, frames):
    if not fps > 0:
        return None, None, f"{clip_id}: no readable fps"
    times, keypoints = [], []
    with create_landmarker(cfg) as landmarker:
        for t, frame, source in frames:
            if frame is None:
                return None, None, f"{clip_id}: unreadable frame {source}"
            times.append(t)
            keypoints.append(detect_pose(landmarker, frame, round(t * 1000), cfg["pose"]["min_visibility"]))
    if not times:
        return None, None, f"{clip_id}: no frames"
    return np.array(times), np.stack(keypoints), None


def save_keypoints(path, t, kp, fps):
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "wb") as handle:
        np.savez(handle, t=t, kp=kp, fps=np.float64(fps))
    os.replace(temp, path)


def pose_fraction(kp):
    return float((~np.isnan(kp[:, :, 2]).all(axis=1)).mean())


def extract_and_save(cfg, clip_id, fps, frames, path):
    start = time.perf_counter()
    t, kp, problem = extract_clip(cfg, clip_id, fps, frames)
    if problem:
        return problem
    seconds = time.perf_counter() - start
    fraction = pose_fraction(kp)
    print(f"{clip_id}: {len(t)} frames at {fps:.3f} fps, pose in {fraction:.3f}, {1000 * seconds / len(t):.1f} ms/frame")
    save_keypoints(path, t, kp, fps)
    if fraction == 0:
        print(f"{clip_id}: zero detected poses, cached as all NaN, it will yield no windows")
    return None


def extract_dataset(cfg, datasets):
    out = Path(cfg["paths"]["keypoints"])
    out.mkdir(parents=True, exist_ok=True)
    problems, clips = [], []
    for dataset in datasets:
        for clip_id, activity, fps, frames in CLIPS[dataset](cfg["paths"]["raw"]):
            path = out / f"{clip_id}.npz"
            problem = None if path.exists() else extract_and_save(cfg, clip_id, fps, frames, path)
            if problem:
                problems.append(problem)
                print(problem)
            clips.append({"dataset": dataset, "clips": 1, "fall_clips": activity.startswith("fall"),
                          "cached": path.exists()})
    return pd.DataFrame(clips, columns=["dataset", "clips", "fall_clips", "cached"]).groupby("dataset").sum(), problems
