import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

WINDOW = "annotate falls"
DISPLAY_WIDTH = 800
HELP = "space play/pause  a/d step  s start  e end  n save+next  r restart  q save+quit"


def read_frames(video):
    capture = cv2.VideoCapture(video)
    fps, frames = capture.get(cv2.CAP_PROP_FPS), []
    ok, frame = capture.read()
    while ok:
        scale = min(1.0, DISPLAY_WIDTH / frame.shape[1])
        frames.append(cv2.resize(frame, None, fx=scale, fy=scale))
        ok, frame = capture.read()
    capture.release()
    return fps, frames


def draw(frame, lines):
    image = frame.copy()
    for row, text in enumerate(lines):
        cv2.putText(image, text, (10, 25 + 25 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(image, text, (10, 25 + 25 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.imshow(WINDOW, image)


def annotate(clip_id, description, fps, frames):
    index, playing, start, end = 0, True, None, None
    while True:
        draw(frames[index], [f"{clip_id}  frame {index}/{len(frames) - 1}  t {index / fps:.3f} s  "
                             f"{'PLAY' if playing else 'PAUSE'}", f"start {start} s  end {end} s", description, HELP])
        key = cv2.waitKey(max(1, round(1000 / fps)) if playing else 30) & 0xFF
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1 or key == ord("q"):
            return start, end, True
        if key == ord("n"):
            return start, end, False
        if key == ord(" "):
            playing = not playing
        elif key == ord("r"):
            index, playing = 0, True
        elif key == ord("s"):
            start = round(index / fps, 3)
        elif key == ord("e"):
            end = round((index + 1) / fps, 3)
        elif key in (ord("a"), ord("d")) and not playing:
            index = min(max(index + (1 if key == ord("d") else -1), 0), len(frames) - 1)
        elif playing:
            index = min(index + 1, len(frames) - 1)
            playing = index < len(frames) - 1


def save(path, table):
    temp = path.with_name(path.name + ".tmp")
    table.to_csv(temp, index=False)
    while True:
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            message = f"cannot write {path.name}: close it in Excel or any editor, retrying"
            print(message, flush=True)
            draw(np.zeros((120, DISPLAY_WIDTH, 3), dtype=np.uint8), [message])
            cv2.waitKey(2000)


def main():
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    path = Path(cfg["paths"]["gmdcsa24_fall_times"])
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    todo = table.index[(table.fall_start_s == "") | (table.fall_end_s == "")]
    print(f"{len(todo)} of {len(table)} clips need times")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    for done, row in enumerate(todo, 1):
        fps, frames = read_frames(table.at[row, "video"])
        if not frames or not fps > 0:
            print(f"{table.at[row, 'clip_id']}: unreadable video, skipped")
            continue
        start, end, stop = annotate(table.at[row, "clip_id"], table.at[row, "description"], fps, frames)
        if start is not None and end is not None and start < end:
            table.loc[row, ["fall_start_s", "fall_end_s"]] = [str(start), str(end)]
            save(path, table)
            print(f"{done}/{len(todo)} {table.at[row, 'clip_id']}: {start} to {end} s saved")
        else:
            print(f"{done}/{len(todo)} {table.at[row, 'clip_id']}: start {start} end {end} not saved")
        if stop:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
