import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.labels import build_labels  # noqa: E402


def main():
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    labels, problems = build_labels(cfg)
    labels["is_fall_clip"] = labels.activity.str.startswith("fall")
    labels["has_interval"] = labels.fall_start_s.notna()
    print("Clips per dataset and activity:")
    print(labels.groupby(["dataset", "activity"]).size().to_string())
    print("Clips, fall clips and falls with an interval per dataset:")
    print(labels.groupby("dataset").agg(clips=("clip_id", "size"), fall_clips=("is_fall_clip", "sum"),
                                        falls_with_interval=("has_interval", "sum")).to_string())
    gmdcsa24_falls = labels[(labels.dataset == "gmdcsa24") & labels.is_fall_clip]
    print(f"GMDCSA-24 fall clips with annotated times: {gmdcsa24_falls.has_interval.sum()}, "
          f"without: {(~gmdcsa24_falls.has_interval).sum()} (file {cfg['paths']['gmdcsa24_fall_times']})")
    cauca_falls = labels[(labels.dataset == "caucafall") & labels.has_interval]
    print("CAUCAFall onset frame and falling interval, first clips:")
    print(cauca_falls[["clip_id", "fps", "onset_frame", "fall_start_s", "fall_end_s"]].head(6).to_string(index=False))
    if problems:
        print(f"{cfg['paths']['labels']} NOT written. {len(problems)} problem(s):")
        print("\n".join(problems))
        sys.exit(1)
    print(f"Wrote {cfg['paths']['labels']}")


if __name__ == "__main__":
    main()
