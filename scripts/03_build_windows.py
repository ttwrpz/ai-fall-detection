import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import windows  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Cut 2 s windows, label them and compute features.")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    summary = windows.build(parser.parse_args().config)
    print(f"clips read per dataset: {summary['clips_read']}")
    print(f"clips in labels without keypoints: {summary['clips_without_keypoints']}")
    print(f"windows per class: {summary['windows_per_class']}")
    print("windows per dataset and class:")
    print(summary["windows_per_dataset_and_class"].to_string())
    print(f"dropped ambiguous windows: {summary['dropped_ambiguous']}")
    print(f"dropped missing pose windows: {summary['dropped_missing_pose']}")
    print(f"clips with zero windows: {len(summary['zero_window_clips'])}")
    for clip_id, frames, reason in summary["zero_window_clips"]:
        print(f"  {clip_id}: {frames} frames, {reason}")
    print(f"feature columns: {summary['feature_count']}, sequences: {summary['sequence_shape']}")
    print("feature NaN fraction on training datasets (nonzero only):")
    print(summary["nan_fraction"].to_string() if len(summary["nan_fraction"]) else "  none")
    print(f"mostly NaN features: {summary['mostly_nan'] or 'none'}")


if __name__ == "__main__":
    main()
