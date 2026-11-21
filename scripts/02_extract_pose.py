import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.pose import CLIPS, extract_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(CLIPS), action="append")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    counts, problems = extract_dataset(cfg, args.dataset or list(CLIPS))
    print("Clips on disk, fall clips (from folder names) and clips cached per dataset:")
    print(counts.to_string())
    if problems:
        print("Clips that need a human (unreadable or no frames):")
        print("\n".join(problems))
        sys.exit(1)


if __name__ == "__main__":
    main()
