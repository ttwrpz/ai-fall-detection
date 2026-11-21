import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.report import CONFIG_PATH, build_test_report  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Figures from the saved single test run, nothing is rescored.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    run = build_test_report(parser.parse_args().config)
    columns = ["name", "level", "n_test", "n_pos", "tp", "fp", "tn", "fn", "recall", "f1"]
    print("test results read from the saved table:")
    print(run["results"][columns].round(3).to_string(index=False))
    for level, path in run["outputs"].items():
        print(f"figure written ({level}): {path}")


if __name__ == "__main__":
    main()
