import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.demo import alert_latency, load_config  # noqa: E402


def main():
    start = time.perf_counter()
    run = alert_latency(load_config())
    clips = run["clips"]
    print(f"fall clips per dataset: {clips.groupby('dataset').size().to_dict()}")
    table = pd.read_csv(run["table"])
    print(f"rows written: {len(table)} to {run['table']}")
    print(table.groupby("detector").agg(clips=("clip_id", "nunique"), alerted=("no_alert", lambda v: int((~v).sum())),
                                        alerts=("alerts", "sum")).to_string())
    summary = pd.read_csv(run["summary"])
    print(f"summary, read back from {run['summary']}:")
    print(summary.drop(columns="note").T.to_string(header=False))
    print(f"note: {summary['note'].tolist()}")
    print(f"runtime: {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
