import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAW = Path("data/raw")
URFD_BASE = "https://fenix.ur.edu.pl/mkepski/ds/data/"
URFD_LABELS = ["urfall-cam0-falls.csv", "urfall-cam0-adls.csv"]
GMDCSA_URL = (
    "https://zenodo.org/records/12921216/files/ekramalam/"
    "GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.0.zip?download=1"
)
CAUCAFALL_PAGE = "https://data.mendeley.com/datasets/7w7fccy7ky"


def urfd_archives():
    falls = [f"fall-{i:02d}-cam0-rgb.zip" for i in range(1, 31)]
    adls = [f"adl-{i:02d}-cam0-rgb.zip" for i in range(1, 41)]
    return falls + adls


def fetch(url, target, attempts=3):
    if target.exists():
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response, open(part, "wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            part.replace(target)
            print(f"ok    {target.name}", flush=True)
            return True
        except Exception as error:
            print(f"retry {target.name} ({attempt}/{attempts}): {error}", flush=True)
            time.sleep(5 * attempt)
    return False


def unzip(archive, folder):
    if folder.exists() and any(folder.iterdir()):
        return
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(folder)


def download_urfd():
    folder = RAW / "urfd"
    names = urfd_archives() + URFD_LABELS
    with ThreadPoolExecutor(max_workers=4) as pool:
        done = list(pool.map(lambda name: fetch(URFD_BASE + name, folder / "zips" / name), names))
    failed = [name for name, ok in zip(names, done) if not ok]
    for name in urfd_archives():
        archive = folder / "zips" / name
        if archive.exists():
            unzip(archive, folder / "frames" / archive.stem)
    for name in URFD_LABELS:
        source = folder / "zips" / name
        if source.exists():
            (folder / name).write_bytes(source.read_bytes())
    clips = len([p for p in (folder / "frames").glob("*") if p.is_dir()]) if (folder / "frames").exists() else 0
    print(f"urfd: {clips} of 70 clips unpacked, {len(failed)} failed")
    return failed


def download_gmdcsa24():
    folder = RAW / "gmdcsa24"
    archive = folder / "gmdcsa24.zip"
    if not fetch(GMDCSA_URL, archive):
        print("gmdcsa24: download failed")
        return ["gmdcsa24.zip"]
    unzip(archive, folder / "files")
    videos = len(list((folder / "files").rglob("*.mp4")))
    print(f"gmdcsa24: {videos} mp4 files unpacked")
    return []


def remind_caucafall():
    folder = RAW / "caucafall"
    videos = len(list(folder.rglob("*.avi"))) if folder.exists() else 0
    print(f"caucafall: {videos} avi files found in {folder}")
    if videos == 0:
        print(f"caucafall: open {CAUCAFALL_PAGE}, press Download All, unzip into {folder}")
    return []


STEPS = {"urfd": download_urfd, "gmdcsa24": download_gmdcsa24, "caucafall": remind_caucafall}


def main(names):
    failed = [item for name in names or list(STEPS) for item in STEPS[name]()]
    if failed:
        print("failed: " + ", ".join(failed))
        print("run the same command again, finished files are kept")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
