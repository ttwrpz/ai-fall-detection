# AI Fall Detection from Camera

CSC340 Artificial Intelligence group project, SIT KMUTT. A camera based system that labels each 2 second window of video as Fall or Not fall, compared against a rule based method with no AI, with a live webcam demo.

This repository starts as an empty skeleton. The code is written into `src/` and `scripts/` by the team.

## Setup

Python 3.12 on every laptop.

```text
python -m venv .venv
.venv\Scripts\activate          (Windows)
source .venv/bin/activate       (macOS, Linux)
pip install -r requirements.txt
pip show mediapipe
```

Download the full Pose Landmarker model from the Models section of the [MediaPipe Pose Landmarker guide](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) and save it as `models/pose_landmarker_full.task`.

Once everything runs, freeze the versions and install from the lock file on the presentation laptop:

```text
pip freeze > requirements.lock.txt
pip install -r requirements.lock.txt
```

Do not install `opencv-python`. mediapipe already installs `opencv-contrib-python`, and two OpenCV packages clash.

The LSTM is trained in Google Colab with the TensorFlow that Colab provides. Do not run `pip install tensorflow` there.

## Data

Datasets are not stored in git. Put them under `data/raw/` like this:

```text
data/raw/urfd/
data/raw/gmdcsa24/
data/raw/caucafall/
```

| Dataset | Source | Role |
| --- | --- | --- |
| URFD | https://fenix.ur.edu.pl/mkepski/ds/uf.html | Training and tuning |
| GMDCSA-24 | https://zenodo.org/records/12921216 | Training and tuning |
| CAUCAFall | https://data.mendeley.com/datasets/7w7fccy7ky | Unseen test, used once |

## Layout

```text
CLAUDE.md               project brief and rules for Claude Code
.claude/                settings, rules, subagents and skills for Claude Code
config.yaml             every tunable number
requirements.txt
data/raw/               downloaded datasets (not in git)
data/labels.csv         one row per clip
data/keypoints/         cached pose per clip (not in git)
data/features/          windows and sequences (not in git)
src/                    labels, pose, windows, features, baseline, models, evaluate, demo
scripts/                01 to 05, thin wrappers around src
notebooks/              lstm_colab.ipynb
models/                 best.joblib, rule.joblib, pose task file
results/tables/         CSV results, the only source of slide numbers
results/figures/        PNG charts
logs/                   demo alerts and errors (not in git)
docs/                   finish plan, rubric map, deployment options
tools/check_style.py    flags em dashes and code comments
```

## Reproduce

```text
python scripts/01_labels.py
python scripts/02_extract_pose.py
python scripts/03_build_windows.py
python scripts/04_cross_validate.py
python scripts/05_test_caucafall.py
python -m src.demo --source 0
```

`05_test_caucafall.py` is run once, after every model is frozen.

## Team rules

1. Split by clip, never by frame or window.
2. CAUCAFall is touched once and never influences a choice.
3. Slide numbers come from `results/tables/` only.
4. No comments in code and no em dashes anywhere. `python tools/check_style.py` must print `0 problem(s)` before a commit.
5. Keep it simple. Ask the team before adding a dependency.

## Using Claude Code in this repository

Open the folder with Claude Code. `CLAUDE.md` and `.claude/` load on their own.

| Ask for | What it does |
| --- | --- |
| `data-lead` subagent | Labels and pose extraction |
| `features-baseline` subagent | Windows, features, rule baseline |
| `modelling-lead` subagent | Models, cross validation, results |
| `demo-builder` subagent | Live demo |
| `integrity-reviewer` subagent | Read only check for leakage, test set isolation and style |
| `presentation-coach` subagent | Drafts `docs/slides.md` against the evaluation form |
| `/check` | Style check plus integrity review before a commit |
| `/gate sat` | Progress against the end of day gate |

Example: `Use the data-lead subagent to write src/labels.py for URFD first.`

Personal settings go in `.claude/settings.local.json` and `CLAUDE.local.md`. Both are ignored by git.
