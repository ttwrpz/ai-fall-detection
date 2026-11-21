# AI Fall Detection from Camera

CSC340 Artificial Intelligence group project, SIT KMUTT.

Each 2 second window of video is classified as Fall or Not fall from MediaPipe body pose. Four AI models are compared with a rule based method (no AI). A live webcam demo raises an alert when someone falls.

## Team

| Student ID | Name | Role |
| --- | --- | --- |
| 67130500802 | Kraiwich Jaiton | Modelling |
| 67130500840 | Theerawat Patthawee | Modelling, GMDCSA-24 annotation |
| 67130500841 | Benyapon Saisong | Demo, features and baseline |
| 67130500853 | Napat Chang-Ekwong | Data, features and baseline, modelling, demo, GMDCSA-24 annotation |

## Results

Cross validation on URFD and GMDCSA-24, grouped by clip, window level.

| Detector | Recall | F1 |
| --- | --- | --- |
| Random Forest (demo model) | 0.917 | 0.851 |
| SVM (RBF) | 0.909 | 0.823 |
| LSTM | 0.895 | 0.737 |
| Naive Bayes | 0.897 | 0.724 |
| Rule, tuned (no AI) | 0.745 | 0.786 |

Unseen test on CAUCAFall, run once. Random Forest caught 40 of 50 falls (clip recall 0.800, F1 0.851). The tuned rule caught 31 (recall 0.620, F1 0.705). McNemar p = 0.004.

Every number comes from `results/tables/`. Charts are in `results/figures/`. The full write up is in `REPORT.md`.

## Setup

Python 3.12.

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.lock.txt
```

Download the full Pose Landmarker model from the [MediaPipe guide](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) and save it as `models/pose_landmarker_full.task`.

Do not install `opencv-python`. mediapipe already brings `opencv-contrib-python`.

## Live demo

```text
python -m src.demo --source 0
python -m src.demo --source 0 --privacy
python -m src.demo --source path/to/video.mp4
python -m src.demo --source 0 --benchmark
```

The rule and the AI run side by side. If the model file fails to load, the rule takes over.

## Data

Datasets are not in git. Put them under `data/raw/urfd`, `data/raw/gmdcsa24` and `data/raw/caucafall`.

| Dataset | Source | Role |
| --- | --- | --- |
| URFD | https://fenix.ur.edu.pl/mkepski/ds/uf.html | Training |
| GMDCSA-24 | https://zenodo.org/records/12921216 | Training |
| CAUCAFall | https://data.mendeley.com/datasets/7w7fccy7ky | Unseen test, used once |

GMDCSA-24 has no fall times. The team marked them by hand in `data/annotations/gmdcsa24_fall_times.csv` with `tools/annotate_falls.py`.

## Pipeline

```text
python scripts/01_labels.py            labels for all clips
python scripts/02_extract_pose.py      pose keypoints per clip
python scripts/03_build_windows.py     windows and features
python scripts/04_cross_validate.py    grouped CV, thresholds, model choice
python scripts/05_test_caucafall.py    single unseen test (already run)
python scripts/06_cv_report.py         CV tables and figures
python scripts/07_test_report.py       test figures
python scripts/08_alert_latency.py     demo alert latency on videos
```

The LSTM runs in Google Colab with `notebooks/lstm_colab.ipynb`.

## Layout

```text
config.yaml        every tunable number
src/               labels, pose, windows, features, baseline, models, evaluate, report, demo
scripts/           thin wrappers around src
notebooks/         LSTM notebook for Colab
models/            frozen Random Forest and rule
results/           tables and figures
docs/              plan, rubric map, deployment options, slide outline
tools/             annotation tool, style check, data download
```

Run `python tools/check_style.py` before each commit. It must report 0 problems.
