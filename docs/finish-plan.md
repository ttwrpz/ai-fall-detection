# Finish plan

Submission: Mon 23 Nov 2026 before 23:59, team target 18:00. Presentation: Tue 24 Nov 2026, 09:00 to 12:00.

## Tasks by day and role

| Day | Data | Features and baseline | Modelling | Demo and presentation |
| --- | --- | --- | --- | --- |
| Fri 20 | Download all data, write `labels.py`, split pose extraction across four laptops | `windows.py`, `features.py`, `baseline.py` on URFD first | `evaluate.py` running end to end on URFD windows | Install the venv on the presentation laptop, benchmark full against lite pose model, skeleton, fps, banner and beep working |
| Sat 21 | Check labels by eye on 10 clips per dataset, dataset slide | Rule tuned in cross validation, activity labels for non fall clips | Cross validation for NB, RF, SVM before and after tuning, LSTM in Colab | Model in the demo, debouncing, side by side, fallback, privacy mode, record the backup video |
| Sun 22 | Error analysis screenshots | Feature importance and activity alarm table | Freeze the best model, run CAUCAFall once, McNemar, PR curves, confusion matrices | Draft all slides, test the supervisor by unplugging the webcam |
| Mon 23 | Proofread data slides | Proofread method slides | Check every slide number against `results/tables/` | Two timed rehearsals, export PDF, upload by 18:00 |

## End of day gates

Fri 20
- `data/labels.csv` exists and covers all three datasets.
- `data/keypoints/` holds one file per clip.
- `data/features/windows.parquet` exists.
- Pose model variant chosen and recorded in `config.yaml`.

Sat 21
- `results/tables/cv_results.csv` covers the rule, NB, RF, SVM and LSTM.
- `results/tables/tuning_before_after.csv` exists.
- `models/best.joblib` exists and the demo runs with it.
- `requirements.lock.txt` exists.

Sun 22
- `results/tables/caucafall_results.csv` exists, written once.
- Figures exist in `results/figures/`: confusion matrices, PR curves, feature importance.
- `docs/slides.md` covers every column of the evaluation form.
- Numbers are frozen. No model changes after this point.

Mon 23
- Slides exported to PDF.
- Backup demo video recorded and copied to a USB stick.
- Demo tested on the presentation laptop with Wi-Fi off.
- Submitted.

## If a gate is missed

Cut in this order, and say so on the limits slide: self recorded clips, the GRU line, the LSTM window length grid, the three seed average, Option B deployment. Never cut grouped cross validation, the rule baseline, the before and after tuning table or the single CAUCAFall test.
