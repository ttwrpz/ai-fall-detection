# Slides: AI Fall Detection from Camera

CSC340, 7 minute talk with the live demo, then 3 minutes of Q&A. Built in Claude Slides from the course template. This file mirrors the deck slide by slide. Presentation: Tue 24 Nov 2026. Submission: Mon 23 Nov 2026, team target 18:00.

## Team and roles

| Role | Members |
| --- | --- |
| Data | Napat Chang-Ekwong |
| GMDCSA-24 hand annotation | Theerawat Patthawee (tool), Napat Chang-Ekwong |
| Features and rule baseline | Napat Chang-Ekwong, Benyapon Saisong |
| Modelling | Kraiwich Jaiton, Napat Chang-Ekwong, Theerawat Patthawee |
| Live demo | Benyapon Saisong, Napat Chang-Ekwong |
| Result tables and figures | Whole group |
| Presentation | Everyone |

Speaking time: Kraiwich 125 s (models, test, conclusion), Napat 120 s (preparation, features, findings and limits), Benyapon 110 s (system, test screenshots, demo, who did what), Theerawat 65 s (problem, datasets and annotation).

## How numbers are shown

Every metric comes from the named file in `results/tables/`. Slides show 3 decimal places, fps 1 decimal place. Design settings (window length, check interval, cooldown, backoff) come from `config.yaml` and are marked as settings, not results.

## Open team tasks

1. Webcam fps. `fps.csv` was measured on two GMDCSA-24 video files. The live fps shows on screen during the demo.
2. Unplug test of the camera supervisor. Until it is done, do not claim on stage that the demo recovers from an unplugged camera.
3. Record the backup demo video and copy it to a USB stick.
4. Choose who performs the fall on the mat. Rehearse the full talk twice with a timer.
5. Export the deck to PDF for the submission.

## Timing plan (420 s)

| Page | Title | Speaker | Seconds |
| --- | --- | --- | --- |
| 01 | AI Fall Detection from Camera | none | 0 |
| 02 | The problem and our questions | Theerawat | 35 |
| 03 | Three datasets, one kept unseen | Theerawat 30, then Napat 30 | 60 |
| 04 | Features and the rule (no AI) | Napat | 40 |
| 05 | Four models, one fair protocol | Kraiwich | 55 |
| 06 | Unseen test on CAUCAFall | Kraiwich | 45 |
| 07 | System design | Benyapon | 25 |
| 08 | Tested on a real fall video | Benyapon | 10 |
| 09 | Live demo | Benyapon (one other member performs) | 65 |
| 10 | Findings and limits | Napat | 50 |
| 11 | Answers to our questions | Kraiwich | 25 |
| 12 | Who did what | Benyapon | 10 |
| 13 | Thank you. Questions? | none, stays on screen during Q&A | 0 |
| | Total | | 420 |

Q&A: 180 s with page 13 on screen. Backups 1 to 3 are shown only if asked.

---

## Page 01. AI Fall Detection from Camera

- Title, course, SIT KMUTT, 24 Nov 2026
- Pose skeleton drawing
- All four members with student IDs

Figure: none.
Speaker: none. Seconds: 0.

Speaker notes:

On screen while the team walks up. No speaking time. Talk: 7 minutes with the live demo, then 3 minutes of questions.

---

## Page 02. The problem and our questions

- Aged society, falls at home alone, limits of wearables and fixed rules
- RQ1: AI against the rule on recall and F1
- RQ2: best of Naive Bayes, Random Forest, SVM (RBF), LSTM
- Goals: recall 0.90, F1 above the rule, 15 fps, live demo

Figure: none.
Speaker: Theerawat. Seconds: 35.

Speaker notes:

Theerawat Patthawee (35 s). Thailand is an aged society and many older people live alone. After a fall the danger is lying on the floor with nobody knowing. A wearable only works if it is worn. A camera does not need that, but the usual camera method is a fixed rule. We ask two questions: does machine learning on body pose beat that rule on recall and F1, and which of four models is best. Our goals: recall at least 0.90, because a missed fall is the costly error, F1 above the rule, 15 frames per second, and a live demo.

---

## Page 03. Three datasets, one kept unseen

- Dataset table: clips, people, windows, role (`dataset_summary.csv`)
- Preparation flow: video, pose, 15 Hz, torso scale, 2 s windows, label rule, folds by clip
- Cards: hand annotation of GMDCSA-24, CAUCAFall falling interval

Figure: drawn in the deck.
Speaker: Theerawat 30, then Napat 30. Seconds: 60.

Speaker notes:

Theerawat Patthawee (30 s): We use three public datasets. URFD and GMDCSA-24 are for training and cross validation. CAUCAFall, 100 clips from 10 people on a different camera, was locked away and run exactly once, after every model was frozen. GMDCSA-24 gives no fall times, so I built an annotation tool and Napat and I marked the start and end of all 79 falls by hand. CAUCAFall marks the moment the person is already down, so we move the label back to cover the fall itself.

Napat Chang-Ekwong (30 s): Every clip goes through the same pose model the demo uses, resampled to 15 frames per second and scaled by torso length. We cut 2 second windows every half second. A window is a fall only if it covers at least half of the fall. Lying still afterwards is not. Falls are rare: 349 of 2,167 training windows. We split by clip, so no clip is in both train and test.

Source: dataset_summary.csv, cv_pooled_metrics.csv.

---

## Page 04. Features and the rule (no AI)

- 7 body quantities, 39 window features
- Three rule checks with published and tuned thresholds
- Published: recall 0.754, F1 0.751. Tuned: recall 0.745, F1 0.786

Figure: drawn in the deck.
Speaker: Napat. Seconds: 40.

Speaker notes:

Napat Chang-Ekwong (40 s). From the skeleton we compute seven body quantities each frame and summarize each window into 39 numbers. The rule baseline uses three of them: is the body wide, is the trunk tilted, did the hips drop fast. With the thresholds from the literature it reaches recall 0.754 and F1 0.751. Tuned inside each training fold only, it picks a 30 degree angle and a faster hip speed in every fold. F1 rises to 0.786, recall stays near 0.75. Even a tuned rule misses about a quarter of fall windows. That is the bar the AI must beat.

Source: tuning_before_after.csv, cv_results.csv. The published speed of 1.0 is our stated default, not a published number.

---

## Page 05. Four models, one fair protocol

- Protocol: one pipeline, outer 5 and inner 3 folds by clip, threshold frozen at training recall 0.90
- Table of 8 detectors: recall, precision, F1 (mean and range over folds), fps
- Random Forest row highlighted as chosen

Figure: table.
Speaker: Kraiwich. Seconds: 55.

Speaker notes:

Kraiwich Jaiton (55 s). Four models under one protocol. Everything that learns from data sits in one pipeline, so nothing is fitted on a test fold, and the grid search runs inside the training part only. A missed fall is worse than a false alarm, so instead of 0.5 we pick the highest threshold that keeps recall at 0.90 on training predictions and freeze it. Random Forest has the best F1, 0.851, with recall 0.917, so it meets the goal in cross validation. Naive Bayes just misses recall at 0.897. SVM is close on recall but lower on F1. The LSTM reaches F1 0.737 and misses the recall goal. Tuning changed nothing for Random Forest, so we kept the default. With pose, Random Forest ran at 27 to 47 frames per second over five runs, above our goal of 15 in every run. The slowest run was on a busy machine: pose alone also dropped to 34 frames per second.

Sources: tuning_before_after.csv, lstm_summary.csv, model_selection.csv, fps.csv. SVM tuned fps was timed with default parameters. LSTM speed was not measured locally (Colab only).

---

## Page 06. Unseen test on CAUCAFall

- 40 of 50 falls (RF, 4 false alarms), 31 of 50 (tuned rule, 7 false alarms), McNemar p = 0.004
- Clip level confusion matrices
- Below the 0.90 goal: 0.917 to 0.694 per window, 0.954 to 0.800 per clip, threshold frozen at 0.43

Figure: `results/figures/caucafall_confusion_clip.png`.
Speaker: Kraiwich. Seconds: 45.

Speaker notes:

Kraiwich Jaiton (45 s). This is the honest result. On CAUCAFall, which the model never saw, Random Forest recall is 0.694 per window and 0.800 per clip. That is below our 0.90 goal. It still found 40 of 50 falls with only 4 false alarms on 50 normal clips. The rule dropped too, to 31 of 50 falls with the tuned thresholds. On the same clips the model was right where the rule was wrong 14 times, the reverse only twice. McNemar gives p 0.004, so the gap is unlikely to be chance. We did not move the threshold after seeing this: that would make the test set training data.

Sources: caucafall_results.csv, caucafall_mcnemar.csv, tuning_before_after.csv (CV clip recall 0.954 counts the 3 windowless URFD falls as missed).

---

## Page 07. System design

- Flow: camera, pose, 2 s buffer, rule and RF side by side, 2 positive checks in a row, banner, beep and log
- Reliability: NO PERSON, RULE FALLBACK, CAMERA OFFLINE with backoff
- Median alert 1.0 s (in sample), 27 to 47 fps over 5 runs, proposed deployment (not built)

Figure: drawn in the deck.
Speaker: Benyapon. Seconds: 25.

Speaker notes:

Benyapon Saisong (25 s). The demo uses the same pose model and feature code as training, runs the rule and the AI side by side, and never pretends to be safe when it cannot see. On 79 fall videos the median alert came 1.0 second after the fall started. These videos were also training data, so that is optimistic. The right box is only a proposal: one box per home, only alerts leave. Switching to the demo now.

Sources: alert_latency_summary.csv (median 1.003 s, 90th percentile 1.781 s), fps.csv (Random Forest 27.3 to 47.2 over 5 runs), config.yaml for the check interval (0.2 s), cooldown (10 s) and backoff settings.

---

## Page 08. Tested on a real fall video

- Screenshot of the demo window at the AI alert on GMDCSA-24 Subject 1 Fall 05 (P(fall) 0.93, threshold 0.43, 46.5 fps)
- `logs/alerts.csv`: AI alert at 2.42 s with beep, rule alert at 2.63 s (2026-11-20 run)
- Four panels: privacy mode, RULE FALLBACK, NO PERSON, lying on a bed calm at P(fall) 0.28
- Training video: shows the product working, not accuracy

Figure: `docs/screenshots/`.
Speaker: Benyapon. Seconds: 10.

Speaker notes:

Benyapon Saisong (10 s). This is our product on one of our test videos, replayed through the demo on our laptop. The AI alerted 2.42 seconds into the video with fall probability 0.93 and beeped, the rule 0.2 seconds later. Below: the same alert in privacy mode, the rule fallback when the model file is missing, NO PERSON, and a person lying on a bed where the AI stays calm. This video is training data, so it shows that the product works, not how accurate it is. Now the live run.

Sources: logs/alerts.csv rows of 2026-11-20 (stream_t 2.424 s, score 0.93, threshold 0.43, and stream_t 2.626 s for the rule). Screenshots of the demo window: Subject 1 Fall 05 at 2.42 s and 2.63 s, Subject 1 ADL 01 at 3.21 s, NO PERSON from a clip with nobody in view. Video frames: GMDCSA-24 dataset.

---

## Page 09. Live demo

- Four steps: sit, fall sideways onto the mat, lie down slowly, step out of view
- Left panel rule, right panel Random Forest, privacy mode

Figure: live webcam, backup video.
Speaker: Benyapon (one other member performs). Seconds: 65.

Speaker notes:

Benyapon Saisong presents (65 s, about 55 s of demo plus 10 s to switch screens). One other member performs on the mat. Command: python -m src.demo --source 0 --privacy. Backup video ready in a second terminal.
1. (10 s) Walk in and sit on a chair. Point at fps and both panels staying low. Say: skeleton only, no image is stored.
2. (20 s) Fall sideways onto the mat, across the camera view, not toward it. Expected: AI banner and beep. Say what the rule panel does, whatever it is. Say why sideways: forward falls were our weakest case on CAUCAFall.
3. (15 s) Get up, then lie down slowly. The AI may alarm: that is the false alarm on lying we found in CV. Say what happens honestly. Do not repeat the scene.
4. (10 s) Step out of view. Point at NO PERSON.
The alert log line and RULE FALLBACK are shown on page 08.
If the webcam fails at any point, switch to the backup video at once and say so.

---

## Page 10. Findings and limits

- Feature importance chart
- Shape change drives the AI, lying down looks alike (14 of 81 normal clips, 7 lying or sleeping)
- 10 missed CAUCAFall falls: 4 forward, 4 to sitting, 2 pose failures
- Recall 0.951 costs 40 of 121 normal clips alarmed, against 24 at 0.903
- Limits strip

Figure: `results/figures/rf_feature_importance.png`.
Speaker: Napat. Seconds: 50.

Speaker notes:

Napat Chang-Ekwong (50 s: findings 30, limits 20). The forest relies most on how fast the body shape changes and the trunk rotates, which is what a fall looks like in 2D. That also explains its errors. In cross validation it alarmed on 14 of 81 normal clips, half of them lying or sleeping: the body ends horizontal, like after a fall. On CAUCAFall the misses are mostly forward falls and falls to sitting. We think viewpoint is the main cause, but we have not proven it. Two misses come from the pose step, not the model: s8 fall forward has no pose, s3 fall forward has no window covering the fall. Lowering the threshold would buy recall with many more false alarms.
Our limits: acted falls by young people, small data, one person in view. Folds are grouped by clip because URFD gives no person IDs, so cross validation is optimistic, and CAUCAFall shows it. Real use needs consent under PDPA, and URFD cannot be used commercially.

Sources: rf_feature_importance.csv, activity_alarm_rates.csv, caucafall_clip_predictions.csv, cv_threshold_tradeoff.csv.

---

## Page 11. Answers to our questions

- RQ1: yes, with CV and unseen test numbers
- RQ2: Random Forest
- Goals checklist: four met, recall 0.90 on unseen data not met
- Not ready to protect a real person yet

Figure: none.
Speaker: Kraiwich. Seconds: 25.

Speaker notes:

Kraiwich Jaiton (25 s). RQ1: yes, the AI beats the rule on recall and F1, in cross validation and on the unseen test, and the clip level gap on CAUCAFall is unlikely to be chance. RQ2: Random Forest is the best of the four. But we did not reach 0.90 recall on unseen data, so it is not ready to protect a real person yet. The price of the AI is more false alarms on lying down.

Sources: tuning_before_after.csv, caucafall_results.csv, caucafall_mcnemar.csv, model_selection.csv, lstm_summary.csv, fps.csv.

---

## Page 12. Who did what

- Role and members table
- Four cards with student ID and name

Figure: table.
Speaker: Benyapon. Seconds: 10.

Speaker notes:

Benyapon Saisong (10 s). Each of us presented the part we built. This table lists who did what. Every number on these slides comes from the results tables in our repository. Next slide: thank you and questions.

---

## Page 13. Thank you. Questions?

- Thank you, questions, project title and members

Figure: disc, floppy and cursor drawing.
Speaker: none, stays on screen during Q&A. Seconds: 0.

Speaker notes:

No speaking time. This slide stays on screen during the 3 minutes of questions. Backup slides follow: cross validation confusion matrices, precision recall curves and the threshold trade off.

---

## Backup 1. Cross validation confusion matrices

- Pooled window counts for the two rules, Naive Bayes, Random Forest and SVM tuned
- Rules miss 86 and 89 fall windows, RF misses 29 with 83 false windows (`cv_pooled_metrics.csv`)

Figure: `results/figures/confusion_matrices_cv.png`.
Speaker: whoever answers. Seconds: 0.

Speaker notes:

Show only if asked. Figure titles show pooled values, for example tuned rule F1 0.785. The slide 5 table shows fold means (0.786). The LSTM is not in this figure because Colab returned only summary scores. Source: confusion_matrices_cv.png, cv_pooled_metrics.csv.

---

## Backup 2. Precision recall curves

- Average precision: RF 0.944, SVM tuned 0.933, Naive Bayes 0.708 (`cv_pooled_metrics.csv`)

Figure: `results/figures/pr_curves_cv.png`.
Speaker: whoever answers. Seconds: 0.

Speaker notes:

Show only if asked. Out of fold window scores pooled across the 5 fold models (training data only). Dots are the per fold thresholds, the star the pooled operating point, the dashed line the 0.90 recall target. The LSTM is not here because Colab returned only summary scores. Source: pr_curves_cv.png, cv_pooled_metrics.csv.

---

## Backup 3. What a lower threshold costs

- Table from `cv_threshold_tradeoff.csv` (below)
- +1 fall for +16 alarmed normal clips at recall 0.95, +2 falls for +46 at 0.98

Figure: table.
Speaker: Kraiwich. Seconds: 0.

Speaker notes:

Show only if asked "why not lower the threshold?". Choosing a threshold on the test set would make it training data, so 0.43 stayed frozen. In training CV, raising Random Forest recall to 0.951 needs threshold 0.257 and raises alarmed normal clips from 24 to 40 of 121. Recall 0.983 raises them to 70. Source: cv_threshold_tradeoff.csv.

| Recall target | Threshold | Window recall | Window false positives | Fall clips found (of 106) | Normal clips alarmed (of 121) |
| --- | --- | --- | --- | --- | --- |
| 0.90 | 0.44 | 0.903 | 59 | 104 | 24 |
| 0.95 | 0.257 | 0.951 | 156 | 105 | 40 |
| 0.98 | 0.117 | 0.983 | 368 | 106 | 70 |

The 0.44 in the first row is one pooled threshold over all out of fold scores. The frozen demo and test threshold is 0.43 (`caucafall_results.csv`). They come from different steps. Do not present them as the same number.

---

## Likely examiner questions

Answers take about 20 s each. The 5 most likely are marked MOST LIKELY. The name in brackets answers.

1. MOST LIKELY. Why did recall drop on CAUCAFall? (Kraiwich)
   New camera, new people and 20 fps. 8 of the 10 missed falls are forward falls (4) or falls to sitting (4) (`caucafall_clip_predictions.csv`). Two misses fail before the model: s8 fall forward has no detected pose and s3 fall forward has no window covering the fall.

2. MOST LIKELY. Why not lower the threshold on CAUCAFall to reach 0.90? (Kraiwich)
   Choosing a threshold on the test set would make it training data, so 0.43 stayed frozen and no lower threshold was tried on CAUCAFall. In training CV, raising RF recall to 0.951 needs threshold 0.257 and raises alarmed normal clips from 24 to 40 of 121. Recall 0.983 raises them to 70 (`cv_threshold_tradeoff.csv`, Backup 3).

3. MOST LIKELY. How fast does it raise an alert? (Benyapon)
   In CV the Random Forest's first positive window ends a median 1.2 s after the annotated fall start, 1.721 s at the 90th percentile, before the demo's 2 in a row check (`cv_detection_delay_summary.csv`). End to end in the demo the median alert comes 1.003 s after the fall starts, 1.781 s at the 90th percentile, with 78 of 79 falls alerted (`alert_latency_summary.csv`). Those GMDCSA-24 fall videos were also in training, so it is optimistic.

4. MOST LIKELY. Why Random Forest and not the LSTM? (Kraiwich)
   RF had the best CV F1, 0.851, among models reaching recall 0.90 (`model_selection.csv`), and the LSTM's F1 was below RF in every fold (`lstm_cv_results.csv`, `cv_results.csv`). The LSTM was run with one seed in Colab and was not tested on CAUCAFall.

5. MOST LIKELY. Why does the AI alarm when someone lies down? (Napat)
   It relies on body shape change and trunk rotation, and lying down ends with the body horizontal like after a fall: 7 of its 14 alarms on 81 normal GMDCSA-24 clips were lying or sleeping (`activity_alarm_rates.csv`). The tuned rule alarmed on 4 of 81 but misses many more falls.

6. Is the McNemar result meaningful with only 100 clips? (Kraiwich)
   It uses only the clips where the two detectors disagree: RF right 14 times, the rule right 2 times, exact p = 0.004 (`caucafall_mcnemar.csv`). It says the difference is unlikely to be chance on this data, not how big it will be elsewhere.

7. Did fine tuning help? (Kraiwich, rule part Napat)
   For the rule, window F1 went from 0.751 to 0.786, and for SVM from 0.808 to 0.823 (`tuning_before_after.csv`). For RF and NB the grid gave identical results, so rf_default was kept.

8. Why group folds by clip and not by person? (Napat)
   URFD gives no person IDs, so clip is the only grouping that works for all training data. One GMDCSA-24 person can then be in train and test, so CV is optimistic, which is why the CAUCAFall test with new people matters.

9. How reliable are your GMDCSA-24 fall times? (Theerawat)
   Theerawat and Napat marked all 79 falls by hand with `tools/annotate_falls.py`, one annotator per clip, with no agreement check. Errors there would shift which windows count as falls, so it is on the limits strip.

10. Is it fast enough? (Benyapon)
    Pose plus RF ran at 27.3 to 47.2 fps over 5 runs on two video files on our laptop against a goal of 15 (`fps.csv`). The 27.3 run had pose alone at 34.2 fps, so the machine was busy. Pose plus rule ran at 45.9 and 48.6, NB at 48.3 and 48.3, SVM at 45.0 and 40.6 fps on the same two videos. Scoring one window takes 17 to 24 ms for RF, about 2 ms for NB and SVM and under 0.5 ms for the rule (`fps.csv`, window_scoring_ms). The webcam fps is shown live on screen and was not saved to a table.

11. What about privacy, and is this deployed? (Benyapon)
    Everything runs on the laptop with no network, and privacy mode shows only the skeleton. Real home use still needs consent under PDPA. The single camera version is built and tested. The multi home design on page 07 is a proposal, not built.

12. Why a threshold for 0.90 recall instead of 0.5? (Kraiwich)
    A missed fall costs far more than a false alarm, so recall is fixed first and the highest threshold that keeps it at 0.90 on training predictions only is taken. For RF that frozen threshold is 0.43 (`caucafall_results.csv`).

## Evaluation form columns against slides

| Column (`docs/rubric-map.md`) | Pages | Gap |
| --- | --- | --- |
| Problem address | 02 | none |
| Realistic / real apply | 07, 08, 09, 10 (limits) | webcam fps not saved to a table |
| Dataset | 03 | none |
| Data preparation | 03 | none |
| System design (flow) | 07 | none |
| Method | 04, 05 | none |
| Experiments: measuring and evaluating | 05, 06 | LSTM fps not measured (Colab only) |
| Finding: assessments | 06, 08, 10, Backups 1 to 3 | no screenshots of CAUCAFall misses, they are described in words |
| Conclusion (interpret) | 06, 10, 11 | none |
| Presentation | 08, 09 (test screenshots, demo and backup video) | rehearse twice |
| Responsibility | 12, every speaker note starts with the speaker's name, all four members speak | none |

Every column has at least one slide.
