# Evaluation form map

Each column of the CSC340 evaluation form, the evidence that answers it and who presents it. The slot is 7 minutes of talk and 3 minutes of Q&A.

| Column | Evidence to show | Presenter |
| --- | --- | --- |
| Problem address | Aged society, older people living alone, late help after an unseen fall, why wearables and simple rules fail, RQ1 and RQ2 on one slide | Demo and presentation |
| Realistic / real apply | Ordinary camera, skeleton only privacy mode, processing on the device, alert within about 2 seconds, the live demo, the proposed deployment slide | Demo and presentation |
| Dataset | Table of the three datasets: clips counted by us, people, camera, fps, role. What the CAUCAFall fall label means | Data |
| Data preparation | Flow: video, pose, 15 Hz resample, normalization, 2 second windows, label rule, folds grouped by clip. Class balance numbers | Data |
| System design (flow) | One block diagram from camera to alert, including smoothing, fallback and logging | Demo and presentation |
| Method | Rule thresholds (published and tuned), four models, feature list, the 90% recall threshold rule | Features and baseline, Modelling |
| Experiments: measuring and evaluating | Results table for five detectors: recall, precision, specificity, F1, accuracy, fps. Cross validation mean and range, CAUCAFall at window and clip level. Before and after tuning table | Modelling |
| Finding: assessments | Confusion matrices, PR curves, alarm rate by activity type, Random Forest feature importance, McNemar result, three failure examples | Modelling, Features and baseline |
| Conclusion (interpret) | Direct answers to RQ1 and RQ2, what the CAUCAFall drop means, limits | Modelling, then all |
| Presentation | Demo in under 90 seconds, clean slides, backup video ready | Demo and presentation |
| Responsibility | Contribution slide listing the files each member owns. Every member presents their own part | All |

## Course score criteria

- Explain where and how AI is applied to the research problem.
- Apply the AI tools correctly and get a better result than the traditional way.
- Interpret the results correctly and fit them to the problem.
- Presentation.

## Requirements from the project brief

- Measurement, evaluation and assessment.
- Comparison with the previous method without AI, and a competition between two or more AI methods.
- Rerun and fine tune the algorithms until the results answer the research question.
