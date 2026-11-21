# Deployment options

The question: for large scale support and zero downtime, do we build something, or do we present a proposed design?

## Decision

Present a proposed design on one slide (Option A). The design we propose is self hosted: one small box per home that runs everything locally. We do not build it before 23 Nov. Option B is a stretch task that may start only after the Sunday gate is met.

Reason: the evaluation form gives marks for dataset, data preparation, method, experiments, findings and conclusion. It has one column, Realistic / real apply, that a deployment slide plus the live demo already covers. A built deployment adds no column and takes time from the ones that score.

## Options

| | A. Proposed design, slide only | B. Self hosted box, built | C. Cloud hosted |
| --- | --- | --- | --- |
| What exists on 24 Nov | Laptop demo with restart loop, rule fallback and logs, plus one design slide | The same demo running unattended on a spare laptop or mini PC as a service | Video or keypoints sent to a server that runs the model |
| Extra work | About 1 hour for the slide | About half a day: headless flag, service setup, a soak test | Several days: server, upload, accounts, security |
| Cost to run | None | None, uses hardware we own | Monthly server cost, grows with every camera |
| Privacy | Strong: nothing leaves the home | Strong: nothing leaves the home | Weak: home video or pose data leaves the home |
| Works without internet | Yes | Yes | No |
| Scales to many homes | By adding one box per home | By adding one box per home | By adding servers, with a shared point of failure |
| Risk to the deadline | None | Medium: competes with experiments and slides | High |
| Verdict | Do this | Stretch only | Rejected |

## What we actually build for reliability

These live inside `src/demo.py` and are part of the graded demo.

- Supervisor loop: the main loop runs inside try and except and restarts with backoff from 1 s to 30 s.
- Camera reconnect: a failed frame read raises, the supervisor reopens the camera.
- Visible states: NO PERSON, CAMERA OFFLINE, RULE FALLBACK. The detector never looks safe when it is blind.
- Graceful degradation: if the model file fails to load, the rule detector takes over.
- Safe model update: save to a temp file, then `os.replace`, so a reload never reads a half written file.
- Logs: `logs/alerts.csv` and `logs/errors.log`.
- Backup: the same app plays a recorded video through `--source`.

## The slide

Title: Proposed deployment, self hosted in the home

- One small box per room runs pose estimation and the classifier on the device. Only an alert event leaves the home, never video.
- Each camera is one independent process, so 1 home or 1,000 homes need no shared server. Scaling means adding boxes.
- Zero downtime for a safety monitor means it never stops silently: automatic restart, camera offline warning, fallback to the rule detector.
- The classifier is cheap. Pose estimation is the load, so the fps we measure on the laptop (`results/tables/fps.csv`) is the cost of one camera. Quote that number here, not one from a paper.
- Not built yet: the box hardware, the caregiver notification, a field test with older adults. Speed on a Raspberry Pi class device is still to be measured.

Say it in one sentence: "We built and tested the single camera version, and this is how the same code would run in many homes."

## Option B, if there is time after Sunday

Scope, nothing more:

1. Add a `--headless` flag to `src/demo.py` that skips the window and keeps the logs.
2. Run it as a service on one spare machine.
3. Leave it running for two hours, unplug the camera once, confirm it recovers and logs the event.
4. Put the uptime and the recovery time on the deployment slide.

Linux service file, saved as `/etc/systemd/system/fall-detection.service`:

```ini
[Unit]
Description=Fall detection
After=network.target

[Service]
WorkingDirectory=/opt/fall-detection
ExecStart=/opt/fall-detection/.venv/bin/python -m src.demo --source 0 --headless
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
```

On Windows, use Task Scheduler with the trigger "At log on" and the same command.

## Questions an examiner may ask

- Why not the cloud? Home video is personal data. Keeping it on the device avoids sending it anywhere, works without internet and costs nothing per month.
- How does it handle many users? Each home is independent. There is no central server in the detection path to overload.
- What happens when it crashes? It restarts itself within seconds and shows a camera offline warning in the meantime.
- Is this deployed? No. The single camera version is built and tested. The multi home design is proposed.
