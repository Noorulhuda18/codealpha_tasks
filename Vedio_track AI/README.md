# VisionTrack AI

**See Everything. Track Anything.**

VisionTrack AI is a local, Python-only web application for **object detection and multi-object tracking**. It runs a pretrained Ultralytics YOLO model on a live webcam stream or an uploaded video, follows every object across frames with **ByteTrack** (default) or **BoT-SORT**, and turns the results into live statistics, charts and exportable records — all inside a ChatGPT-inspired dark dashboard built with Streamlit.

No paid API, no cloud service, no JavaScript frontend: the model and the tracker run on your own computer, on CPU if you have no GPU.

## Project objectives

1. Detect multiple objects per frame with a pretrained YOLO model (80 COCO classes).
2. Assign each object a persistent tracking ID that comes from a real tracking algorithm — never random numbers.
3. Support both continuous webcam streaming (WebRTC) and video-file analysis.
4. Compute every statistic from frames that were actually processed.
5. Export results: processed video, detection records (CSV) and an analysis summary.
6. Look and behave like a polished product, not a demo.

## Key features

| Area | What you get |
|---|---|
| **Input** | Real-time webcam through `streamlit-webrtc` (start / stop / pause / reset); video upload (MP4, AVI, MOV) with file information, progress bar, ETA and start / pause / resume / stop controls |
| **Detection** | Pretrained YOLO (YOLO11n by default, lightweight for CPU), adjustable confidence and IoU, per-class filter, image-size and frame-interval controls |
| **Tracking** | ByteTrack or BoT-SORT from Ultralytics, consistent colours per ID, optional movement trails, track memory and matching threshold |
| **Overlays** | Toggle bounding boxes, labels, tracking IDs, confidence scores and trails — live, while processing |
| **Records** | Class, tracking ID, confidence %, box coordinates, frame number, timestamp — kept in Pandas DataFrames |
| **Analytics** | Detections by class, unique tracked objects, most frequent object, distribution / activity / FPS / latency / confidence charts, track lifetimes, tracking history |
| **Export** | Processed H.264 MP4, detections CSV, analysis summary (Markdown / JSON) |
| **History** | Finished sessions are saved under `outputs/sessions/` and can be reopened, analysed, exported or deleted |
| **Robustness** | Missing weights, unsupported or corrupt videos, camera errors and empty frames produce readable messages instead of crashes; capture / writer resources are always released |

## Technologies

- **Interface:** Streamlit, custom CSS, inline Lucide-style SVG icons, Plotly
- **AI:** Ultralytics YOLO (PyTorch), ByteTrack / BoT-SORT
- **Video:** OpenCV, FFmpeg (bundled through `imageio-ffmpeg`) for browser-playable H.264 output
- **Webcam:** `streamlit-webrtc` (WebRTC, `av`)
- **Data:** NumPy, Pandas

## Installation (Windows)

Requirements: **Python 3.10, 3.11 or 3.12** (64-bit) and about 2 GB of free disk space. Python 3.13 is not yet supported by every dependency.

Open **PowerShell** in the folder where you want the project:

```powershell
git clone <your-repository-url> VisionTrack-AI
cd VisionTrack-AI

py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the activation script, run `Set-ExecutionPolicy -Scope Process Bypass` once and activate again.

### Install the dependencies

Installing the small **CPU-only** build of PyTorch first avoids a multi-gigabyte GPU download:

```powershell
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Have an NVIDIA GPU? Skip the first `pip install` line and install the CUDA build of PyTorch from <https://pytorch.org/get-started/locally/>; choose *GPU* in **Settings → Compute device**.

macOS / Linux: the same steps work with `python3 -m venv .venv && source .venv/bin/activate`.

### Model download

Nothing to do manually: the first time you start a video or the webcam, Ultralytics downloads `yolo11n.pt` (about 6 MB) into the `models/` folder. To download ahead of time, or to check your setup:

```powershell
python -m backend.model_loader yolo11n.pt
```

Working offline? Download the `.pt` file from the Ultralytics releases on another computer and copy it into `models/`.

## Run the application

```powershell
streamlit run app.py
```

The app opens at <http://localhost:8501>. Use Chrome or Edge.

## Using the app

### Webcam (Live Detection)

1. Open **Live Detection**.
2. Click **Start camera** in the video box and choose **Allow** when the browser asks for camera access.
3. Boxes, labels and tracking IDs appear on the live image. Adjust confidence, classes and overlays in the *Detection configuration* panel; they apply immediately.
4. **Pause** shows the raw camera image without detection, **Reset** clears the log and restarts IDs from 1, **Save session** stores the session, **Export CSV** downloads the records.
5. Click **Stop camera** to end the stream. A session is saved automatically when a new stream starts.

The webcam only streams while the Live Detection page is open; switching pages stops it (the collected data stays available).

### Video upload (Video Analysis)

1. Open **Video Analysis** and drop an MP4, AVI or MOV file into the uploader. Resolution, frame rate, duration and size are shown.
2. Press **Start**. Frames are processed one at a time in a background thread, so the interface stays responsive. Progress, ETA, tracking IDs and live statistics update as it runs; **Pause**, **Resume** and **Stop** work at any time.
3. When processing finishes, play the processed video and download it, the detections CSV and the analysis summary.
4. Finished sessions are listed under **Detection History**.

### Other pages

- **Overview** – workspace summary, preview of the active session, statistics and charts.
- **Object Tracking** – preview with trails, current IDs, track lifetimes and the tracking history table.
- **Detection History** – filter the current records by class / ID / confidence; reopen, export or delete saved sessions.
- **Analytics** – full statistics for the current or a saved session, with exports.
- **Settings** – model, device, tracker, thresholds, frame interval, frame width, classes, storage and model cache.

For slow computers: keep **YOLO11 Nano**, lower the *Maximum frame width*, choose a smaller *Inference image size* or raise *Run detection every Nth frame*. Skipped frames reuse the latest tracks, so the processed video keeps its full length.

## Screenshots

Screenshots are not committed yet. After your first run, capture each page and save the images in `assets/screenshots/`, then reference them here:

```markdown
![Overview](assets/screenshots/overview.png)
![Live Detection](assets/screenshots/live-detection.png)
![Analytics](assets/screenshots/analytics.png)
```

## Project structure

```
VisionTrack-AI/
├── app.py                  # Streamlit entry point (streamlit run app.py)
├── config.py               # constants, theme, COCO classes, ProcessingSettings
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/config.toml  # dark theme, upload limit
├── assets/
│   ├── logo.png
│   └── style.css           # custom CSS
├── backend/                # computer vision — never imports Streamlit
│   ├── model_loader.py     # weight download, device choice, process-wide model cache
│   ├── detector.py         # YOLO wrapper -> Detections
│   ├── tracker.py          # ByteTrack / BoT-SORT adapter, movement trails
│   ├── video_processor.py  # frame pipeline, background VideoJob, webcam processor
│   ├── analytics.py        # records, statistics, summaries, session storage
│   └── utils.py            # video probing, drawing, H.264 writer, file helpers
├── components/             # Streamlit UI
│   ├── __init__.py         # icons, CSS injection, session state, settings
│   ├── sidebar.py          # ChatGPT-style navigation + status footer
│   ├── dashboard.py        # the seven pages
│   ├── video_panel.py      # upload / job controls / webcam panel
│   ├── metrics.py          # metric cards, chips, live lists
│   └── charts.py           # Plotly figures
├── outputs/                # processed videos and saved sessions (git-ignored)
└── tests/                  # pytest suite (see below)
```

## Tests

```powershell
pytest -q
```

The suite runs the real pipeline (probing, resizing, drawing, record keeping, H.264 export, session saving, pause / resume / stop, resource cleanup) on a synthetic video using a colour-based detector and an IoU tracker as stand-ins, so it needs neither PyTorch nor model weights. Tests that need the real libraries (`ultralytics`, `lap`, `av`, model weights) use `pytest.importorskip` and are **skipped**, not faked, when those are unavailable — with everything installed they check that real ByteTrack / BoT-SORT IDs stay consistent across frames.

Things the automated tests cannot cover and you should check by hand on first run: the Streamlit interface itself, webcam streaming (needs a camera, a browser and WebRTC connectivity), and detection quality on your own footage.

## Troubleshooting

### Webcam

| Symptom | Fix |
|---|---|
| Browser never asks for permission | Open the app at `http://localhost:8501` (secure context). Remote hosts need HTTPS. |
| “Camera access was denied” | Click the padlock in the address bar → *Camera* → *Allow*, then reload. |
| “No camera was found” / black image | Close Zoom, Teams, Windows Camera and other tabs using the camera. Check *Windows Settings → Privacy & security → Camera*. |
| Stuck on connecting | A VPN or firewall may block WebRTC. Disable it or try another network. |
| Very low FPS | Use YOLO11 Nano, a smaller image size, a smaller frame width or a higher frame interval. |
| No frames for several seconds | The status bar shows “No frames”; click **Stop camera**, then **Start camera**. |

### Model loading

| Symptom | Fix |
|---|---|
| “ultralytics package is not installed” | Activate the virtual environment and run `pip install -r requirements.txt`. |
| Download fails | Check the internet connection, or place the `.pt` file in `models/` manually. |
| `yolo26n.pt` fails to load | Needs a recent Ultralytics: `pip install -U ultralytics`, or use YOLO11 / YOLOv8. |
| “GPU inference was requested but no CUDA device” | Set **Settings → Compute device** to *Auto* or *CPU*. |
| Tracking error mentioning `lap` | `pip install lapx`. |

### Video processing

| Symptom | Fix |
|---|---|
| “Unsupported format” | Use MP4, AVI or MOV. Convert other formats with FFmpeg or HandBrake. |
| “could not be opened” / “no readable frames” | The file is corrupt or uses a codec OpenCV cannot decode; re-encode it to H.264 MP4. |
| Processed video downloads but does not preview | No H.264 encoder was found. `pip install imageio-ffmpeg`, then process again. |
| Upload rejected as too large | Raise `maxUploadSize` in `.streamlit/config.toml` (and `MAX_UPLOAD_MB` in `config.py`). |
| Processing is slow | Lower the frame width or raise the frame interval; videos are processed on the CPU unless a GPU is selected. |

## Limitations and future improvements

- Tracking IDs belong to one session; identities are not matched across different videos or cameras.
- BoT-SORT runs without its appearance (Re-ID) model, so it relies on motion and box overlap like ByteTrack.
- Webcam frames are processed in the browser-connected worker; on a slow CPU the stream drops frames rather than building a delay.
- Only the 80 COCO classes of the pretrained models are available; custom-trained weights are not selectable in the UI yet.
- Processed videos have no audio track.
- Ideas: custom model upload, zones and line-crossing counters, RTSP / IP camera input, Re-ID tracking, heat maps, ONNX / OpenVINO acceleration, user-defined alert rules.
