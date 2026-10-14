# SignBridge AI — Real-Time Sign Language Recognition & Communication Assistant

> **SignBridge AI** is a professional-grade Deep Learning and Computer Vision application designed to translate American Sign Language (ASL) manual alphabet hand gestures into text and audible speech in real time.

Built with **MediaPipe Tasks Vision**, **PyTorch**, **FastAPI**, **React 18 + Vite**, **Tailwind CSS**, and the browser **Web Speech API**.

---

## Key Features

1. **Real-Time Webcam Hand Landmark Tracking**:
   - Client-side 3D tracking of all 21 hand joints using MediaPipe Tasks Vision.
   - Privacy-first: Raw video frames never leave the user's browser; only normalized 3D joint coordinates are transmitted.
   - Real-time glowing visual skeleton overlay on canvas.

2. **Invariant Landmark Preprocessing**:
   - Automatic translation to wrist origin $(0, 0, 0)$.
   - Palm scale normalization (invariant to distance from camera lens).
   - Dynamic left-hand coordinate mirroring: supports both right-handed and left-handed signers seamlessly with a single model.

3. **Trainable PyTorch Deep Learning Models**:
   - **Static Model**: `HandLandmarkMLP` (Dense layers with BatchNorm1d, Dropout, and AdamW optimization) supporting 26 ASL alphabet classes (`A–Y`, `SPACE`, `DEL`).
   - **Sequential Extension**: `GestureGRU` recurrent neural network for dynamic, motion-based sign gestures.
   - **Confidence & Uncertainty Rejection**: Softmax probability thresholding marks uncertain or ambiguous signs as `UNCERTAIN` instead of outputting false positives.

4. **Temporal Smoothing & Token Stabilization**:
   - Rolling majority voting eliminates single-frame hand flutter and visual noise.
   - Configurable hold-confirmation time prevents flooding duplicate characters into sentences.

5. **Sentence Builder & Voice Synthesis**:
   - Continuous token accumulator with Space, Backspace, and Clear operations.
   - Browser Web Speech API text-to-speech (TTS) playback with pitch, volume, and rate control.
   - One-click copy to clipboard and transcript export.

6. **Interactive Learning & Practice Quiz**:
   - Comprehensive ASL Alphabet Dictionary with anatomical finger placement guides.
   - Gamified Practice Mode: System prompts a random letter, tracks live attempts, validates sign formation, and awards points and streaks.

7. **Complete ML Studio & Analytics**:
   - Built-in webcam data collection interface to record and label new samples.
   - Interactive training dashboard with hyperparameter tuning and live loss/accuracy curves.
   - Independent test-set evaluation report with Confusion Matrix heatmap and per-class precision, recall, and F1-score.
   - Full telemetry dashboard tracking latency, inference volume, and dataset balance.

---

## Technology Stack

- **Frontend**: React 18, Vite, TypeScript, Tailwind CSS, React Router, Lucide Icons, Recharts, MediaPipe Tasks Vision, Web Speech API.
- **Backend**: Python 3.12, FastAPI, Pydantic v2, SQLAlchemy, SQLite local database, Uvicorn.
- **Machine Learning**: PyTorch 2.14+, NumPy, scikit-learn, SciPy.

---

## Project Structure

```
signbridge-ai/
├── frontend/                     # React + Vite + TypeScript + Tailwind CSS
│   ├── src/
│   │   ├── components/           # Navbar, LandmarkCanvas, ConfidenceMeter, TopPredictionsCard, SentenceBuilder
│   │   ├── pages/                # Live Studio, Learning, Quiz, Datasets, Training, Evaluation, Analytics, Settings
│   │   ├── services/             # API client, MediaPipe HandLandmarker wrapper, Speech service
│   │   └── types/                # TypeScript interface definitions
│   └── package.json
├── backend/                      # Python FastAPI server
│   ├── app/
│   │   ├── api/                  # Endpoints: /health, /predict, /datasets, /training, /evaluation, /history
│   │   ├── core/                 # Config, logger, CORS settings
│   │   ├── database/             # SQLAlchemy SQLite models and session
│   │   ├── models/               # PyTorch HandLandmarkMLP & GestureGRU architectures
│   │   ├── schemas/              # Pydantic v2 schemas
│   │   └── services/             # Inference, Dataset, and Training workers
│   └── tests/                    # Automated PyTest test suite
├── ml/                           # ML pipeline scripts and artifacts
│   ├── data/                     # Labeled landmark datasets (JSON)
│   ├── artifacts/                # Model weights (.pth), classes.json, evaluation_report.json
│   └── scripts/                  # Normalization, dataset generator, and training scripts
├── docs/                         # Architecture, model card, API guide, dataset guide
├── start-backend.ps1             # One-click Windows PowerShell starter for backend
├── start-frontend.ps1            # One-click Windows PowerShell starter for frontend
├── run-tests.ps1                 # Runs PyTest automated test suite
├── train-model.ps1               # Runs PyTorch model training
└── README.md
```

---

## Local Setup & Quick Start (Windows)

### Prerequisites
- Windows 10/11
- Python 3.12+ installed
- Node.js v18+ and npm installed

### 1. Start the FastAPI Backend
Open a PowerShell terminal in the project directory:
```powershell
cd backend
.\venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000 --reload
```
Or simply run:
```powershell
.\start-backend.ps1
```
The backend API will start at **http://localhost:8000**.
Interactive API documentation is available at **http://localhost:8000/docs**.

### 2. Start the Frontend
In a second PowerShell terminal:
```powershell
cd frontend
npm run dev
```
Or simply run:
```powershell
.\start-frontend.ps1
```
Open **http://localhost:5173** in your browser.

### 3. Run Automated Tests
```powershell
.\run-tests.ps1
```
All 13 unit and integration tests will execute and report status.

### 4. Retrain the PyTorch Model (Optional)
```powershell
.\train-model.ps1
```

---

## Language Scope & Limitations

> [!IMPORTANT]
> - **Language Scope**: SignBridge AI is strictly calibrated on the **American Sign Language (ASL) manual alphabet** (26 static classes: `A–Y`, `SPACE`, `DEL`).
> - **Continuous Signing vs. Fingerspelling**: Fingerspelling transcribes English orthographic letters. Natural ASL is an independent language with distinct spatial grammar and facial prosody.
> - **Motion Signs**: Traced signs (`J` and `Z`) require temporal trajectory modeling and are handled via the sequential extension.

---

## Verification & Acceptance Checklist

- [x] Backend starts and `/api/health` responds with HTTP 200.
- [x] Model weights (`best_model.pth`) and classes (`classes.json`) loaded.
- [x] PyTorch model trained on real stratified train/val/test splits (80.51% test accuracy).
- [x] Normalized landmark inputs tested for translation, scale, and left-hand mirroring invariance.
- [x] Frontend compiled with zero errors (`npm run build`).
- [x] Real-time video preview with 21 MediaPipe hand joint skeleton overlay.
- [x] Confidence thresholding and UNCERTAIN state rejection.
- [x] Debounced sentence building with space, delete, and copy features.
- [x] Web Speech API text-to-speech audio synthesis.
- [x] Interactive Quiz Mode with scoring and streaks.
- [x] 13 automated tests passing in PyTest.
