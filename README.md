# OSCE Automated Assessment Model for Male Foley's Catheterization
**Institution:** D Y Patil Medical College, Department of Surgery & Medical Education Unit  
**Technology Stack:** Pure Python Standard Library + NumPy (100% train-from-scratch, zero third-party ML/DL frameworks)  
**Repository:** [https://github.com/moin08s/foley-osce-assessment](https://github.com/moin08s/foley-osce-assessment)

---

## 🚀 Quick Start: How to Run

### Prerequisites
- **Python 3.10+** (tested on Python 3.12).
- **NumPy**: `pip install numpy`
- **FFmpeg**: Included in project (`.venv/Scripts/ffmpeg.exe`) or install via `winget install Gyan.FFmpeg` / `choco install ffmpeg`.

---

### 1. Run the Web Application Dashboard (Recommended)

The easiest way to use the platform is via the interactive web suite, featuring **Live USB Camera evaluation**, **Interactive 13-step checklist**, **Synchronized step video slices with authentic audio**, and **Teacher reassessment with instant active learning**:

#### Method A: Double-Click Batch File (Windows)
Double-click:
```cmd
START_OSCE_WEB_APP.bat
```
This automatically starts the local edge AI server and opens your browser at **`http://127.0.0.1:8000`**.

#### Method B: Command Line
```powershell
python web_app.py
```
*(Or with explicit Python 3.12: `& "C:\Users\Moin\AppData\Local\Programs\Python\Python312\python.exe" web_app.py`)*

Then open your browser at:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

#### Web Suite Capabilities:
1. **Live Camera / USB Cam Mode:** Select any connected USB webcam, record a candidate in real time with an active timer, and receive an instant automated OSCE evaluation.
2. **Benchmark / File Upload Mode:** Upload student `.mp4` video clips or evaluate the preloaded benchmark video with 1 click.
3. **Step Evidence Video Clips (with Audio):** Each of the 13 steps has an individual carved clip with clean synchronized audio so teachers can inspect student verbal communication and surgical technique.
4. **Procedural Defect Inspection:** Real-time diagnostic flags (e.g., *Missing verbal consent*, *Inadequate antiseptic dwell time*, *Balloon inflated before urine flow*).
5. **Teacher Reassessment & Active Feedback:** Faculty can override step results and add remarks. Clicking **"Save & Fine-Tune"** retrains the model on the fly so it never repeats the mistake!
6. **Download Official Reports:** Download printable text scorecards (`.txt`) and structured clinical JSON reports (`.json`).

---

### 2. Run Assessment from Command Line (`predict.py`)

To evaluate any candidate video clip (`.mp4`), automatically extract time-aligned 16 kHz audio, segment the 13 steps, slice individual step video clips, and print the official D Y Patil scorecard:

```powershell
python predict.py --video "dataset/test/KaiV/session_001/clip_001.mp4" \
                  --student-id "Resident-001" \
                  --session-id "session_001" \
                  --out-report "reports/candidate_scorecard.txt" \
                  --out-json "reports/candidate_report.json"
```

---

### 3. Fine-Tune the Model (Zero-Mistake Mode)

To fine-tune and adapt neural network weights over annotated clinical sessions with $10\times$ data repetition until reaching **100.0% accuracy (0 mistakes)**:

```powershell
python fine_tune.py --manifest foley_clip_manifest.csv --max-epochs 200 --repeat 10 --lr 0.003
```
This updates [`models/foley_osce_mlp.npz`](file:///C:/Medical/models/foley_osce_mlp.npz) directly.

---

### 4. Train from Scratch (`train.py`)

To train a new classifier from scratch with train/val/test splits strictly partitioned by student session (never by frame), calculating per-step Precision, Recall, F1, and the $14 \times 14$ confusion matrix:

```powershell
python train.py --manifest foley_clip_manifest.csv --epochs 35 --model-out models/foley_osce_mlp.npz
```

---

### 5. Run the Automated Test Suite

To verify all system modules (data loader, feature extractor, classifier, sequence checker, video slicer, and grader) using synthetic signals:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 📋 The 13 OSCE Steps

The model continuously analyzes video and audio from a student's examination clip, segmenting and verifying all 13 standard steps of the male urinary catheterization procedure:

| Step ID | Procedural Step Label | Clinical Function & Modality | Critical Sequence Constraints |
|:-------:|:----------------------|:-----------------------------|:------------------------------|
| **1** | `introduce_explain` | Doctor self-introduction & explanation (**Audio Primary**) | Must be performed **first** |
| **2** | `informed_consent` | Verbal consent verification (**Audio Primary**) | Must follow Step 1; must precede Step 4 |
| **3** | `check_instruments` | Inspect sterile tray, catheter pack, syringe | Pre-procedure check |
| **4** | `hand_hygiene_gloves` | Hand sanitization & sterile glove donning | Must precede Step 6 (painting/draping) |
| **5** | `privacy_position_exposure`| Screen curtains drawn, supine position | Patient privacy & dignity |
| **6** | `painting_draping` | Antiseptic swab circular painting & drape | Requires sterile gloves (Step 4) |
| **7** | `lignocaine_jelly` | Meatal instillation of 2% lignocaine jelly | Allow adequate anesthetic dwell time |
| **8** | `open_catheter_insert` | Sterile sheath opened, advance to Y-junction | Advance fully before balloon inflation |
| **9** | `attach_urobag_confirm_urine`| Drainage bag attached; urine flashback | **CRITICAL SAFETY:** Must precede Step 10 |
| **10** | `inflate_balloon` | 10 mL distilled water in retention balloon | **CRITICAL SAFETY:** Only after urine confirmation |
| **11** | `traction_until_resistance` | Gentle backward traction to bladder neck | Only after Step 10 |
| **12** | `reposition_prepuce` | Draw foreskin forward over glans penis | **CRITICAL SAFETY:** Prevents paraphimosis |
| **13** | `tip_dressing` | Sterile meatal dressing & tubing securement| Terminal procedural step (**must be last**) |

---

## 🎓 Grading Scheme

- **Grade `P` (Pass / Advance Beginner):**  
  All 13 procedural steps performed correctly and safely in chronological order without verbal prompting.
- **Grade `P-1` (Pass with 1 Prompt / Beginner):**  
  All 13 steps completed safely, but exactly one step required a faculty verbal prompt or reminder.
- **Grade `R` (Needs Remediation / Novice):**  
  Any omitted step, critical safety violation (e.g., inflating balloon before urine flow, failure to reduce foreskin, aseptic breach), or more than one prompted step.

---

## 📹 Recording Guidelines for Faculty & Staff

To ensure high-quality automated assessment:
1. **Camera Angle:**
   - Mount the camera at the standard side-on oblique angle (45°–60° to the operating couch), clearly capturing the patient pelvic manikin, instrument tray, and student hand movements.
   - Ensure the student does not entirely block the procedural field with their back.
2. **Video Specifications:**
   - Format: `.mp4`, `.avi`, or `.webm`
   - Resolution: 1080p (1920x1080) or 720p (1280x720)
   - Frame rate: 25–60 fps
3. **Audio Specifications:**
   - The student must speak clearly in English, Hindi, or Marathi during Step 1 (Introduction) and Step 2 (Informed Consent).
   - Ensure minimal background clinic chatter to allow accurate short-time energy (STE) voice activity detection.

---

## 🔬 Sample Video Benchmark Validation

The model was validated against the faculty assessment of the benchmark video (`001.MP4`, Resident KaiV, duration 2:38):
- **Faculty Assigned Grade:** `P` (Advance Beginner - Pass)
- **Model Output:** `P` (13/13 steps detected, 0 safety violations, 100% agreement)
- **Per-Step Timeline:**
  - Step 1 (`introduce_explain`): 0.0s – 8.0s (Speech confirmed)
  - Step 2 (`informed_consent`): 6.0s – 16.0s (Speech confirmed)
  - Step 3 (`check_instruments`): 14.0s – 28.0s
  - Step 4 (`hand_hygiene_gloves`): 26.0s – 40.0s
  - Step 5 (`privacy_position_exposure`): 38.0s – 52.0s
  - Step 6 (`painting_draping`): 50.0s – 72.0s
  - Step 7 (`lignocaine_jelly`): 70.0s – 88.0s
  - Step 8 (`open_catheter_insert`): 86.0s – 114.0s
  - Step 9 (`attach_urobag_confirm_urine`): 112.0s – 124.0s
  - Step 10 (`inflate_balloon`): 122.0s – 136.0s
  - Step 11 (`traction_until_resistance`): 134.0s – 142.0s
  - Step 12 (`reposition_prepuce`): 140.0s – 148.0s
  - Step 13 (`tip_dressing`): 146.0s – 158.0s

*Note: Per clinical guidelines, this single sample video is used as an end-to-end format and sanity benchmark, not for claiming statistical generalization.*
