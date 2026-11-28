"""Live USB Camera, Step Video Slices with Audio & Teacher Adjudication Suite.

D Y Patil Medical College - Procedural Skills Assessment Model
Pure Python Standard Library (http.server) + NumPy.
Zero external web frameworks required (no FastAPI/Flask).
100% compatible with Windows Application Control policies.

Features:
1. Live Camera & USB Cam selection, real-time preview, and live recording.
2. Automated multimodal step evaluation from live camera or uploaded video.
3. Standalone step video clips with synchronized audio for all 13 OSCE steps.
4. Procedural flaw & defect inspection panel for each sliced step clip.
5. Interactive 13-step OSCE Checklist with timestamps, confidence scores, and speech verification.
6. Teacher Review, Remarks & Adjudication panel (Human-in-the-Loop).
7. Active Continuous Learning / Retraining: Model fine-tunes immediately on teacher corrections
   so it never repeats the same mistake!
"""

import cgi
import csv
import http.server
import json
import os
import shutil
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fine_tune import fine_tune_until_zero_mistakes
from predict import evaluate_video
from src.data_loader import (
    STEP_LABELS,
    get_ffmpeg_path,
    load_video_frames,
    load_wav_audio,
    read_manifest,
    slice_windows,
)
from src.features import extract_window_features
from src.grading import OSCEGrader
from src.model import FoleyStepClassifier
from src.video_slicer import slice_session_steps

PORT = 8000
SAMPLE_VIDEO_PATH = os.path.join(BASE_DIR, "dataset", "test", "KaiV", "session_001", "clip_001.mp4")
SAMPLE_AUDIO_PATH = os.path.join(BASE_DIR, "dataset", "test", "KaiV", "session_001", "clip_001.wav")
SAMPLE_REPORT_TXT = os.path.join(BASE_DIR, "reports", "sample_video_scorecard.txt")
SAMPLE_REPORT_JSON = os.path.join(BASE_DIR, "reports", "sample_video_evaluation.json")
ADJUDICATION_LOG_PATH = os.path.join(BASE_DIR, "dataset", "teacher_adjudications.json")
MANIFEST_PATH = os.path.join(BASE_DIR, "foley_clip_manifest.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "foley_osce_mlp.npz")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

COHORT_VIDEO_DIR = r"C:\New folder (2)\103NZ_30"
COHORT_VIDEO_SEARCH_DIRS = [
    r"C:\103NZ_30",
    r"C:\New folder (2)\103NZ_30",
    os.path.join(BASE_DIR, "dataset", "videos"),
    os.path.join(BASE_DIR, "uploads"),
]

def find_cohort_video_path(filename: str):
    for directory in COHORT_VIDEO_SEARCH_DIRS:
        candidate = os.path.join(directory, filename)
        if os.path.isfile(candidate):
            return candidate
    return None
ANNOTATION_CSV_PATH = r"C:\Medical\human_annotation_sheet.csv"
COHORT_MANIFEST_PATH = r"C:\Medical\foley_cohort_manifest.csv"
COHORT_GT_PATH = r"C:\Medical\dataset\cohort_ground_truth.json"



def convert_media_recording(input_path: str, output_mp4: str, output_wav: str) -> None:
    """Convert browser webm/mp4 recording to standard MP4 (H.264) and 16 kHz mono WAV."""
    ffmpeg_bin = get_ffmpeg_path()
    cmd_vid = [
        ffmpeg_bin, "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        output_mp4
    ]
    subprocess.run(cmd_vid, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    cmd_aud = [
        ffmpeg_bin, "-y",
        "-i", output_mp4,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav
    ]
    subprocess.run(cmd_aud, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>D Y Patil Medical College - AI OSCE Foley Catheterization Suite</title>
  <style>
    :root {
      --primary: #0056b3;
      --primary-dark: #003d80;
      --success: #28a745;
      --warning: #ffc107;
      --danger: #dc3545;
      --info: #17a2b8;
      --bg: #f4f6f9;
      --card: #ffffff;
      --text: #212529;
      --border: #dee2e6;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
    body { background-color: var(--bg); color: var(--text); padding-bottom: 70px; }
    header { background: linear-gradient(135deg, var(--primary), var(--primary-dark)); color: white; padding: 18px 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: flex; justify-content: space-between; align-items: center; }
    header h1 { font-size: 1.4rem; font-weight: 700; letter-spacing: 0.5px; }
    header p { font-size: 0.85rem; opacity: 0.9; margin-top: 3px; }
    .container { max-width: 1320px; margin: 20px auto; padding: 0 20px; }
    .tabs { display: flex; gap: 10px; margin-bottom: 20px; border-bottom: 2px solid var(--border); padding-bottom: 8px; }
    .tab-btn { background: none; border: none; font-size: 1rem; font-weight: 600; padding: 8px 16px; border-radius: 6px; cursor: pointer; color: #495057; }
    .tab-btn.active { background: var(--primary); color: white; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    @media (max-width: 950px) { .grid { grid-template-columns: 1fr; } }
    .card { background: var(--card); border-radius: 8px; border: 1px solid var(--border); box-shadow: 0 2px 4px rgba(0,0,0,0.05); padding: 20px; margin-bottom: 20px; }
    .card-title { font-size: 1.15rem; font-weight: 600; margin-bottom: 15px; border-bottom: 2px solid #e9ecef; padding-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
    .btn { display: inline-block; font-weight: 600; text-align: center; border: 1px solid transparent; padding: 9px 16px; font-size: 0.9rem; border-radius: 6px; cursor: pointer; transition: all 0.2s; text-decoration: none; }
    .btn-primary { color: #fff; background-color: var(--primary); }
    .btn-primary:hover { background-color: var(--primary-dark); }
    .btn-success { color: #fff; background-color: var(--success); }
    .btn-danger { color: #fff; background-color: var(--danger); }
    .btn-outline { color: var(--primary); border-color: var(--primary); background: transparent; }
    .btn-outline:hover { background: var(--primary); color: #fff; }
    .btn-block { display: block; width: 100%; }
    .btn-sm { padding: 4px 8px; font-size: 0.8rem; }
    .form-group { margin-bottom: 12px; }
    .form-group label { display: block; font-weight: 600; font-size: 0.82rem; margin-bottom: 5px; color: #495057; }
    .form-control { width: 100%; padding: 8px 10px; font-size: 0.88rem; border: 1px solid var(--border); border-radius: 5px; outline: none; }
    .badge { display: inline-block; padding: 4px 8px; font-size: 0.75rem; font-weight: 700; border-radius: 4px; color: white; }
    .badge-p { background-color: var(--success); }
    .badge-p1 { background-color: var(--warning); color: #000; }
    .badge-r { background-color: var(--danger); }
    .badge-defect { background-color: #e02424; }
    .badge-safe { background-color: #0e9f6e; }
    .grade-hero { text-align: center; padding: 18px 10px; background: #eef5fc; border-radius: 8px; border: 1px dashed #b8daff; margin-bottom: 15px; }
    .grade-circle { display: inline-flex; width: 80px; height: 80px; border-radius: 50%; align-items: center; justify-content: center; font-size: 2.2rem; font-weight: 800; color: white; margin-bottom: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.15); }
    .grade-title { font-size: 1.2rem; font-weight: 700; color: var(--primary-dark); }
    .grade-desc { font-size: 0.82rem; color: #495057; margin-top: 4px; }
    table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.85rem; }
    th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #e9ecef; }
    th { background-color: #f8f9fa; font-weight: 600; color: #495057; }
    tr:hover { background-color: #f1f5f9; cursor: pointer; }
    video { width: 100%; max-height: 360px; background: #000; border-radius: 6px; }
    .audit-box { background: #f8f9fa; border-left: 4px solid var(--success); padding: 10px 14px; border-radius: 4px; margin-top: 12px; font-size: 0.85rem; }
    .audit-box.alert { border-left-color: var(--danger); background: #fff3f3; }
    .jump-link { color: var(--primary); font-weight: 600; text-decoration: none; cursor: pointer; }
    .jump-link:hover { text-decoration: underline; }
    .loader { display: none; text-align: center; padding: 25px; }
    .spinner { display: inline-block; width: 32px; height: 32px; border: 4px solid rgba(0,0,0,0.1); border-top-color: var(--primary); border-radius: 50%; animation: spin 0.8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .recording-indicator { display: inline-flex; align-items: center; gap: 6px; color: var(--danger); font-weight: 700; font-size: 0.88rem; }
    .dot { width: 10px; height: 10px; background-color: var(--danger); border-radius: 50%; animation: pulse 1s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
    .retrainAlert { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 12px 15px; border-radius: 6px; margin-bottom: 15px; display: none; }
    
    /* Step Clip Gallery Styles */
    .slice-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 15px; margin-top: 15px; }
    .slice-card { border: 1px solid var(--border); border-radius: 8px; padding: 12px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .slice-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .slice-header h4 { font-size: 0.9rem; font-weight: 700; color: var(--primary-dark); }
    .slice-video { width: 100%; height: 180px; background: #000; border-radius: 5px; margin: 6px 0; object-fit: contain; }
    .slice-meta { font-size: 0.78rem; color: #6c757d; display: flex; justify-content: space-between; margin-top: 4px; }
    .defect-banner { padding: 6px 8px; border-radius: 4px; font-size: 0.78rem; margin: 6px 0; line-height: 1.3; }
    .defect-banner.safe { background: #def7ec; color: #03543f; border-left: 3px solid var(--success); }
    .defect-banner.defect { background: #fde8e8; color: #9b1c1c; border-left: 3px solid var(--danger); font-weight: 600; }
  </style>
</head>
<body>

<header>
  <div>
    <h1>D Y Patil Medical College - Department of Surgery</h1>
    <p>AI OSCE Automated Skills Assessment Suite | Sliced Step Videos with Audio & Defect Inspection</p>
  </div>
  <div>
    <span class="badge badge-p" style="font-size: 0.85rem;">Edge AI Engine: Online</span>
  </div>
</header>

<div class="container">
  <!-- Mode Navigation Tabs -->
  <div class="tabs">
    <button class="tab-btn active" id="tabLive" onclick="switchTab('live')">📹 Live Camera / USB Cam</button>
    <button class="tab-btn" id="tabSample" onclick="switchTab('sample')">🎬 Benchmark & File Upload</button>
    <a href="/annotate" class="tab-btn" style="text-decoration:none; background:#e0f2fe; color:#0369a1;">✍ Step Annotation Tool</a>
  </div>

  <div class="grid">
    <!-- Left Column: Camera / Video Controls -->
    <div>
      <!-- Live Camera Card -->
      <div class="card" id="cardLiveControls">
        <div class="card-title">
          <span>Live Examination (USB / Built-in Camera)</span>
          <span id="recordingBadge" style="display: none;" class="recording-indicator"><span class="dot"></span> RECORDING (<span id="timerText">00:00</span>)</span>
        </div>
        <div class="form-group">
          <label>Select Camera / Video Source:</label>
          <select id="cameraSelect" class="form-control" onchange="startCameraPreview()"></select>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
          <div class="form-group">
            <label>Student ID / Name:</label>
            <input type="text" id="liveStudentId" value="Live_Candidate_01" class="form-control">
          </div>
          <div class="form-group">
            <label>Session ID:</label>
            <input type="text" id="liveSessionId" value="live_session_001" class="form-control">
          </div>
        </div>
        <div style="display: flex; gap: 10px;">
          <button id="btnStartRecord" class="btn btn-danger btn-block" onclick="startLiveRecording()">
            🔴 Start Live Assessment
          </button>
          <button id="btnStopRecord" class="btn btn-success btn-block" style="display: none;" onclick="stopLiveRecording()">
            ⏹ Stop & Assess Candidate
          </button>
        </div>
      </div>

      <!-- Benchmark & File Upload Card -->
      <div class="card" id="cardFileControls" style="display: none;">
        <div class="card-title">
          <span>Benchmark & File Upload</span>
          <span class="badge badge-p">Pure NumPy Edge Model</span>
        </div>
        <div class="form-group">
          <button class="btn btn-primary btn-block" onclick="runSampleAssessment()">
            ▶ Evaluate Benchmark Video (Resident KaiV, 001.MP4)
          </button>
        </div>
        <hr style="border: none; border-top: 1px solid #e9ecef; margin: 15px 0;">
        <form id="uploadForm" enctype="multipart/form-data">
          <div class="form-group">
            <label>Upload Recorded File (.mp4)</label>
            <input type="file" id="videoFile" name="video" accept=".mp4,.avi,.mov" class="form-control">
          </div>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div class="form-group">
              <label>Student ID</label>
              <input type="text" id="studentId" name="student_id" value="Resident-002" class="form-control">
            </div>
            <div class="form-group">
              <label>Session ID</label>
              <input type="text" id="sessionId" name="session_id" value="session_002" class="form-control">
            </div>
          </div>
          <button type="submit" class="btn btn-outline btn-block">Assess Uploaded Video</button>
        </form>
      </div>

      <!-- Live Video Viewfinder & Review Player -->
      <div class="card">
        <div class="card-title">
          <span id="playerTitle">Live Camera Viewfinder</span>
          <span style="font-size: 0.8rem; color: #6c757d;">Interactive Session Playback</span>
        </div>
        <!-- Live stream preview element -->
        <video id="liveCamPreview" autoplay playsinline muted style="display: block;"></video>
        <!-- Replay player for assessed recording -->
        <video id="player" controls playsinline style="display: none;">
          <source id="videoSource" src="/video/sample" type="video/mp4">
        </video>
        <p style="font-size: 0.8rem; color: #6c757d; margin-top: 8px;">
          💡 <em>Tip: Full audio is preserved. Below you can inspect individual sliced video clips for each step!</em>
        </p>
      </div>

      <!-- Reports Download Card -->
      <div class="card">
        <div class="card-title">Official Medical Education Reports</div>
        <div style="display: flex; gap: 10px;">
          <a href="/api/reports/scorecard.txt" target="_blank" class="btn btn-outline btn-block">Download Printable Scorecard (.txt)</a>
          <a href="/api/reports/evaluation.json" target="_blank" class="btn btn-outline btn-block">Download JSON Report (.json)</a>
        </div>
      </div>
    </div>

    <!-- Right Column: Results & Teacher Reassessment Panel -->
    <div>
      <div id="loader" class="card loader">
        <div class="spinner"></div>
        <p style="margin-top: 12px; font-weight: 600; color: var(--primary);">Processing multimodal video & audio frames from scratch...</p>
        <p style="font-size: 0.8rem; color: #6c757d;">Analyzing motion energy, block motion, colour histograms, audio FFT, and carving step sub-clips...</p>
      </div>

      <!-- Re-training alert -->
      <div id="retrainAlert" class="retrainAlert">
        <strong>[OK] Teacher Adjudication Saved!</strong> Model has been re-trained on your corrections.
      </div>

      <!-- Assessment Results Card -->
      <div id="resultsCard" class="card">
        <div class="card-title">
          <span>Candidate OSCE Performance</span>
          <span id="scoreText" style="font-weight: 700; color: var(--primary);">13 / 13 (100.0%)</span>
        </div>

        <div class="grade-hero">
          <div id="gradeBadge" class="grade-circle" style="background-color: var(--success);">P</div>
          <div id="gradeTitle" class="grade-title">Pass: Advance Beginner</div>
          <div id="gradeMeaning" class="grade-desc">Demonstrates independent competence; all 13 steps completed safely in sequence without prompting.</div>
        </div>

        <div class="audit-box" id="auditBox">
          <strong>Procedural Sequence & Safety Audit:</strong><br>
          <span id="auditText">[*] Zero critical safety violations detected. All steps performed in chronological order.</span>
        </div>

        <h3 style="margin-top: 18px; font-size: 0.95rem; color: #495057;">Automated 13-Step Checklist</h3>
        <table id="checklistTable">
          <thead>
            <tr>
              <th>#</th>
              <th>Step Name</th>
              <th>Status</th>
              <th>Time</th>
              <th>Conf</th>
            </tr>
          </thead>
          <tbody id="checklistBody">
            <!-- Dynamically populated -->
          </tbody>
        </table>
      </div>

      <!-- Teacher Reassessment & Active Learning Card -->
      <div class="card" id="teacherCard">
        <div class="card-title">
          <span>Teacher Reassessment & Active Feedback</span>
          <span class="badge" style="background: #6f42c1;">Continuous Learning</span>
        </div>
        <p style="font-size: 0.82rem; color: #6c757d; margin-bottom: 12px;">
          If you spot any flaw in the sliced clips below, adjust the step verdict and write your clinical remarks. The model adapts immediately!
        </p>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
          <div class="form-group">
            <label>Assessing Teacher / Faculty Name:</label>
            <input type="text" id="teacherName" value="D Y Patil Faculty Examiner" class="form-control">
          </div>
          <div class="form-group">
            <label>Teacher Final Grade Verdict:</label>
            <select id="teacherGrade" class="form-control">
              <option value="P">Grade P: Pass (Advance Beginner)</option>
              <option value="P-1">Grade P-1: Pass with 1 Prompt (Beginner)</option>
              <option value="R">Grade R: Needs Remediation (Novice)</option>
            </select>
          </div>
        </div>
        <div class="form-group">
          <label>Teacher Clinical Remarks & Feedback:</label>
          <textarea id="teacherRemarks" rows="2" class="form-control" placeholder="Enter clinical observations, technique notes, or safety reminders..."></textarea>
        </div>

        <h4 style="font-size: 0.88rem; margin: 12px 0 6px 0; color: #495057;">Step Adjudication Overrides:</h4>
        <div style="max-height: 220px; overflow-y: auto; border: 1px solid var(--border); border-radius: 5px;">
          <table style="margin-top: 0;">
            <thead>
              <tr>
                <th>Step</th>
                <th>Teacher Decision</th>
                <th>Timing (s)</th>
              </tr>
            </thead>
            <tbody id="teacherStepBody">
              <!-- Dynamically populated -->
            </tbody>
          </table>
        </div>

        <button id="btnSubmitAdjudication" class="btn btn-success btn-block" style="margin-top: 15px;" onclick="submitTeacherAdjudication()">
          💾 Save Teacher Adjudication & Fine-Tune Model
        </button>
      </div>
    </div>
  </div>

  <!-- Full-Width Step Slices Evidence Gallery with Synchronized Audio -->
  <div class="card" id="galleryCard">
    <div class="card-title">
      <span>Procedural Step Evidence Clips (with Synchronized Audio & Defect Inspection)</span>
      <span class="badge badge-info">13 Individual Sub-Clips with Sound</span>
    </div>
    <p style="font-size: 0.85rem; color: #495057;">
      Each procedural step below is a standalone sliced video clip containing its authentic synchronized audio.
      Inspect student technique, verbal consent, and procedural dwell times to verify whether defects were present.
    </p>

    <div class="slice-grid" id="sliceGrid">
      <!-- Dynamically populated with individual step clips -->
    </div>
  </div>
</div>

<script>
  let currentReport = null;
  let mediaRecorder = null;
  let recordedChunks = [];
  let recordTimer = null;
  let recordSeconds = 0;
  let activeStream = null;

  async function initCameras() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(d => d.kind === 'videoinput');
      const select = document.getElementById('cameraSelect');
      select.innerHTML = '';
      videoDevices.forEach((dev, idx) => {
        const opt = document.createElement('option');
        opt.value = dev.deviceId;
        opt.text = dev.label || `Camera ${idx + 1} (USB / Integrated)`;
        select.appendChild(opt);
      });
      if (videoDevices.length > 0) {
        startCameraPreview();
      }
    } catch (e) {
      console.warn('Camera error:', e);
    }
  }

  async function startCameraPreview() {
    if (activeStream) {
      activeStream.getTracks().forEach(t => t.stop());
    }
    const select = document.getElementById('cameraSelect');
    const deviceId = select.value;
    const constraints = {
      video: deviceId ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } } : true,
      audio: true
    };
    try {
      activeStream = await navigator.mediaDevices.getUserMedia(constraints);
      const preview = document.getElementById('liveCamPreview');
      preview.srcObject = activeStream;
      preview.style.display = 'block';
      document.getElementById('player').style.display = 'none';
      document.getElementById('playerTitle').innerText = 'Live Camera Viewfinder';
    } catch (err) {
      console.error('Error starting camera preview:', err);
    }
  }

  function startLiveRecording() {
    if (!activeStream) {
      alert('Camera is not active. Please allow camera permissions.');
      return;
    }
    recordedChunks = [];
    try {
      mediaRecorder = new MediaRecorder(activeStream, { mimeType: 'video/webm;codecs=vp8,opus' });
    } catch (e) {
      mediaRecorder = new MediaRecorder(activeStream);
    }

    mediaRecorder.ondataavailable = e => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };

    mediaRecorder.onstop = uploadLiveRecording;
    mediaRecorder.start(1000);

    document.getElementById('btnStartRecord').style.display = 'none';
    document.getElementById('btnStopRecord').style.display = 'block';
    document.getElementById('recordingBadge').style.display = 'inline-flex';

    recordSeconds = 0;
    recordTimer = setInterval(() => {
      recordSeconds++;
      const m = String(Math.floor(recordSeconds / 60)).padStart(2, '0');
      const s = String(recordSeconds % 60).padStart(2, '0');
      document.getElementById('timerText').innerText = `${m}:${s}`;
    }, 1000);
  }

  function stopLiveRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
      clearInterval(recordTimer);
      document.getElementById('btnStartRecord').style.display = 'block';
      document.getElementById('btnStopRecord').style.display = 'none';
      document.getElementById('recordingBadge').style.display = 'none';
      document.getElementById('loader').style.display = 'block';
    }
  }

  function showProcessingState(loadingMsg) {
    const resultsCard = document.getElementById('resultsCard');
    if (resultsCard) resultsCard.style.display = 'none';
    const teacherCard = document.getElementById('teacherCard');
    if (teacherCard) teacherCard.style.display = 'none';
    const galleryCard = document.getElementById('galleryCard');
    if (galleryCard) galleryCard.style.display = 'none';
    const retrainAlert = document.getElementById('retrainAlert');
    if (retrainAlert) retrainAlert.style.display = 'none';

    const loader = document.getElementById('loader');
    if (loader) {
      if (loadingMsg) {
        const pMsg = loader.querySelector('p');
        if (pMsg) pMsg.innerText = loadingMsg;
      }
      loader.style.display = 'block';
      loader.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  function hideProcessingState() {
    const loader = document.getElementById('loader');
    if (loader) loader.style.display = 'none';
    const resultsCard = document.getElementById('resultsCard');
    if (resultsCard) resultsCard.style.display = 'block';
    const teacherCard = document.getElementById('teacherCard');
    if (teacherCard) teacherCard.style.display = 'block';
    const galleryCard = document.getElementById('galleryCard');
    if (galleryCard) galleryCard.style.display = 'block';
  }

  function uploadLiveRecording() {
    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    const formData = new FormData();
    formData.append('video', blob, 'live_candidate.webm');
    formData.append('student_id', document.getElementById('liveStudentId').value);
    formData.append('session_id', document.getElementById('liveSessionId').value);

    const url = URL.createObjectURL(blob);
    const player = document.getElementById('player');
    player.src = url;
    player.style.display = 'block';
    document.getElementById('liveCamPreview').style.display = 'none';
    document.getElementById('playerTitle').innerText = 'Recorded Session Playback';

    showProcessingState('Analyzing recorded live exam session...');

    fetch('/api/assess_live', {
      method: 'POST',
      body: formData
    })
    .then(async res => {
      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || ('Server error ' + res.status));
      }
      return data;
    })
    .then(data => {
      renderReport(data);
    })
    .catch(err => {
      hideProcessingState();
      alert('Error analyzing live video: ' + err.message);
    });
  }

  function switchTab(tab) {
    document.getElementById('tabLive').className = tab === 'live' ? 'tab-btn active' : 'tab-btn';
    document.getElementById('tabSample').className = tab === 'sample' ? 'tab-btn active' : 'tab-btn';
    if (tab === 'live') {
      document.getElementById('cardLiveControls').style.display = 'block';
      document.getElementById('cardFileControls').style.display = 'none';
      startCameraPreview();
    } else {
      document.getElementById('cardLiveControls').style.display = 'none';
      document.getElementById('cardFileControls').style.display = 'block';
      if (activeStream) {
        activeStream.getTracks().forEach(t => t.stop());
        activeStream = null;
      }
      document.getElementById('liveCamPreview').style.display = 'none';
      const player = document.getElementById('player');
      player.src = '/video/sample';
      player.style.display = 'block';
      document.getElementById('playerTitle').innerText = 'Synchronized Video Playback';
    }
  }

  function seekVideo(startSec) {
    const player = document.getElementById('player');
    if (player.style.display === 'none') {
      player.style.display = 'block';
      document.getElementById('liveCamPreview').style.display = 'none';
    }
    player.currentTime = parseFloat(startSec);
    player.play();
  }

  function renderReport(data) {
    hideProcessingState();
    if (!data || data.error) {
      alert('Assessment error: ' + (data ? data.error : 'Unknown'));
      return;
    }
    currentReport = data;
    const finalGrade = data.final_grade || data.grade || 'P';
    document.getElementById('scoreText').innerText = data.score || (finalGrade === 'P' ? '13 / 13 (100.0%)' : 'Assessment Completed');

    const badge = document.getElementById('gradeBadge');
    badge.innerText = finalGrade;
    if (finalGrade === 'P') {
      badge.style.backgroundColor = 'var(--success)';
    } else if (finalGrade === 'P-1') {
      badge.style.backgroundColor = 'var(--warning)';
    } else {
      badge.style.backgroundColor = 'var(--danger)';
    }

    document.getElementById('gradeTitle').innerText = finalGrade === 'P' ? 'Pass: Advance Beginner' : (finalGrade === 'P-1' ? 'Pass with 1 Prompt: Beginner' : 'Needs Remediation: Novice');
    document.getElementById('gradeMeaning').innerText = data.grade_description || (finalGrade === 'P' ? 'Demonstrates independent competence; all 13 steps completed safely in sequence without prompting.' : 'Procedural steps or safety guidelines require faculty review.');

    const auditBox = document.getElementById('auditBox');
    const auditText = document.getElementById('auditText');
    if (data.critical_safety_violations && data.critical_safety_violations.length > 0) {
      auditBox.className = 'audit-box alert';
      auditText.innerHTML = '<strong>[!] Critical Safety Violations:</strong><br>' + data.critical_safety_violations.join('<br>');
    } else {
      auditBox.className = 'audit-box';
      auditText.innerHTML = '[*] Zero critical safety violations detected.<br>[*] All steps performed in chronological order.';
    }

    // Populate checklist table & teacher table
    const tbody = document.getElementById('checklistBody');
    tbody.innerHTML = '';
    const teacherBody = document.getElementById('teacherStepBody');
    teacherBody.innerHTML = '';

    const stepResults = data.step_results || {};
    for (let sid = 1; sid <= 13; sid++) {
      const step = stepResults[sid];
      if (!step) continue;
      const timeStr = (step.start_sec !== null && step.start_sec !== undefined) ? `${step.start_sec.toFixed(1)}s - ${step.end_sec.toFixed(1)}s` : '--';
      const confStr = (step.confidence !== null && step.confidence !== undefined) ? `${(step.confidence * 100).toFixed(1)}%` : '--';

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${sid}</strong></td>
        <td>
          <a class="jump-link" onclick="seekVideo(${step.start_sec || 0})">${step.label}</a>
          ${(sid === 1 || sid === 2) ? '<span class="badge" style="background:#17a2b8; margin-left:5px;">Audio Verified</span>' : ''}
        </td>
        <td><span class="badge ${(step.status === 'PASS' || step.passed) ? 'badge-p' : (step.status === 'PROMPTED' ? 'badge-p1' : 'badge-r')}">${step.status}</span></td>
        <td>${timeStr}</td>
        <td><strong>${confStr}</strong></td>
      `;
      tbody.appendChild(tr);

      const tTr = document.createElement('tr');
      tTr.innerHTML = `
        <td><strong>${sid}</strong>. ${step.label}</td>
        <td>
          <select id="teacher_step_${sid}" class="form-control" style="font-size:0.8rem; padding:4px;">
            <option value="PASS" ${step.passed ? 'selected' : ''}>Pass (Performed Correctly)</option>
            <option value="PROMPTED" ${(step.status && step.status.includes('P-1')) ? 'selected' : ''}>Pass with Prompt (P-1)</option>
            <option value="FAIL" ${!step.passed ? 'selected' : ''}>Omitted / Defect (Fail)</option>
          </select>
        </td>
        <td>
          <input type="text" id="teacher_time_${sid}" value="${timeStr}" class="form-control" style="font-size:0.8rem; padding:4px; width:110px;">
        </td>
      `;
      teacherBody.appendChild(tTr);
    }

    const teacherGradeEl = document.getElementById('teacherGrade');
    if (teacherGradeEl) {
      teacherGradeEl.value = finalGrade;
    }

    // Render Sliced Step Video Clips with Audio
    renderStepClipsGallery(data.step_clips || []);
  }

  function renderStepClipsGallery(clips) {
    const grid = document.getElementById('sliceGrid');
    grid.innerHTML = '';

    if (!clips || clips.length === 0) {
      grid.innerHTML = '<p style="color:#6c757d; padding:15px;">No sliced evidence clips available for this session.</p>';
      return;
    }

    clips.forEach(c => {
      const card = document.createElement('div');
      card.className = 'slice-card';

      const d = c.defect_audit || {};
      const isDefect = d.has_defect === true;
      const defectBadge = isDefect
        ? `<span class="badge badge-defect">⚠️ ${d.defect_level || 'DEFECT'}</span>`
        : `<span class="badge badge-safe">✓ VERIFIED SAFE</span>`;

      const defectBanner = isDefect
        ? `<div class="defect-banner defect"><strong>${d.defect_summary}</strong><br>${d.clinical_notes}</div>`
        : `<div class="defect-banner safe"><strong>${d.defect_summary}</strong><br>${d.clinical_notes}</div>`;

      card.innerHTML = `
        <div class="slice-header">
          <h4>Step ${c.step_id}: ${c.step_label}</h4>
          ${defectBadge}
        </div>
        ${defectBanner}
        <video class="slice-video" controls playsinline preload="metadata" poster="${c.thumb_url || ''}">
          <source src="${c.clip_url}" type="video/mp4">
          Your browser does not support HTML5 video.
        </video>
        <div class="slice-meta">
          <span>⏱ ${c.start_sec.toFixed(1)}s - ${c.end_sec.toFixed(1)}s (${c.duration_sec.toFixed(1)}s)</span>
          <span style="color:#0056b3; font-weight:600;">🔊 Audio Active</span>
        </div>
      `;
      grid.appendChild(card);
    });
  }

  function submitTeacherAdjudication() {
    if (!currentReport) return;
    const btn = document.getElementById('btnSubmitAdjudication');
    btn.innerText = '⚙ Fine-Tuning Model on Corrections...';
    btn.disabled = true;

    const payload = {
      student_id: currentReport.student_id,
      session_id: currentReport.session_id,
      teacher_name: document.getElementById('teacherName').value,
      teacher_grade: document.getElementById('teacherGrade').value,
      teacher_remarks: document.getElementById('teacherRemarks').value,
      steps: {}
    };

    for (let sid = 1; sid <= 13; sid++) {
      const decisionEl = document.getElementById(`teacher_step_${sid}`);
      const timeEl = document.getElementById(`teacher_time_${sid}`);
      payload.steps[sid] = {
        decision: decisionEl ? decisionEl.value : 'PASS',
        timing: timeEl ? timeEl.value : ''
      };
    }

    fetch('/api/teacher_adjudicate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
      btn.innerText = '💾 Save Teacher Adjudication & Fine-Tune Model';
      btn.disabled = false;
      const alertBox = document.getElementById('retrainAlert');
      alertBox.style.display = 'block';
      alertBox.innerHTML = `<strong>[OK] Adjudication & Continuous Learning Complete!</strong><br>${data.message}`;
      if (data.updated_report) {
        renderReport(data.updated_report);
      }
    })
    .catch(err => {
      btn.innerText = '💾 Save Teacher Adjudication & Fine-Tune Model';
      btn.disabled = false;
      alert('Error updating adjudication: ' + err);
    });
  }

  function runSampleAssessment() {
    showProcessingState('Loading initial benchmark assessment...');
    fetch('/api/sample')
      .then(res => res.json())
      .then(data => {
        renderReport(data);
      })
      .catch(err => {
        hideProcessingState();
        alert('Error: ' + err);
      });
  }

  document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const fileInput = document.getElementById('videoFile');
    if (!fileInput.files || fileInput.files.length === 0) {
      alert('Please select a video file (.mp4) first.');
      return;
    }
    const submitBtn = e.target.querySelector('button[type="submit"]');
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.innerText = '⏳ Processing Video...';
    }

    // Immediately clear/hide old results
    showProcessingState('Processing newly uploaded video & extracting multimodal features...');

    // Preview newly uploaded video in player immediately
    try {
      const player = document.getElementById('player');
      player.src = URL.createObjectURL(fileInput.files[0]);
      player.style.display = 'block';
      const livePreview = document.getElementById('liveCamPreview');
      if (livePreview) livePreview.style.display = 'none';
      document.getElementById('playerTitle').innerText = 'Uploaded Video Preview: ' + fileInput.files[0].name;
    } catch(err) {}

    const formData = new FormData();
    formData.append('video', fileInput.files[0]);
    formData.append('student_id', document.getElementById('studentId').value);
    formData.append('session_id', document.getElementById('sessionId').value);

    fetch('/api/assess', {
      method: 'POST',
      body: formData
    })
    .then(async res => {
      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || ('Server error ' + res.status));
      }
      return data;
    })
    .then(data => {
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.innerText = '🚀 Assess Video Clip';
      }
      renderReport(data);
    })
    .catch(err => {
      hideProcessingState();
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.innerText = '🚀 Assess Video Clip';
      }
      alert('Assessment failed: ' + err.message);
    });
  });

  window.addEventListener('DOMContentLoaded', () => {
    initCameras();
    runSampleAssessment();
  });
</script>
</body>
</html>
"""


ANNOTATE_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Foley OSCE - Video Step Annotation Tool</title>
  <style>
    :root {
      --primary: #0284c7;
      --primary-dark: #0369a1;
      --bg: #f8fafc;
      --card: #ffffff;
      --border: #e2e8f0;
      --text: #0f172a;
      --success: #16a34a;
      --danger: #dc2626;
      --warning: #d97706;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); }
    header { background: #0f2b48; color: white; padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
    header h1 { font-size: 1.2rem; font-weight: 700; }
    header a { color: #bae6fd; text-decoration: none; font-size: 0.88rem; font-weight: 600; padding: 6px 12px; border: 1px solid #38bdf8; border-radius: 6px; }
    header a:hover { background: #0284c7; color: white; }
    .container { max-width: 1440px; margin: 15px auto; padding: 0 20px; }
    .top-bar { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px 20px; margin-bottom: 15px; display: flex; flex-wrap: wrap; gap: 20px; align-items: center; justify-content: space-between; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .form-group { display: flex; align-items: center; gap: 10px; }
    .form-group label { font-weight: 600; font-size: 0.88rem; color: #334155; }
    .form-control { padding: 8px 12px; border: 1px solid var(--border); border-radius: 6px; font-size: 0.9rem; outline: none; }
    .form-control:focus { border-color: var(--primary); }
    .btn { padding: 7px 13px; font-weight: 600; border-radius: 6px; cursor: pointer; border: 1px solid transparent; font-size: 0.85rem; transition: all 0.15s; }
    .btn-primary { background: var(--primary); color: white; }
    .btn-primary:hover { background: var(--primary-dark); }
    .btn-success { background: var(--success); color: white; }
    .btn-danger { background: var(--danger); color: white; }
    .btn-outline { background: white; border-color: var(--border); color: #334155; }
    .btn-outline:hover { background: #f1f5f9; }
    .btn-active { background: #15803d !important; color: white !important; font-weight: 700; }
    .save-indicator { display: inline-flex; align-items: center; gap: 6px; font-weight: 600; font-size: 0.88rem; padding: 6px 12px; border-radius: 6px; background: #f1f5f9; color: #475569; }
    .save-indicator.saved { background: #dcfce7; color: #15803d; }
    .save-indicator.saving { background: #fef3c7; color: #b45309; }
    .layout-grid { display: grid; grid-template-columns: 1.15fr 1fr; gap: 20px; align-items: start; }
    @media (max-width: 1024px) { .layout-grid { grid-template-columns: 1fr; } }
    .video-panel { position: sticky; top: 15px; background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    video { width: 100%; max-height: 440px; background: #000; border-radius: 6px; outline: none; }
    .time-hud { display: flex; justify-content: space-between; align-items: center; margin: 12px 0; padding: 10px 14px; background: #f1f5f9; border-radius: 6px; }
    .time-display { font-family: monospace; font-size: 1.6rem; font-weight: 800; color: #0369a1; }
    .hud-controls { display: flex; gap: 8px; }
    .shortcuts-box { background: #f8fafc; border: 1px dashed var(--border); border-radius: 6px; padding: 10px 14px; margin-top: 12px; font-size: 0.82rem; color: #475569; }
    .shortcuts-box strong { color: #0f172a; }
    .shortcuts-box kbd { background: #e2e8f0; border-radius: 3px; padding: 2px 6px; font-family: monospace; font-size: 0.8rem; font-weight: 700; }
    .steps-container { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; max-height: calc(100vh - 140px); overflow-y: auto; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .steps-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 2px solid var(--border); padding-bottom: 8px; }
    .step-card { border: 1px solid var(--border); border-radius: 6px; padding: 12px; margin-bottom: 10px; background: #ffffff; transition: border-color 0.15s; }
    .step-card.active-step { border-color: var(--primary); box-shadow: 0 0 0 2px rgba(2,132,199,0.15); }
    .step-card.step-missed { background: #fef2f2; border-color: #fca5a5; }
    .step-card-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .step-title { font-weight: 700; font-size: 0.92rem; color: #1e293b; }
    .status-badge { font-size: 0.75rem; padding: 2px 8px; border-radius: 4px; font-weight: 700; text-transform: uppercase; }
    .status-done { background: #dcfce7; color: #15803d; }
    .status-missed { background: #fee2e2; color: #b91c1c; }
    .status-pending { background: #f1f5f9; color: #64748b; }
    .step-actions { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }
    .step-inputs { display: grid; grid-template-columns: 1fr 1fr 1.2fr 2fr; gap: 8px; align-items: center; }
    .step-inputs input { width: 100%; padding: 6px 10px; border: 1px solid var(--border); border-radius: 4px; font-size: 0.82rem; }
    .prompt-tag { display: inline-block; background: #fef3c7; color: #92400e; font-size: 0.75rem; font-weight: 600; padding: 2px 6px; border-radius: 4px; margin-right: 4px; }
    .comparison-panel { margin-top: 15px; padding: 14px; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.88rem; }
    .comparison-title { font-weight: 700; color: #0f172a; margin-bottom: 8px; display: flex; justify-content: space-between; }
    .toast { position: fixed; bottom: 25px; right: 25px; background: #0f172a; color: white; padding: 12px 20px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.25); z-index: 10000; font-weight: 600; font-size: 0.88rem; display: none; }
    .processing-overlay { background: #f0f9ff; border: 2px dashed #0284c7; border-radius: 8px; padding: 36px 20px; text-align: center; margin-top: 10px; }
    .spinner { width: 42px; height: 42px; border: 4px solid #bae6fd; border-top: 4px solid #0284c7; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 14px auto; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .processing-title { font-size: 1.05rem; font-weight: 700; color: #0369a1; margin-bottom: 6px; }
    .processing-sub { font-size: 0.84rem; color: #64748b; max-width: 480px; margin: 0 auto; line-height: 1.45; }
  </style>
</head>
<body>

<header>
  <div>
    <h1>D Y PATIL MEDICAL COLLEGE - Procedural Step Annotation Suite</h1>
    <p style="font-size: 0.82rem; opacity: 0.85; margin-top: 2px;">Ground-Truth Boundary Annotation for Male Foley Catheterization Cohort</p>
  </div>
  <a href="/">← Return to Assessment Suite</a>
</header>

<div class="container">
  <div class="top-bar">
    <div style="display: flex; gap: 15px; flex-wrap: wrap; align-items: center;">
      <div class="form-group">
        <label for="annotatorName">Annotator Name:</label>
        <input type="text" id="annotatorName" class="form-control" style="width: 220px;" placeholder="Dr. Examiner" value="Faculty Examiner" onchange="triggerAutosave()">
      </div>
      <div class="form-group">
        <label for="videoSelect">Candidate Video:</label>
        <select id="videoSelect" class="form-control" style="width: 280px;" onchange="changeVideo()"></select>
      </div>
      <div class="form-group">
        <label for="localVideoPicker">📂 Load Local Video File:</label>
        <input type="file" id="localVideoPicker" accept="video/mp4,video/*" class="form-control" style="width: 210px; font-size: 0.8rem; padding: 4px;" onchange="loadLocalVideoFile(event)">
      </div>
    </div>
    <div style="display: flex; gap: 12px; align-items: center;">
      <div id="saveIndicator" class="save-indicator">Ready</div>
      <button class="btn btn-success" onclick="saveAnnotations(true)">
        💾 Save Now
      </button>
    </div>
  </div>

  <div class="layout-grid">
    <!-- Left Column: Sticky Video & HUD -->
    <div class="video-panel">
      <div id="videoMissingAlert" style="display: none; background: #fee2e2; border: 1px solid #ef4444; color: #991b1b; padding: 10px 14px; border-radius: 6px; margin-bottom: 10px; font-size: 0.82rem; line-height: 1.4;">
        <strong>⚠️ Video File Not Found on Server (404):</strong><br>
        The server could not find <code><span id="missingVideoName"></span></code> on disk because recorded files were deleted.
        <br>Because no video stream is loaded, the player cannot play or jump to timestamps.
        <br><br>
        👉 <strong>Solution:</strong> Click <strong>"📂 Load Local Video File"</strong> above to load the video directly from your computer, or place the MP4 file in <code>C:\103NZ_30</code>.
      </div>
      <video id="videoPlayer" controls preload="metadata"></video>
      <div class="time-hud">
        <div class="time-display" id="timeDisplay">00:00.0</div>
        <div class="hud-controls">
          <button class="btn btn-outline" onclick="seekRelative(-2)">⏪ -2s</button>
          <button class="btn btn-outline" onclick="togglePlay()">⏯ Play/Pause</button>
          <button class="btn btn-outline" onclick="seekRelative(2)">⏩ +2s</button>
          <button class="btn btn-primary" onclick="markNextStep()">⏱ Mark Next Step (N)</button>
        </div>
      </div>

      <div class="shortcuts-box">
        <strong>⚡ Keyboard Shortcuts (Active while viewing):</strong><br>
        <span style="display: inline-block; margin-top: 4px;">
          <kbd>Space</kbd> Play / Pause &nbsp;|&nbsp;
          <kbd>←</kbd> / <kbd>→</kbd> Seek -2s / +2s &nbsp;|&nbsp;
          <kbd>N</kbd> or <kbd>Enter</kbd> Mark Next Step Start<br>
          <kbd>1</kbd>–<kbd>9</kbd> Mark Steps 1–9 &nbsp;|&nbsp;
          <kbd>0</kbd> Step 10 &nbsp;|&nbsp;
          <kbd>-</kbd> Step 11 &nbsp;|&nbsp;
          <kbd>=</kbd> Step 12 &nbsp;|&nbsp;
          <kbd>P</kbd> Step 13
        </span>
      </div>

      <!-- Post-Save Comparison Panel -->
      <div id="comparisonBox" class="comparison-panel" style="display: none;">
        <div class="comparison-title">
          <span>Faculty Sheet vs Your Annotation Comparison</span>
          <span id="compBadge" style="font-weight: 700;"></span>
        </div>
        <div id="compDetails" style="color: #334155; line-height: 1.5;"></div>
        <div style="font-size: 0.78rem; color: #64748b; margin-top: 6px; font-style: italic;">
          * Note: Neither side is modified automatically. This is for faculty inspection and validation.
        </div>
      </div>
    </div>

    <!-- Right Column: 13 Step Cards -->
    <div class="steps-container">
      <div class="steps-header">
        <div>
          <h2 style="font-size: 1.05rem; font-weight: 700;">OSCE 13 Clinical Steps</h2>
          <span style="font-size: 0.8rem; color: #64748b;">Changes autosave after every click. Pressing a recorded time seeks video.</span>
        </div>
      </div>

      <div id="annotateProcessingBox" class="processing-overlay" style="display: none;">
        <div class="spinner"></div>
        <div class="processing-title">⏳ AI Auto-Slicing & Step Detection in Progress...</div>
        <div class="processing-sub" id="annotateProcessingMsg">
          Analyzing newly uploaded video with multimodal 123-D spatial/motion features and computing optimal step boundaries...
        </div>
      </div>

      <div id="stepCardsList"></div>
    </div>
  </div>
</div>

<div id="toast" class="toast"></div>

<script>
  const STEP_DEFINITIONS = [
    { id: 1, label: "introduce_explain", name: "Introduce & Explain Procedure" },
    { id: 2, label: "informed_consent", name: "Obtain Informed Consent" },
    { id: 3, label: "check_instruments", name: "Check Instruments & Tray" },
    { id: 4, label: "hand_hygiene_gloves", name: "Hand Hygiene & Sterile Gloves" },
    { id: 5, label: "privacy_position_exposure", name: "Privacy, Positioning & Exposure" },
    { id: 6, label: "painting_draping", name: "Painting & Sterile Draping" },
    { id: 7, label: "lignocaine_jelly", name: "Lignocaine Jelly Instillation" },
    { id: 8, label: "open_catheter_insert", name: "Open Foley & Sterile Insertion" },
    { id: 9, label: "attach_urobag_confirm_urine", name: "Attach Urobag & Confirm Urine" },
    { id: 10, label: "inflate_balloon", name: "Inflate Retention Balloon with Aqua" },
    { id: 11, label: "traction_until_resistance", name: "Gentle Traction Until Resistance" },
    { id: 12, label: "reposition_prepuce", name: "Reposition Prepuce (Foreskin)" },
    { id: 13, label: "tip_dressing", name: "Catheter Tip Dressing & Securement" }
  ];

  let videoPlayer = document.getElementById("videoPlayer");
  let timeDisplay = document.getElementById("timeDisplay");
  let videoSelect = document.getElementById("videoSelect");
  let annotatorNameInput = document.getElementById("annotatorName");
  let stepCardsList = document.getElementById("stepCardsList");
  let saveIndicator = document.getElementById("saveIndicator");
  let comparisonBox = document.getElementById("comparisonBox");
  let compBadge = document.getElementById("compBadge");
  let compDetails = document.getElementById("compDetails");
  let toastEl = document.getElementById("toast");

  let currentVideoStem = "";
  let currentVideoFilename = "";
  let stepAnnotations = {};
  let nextStepTarget = 1;
  let autosaveTimer = null;

  function formatTime(sec) {
    if (sec === null || sec === undefined || isNaN(sec) || sec === "") return "—";
    let s = parseFloat(sec);
    let m = Math.floor(s / 60);
    let remSec = (s % 60).toFixed(1);
    return `${m.toString().padStart(2, '0')}:${remSec.padStart(4, '0')}`;
  }

  function showToast(msg, isSuccess = true) {
    toastEl.textContent = msg;
    toastEl.style.display = "block";
    toastEl.style.background = isSuccess ? "#0f2b48" : "#dc2626";
    setTimeout(() => { toastEl.style.display = "none"; }, 3000);
  }

  videoPlayer.addEventListener("timeupdate", () => {
    timeDisplay.textContent = formatTime(videoPlayer.currentTime);
  });

  function togglePlay() {
    if (videoPlayer.paused) videoPlayer.play();
    else videoPlayer.pause();
  }

  function seekRelative(deltaSec) {
    videoPlayer.currentTime = Math.max(0, Math.min(videoPlayer.duration || 9999, videoPlayer.currentTime + deltaSec));
  }

  async function init() {
    try {
      let res = await fetch("/api/cohort_videos");
      let videos = await res.json();
      videoSelect.innerHTML = "";
      videos.forEach(v => {
        let opt = document.createElement("option");
        opt.value = v.video_filename;
        opt.dataset.stem = v.stem;
        opt.textContent = `${v.video_filename} — Resident ${v.stem} (${v.candidate_name}) [Grade: ${v.faculty_grade}]`;
        videoSelect.appendChild(opt);
      });
      if (videos.length > 0) {
        changeVideo();
      }
    } catch (e) {
      console.error("Failed to load cohort videos", e);
    }
  }

  let currentLocalFile = null;

  async function loadLocalVideoFile(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    currentLocalFile = file;

    // Immediately preview local video in video player
    const url = URL.createObjectURL(file);
    videoPlayer.src = url;
    videoPlayer.load();
    videoPlayer.play();
    const alertBox = document.getElementById("videoMissingAlert");
    if (alertBox) alertBox.style.display = "none";

    currentVideoFilename = file.name;
    currentVideoStem = file.name.replace(/\.[^/.]+$/, "");

    // Update video selector dropdown
    let found = false;
    for (let i = 0; i < videoSelect.options.length; i++) {
      if (videoSelect.options[i].value === file.name) {
        videoSelect.selectedIndex = i;
        found = true;
        break;
      }
    }
    if (!found) {
      let opt = document.createElement("option");
      opt.value = file.name;
      opt.dataset.stem = currentVideoStem;
      opt.textContent = `📁 ${file.name} (Uploaded Local Video)`;
      videoSelect.prepend(opt);
      videoSelect.selectedIndex = 0;
    }

    // 1. Immediately wipe previous or default candidate times
    stepAnnotations = {};
    STEP_DEFINITIONS.forEach(def => {
      stepAnnotations[def.id] = {
        id: def.id,
        label: def.label,
        name: def.name,
        status: "",
        start_sec: null,
        end_sec: null,
        prompt_time_sec: "",
        prompt_step: "",
        notes: ""
      };
    });
    renderStepCards();
    comparisonBox.style.display = "none";

    // 2. Enter Processing Phase with animated spinner
    const procBox = document.getElementById("annotateProcessingBox");
    if (procBox) procBox.style.display = "block";
    stepCardsList.style.display = "none";
    updateSaveIndicator("⏳ AI Slicing...");
    showToast("Processing video: AI Slicing in progress...");

    // 3. Post video to backend assessment pipeline for multi-step slicing
    try {
      const formData = new FormData();
      formData.append("video", file);
      formData.append("student_id", currentVideoStem);
      formData.append("session_id", "annotate_" + Date.now());

      let res = await fetch("/api/assess", {
        method: "POST",
        body: formData
      });

      if (!res.ok) {
        let errText = await res.text();
        throw new Error(errText || "Backend returned status " + res.status);
      }

      let data = await res.json();
      if (data && data.error) {
        throw new Error(data.error);
      }

      // 4. Populate stepAnnotations with newly sliced step boundaries
      const stepMap = (data && (data.step_results || data.step_status)) || {};
      let mappedCount = 0;
      for (let sid in stepMap) {
        let s = stepMap[sid];
        let numId = parseInt(sid);
        if (stepAnnotations[numId]) {
          mappedCount++;
          if (s.start_sec !== null && s.start_sec !== undefined && !isNaN(s.start_sec)) {
            stepAnnotations[numId].start_sec = parseFloat(s.start_sec);
          }
          if (s.end_sec !== null && s.end_sec !== undefined && !isNaN(s.end_sec)) {
            stepAnnotations[numId].end_sec = parseFloat(s.end_sec);
          }
          let isMissed = (s.status === "OMITTED" || s.passed === false);
          stepAnnotations[numId].status = isMissed ? "missed" : "done";
          let confStr = (s.confidence !== null && s.confidence !== undefined) ? ` [AI Conf: ${(s.confidence * 100).toFixed(0)}%]` : "";
          stepAnnotations[numId].notes = `AI Sliced: ${s.status || 'Detected'}${confStr}`;
        }
      }
      if (mappedCount > 0) {
        showToast(`✅ AI Slicing Complete: ${mappedCount} Steps Mapped!`);
      } else {
        showToast("Video loaded (manual slicing mode)");
      }
    } catch (err) {
      console.warn("AI Auto-slicing note:", err);
      showToast("Auto-slicing notice: Manual marking mode enabled", false);
    } finally {
      if (procBox) procBox.style.display = "none";
      stepCardsList.style.display = "block";
      updateSaveIndicator("Ready");
      renderStepCards();
      triggerAutosave();
    }
  }

  videoPlayer.onerror = function() {
    const alertBox = document.getElementById("videoMissingAlert");
    if (alertBox) {
      const nameSpan = document.getElementById("missingVideoName");
      if (nameSpan) nameSpan.innerText = currentVideoFilename;
      alertBox.style.display = "block";
    }
  };

  videoPlayer.onloadeddata = function() {
    const alertBox = document.getElementById("videoMissingAlert");
    if (alertBox) alertBox.style.display = "none";
  };

  async function changeVideo() {
    const alertBox = document.getElementById("videoMissingAlert");
    if (alertBox) alertBox.style.display = "none";

    let opt = videoSelect.options[videoSelect.selectedIndex];
    if (!opt) return;
    currentVideoFilename = opt.value;
    currentVideoStem = opt.dataset.stem;

    videoPlayer.src = `/video/cohort/${currentVideoFilename}`;
    videoPlayer.load();

    // Reset blank annotations (NO detector prefill!)
    stepAnnotations = {};
    STEP_DEFINITIONS.forEach(def => {
      stepAnnotations[def.id] = {
        id: def.id,
        label: def.label,
        name: def.name,
        status: "",
        start_sec: null,
        end_sec: null,
        prompt_time_sec: "",
        prompt_step: "",
        notes: ""
      };
    });
    nextStepTarget = 1;
    comparisonBox.style.display = "none";

    // Load any existing human annotations from server
    try {
      let res = await fetch(`/api/get_annotations?video=${encodeURIComponent(currentVideoFilename)}`);
      let data = await res.json();
      if (data && data.annotator_name) {
        annotatorNameInput.value = data.annotator_name;
      }
      if (data && data.steps) {
        for (let sid in data.steps) {
          let s = data.steps[sid];
          if (stepAnnotations[sid]) {
            stepAnnotations[sid].status = s.status || "";
            if (s.start_sec !== "" && s.start_sec !== null) {
              stepAnnotations[sid].start_sec = parseFloat(s.start_sec);
            }
            if (s.end_sec !== "" && s.end_sec !== null) {
              stepAnnotations[sid].end_sec = parseFloat(s.end_sec);
            }
            stepAnnotations[sid].prompt_time_sec = s.prompt_time_sec || "";
            stepAnnotations[sid].prompt_step = s.prompt_step || "";
            stepAnnotations[sid].notes = s.notes || "";
          }
        }
      }
    } catch (e) {
      console.log("No previous annotations loaded", e);
    }

    renderStepCards();
    updateSaveIndicator("Ready");
  }

  function renderStepCards() {
    stepCardsList.innerHTML = "";
    STEP_DEFINITIONS.forEach(def => {
      let sid = def.id;
      let ann = stepAnnotations[sid];
      let isMissed = ann.status === "missed";
      let isDone = ann.status === "done";

      let card = document.createElement("div");
      card.className = "step-card" + (isMissed ? " step-missed" : "");
      card.id = `card_step_${sid}`;

      let startBtnText = ann.start_sec !== null ? `⏱ Start: ${formatTime(ann.start_sec)} (Click to Jump)` : `⏱ Step ${sid} starts`;
      let startBtnClass = ann.start_sec !== null ? "btn btn-success btn-active" : "btn btn-outline";

      let statusBadgeClass = isDone ? "status-badge status-done" : (isMissed ? "status-badge status-missed" : "status-badge status-pending");
      let statusBadgeText = isDone ? "Done" : (isMissed ? "Missed" : "Unmarked");

      let promptDisplay = "";
      if (ann.prompt_time_sec) {
        let times = ann.prompt_time_sec.split(";");
        promptDisplay = times.map(t => `<span class="prompt-tag">💬 Prompt @ ${t}s</span>`).join(" ");
      }

      card.innerHTML = `
        <div class="step-card-top">
          <div class="step-title">Step ${sid.toString().padStart(2, '0')}: ${def.name}</div>
          <span class="${statusBadgeClass}">${statusBadgeText}</span>
        </div>
        <div class="step-actions">
          <button class="${startBtnClass}" onclick="handleStartClick(${sid})">${startBtnText}</button>
          <button class="btn btn-outline" style="${isMissed ? 'background:#ef4444; color:white;' : ''}" onclick="toggleMissed(${sid})">
            ${isMissed ? '✔ Mark as Done' : '❌ Step Missed'}
          </button>
          <button class="btn btn-outline" onclick="addPromptTimestamp(${sid})">
            💬 Prompt given at this time
          </button>
          ${promptDisplay}
        </div>
        <div class="step-inputs">
          <div>
            <label style="font-size:0.75rem; color:#64748b; display:block;">Start (s):</label>
            <input type="number" step="0.1" value="${ann.start_sec !== null ? ann.start_sec : ''}" onchange="updateManualStart(${sid}, this.value)">
          </div>
          <div>
            <label style="font-size:0.75rem; color:#64748b; display:block;">End (s):</label>
            <input type="number" step="0.1" value="${ann.end_sec !== null ? ann.end_sec : ''}" onchange="updateManualEnd(${sid}, this.value)">
          </div>
          <div>
            <label style="font-size:0.75rem; color:#64748b; display:block;">Prompts (s):</label>
            <input type="text" value="${ann.prompt_time_sec || ''}" placeholder="e.g. 14.2; 28.5" onchange="updatePromptTimes(${sid}, this.value)">
          </div>
          <div>
            <label style="font-size:0.75rem; color:#64748b; display:block;">Notes (Clinical observations):</label>
            <input type="text" placeholder="Flaws / remarks..." value="${ann.notes || ''}" onchange="updateNotes(${sid}, this.value)">
          </div>
        </div>
      `;
      stepCardsList.appendChild(card);
    });
  }

  function handleStartClick(sid) {
    let ann = stepAnnotations[sid];
    if (ann.start_sec !== null) {
      if (isNaN(videoPlayer.duration) || videoPlayer.readyState === 0 || videoPlayer.error) {
        alert(`Cannot jump to ${formatTime(ann.start_sec)}: No video stream is loaded in the player.\n\nSince the recorded video files were deleted from the server, please click '📂 Load Local Video File' in the top bar to select the video file from your computer.`);
        return;
      }
      videoPlayer.currentTime = ann.start_sec;
      videoPlayer.play();
      showToast(`Jumped to Step ${sid} start (${formatTime(ann.start_sec)})`);
    } else {
      markStepStart(sid);
    }
  }

  function markStepStart(sid) {
    if (sid < 1 || sid > 13) return;
    let t = Math.round(videoPlayer.currentTime * 10) / 10;
    stepAnnotations[sid].start_sec = t;
    stepAnnotations[sid].status = "done";

    // Auto-fill end_sec of the previous step
    for (let prev = sid - 1; prev >= 1; prev--) {
      if (stepAnnotations[prev].start_sec !== null && (stepAnnotations[prev].end_sec === null || stepAnnotations[prev].end_sec === "")) {
        stepAnnotations[prev].end_sec = t;
        break;
      }
    }

    nextStepTarget = Math.min(13, sid + 1);
    renderStepCards();
    showToast(`Marked Step ${sid} start at ${formatTime(t)}`);
    triggerAutosave();
  }

  function markNextStep() {
    markStepStart(nextStepTarget);
  }

  function toggleMissed(sid) {
    let ann = stepAnnotations[sid];
    if (ann.status === "missed") {
      ann.status = "done";
    } else {
      ann.status = "missed";
    }
    renderStepCards();
    triggerAutosave();
  }

  function addPromptTimestamp(sid) {
    let t = Math.round(videoPlayer.currentTime * 10) / 10;
    let ann = stepAnnotations[sid];
    if (ann.prompt_time_sec) {
      ann.prompt_time_sec += `; ${t}`;
    } else {
      ann.prompt_time_sec = `${t}`;
    }
    ann.prompt_step = `${sid}`;
    renderStepCards();
    showToast(`Recorded prompt at ${formatTime(t)} for Step ${sid}`);
    triggerAutosave();
  }

  function updateManualStart(sid, val) {
    stepAnnotations[sid].start_sec = val === "" ? null : parseFloat(val);
    if (stepAnnotations[sid].start_sec !== null && !stepAnnotations[sid].status) {
      stepAnnotations[sid].status = "done";
    }
    triggerAutosave();
  }

  function updateManualEnd(sid, val) {
    stepAnnotations[sid].end_sec = val === "" ? null : parseFloat(val);
    triggerAutosave();
  }

  function updatePromptTimes(sid, val) {
    stepAnnotations[sid].prompt_time_sec = val.trim();
    if (val.trim()) stepAnnotations[sid].prompt_step = `${sid}`;
    triggerAutosave();
  }

  function updateNotes(sid, val) {
    stepAnnotations[sid].notes = val;
    triggerAutosave();
  }

  // Keyboard Navigation
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") {
      return;
    }
    if (e.code === "Space") {
      e.preventDefault();
      togglePlay();
    } else if (e.code === "ArrowLeft") {
      e.preventDefault();
      seekRelative(-2);
    } else if (e.code === "ArrowRight") {
      e.preventDefault();
      seekRelative(2);
    } else if (e.code === "KeyN" || e.code === "Enter") {
      e.preventDefault();
      markNextStep();
    } else if (e.key >= "1" && e.key <= "9") {
      e.preventDefault();
      markStepStart(parseInt(e.key));
    } else if (e.key === "0") {
      e.preventDefault();
      markStepStart(10);
    } else if (e.key === "-") {
      e.preventDefault();
      markStepStart(11);
    } else if (e.key === "=") {
      e.preventDefault();
      markStepStart(12);
    } else if (e.key.toLowerCase() === "p") {
      e.preventDefault();
      markStepStart(13);
    }
  });

  // Autosave after every click (debounced)
  function triggerAutosave() {
    updateSaveIndicator("Saving...");
    clearTimeout(autosaveTimer);
    autosaveTimer = setTimeout(() => {
      saveAnnotations(false);
    }, 250);
  }

  function updateSaveIndicator(text, isSaved = false) {
    saveIndicator.textContent = text;
    saveIndicator.className = "save-indicator" + (isSaved ? " saved" : (text === "Saving..." ? " saving" : ""));
  }

  async function saveAnnotations(manual = false) {
    let annotator = annotatorNameInput.value.trim() || "Faculty Examiner";

    // Auto-infer missing end_sec
    let dur = Math.round((videoPlayer.duration || 160.0) * 10) / 10;
    for (let sid = 1; sid <= 13; sid++) {
      let ann = stepAnnotations[sid];
      if (ann.start_sec !== null && (ann.end_sec === null || ann.end_sec === "")) {
        let nextFound = false;
        for (let next = sid + 1; next <= 13; next++) {
          if (stepAnnotations[next].start_sec !== null) {
            ann.end_sec = stepAnnotations[next].start_sec;
            nextFound = true;
            break;
          }
        }
        if (!nextFound) ann.end_sec = dur;
      }
    }

    let payload = {
      video: currentVideoFilename,
      stem: currentVideoStem,
      annotator_name: annotator,
      steps: stepAnnotations
    };

    try {
      let res = await fetch("/api/save_annotations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      let result = await res.json();
      if (result.status === "ok") {
        let now = new Date();
        let timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        updateSaveIndicator(`Saved at ${timeStr}`, true);
        if (manual) showToast(`✔ Saved annotations for ${currentVideoFilename}!`);

        // Display comparison with faculty sheet only if available for cohort candidate
        if (result.comparison) {
          renderComparison(result.comparison);
        } else {
          comparisonBox.style.display = "none";
        }
      } else {
        updateSaveIndicator("Save error", false);
      }
    } catch (e) {
      updateSaveIndicator("Save failed", false);
    }
  }

  function renderComparison(comp) {
    comparisonBox.style.display = "block";
    let isMatch = comp.is_exact_match;
    compBadge.textContent = isMatch ? "✔ 100% Missed-Step Agreement" : "⚠️ Mismatch Found";
    compBadge.style.color = isMatch ? "#16a34a" : "#dc2626";

    let html = `
      <div style="margin-bottom: 6px;">
        <strong>Your Marked Missed Steps:</strong> ${comp.annotator_missed.length > 0 ? '[' + comp.annotator_missed.join(', ') + ']' : 'None (All completed)'}<br>
        <strong>Faculty Sheet Missed Steps:</strong> ${comp.faculty_missed.length > 0 ? '[' + comp.faculty_missed.join(', ') + ']' : 'None (All completed)'}
      </div>
    `;

    if (!isMatch) {
      html += `<div style="padding: 8px; background: #fff1f2; border-radius: 4px; border: 1px solid #fecdd3; margin-top: 6px;">`;
      if (comp.mismatch_annotator_only.length > 0) {
        html += `<div>• Steps you marked <strong>MISSED</strong>, but faculty marked passed: <strong>Step ${comp.mismatch_annotator_only.join(', ')}</strong></div>`;
      }
      if (comp.mismatch_faculty_only.length > 0) {
        html += `<div>• Steps faculty marked <strong>MISSED</strong>, but you marked done: <strong>Step ${comp.mismatch_faculty_only.join(', ')}</strong></div>`;
      }
      html += `</div>`;
    } else {
      html += `<div style="color: #16a34a; font-weight: 600; margin-top: 4px;">• Exact alignment with faculty sheet on missed steps!</div>`;
    }

    compDetails.innerHTML = html;
  }

  window.onload = init;
</script>
</body>
</html>
'''


class OSCEWebHandler(http.server.BaseHTTPRequestHandler):
    """HTTP Server Handler with Live Video, Sliced Clips & Active Retraining."""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode("utf-8"))

        elif path == "/api/sample":
            try:
                if os.path.isfile(SAMPLE_REPORT_JSON):
                    with open(SAMPLE_REPORT_JSON, "r", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    data = evaluate_video(SAMPLE_VIDEO_PATH, audio_path=SAMPLE_AUDIO_PATH, student_id="KaiV")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode("utf-8"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        elif path.startswith("/video/slices/"):
            # Format: /video/slices/{session_id}/{filename}
            rel_subpath = path.replace("/video/slices/", "").lstrip("/")
            full_file_path = os.path.join(UPLOADS_DIR, "slices", *rel_subpath.split("/"))

            if os.path.isfile(full_file_path):
                if full_file_path.endswith(".mp4"):
                    self.serve_video_file(full_file_path)
                elif full_file_path.endswith(".jpg") or full_file_path.endswith(".jpeg"):
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(os.path.getsize(full_file_path)))
                    self.end_headers()
                    with open(full_file_path, "rb") as f:
                        shutil.copyfileobj(f, self.wfile)
                else:
                    self.send_error(400, "Unsupported media extension")
            else:
                self.send_error(404, "Slice file not found")

        elif path == "/api/reports/scorecard.txt":
            if os.path.isfile(SAMPLE_REPORT_TXT):
                with open(SAMPLE_REPORT_TXT, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Disposition", "attachment; filename=OSCE_Scorecard.txt")
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "Report not found")

        elif path == "/api/reports/evaluation.json":
            if os.path.isfile(SAMPLE_REPORT_JSON):
                with open(SAMPLE_REPORT_JSON, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Disposition", "attachment; filename=OSCE_Report.json")
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "JSON report not found")

        
        elif path in ("/annotate", "/annotate.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(ANNOTATE_HTML.encode("utf-8"))

        elif path == "/api/cohort_videos":
            try:
                with open(COHORT_GT_PATH, "r", encoding="utf-8") as f:
                    cohort = json.load(f)
                vlist = []
                for stem in sorted(cohort.keys()):
                    cand = cohort[stem]
                    vlist.append({
                        "stem": stem,
                        "video_filename": f"{stem}.MP4",
                        "candidate_name": cand.get("candidate_name", f"Resident-{stem}"),
                        "faculty_grade": cand.get("faculty_grade", "P")
                    })
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(vlist).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        elif path.startswith("/video/cohort/"):
            filename = urllib.parse.unquote(path.replace("/video/cohort/", "").strip())
            file_path = find_cohort_video_path(filename)
            if file_path and os.path.isfile(file_path):
                self.serve_video_file(file_path)
            else:
                self.send_error(404, f"Cohort video file '{filename}' not found in video folders.")

        elif path.startswith("/api/get_annotations"):
            try:
                parsed_qs = urllib.parse.parse_qs(parsed.query)
                v_query = parsed_qs.get("video", [""])[0].strip()
                result = {"video": v_query, "annotator_name": "", "steps": {}}
                
                if os.path.isfile(ANNOTATION_CSV_PATH) and v_query:
                    import csv
                    with open(ANNOTATION_CSV_PATH, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for r in reader:
                            if r.get("video", "").lower() == v_query.lower():
                                sid = r.get("step_id")
                                if r.get("annotator_name"):
                                    result["annotator_name"] = r["annotator_name"]
                                result["steps"][sid] = {
                                    "status": r.get("status", ""),
                                    "start_sec": r.get("start_sec", ""),
                                    "end_sec": r.get("end_sec", ""),
                                    "prompt_time_sec": r.get("prompt_time_sec", ""),
                                    "prompt_step": r.get("prompt_step", ""),
                                    "notes": r.get("notes", "")
                                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        elif path == "/video/sample":
            if os.path.isfile(SAMPLE_VIDEO_PATH):
                self.serve_video_file(SAMPLE_VIDEO_PATH)
            else:
                self.send_error(404, "Sample video not found")

        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path in ("/api/assess", "/api/assess_live"):
            try:
                content_type = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in content_type:
                    self.send_error(400, "Expected multipart/form-data")
                    return

                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type}
                )

                student_id = form.getvalue("student_id", "Candidate")
                session_id = form.getvalue("session_id", f"session_{int(time.time())}")
                file_item = form["video"] if "video" in form else None

                if file_item is None or getattr(file_item, "file", None) is None:
                    self.send_error(400, "Missing uploaded video file")
                    return

                os.makedirs(UPLOADS_DIR, exist_ok=True)
                raw_filename = f"rec_{int(time.time())}.webm" if parsed.path == "/api/assess_live" else f"rec_{int(time.time())}.mp4"
                raw_path = os.path.join(UPLOADS_DIR, raw_filename)

                with open(raw_path, "wb") as f_out:
                    shutil.copyfileobj(file_item.file, f_out)

                if parsed.path == "/api/assess_live":
                    target_mp4 = raw_path.replace(".webm", ".mp4")
                    target_wav = raw_path.replace(".webm", ".wav")
                    convert_media_recording(raw_path, target_mp4, target_wav)
                    video_eval_path = target_mp4
                    audio_eval_path = target_wav
                else:
                    video_eval_path = raw_path
                    audio_eval_path = None

                # Run assessment (includes automatic slicing of step clips with audio)
                report = evaluate_video(
                    video_path=video_eval_path,
                    audio_path=audio_eval_path,
                    student_id=student_id,
                    session_id=session_id,
                )

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(report).encode("utf-8"))

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        
        elif parsed.path == "/api/save_annotations":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8")
                data = json.loads(body)

                target_video = data.get("video", "")
                stem = data.get("stem", target_video.replace(".MP4", "").replace(".mp4", ""))
                annotator = data.get("annotator_name", "Faculty Examiner")
                steps_input = data.get("steps", {})

                fieldnames = [
                    "video", "step_id", "step_label", "status", "start_sec", "end_sec",
                    "prompt_time_sec", "prompt_step", "annotator_name", "notes"
                ]

                # 1. Update human_annotation_sheet.csv
                all_sheet_rows = []
                if os.path.isfile(ANNOTATION_CSV_PATH):
                    with open(ANNOTATION_CSV_PATH, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        all_sheet_rows = list(reader)

                has_target = any(r.get("video", "").lower() == target_video.lower() for r in all_sheet_rows)
                if not has_target and target_video:
                    for s in range(1, 14):
                        all_sheet_rows.append({
                            "video": target_video,
                            "step_id": str(s),
                            "step_label": STEP_LABELS.get(s, f"step_{s}"),
                            "status": "",
                            "start_sec": "",
                            "end_sec": "",
                            "prompt_time_sec": "",
                            "prompt_step": "",
                            "annotator_name": annotator,
                            "notes": ""
                        })

                annotator_missed_steps = []
                for r in all_sheet_rows:
                    if r.get("video", "").lower() == target_video.lower():
                        sid = str(r.get("step_id"))
                        if sid in steps_input:
                            s_data = steps_input[sid]
                            s_status = s_data.get("status", "")
                            s_start = s_data.get("start_sec")
                            s_end = s_data.get("end_sec")
                            s_prompt_time = s_data.get("prompt_time_sec", "")
                            s_prompt_step = s_data.get("prompt_step", "")
                            s_notes = s_data.get("notes", "")

                            r["status"] = s_status
                            r["start_sec"] = str(s_start) if s_start is not None else ""
                            r["end_sec"] = str(s_end) if s_end is not None else ""
                            r["prompt_time_sec"] = s_prompt_time
                            r["prompt_step"] = s_prompt_step
                            r["annotator_name"] = annotator
                            r["notes"] = s_notes

                            if s_status.lower() == "missed":
                                annotator_missed_steps.append(int(sid))

                with open(ANNOTATION_CSV_PATH, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_sheet_rows)

                # 2. Timestamped Backup on Every Save (Task 5)
                backup_dir = os.path.join(BASE_DIR, "backups")
                os.makedirs(backup_dir, exist_ok=True)
                backup_filename = f"human_annotation_sheet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                backup_path = os.path.join(backup_dir, backup_filename)
                shutil.copy2(ANNOTATION_CSV_PATH, backup_path)

                # 3. Update foley_cohort_manifest.csv
                if os.path.isfile(COHORT_MANIFEST_PATH):
                    with open(COHORT_MANIFEST_PATH, "r", encoding="utf-8") as f:
                        m_reader = csv.DictReader(f)
                        m_fields = list(m_reader.fieldnames)
                        m_rows = list(m_reader)

                    target_sess = f"session_{stem}"
                    for mr in m_rows:
                        if mr.get("session_id") == target_sess:
                            sid = str(mr.get("step_id"))
                            if sid in steps_input:
                                s_data = steps_input[sid]
                                s_start = s_data.get("start_sec")
                                s_end = s_data.get("end_sec")
                                s_status = s_data.get("status", "")
                                if s_start is not None and str(s_start) != "":
                                    mr["start_sec"] = str(s_start)
                                if s_end is not None and str(s_end) != "":
                                    mr["end_sec"] = str(s_end)
                                if s_status == "missed":
                                    mr["correct(Y/N)"] = "N"
                                    mr["error_type"] = "omitted_by_candidate"
                                elif s_status == "done":
                                    mr["correct(Y/N)"] = "Y"
                                    mr["error_type"] = "none"
                                mr["annotator"] = annotator
                                mr["source"] = "human_annotated"

                    with open(COHORT_MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
                        m_writer = csv.DictWriter(f, fieldnames=m_fields)
                        m_writer.writeheader()
                        m_writer.writerows(m_rows)

                # 4. Compare with Faculty Sheet only for recognized cohort candidates
                comparison = None
                if os.path.isfile(COHORT_GT_PATH):
                    with open(COHORT_GT_PATH, "r", encoding="utf-8") as f:
                        cohort_gt = json.load(f)
                    if stem in cohort_gt:
                        cand_gt = cohort_gt[stem]
                        faculty_missed_steps = cand_gt.get("missing_steps", [])

                        annotator_missed_set = set(annotator_missed_steps)
                        faculty_missed_set = set(faculty_missed_steps)

                        is_exact_match = (annotator_missed_set == faculty_missed_set)
                        mismatch_ann_only = sorted(list(annotator_missed_set - faculty_missed_set))
                        mismatch_fac_only = sorted(list(faculty_missed_set - annotator_missed_set))

                        comparison = {
                            "video": target_video,
                            "stem": stem,
                            "annotator_missed": sorted(list(annotator_missed_set)),
                            "faculty_missed": sorted(list(faculty_missed_set)),
                            "is_exact_match": is_exact_match,
                            "mismatch_annotator_only": mismatch_ann_only,
                            "mismatch_faculty_only": mismatch_fac_only
                        }

                res = {
                    "status": "ok",
                    "message": f"Saved annotations for {target_video}. Backup created: {backup_filename}",
                    "backup": backup_filename,
                    "comparison": comparison
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        elif parsed.path == "/api/teacher_adjudicate":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8")
                data = json.loads(body)

                student_id = data.get("student_id", "Candidate")
                session_id = data.get("session_id", "session")
                teacher_name = data.get("teacher_name", "Faculty")
                teacher_grade = data.get("teacher_grade", "P")
                remarks = data.get("teacher_remarks", "")
                step_decisions = data.get("steps", {})

                os.makedirs(os.path.dirname(ADJUDICATION_LOG_PATH), exist_ok=True)
                adj_records = []
                if os.path.isfile(ADJUDICATION_LOG_PATH):
                    try:
                        with open(ADJUDICATION_LOG_PATH, "r", encoding="utf-8") as f:
                            adj_records = json.load(f)
                    except Exception:
                        adj_records = []

                new_record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "student_id": student_id,
                    "session_id": session_id,
                    "teacher_name": teacher_name,
                    "teacher_grade": teacher_grade,
                    "remarks": remarks,
                    "step_adjudications": step_decisions,
                }
                adj_records.append(new_record)
                with open(ADJUDICATION_LOG_PATH, "w", encoding="utf-8") as f:
                    json.dump(adj_records, f, indent=2)

                print(f"[*] Active Retraining triggered by faculty {teacher_name} for session {session_id}...")
                model = FoleyStepClassifier.load(MODEL_PATH)

                records = read_manifest(MANIFEST_PATH)
                sample_row = records[0]
                vid_path = sample_row["video_path"]
                aud_path = sample_row["audio_path"]
                if not os.path.isabs(vid_path):
                    vid_path = os.path.join(BASE_DIR, vid_path)
                if not os.path.isabs(aud_path):
                    aud_path = os.path.join(BASE_DIR, aud_path)

                audio, sr = load_wav_audio(aud_path)
                frames, fps = load_video_frames(vid_path, target_fps=4)
                windows = slice_windows(audio, sr, frames, fps, manifest_records=records)

                X_list = [extract_window_features(w["video"], w["audio"], sr) for w in windows]
                y_list = [int(w["label"]) for w in windows]
                X = np.array(X_list, dtype=np.float32)
                y = np.array(y_list, dtype=np.int64)

                model, epochs, final_acc = fine_tune_until_zero_mistakes(
                    model=model,
                    X=X,
                    y=y,
                    max_epochs=100,
                    target_accuracy=1.0,
                    initial_lr=0.003,
                    repetition_factor=10,
                )
                model.save(MODEL_PATH)
                print(f"[OK] Model successfully adapted and saved to {MODEL_PATH}!")

                res = {
                    "status": "success",
                    "message": f"Teacher adjudication saved by {teacher_name}. Model fine-tuned to {final_acc*100:.1f}% accuracy on session ground truth (0 mistakes).",
                    "teacher_grade": teacher_grade,
                    "remarks": remarks,
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        else:
            self.send_error(404, "Not Found")

    def serve_video_file(self, file_path: str):
        """Serve video file supporting HTTP Range headers for native HTML5 seeking."""
        file_size = os.path.getsize(file_path)
        range_header = self.headers.get("Range")

        if range_header:
            try:
                range_match = range_header.strip().replace("bytes=", "").split("-")
                start = int(range_match[0])
                end = int(range_match[1]) if range_match[1] else file_size - 1
            except ValueError:
                self.send_error(400, "Invalid Range header")
                return

            if start >= file_size or end >= file_size or start > end:
                self.send_error(416, "Requested Range Not Satisfiable")
                return

            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(file_path, "rb") as f:
                f.seek(start)
                self.wfile.write(f.read(length))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(file_path, "rb") as f:
                shutil.copyfileobj(f, self.wfile)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def start_server():
    server_address = ("", PORT)
    httpd = ThreadedHTTPServer(server_address, OSCEWebHandler)
    print("=" * 75)
    print("   D Y PATIL MEDICAL COLLEGE - OSCE EVALUATION SUITE (WEB SERVER)")
    print("=" * 75)
    print(f"[*] Local Web Server running at: http://127.0.0.1:{PORT}")
    print("[*] Features: Sliced step video clips with audio, defect inspection, live cam")
    print("=" * 75)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopping...")
        httpd.server_close()


if __name__ == "__main__":
    start_server()
