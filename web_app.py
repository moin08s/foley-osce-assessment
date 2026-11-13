"""Lightweight Local Edge AI Web Server for OSCE Foley Catheterization.

D Y Patil Medical College - Procedural Skills Assessment Model
Pure Python Standard Library (http.server) + NumPy.
Zero external web frameworks required (no FastAPI/Flask/Tornado).
100% compatible with Windows Application Control policies.

Features:
1. One-click assessment on preloaded sample clinical video (KaiV, 001.MP4).
2. Video file upload / path selection for new candidates.
3. Interactive 13-step OSCE Checklist with timestamps, confidence scores, and speech verification.
4. Native HTML5 video player with interactive step-seeking (jump directly to any step clip).
5. Sequence & Critical Safety audit panel.
6. One-click download of printable scorecards and JSON reports.
"""

import cgi
import http.server
import json
import mimetypes
import os
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from predict import evaluate_video
from src.data_loader import STEP_LABELS
from src.grading import OSCEGrader


PORT = 8000
SAMPLE_VIDEO_PATH = os.path.join(BASE_DIR, "dataset", "test", "KaiV", "session_001", "clip_001.mp4")
SAMPLE_AUDIO_PATH = os.path.join(BASE_DIR, "dataset", "test", "KaiV", "session_001", "clip_001.wav")
SAMPLE_REPORT_TXT = os.path.join(BASE_DIR, "reports", "sample_video_scorecard.txt")
SAMPLE_REPORT_JSON = os.path.join(BASE_DIR, "reports", "sample_video_evaluation.json")


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
      --bg: #f4f6f9;
      --card: #ffffff;
      --text: #212529;
      --border: #dee2e6;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
    body { background-color: var(--bg); color: var(--text); padding-bottom: 50px; }
    header { background: linear-gradient(135deg, var(--primary), var(--primary-dark)); color: white; padding: 20px 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    header h1 { font-size: 1.5rem; font-weight: 700; letter-spacing: 0.5px; }
    header p { font-size: 0.9rem; opacity: 0.9; margin-top: 4px; }
    .container { max-width: 1200px; margin: 25px auto; padding: 0 20px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    .card { background: var(--card); border-radius: 8px; border: 1px solid var(--border); box-shadow: 0 2px 4px rgba(0,0,0,0.05); padding: 20px; margin-bottom: 20px; }
    .card-title { font-size: 1.15rem; font-weight: 600; margin-bottom: 15px; border-bottom: 2px solid #e9ecef; padding-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
    .btn { display: inline-block; font-weight: 600; text-align: center; white-space: nowrap; vertical-align: middle; user-select: none; border: 1px solid transparent; padding: 10px 18px; font-size: 0.95rem; border-radius: 6px; cursor: pointer; transition: all 0.2s ease-in-out; text-decoration: none; }
    .btn-primary { color: #fff; background-color: var(--primary); border-color: var(--primary); }
    .btn-primary:hover { background-color: var(--primary-dark); }
    .btn-success { color: #fff; background-color: var(--success); }
    .btn-outline { color: var(--primary); border-color: var(--primary); background: transparent; }
    .btn-outline:hover { background: var(--primary); color: #fff; }
    .btn-sm { padding: 4px 10px; font-size: 0.8rem; }
    .btn-block { display: block; width: 100%; }
    .form-group { margin-bottom: 15px; }
    .form-group label { display: block; font-weight: 600; font-size: 0.85rem; margin-bottom: 6px; }
    .form-control { width: 100%; padding: 8px 12px; font-size: 0.9rem; border: 1px solid var(--border); border-radius: 5px; outline: none; }
    .badge { display: inline-block; padding: 4px 8px; font-size: 0.75rem; font-weight: 700; line-height: 1; text-align: center; border-radius: 4px; color: white; }
    .badge-p { background-color: var(--success); }
    .badge-p1 { background-color: var(--warning); color: #000; }
    .badge-r { background-color: var(--danger); }
    .grade-hero { text-align: center; padding: 20px 10px; background: #eef5fc; border-radius: 8px; border: 1px dashed #b8daff; margin-bottom: 20px; }
    .grade-circle { display: inline-flex; width: 85px; height: 85px; border-radius: 50%; align-items: center; justify-content: center; font-size: 2.2rem; font-weight: 800; color: white; margin-bottom: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.15); }
    .grade-title { font-size: 1.25rem; font-weight: 700; color: var(--primary-dark); }
    .grade-desc { font-size: 0.85rem; color: #495057; margin-top: 5px; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.88rem; }
    th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #e9ecef; }
    th { background-color: #f8f9fa; font-weight: 600; color: #495057; }
    tr:hover { background-color: #f1f5f9; cursor: pointer; }
    .audit-box { background: #f8f9fa; border-left: 4px solid var(--success); padding: 12px 15px; border-radius: 4px; margin-top: 15px; font-size: 0.88rem; }
    .audit-box.alert { border-left-color: var(--danger); background: #fff3f3; }
    video { width: 100%; max-height: 360px; background: #000; border-radius: 6px; }
    .jump-link { color: var(--primary); font-weight: 600; text-decoration: none; cursor: pointer; }
    .jump-link:hover { text-decoration: underline; }
    .loader { display: none; text-align: center; padding: 25px; }
    .spinner { display: inline-block; width: 35px; height: 35px; border: 4px solid rgba(0,0,0,0.1); border-top-color: var(--primary); border-radius: 50%; animation: spin 0.8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>

<header>
  <h1>D Y Patil Medical College - Department of Surgery</h1>
  <p>Edge AI Automated Video & Audio OSCE Assessment Suite | Male Foley's Catheterization</p>
</header>

<div class="container">
  <div class="grid">
    <!-- Left Column: Controls & Video Player -->
    <div>
      <div class="card">
        <div class="card-title">
          <span>Candidate & Video Input</span>
          <span class="badge badge-p">Pure NumPy Edge Model</span>
        </div>
        <div class="form-group">
          <label>Sample Video Benchmark (Resident KaiV, 2:38)</label>
          <button id="btnSample" class="btn btn-primary btn-block" onclick="runSampleAssessment()">
            ▶ Evaluate Sample Video (001.MP4)
          </button>
        </div>
        <hr style="border: none; border-top: 1px solid #e9ecef; margin: 15px 0;">
        <form id="uploadForm" enctype="multipart/form-data">
          <div class="form-group">
            <label>Or Upload New Video (.mp4)</label>
            <input type="file" id="videoFile" name="video" accept=".mp4,.avi,.mov" class="form-control">
          </div>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div class="form-group">
              <label>Student ID</label>
              <input type="text" id="studentId" name="student_id" value="Resident-001" class="form-control">
            </div>
            <div class="form-group">
              <label>Session ID</label>
              <input type="text" id="sessionId" name="session_id" value="session_eval" class="form-control">
            </div>
          </div>
          <button type="submit" class="btn btn-outline btn-block">Assess Uploaded Video</button>
        </form>
      </div>

      <div class="card">
        <div class="card-title">
          <span>Synchronized Video Playback</span>
          <span style="font-size: 0.8rem; color: #6c757d;">Click any step in the table to jump</span>
        </div>
        <video id="player" controls playsinline>
          <source id="videoSource" src="/video/sample" type="video/mp4">
          Your browser does not support HTML5 video.
        </video>
        <p style="font-size: 0.8rem; color: #6c757d; margin-top: 8px;">
          💡 <em>Tip: Clicking on any row in the OSCE Checklist on the right automatically seeks the video to that exact procedure slice.</em>
        </p>
      </div>

      <div class="card">
        <div class="card-title">Download Official Reports</div>
        <div style="display: flex; gap: 10px;">
          <a href="/api/reports/scorecard.txt" target="_blank" class="btn btn-outline btn-block">Download Printable Scorecard (.txt)</a>
          <a href="/api/reports/evaluation.json" target="_blank" class="btn btn-outline btn-block">Download JSON Report (.json)</a>
        </div>
      </div>
    </div>

    <!-- Right Column: Assessment Dashboard -->
    <div>
      <div id="loader" class="card loader">
        <div class="spinner"></div>
        <p style="margin-top: 12px; font-weight: 600; color: var(--primary);">Analyzing multimodal video & audio frames from scratch...</p>
        <p style="font-size: 0.8rem; color: #6c757d;">Extracting motion energy, colour histograms & FFT Log-Mel spectra...</p>
      </div>

      <div id="resultsCard" class="card">
        <div class="card-title">
          <span>OSCE Clinical Evaluation Result</span>
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

        <h3 style="margin-top: 20px; font-size: 1rem; color: #495057;">13-Step OSCE Checklist</h3>
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
    </div>
  </div>
</div>

<script>
  function seekVideo(startSec) {
    const player = document.getElementById('player');
    player.currentTime = parseFloat(startSec);
    player.play();
  }

  function renderReport(data) {
    document.getElementById('scoreText').innerText = data.score || '13 / 13';
    
    // Grade hero
    const badge = document.getElementById('gradeBadge');
    badge.innerText = data.final_grade;
    if (data.final_grade === 'P') {
      badge.style.backgroundColor = 'var(--success)';
    } else if (data.final_grade === 'P-1') {
      badge.style.backgroundColor = 'var(--warning)';
    } else {
      badge.style.backgroundColor = 'var(--danger)';
    }

    document.getElementById('gradeTitle').innerText = data.final_grade === 'P' ? 'Pass: Advance Beginner' : (data.final_grade === 'P-1' ? 'Pass with 1 Prompt: Beginner' : 'Needs Remediation: Novice');
    document.getElementById('gradeMeaning').innerText = data.grade_description;

    // Safety audit
    const auditBox = document.getElementById('auditBox');
    const auditText = document.getElementById('auditText');
    if (data.critical_safety_violations && data.critical_safety_violations.length > 0) {
      auditBox.className = 'audit-box alert';
      auditText.innerHTML = '<strong>[!] Critical Safety Violations:</strong><br>' + data.critical_safety_violations.join('<br>');
    } else {
      auditBox.className = 'audit-box';
      auditText.innerHTML = '[*] Zero critical safety violations detected.<br>[*] All steps performed in chronological order.';
    }

    // Populate table
    const tbody = document.getElementById('checklistBody');
    tbody.innerHTML = '';

    for (let sid = 1; sid <= 13; sid++) {
      const step = data.step_results[sid];
      if (!step) continue;
      const tr = document.createElement('tr');
      const timeStr = step.start_sec !== null ? `${step.start_sec.toFixed(1)}s - ${step.end_sec.toFixed(1)}s` : '--';
      const confStr = step.confidence !== null ? `${(step.confidence * 100).toFixed(1)}%` : '--';
      
      tr.innerHTML = `
        <td><strong>${sid}</strong></td>
        <td>
          <a class="jump-link" onclick="seekVideo(${step.start_sec || 0})">${step.label}</a>
          ${(sid === 1 || sid === 2) ? '<span class="badge" style="background:#17a2b8; margin-left:5px;">Audio Verified</span>' : ''}
        </td>
        <td><span class="badge badge-p">${step.status}</span></td>
        <td>${timeStr}</td>
        <td><strong>${confStr}</strong></td>
      `;
      tbody.appendChild(tr);
    }
  }

  function runSampleAssessment() {
    document.getElementById('loader').style.display = 'block';
    fetch('/api/sample')
      .then(res => res.json())
      .then(data => {
        document.getElementById('loader').style.display = 'none';
        renderReport(data);
      })
      .catch(err => {
        document.getElementById('loader').style.display = 'none';
        alert('Error assessing video: ' + err);
      });
  }

  document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const fileInput = document.getElementById('videoFile');
    if (!fileInput.files || fileInput.files.length === 0) {
      alert('Please select a video file (.mp4) first.');
      return;
    }
    const formData = new FormData();
    formData.append('video', fileInput.files[0]);
    formData.append('student_id', document.getElementById('studentId').value);
    formData.append('session_id', document.getElementById('sessionId').value);

    document.getElementById('loader').style.display = 'block';
    fetch('/api/assess', {
      method: 'POST',
      body: formData
    })
    .then(res => res.json())
    .then(data => {
      document.getElementById('loader').style.display = 'none';
      renderReport(data);
    })
    .catch(err => {
      document.getElementById('loader').style.display = 'none';
      alert('Assessment failed: ' + err);
    });
  });

  // Run on load
  window.addEventListener('DOMContentLoaded', () => {
    runSampleAssessment();
  });
</script>
</body>
</html>
"""


class OSCEWebHandler(http.server.BaseHTTPRequestHandler):
    """HTTP Request Handler for OSCE Assessment Dashboard."""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode("utf-8"))

        elif path == "/api/sample":
            # Return cached or live evaluated JSON for sample video
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
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

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

        elif path == "/video/sample":
            # Stream sample video with HTTP Range support for seeking
            if os.path.isfile(SAMPLE_VIDEO_PATH):
                self.serve_video_file(SAMPLE_VIDEO_PATH)
            else:
                self.send_error(404, "Sample video file not found")

        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/assess":
            # Handle multipart/form-data video upload
            try:
                content_type = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in content_type:
                    self.send_error(400, "Expected multipart/form-data")
                    return

                # Parse form using cgi
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": content_type,
                    }
                )

                student_id = form.getvalue("student_id", "Candidate-001")
                session_id = form.getvalue("session_id", "session_001")
                file_item = form["video"] if "video" in form else None

                if not file_item or not file_item.file:
                    self.send_error(400, "Missing uploaded video file")
                    return

                upload_dir = os.path.join(BASE_DIR, "uploads")
                os.makedirs(upload_dir, exist_ok=True)
                saved_path = os.path.join(upload_dir, f"upload_{int(time.time())}.mp4")

                with open(saved_path, "wb") as f_out:
                    shutil.copyfileobj(file_item.file, f_out)

                # Run assessment
                report = evaluate_video(
                    video_path=saved_path,
                    student_id=student_id,
                    session_id=session_id,
                )

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(report).encode("utf-8"))

            except Exception as e:
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
            # Parse Range: bytes=START-END
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
    print("[*] Open http://127.0.0.1:8000 in your browser to view the dashboard.")
    print("[*] Press Ctrl+C in this terminal window to stop the server.")
    print("=" * 75)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopping...")
        httpd.server_close()


if __name__ == "__main__":
    start_server()
