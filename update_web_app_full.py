"""Update web_app.py with:
1. New columns (status, prompt_time_sec, prompt_step, annotator_name, notes).
2. Autosave after every click with 'Saved at hh:mm' indicator.
3. Post-save comparison with faculty sheet (annotator missed vs faculty missed).
4. Timestamped CSV backup on every save.
"""

import os
import json
import csv
import shutil
from datetime import datetime

WEB_APP_PATH = r"C:\Medical\web_app.py"

ANNOTATE_HTML_UPGRADED = r'''<!DOCTYPE html>
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

  async function changeVideo() {
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
      videoPlayer.currentTime = ann.start_sec;
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

        // Display comparison with faculty sheet
        if (result.comparison) {
          renderComparison(result.comparison);
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

print("ANNOTATE_HTML_UPGRADED defined.")

import re

def apply_full_update():
    with open(WEB_APP_PATH, "r", encoding="utf-8") as f:
        code = f.read()

    # Replace ANNOTATE_HTML
    pattern = r"ANNOTATE_HTML = r'''.*?'''"
    replacement = f"ANNOTATE_HTML = r'''{ANNOTATE_HTML_UPGRADED}'''"
    code = re.sub(pattern, replacement, code, flags=re.DOTALL)

    # Replace save_annotations handler in do_POST
    old_post_handler = r'''        elif parsed.path == "/api/save_annotations":.*?self\.wfile\.write\(json\.dumps\(res\)\.encode\("utf-8"\)\)'''
    
    new_post_handler = '''        elif parsed.path == "/api/save_annotations":
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

                # 4. Compare with Faculty Sheet (Task 3)
                faculty_missed_steps = []
                if os.path.isfile(COHORT_GT_PATH):
                    with open(COHORT_GT_PATH, "r", encoding="utf-8") as f:
                        cohort_gt = json.load(f)
                    cand_gt = cohort_gt.get(stem, {})
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
                self.wfile.write(json.dumps(res).encode("utf-8"))'''

    code = re.sub(old_post_handler, new_post_handler, code, flags=re.DOTALL)

    # Also update get_annotations to read status and prompt fields
    old_get_handler = r'''        elif path\.startswith\("/api/get_annotations"\):.*?self\.wfile\.write\(json\.dumps\(result\)\.encode\("utf-8"\)\)'''
    new_get_handler = '''        elif path.startswith("/api/get_annotations"):
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
                self.wfile.write(json.dumps(result).encode("utf-8"))'''

    code = re.sub(old_get_handler, new_get_handler, code, flags=re.DOTALL)

    with open(WEB_APP_PATH, "w", encoding="utf-8") as f:
        f.write(code)

    print("web_app.py upgraded with full autosave, backups, and faculty sheet comparison!")

if __name__ == "__main__":
    apply_full_update()
