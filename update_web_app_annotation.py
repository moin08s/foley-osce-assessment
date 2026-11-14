"""Add HTML5 Step Annotation Page and Handlers to web_app.py.

Includes:
- ANNOTATE_HTML template (HTML5 video, vanilla JS, hotkeys, step buttons, prompt & missed controls).
- Endpoints:
  GET /annotate
  GET /api/cohort_videos
  GET /video/cohort/<filename>
  GET /api/get_annotations?video=<filename>
  POST /api/save_annotations
"""

import re

WEB_APP_PATH = r"C:\Medical\web_app.py"

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
    header h1 { font-size: 1.25rem; font-weight: 700; }
    header a { color: #bae6fd; text-decoration: none; font-size: 0.9rem; font-weight: 600; padding: 6px 12px; border: 1px solid #38bdf8; border-radius: 6px; }
    header a:hover { background: #0284c7; color: white; }
    .container { max-width: 1440px; margin: 15px auto; padding: 0 20px; }
    .top-bar { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 14px 20px; margin-bottom: 15px; display: flex; flex-wrap: wrap; gap: 20px; align-items: center; justify-content: space-between; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .form-group { display: flex; align-items: center; gap: 10px; }
    .form-group label { font-weight: 600; font-size: 0.88rem; color: #334155; }
    .form-control { padding: 8px 12px; border: 1px solid var(--border); border-radius: 6px; font-size: 0.9rem; outline: none; }
    .form-control:focus { border-color: var(--primary); }
    .btn { padding: 8px 14px; font-weight: 600; border-radius: 6px; cursor: pointer; border: 1px solid transparent; font-size: 0.88rem; transition: all 0.15s; }
    .btn-primary { background: var(--primary); color: white; }
    .btn-primary:hover { background: var(--primary-dark); }
    .btn-success { background: var(--success); color: white; }
    .btn-success:hover { background: #15803d; }
    .btn-danger { background: var(--danger); color: white; }
    .btn-outline { background: white; border-color: var(--border); color: #334155; }
    .btn-outline:hover { background: #f1f5f9; }
    .btn-active { background: #15803d !important; color: white !important; font-weight: 700; }
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
    .steps-container { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; max-height: calc(100vh - 150px); overflow-y: auto; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .steps-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 2px solid var(--border); padding-bottom: 8px; }
    .step-card { border: 1px solid var(--border); border-radius: 6px; padding: 12px; margin-bottom: 10px; background: #ffffff; transition: border-color 0.15s; }
    .step-card.active-step { border-color: var(--primary); box-shadow: 0 0 0 2px rgba(2,132,199,0.15); }
    .step-card.step-missed { background: #fef2f2; border-color: #fca5a5; opacity: 0.75; }
    .step-card-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .step-title { font-weight: 700; font-size: 0.92rem; color: #1e293b; }
    .step-badge { font-size: 0.75rem; padding: 2px 8px; border-radius: 4px; font-weight: 600; background: #e2e8f0; color: #475569; }
    .step-actions { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }
    .step-inputs { display: grid; grid-template-columns: 1fr 1fr 2fr; gap: 8px; align-items: center; }
    .step-inputs input { width: 100%; padding: 6px 10px; border: 1px solid var(--border); border-radius: 4px; font-size: 0.82rem; }
    .toast { position: fixed; bottom: 25px; right: 25px; background: #0f172a; color: white; padding: 14px 22px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.25); z-index: 10000; font-weight: 600; font-size: 0.9rem; display: none; }
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
        <input type="text" id="annotatorName" class="form-control" style="width: 220px;" placeholder="Dr. Examiner / Reviewer" value="Faculty Examiner">
      </div>
      <div class="form-group">
        <label for="videoSelect">Candidate Video:</label>
        <select id="videoSelect" class="form-control" style="width: 280px;" onchange="changeVideo()"></select>
      </div>
    </div>
    <div>
      <button class="btn btn-success" style="font-size: 1rem; padding: 10px 20px;" onclick="saveAnnotations()">
        💾 Save to human_annotation_sheet.csv
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
    </div>

    <!-- Right Column: 13 Step Annotation Cards -->
    <div class="steps-container">
      <div class="steps-header">
        <div>
          <h2 style="font-size: 1.1rem; font-weight: 700;">OSCE 13 Clinical Steps</h2>
          <span style="font-size: 0.8rem; color: #64748b;">Click start buttons to record current video timestamp. Click timestamp to jump.</span>
        </div>
        <div id="saveBadge" style="font-size: 0.8rem; font-weight: 600; color: #0284c7;"></div>
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
  let toastEl = document.getElementById("toast");

  let currentVideoStem = "";
  let currentVideoFilename = "";
  let stepAnnotations = {};
  let nextStepTarget = 1;

  function formatTime(sec) {
    if (sec === null || sec === undefined || isNaN(sec) || sec === "") return "—";
    let s = parseFloat(sec);
    let m = Math.floor(s / 60);
    let remSec = (s % 60).toFixed(1);
    return `${m.toString().padStart(2, '0')}:${remSec.padStart(4, '0')}`;
  }

  function parseTimeStr(str) {
    if (!str || str === "—") return "";
    str = String(str).trim();
    if (str.includes(":")) {
      let parts = str.split(":");
      return Math.round((parseFloat(parts[0]) * 60 + parseFloat(parts[1])) * 10) / 10;
    }
    return Math.round(parseFloat(str) * 10) / 10;
  }

  function showToast(msg, isSuccess = true) {
    toastEl.textContent = msg;
    toastEl.style.display = "block";
    toastEl.style.background = isSuccess ? "#0f2b48" : "#dc2626";
    setTimeout(() => { toastEl.style.display = "none"; }, 3500);
  }

  // Update Time HUD
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

  // Load list of 23 verified cohort videos
  async function init() {
    try {
      let res = await fetch("/api/cohort_videos");
      let videos = await res.json();
      videoSelect.innerHTML = "";
      videos.forEach((v, idx) => {
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
        start_sec: null,
        end_sec: null,
        missed: false,
        prompt: false,
        notes: ""
      };
    });
    nextStepTarget = 1;

    // Load any existing human-annotated rows from server
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
            if (s.start_sec !== "" && s.start_sec !== null) {
              stepAnnotations[sid].start_sec = parseFloat(s.start_sec);
            }
            if (s.end_sec !== "" && s.end_sec !== null) {
              stepAnnotations[sid].end_sec = parseFloat(s.end_sec);
            }
            stepAnnotations[sid].notes = s.notes || "";
            if (s.notes && s.notes.includes("[MISSED]")) {
              stepAnnotations[sid].missed = true;
            }
          }
        }
      }
    } catch (e) {
      console.log("No previous annotations loaded", e);
    }

    renderStepCards();
  }

  function renderStepCards() {
    stepCardsList.innerHTML = "";
    STEP_DEFINITIONS.forEach(def => {
      let sid = def.id;
      let ann = stepAnnotations[sid];
      let card = document.createElement("div");
      card.className = "step-card" + (ann.missed ? " step-missed" : "");
      card.id = `card_step_${sid}`;

      let startBtnText = ann.start_sec !== null ? `⏱ Start: ${formatTime(ann.start_sec)} (Click to Jump)` : `⏱ Step ${sid} starts`;
      let startBtnClass = ann.start_sec !== null ? "btn btn-success btn-active" : "btn btn-outline";

      card.innerHTML = `
        <div class="step-card-top">
          <div class="step-title">Step ${sid.toString().padStart(2, '0')}: ${def.name}</div>
          <span class="step-badge">${def.label}</span>
        </div>
        <div class="step-actions">
          <button class="${startBtnClass}" onclick="handleStartClick(${sid})">${startBtnText}</button>
          <button class="btn btn-outline" style="${ann.missed ? 'background:#ef4444; color:white;' : ''}" onclick="toggleMissed(${sid})">
            ${ann.missed ? '✔ Mark as Completed' : '❌ Step Missed'}
          </button>
          <button class="btn btn-outline" onclick="addPromptTimestamp(${sid})">
            💬 Prompt given at this time
          </button>
        </div>
        <div class="step-inputs">
          <div>
            <label style="font-size:0.75rem; color:#64748b; display:block;">Start (sec):</label>
            <input type="number" step="0.1" id="input_start_${sid}" value="${ann.start_sec !== null ? ann.start_sec : ''}" onchange="updateManualStart(${sid}, this.value)">
          </div>
          <div>
            <label style="font-size:0.75rem; color:#64748b; display:block;">End (sec):</label>
            <input type="number" step="0.1" id="input_end_${sid}" value="${ann.end_sec !== null ? ann.end_sec : ''}" onchange="updateManualEnd(${sid}, this.value)">
          </div>
          <div>
            <label style="font-size:0.75rem; color:#64748b; display:block;">Clinical Notes / Error Remarks:</label>
            <input type="text" id="input_notes_${sid}" placeholder="Optional notes / error observations" value="${ann.notes || ''}" onchange="updateNotes(${sid}, this.value)">
          </div>
        </div>
      `;
      stepCardsList.appendChild(card);
    });
  }

  function handleStartClick(sid) {
    let ann = stepAnnotations[sid];
    if (ann.start_sec !== null) {
      // If already recorded, jumping back to it on click as requested
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
    stepAnnotations[sid].missed = false;

    // Auto-fill end_sec of the previous step if it was missing
    for (let prev = sid - 1; prev >= 1; prev--) {
      if (stepAnnotations[prev].start_sec !== null && (stepAnnotations[prev].end_sec === null || stepAnnotations[prev].end_sec === "")) {
        stepAnnotations[prev].end_sec = t;
        break;
      }
    }

    nextStepTarget = Math.min(13, sid + 1);
    renderStepCards();
    showToast(`Marked Step ${sid} start at ${formatTime(t)}`);
  }

  function markNextStep() {
    markStepStart(nextStepTarget);
  }

  function toggleMissed(sid) {
    stepAnnotations[sid].missed = !stepAnnotations[sid].missed;
    if (stepAnnotations[sid].missed) {
      if (!stepAnnotations[sid].notes.includes("[MISSED]")) {
        stepAnnotations[sid].notes = (stepAnnotations[sid].notes + " [MISSED]").trim();
      }
    } else {
      stepAnnotations[sid].notes = stepAnnotations[sid].notes.replace("[MISSED]", "").trim();
    }
    renderStepCards();
  }

  function addPromptTimestamp(sid) {
    let t = Math.round(videoPlayer.currentTime * 10) / 10;
    let promptTag = `[Prompt given at ${formatTime(t)}]`;
    stepAnnotations[sid].notes = (stepAnnotations[sid].notes + " " + promptTag).trim();
    renderStepCards();
    showToast(`Recorded prompt at ${formatTime(t)} for Step ${sid}`);
  }

  function updateManualStart(sid, val) {
    stepAnnotations[sid].start_sec = val === "" ? null : parseFloat(val);
  }

  function updateManualEnd(sid, val) {
    stepAnnotations[sid].end_sec = val === "" ? null : parseFloat(val);
  }

  function updateNotes(sid, val) {
    stepAnnotations[sid].notes = val;
  }

  // Keyboard Navigation
  window.addEventListener("keydown", (e) => {
    // Ignore hotkeys when user is typing into text inputs
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

  // Save to human_annotation_sheet.csv
  async function saveAnnotations() {
    let annotator = annotatorNameInput.value.trim();
    if (!annotator) {
      alert("Please enter the Annotator Name before saving.");
      annotatorNameInput.focus();
      return;
    }

    // Auto-compute missing end_sec values
    let dur = Math.round((videoPlayer.duration || 160.0) * 10) / 10;
    for (let sid = 1; sid <= 13; sid++) {
      let ann = stepAnnotations[sid];
      if (ann.start_sec !== null && (ann.end_sec === null || ann.end_sec === "")) {
        // Look ahead for next step start
        let nextFound = false;
        for (let next = sid + 1; next <= 13; next++) {
          if (stepAnnotations[next].start_sec !== null) {
            ann.end_sec = stepAnnotations[next].start_sec;
            nextFound = true;
            break;
          }
        }
        if (!nextFound) {
          ann.end_sec = dur;
        }
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
        showToast(`✔ Saved annotations for ${currentVideoFilename} to human_annotation_sheet.csv!`);
        document.getElementById("saveBadge").textContent = `Last saved: ${new Date().toLocaleTimeString()} by ${annotator}`;
        renderStepCards();
      } else {
        alert("Error saving: " + result.error);
      }
    } catch (e) {
      alert("Save failed: " + e.message);
    }
  }

  window.onload = init;
</script>
</body>
</html>
'''

print("ANNOTATE_HTML template ready.")
