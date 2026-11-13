import os
import json

base_video_dir = r"C:\New folder (2)\103NZ_30"
audio_dir = r"C:\Medical\dataset\audio"

# Base ground truth from batch2_faculty_ground_truth.json
with open(r"C:\Medical\catheter_ai_project\dataset\batch2_faculty_ground_truth.json") as f:
    batch2_gt = json.load(f)

cohort = {}

# Process 001 through 020
for k, v in batch2_gt.items():
    stem = k
    vid_name = f"{stem}.MP4"
    vid_path = os.path.join(base_video_dir, vid_name)
    aud_path = os.path.join(audio_dir, f"{stem}.wav")
    
    code = v.get("adjudication_code", "P")
    cand_name = v.get("candidate_name", f"Resident-{stem}")
    
    # Step results
    step_results = {}
    missing = []
    prompted = []
    
    for s_idx in range(1, 14):
        s_data = v.get("steps", {}).get(str(s_idx), {})
        status = s_data.get("status", "PASS (✔)")
        if "MISSED" in status or "✘" in status:
            step_results[s_idx] = "Missed"
            missing.append(s_idx)
        elif "P-1" in status or "Prompt" in status:
            step_results[s_idx] = "P-1"
            prompted.append(s_idx)
        else:
            step_results[s_idx] = "Pass"
            
    cohort[stem] = {
        "stem": stem,
        "video_file": vid_path,
        "audio_file": aud_path,
        "candidate_name": cand_name,
        "faculty_grade": code,
        "missing_steps": missing,
        "prompted_steps": prompted,
        "step_results": step_results,
    }

# Candidate 021 (Page 21)
# Missed step 12 and 13 -> Grade R
cohort["021"] = {
    "stem": "021",
    "video_file": os.path.join(base_video_dir, "021.MP4"),
    "audio_file": os.path.join(audio_dir, "021.wav"),
    "candidate_name": "VoiswwaWi",
    "faculty_grade": "R",
    "missing_steps": [12, 13],
    "prompted_steps": [],
    "step_results": {s: ("Missed" if s in [12, 13] else "Pass") for s in range(1, 14)},
}

# Candidate 022 (Page 22)
# All 13 steps completed -> Grade P
cohort["022"] = {
    "stem": "022",
    "video_file": os.path.join(base_video_dir, "022.MP4"),
    "audio_file": os.path.join(audio_dir, "022.wav"),
    "candidate_name": "KA-+",
    "faculty_grade": "P",
    "missing_steps": [],
    "prompted_steps": [],
    "step_results": {s: "Pass" for s in range(1, 14)},
}

# Candidate 023 (Page 23)
# Missed steps 3, 4, 5, 7 -> Grade R
cohort["023"] = {
    "stem": "023",
    "video_file": os.path.join(base_video_dir, "023.MP4"),
    "audio_file": os.path.join(audio_dir, "023.wav"),
    "candidate_name": "Candidate_023",
    "faculty_grade": "R",
    "missing_steps": [3, 4, 5, 7],
    "prompted_steps": [],
    "step_results": {s: ("Missed" if s in [3, 4, 5, 7] else "Pass") for s in range(1, 14)},
}

# Candidate 024 (Page 24)
# Step 4 prompted -> Grade P-1
cohort["024"] = {
    "stem": "024",
    "video_file": os.path.join(base_video_dir, "024.MP4"),
    "audio_file": os.path.join(audio_dir, "024.wav"),
    "candidate_name": "Candidate_024",
    "faculty_grade": "P-1",
    "missing_steps": [],
    "prompted_steps": [4],
    "step_results": {s: ("P-1" if s == 4 else "Pass") for s in range(1, 14)},
}

# Candidate 025 (Part 1 and 2)
# Completed -> Grade P
cohort["025"] = {
    "stem": "025",
    "video_file": os.path.join(base_video_dir, "025(1).MP4"),
    "audio_file": os.path.join(audio_dir, "025(1).wav"),
    "candidate_name": "Candidate_025",
    "faculty_grade": "P",
    "missing_steps": [],
    "prompted_steps": [],
    "step_results": {s: "Pass" for s in range(1, 14)},
}

out_path = r"C:\Medical\dataset\cohort_ground_truth.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(cohort, f, indent=2)

print(f"Saved cohort ground truth with {len(cohort)} candidates to {out_path}:")
for k, v in cohort.items():
    print(f"  {k:5s} | Name: {v['candidate_name']:15s} | Grade: {v['faculty_grade']:3s} | Missing: {v['missing_steps']} | Prompted: {v['prompted_steps']}")
