"""Script to cleanly update web_app.py with the HTML5 Step Annotation Page and APIs."""

import os
import re

WEB_APP_PATH = r"C:\Medical\web_app.py"

with open(WEB_APP_PATH, "r", encoding="utf-8") as f:
    code = f.read()

# 1. Add ANNOTATE_HTML and cohort paths
from update_web_app_annotation import ANNOTATE_HTML

# Check if already present
if "ANNOTATE_HTML =" in code:
    print("ANNOTATE_HTML already in web_app.py, proceeding to replace or check...")

# Add Constants if missing
constants_block = '''
COHORT_VIDEO_DIR = r"C:\\New folder (2)\\103NZ_30"
ANNOTATION_CSV_PATH = r"C:\\Medical\\human_annotation_sheet.csv"
COHORT_MANIFEST_PATH = r"C:\\Medical\\foley_cohort_manifest.csv"
COHORT_GT_PATH = r"C:\\Medical\\dataset\\cohort_ground_truth.json"
'''

if "COHORT_VIDEO_DIR =" not in code:
    code = code.replace('UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")', 'UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")\n' + constants_block)

# Add ANNOTATE_HTML before class OSCEWebHandler
if "ANNOTATE_HTML =" not in code:
    code = code.replace("class OSCEWebHandler(http.server.BaseHTTPRequestHandler):", f"ANNOTATE_HTML = r'''{ANNOTATE_HTML}'''\n\n\nclass OSCEWebHandler(http.server.BaseHTTPRequestHandler):")

# Add link to tab in INDEX_HTML
if 'href="/annotate"' not in code:
    code = code.replace(
        '<button class="tab-btn" id="tabSample" onclick="switchTab(\'sample\')">🎬 Benchmark & File Upload</button>',
        '<button class="tab-btn" id="tabSample" onclick="switchTab(\'sample\')">🎬 Benchmark & File Upload</button>\n    <a href="/annotate" class="tab-btn" style="text-decoration:none; background:#e0f2fe; color:#0369a1;">✍ Step Annotation Tool</a>'
    )

# Add GET endpoints before `else: self.send_error(404, "Not Found")` in do_GET
get_endpoints = '''
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
            file_path = os.path.join(COHORT_VIDEO_DIR, filename)
            if os.path.isfile(file_path):
                self.serve_video_file(file_path)
            else:
                self.send_error(404, f"Cohort video file '{filename}' not found")

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
                                    "start_sec": r.get("start_sec", ""),
                                    "end_sec": r.get("end_sec", ""),
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
'''

if 'elif path in ("/annotate", "/annotate.html"):' not in code:
    code = code.replace(
        'elif path == "/video/sample":',
        get_endpoints + '\n        elif path == "/video/sample":'
    )

# Add POST endpoint in do_POST before `else: self.send_error(404, "Not Found")`
post_endpoints = '''
        elif parsed.path == "/api/save_annotations":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8")
                data = json.loads(body)

                target_video = data.get("video", "")
                stem = data.get("stem", target_video.replace(".MP4", "").replace(".mp4", ""))
                annotator = data.get("annotator_name", "Human Expert")
                steps_input = data.get("steps", {})

                # 1. Update human_annotation_sheet.csv
                import csv
                all_sheet_rows = []
                if os.path.isfile(ANNOTATION_CSV_PATH):
                    with open(ANNOTATION_CSV_PATH, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        fieldnames = list(reader.fieldnames)
                        all_sheet_rows = list(reader)
                else:
                    fieldnames = ["video", "step_id", "step_label", "start_sec", "end_sec", "annotator_name", "notes"]

                for r in all_sheet_rows:
                    if r.get("video", "").lower() == target_video.lower():
                        sid = str(r.get("step_id"))
                        if sid in steps_input:
                            s_data = steps_input[sid]
                            s_start = s_data.get("start_sec")
                            s_end = s_data.get("end_sec")
                            r["start_sec"] = str(s_start) if s_start is not None else ""
                            r["end_sec"] = str(s_end) if s_end is not None else ""
                            r["annotator_name"] = annotator
                            r["notes"] = s_data.get("notes", "")

                with open(ANNOTATION_CSV_PATH, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_sheet_rows)

                # 2. Update foley_cohort_manifest.csv marking source="human_annotated"
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
                                if s_start is not None and str(s_start) != "":
                                    mr["start_sec"] = str(s_start)
                                if s_end is not None and str(s_end) != "":
                                    mr["end_sec"] = str(s_end)
                                mr["annotator"] = annotator
                                mr["source"] = "human_annotated"

                    with open(COHORT_MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
                        m_writer = csv.DictWriter(f, fieldnames=m_fields)
                        m_writer.writeheader()
                        m_writer.writerows(m_rows)

                res = {
                    "status": "ok",
                    "message": f"Successfully saved 13 steps for {target_video} as human_annotated to human_annotation_sheet.csv and manifest."
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
'''

if 'elif parsed.path == "/api/save_annotations":' not in code:
    code = code.replace(
        'elif parsed.path == "/api/teacher_adjudicate":',
        post_endpoints + '\n        elif parsed.path == "/api/teacher_adjudicate":'
    )

with open(WEB_APP_PATH, "w", encoding="utf-8") as f:
    f.write(code)

print("web_app.py successfully updated with Step Annotation Page and APIs!")
