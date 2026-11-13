import os
import sys
from catheter_ai_project.src.audio_verifier import AudioConsentVerifier

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

verifier = AudioConsentVerifier(model_size="tiny")

videos = [
    r"C:\text\001.MP4",
    r"C:\text\002.MP4",
    r"C:\text\003.MP4",
    r"C:\text\007.MP4",
]

for v in videos:
    res = verifier.verify_speech(v, duration_sec=40.0)
    print(f"\nVideo: {os.path.basename(v)}")
    print(f"  Candidate Name: {res.get('candidate_name')}")
    print(f"  Step 1 Verified: {res.get('step_1_verified')}")
    print(f"  Step 2 Verified: {res.get('step_2_verified')}")
    print(f"  Transcript: {res.get('transcription')[:150]}")
