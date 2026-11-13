import os
import subprocess

video_dir = r"C:\New folder (2)\103NZ_30"
out_audio_dir = r"C:\Medical\dataset\audio"
os.makedirs(out_audio_dir, exist_ok=True)

ffmpeg = r"C:\Medical\.venv\Scripts\ffmpeg.exe"

files = sorted([f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")])
print(f"Extracting 16kHz mono WAV for {len(files)} videos...")

for f in files:
    stem = os.path.splitext(f)[0]
    in_video = os.path.join(video_dir, f)
    out_wav = os.path.join(out_audio_dir, f"{stem}.wav")
    
    if os.path.exists(out_wav) and os.path.getsize(out_wav) > 10000:
        print(f"Already exists: {stem}.wav ({os.path.getsize(out_wav):,} bytes)")
        continue
        
    cmd = [
        ffmpeg, "-y", "-i", in_video,
        "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
        out_wav
    ]
    res = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    if res.returncode == 0:
        print(f"Extracted: {stem}.wav ({os.path.getsize(out_wav):,} bytes)")
    else:
        print(f"Error on {f}: {res.stderr.decode('utf-8', errors='ignore')[:200]}")

print("Audio extraction complete.")
