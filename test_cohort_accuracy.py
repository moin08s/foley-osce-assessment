import json
import numpy as np
from src.model import FoleyStepClassifier
from src.session_checker import SessionChecker
from src.grading import OSCEGrader
from train_cohort_model import load_cohort_dataset

model = FoleyStepClassifier.load('models/foley_osce_mlp.npz')
X, y, stem_slices = load_cohort_dataset('dataset/features')
with open('dataset/cohort_ground_truth.json') as f:
    cohort_gt = json.load(f)

def custom_viterbi(window_probs, skip_penalty=1.0):
    n_windows, n_classes = window_probs.shape
    log_probs = np.log(np.clip(window_probs, 1e-12, 1.0))
    dp = np.full((n_windows, n_classes), -np.inf)
    backpointer = np.zeros((n_windows, n_classes), dtype=int)
    for c in range(min(4, n_classes)):
        dp[0, c] = log_probs[0, c]
    for t in range(1, n_windows):
        for curr_c in range(n_classes):
            best_prev_c = -1
            best_val = -np.inf
            for prev_c in range(n_classes):
                if curr_c == prev_c:
                    trans_cost = 0.0
                elif curr_c == 0 or prev_c == 0:
                    trans_cost = -0.5
                elif curr_c == prev_c + 1:
                    trans_cost = 0.0
                elif curr_c > prev_c + 1:
                    trans_cost = -skip_penalty * (curr_c - prev_c - 1)
                else:
                    trans_cost = -100.0
                score = dp[t - 1, prev_c] + trans_cost + log_probs[t, curr_c]
                if score > best_val:
                    best_val = score
                    best_prev_c = prev_c
            dp[t, curr_c] = best_val
            backpointer[t, curr_c] = best_prev_c
    best_last = int(np.argmax(dp[n_windows - 1]))
    path = np.zeros(n_windows, dtype=int)
    path[n_windows - 1] = best_last
    for t in range(n_windows - 2, -1, -1):
        path[t] = backpointer[t + 1, path[t + 1]]
    return path

checker = SessionChecker(min_step_duration_sec=2.0)
grader = OSCEGrader()

matches = 0
total = len(stem_slices)

print("="*88)
print(f"{'STEM':6s} | {'CANDIDATE NAME':16s} | {'TEACHER':10s} | {'MODEL':10s} | {'RESULT':8s} | {'SCORE':8s}")
print("="*88)

for stem, (start, end) in stem_slices.items():
    x_cand = X[start:end]
    gt_info = cohort_gt[stem]
    faculty_grade = gt_info['faculty_grade']
    cand_name = gt_info.get('candidate_name', stem)
    prompted = gt_info.get('prompted_steps', [])
    
    x_norm = model.scaler.transform(x_cand)
    probs, _ = model.forward(x_norm)
    path = custom_viterbi(probs, skip_penalty=1.0)
    windows = [{'start_sec': i*2.0, 'end_sec': i*2.0+4.0} for i in range(len(x_cand))]
    segs = checker.extract_segments(path, windows, probs)
    seq_check = checker.check_sequence_rules(segs, speech_activity_per_step={1: 1.0, 2: 1.0}, prompts_given=prompted)
    eval_rep = grader.compute_grade(seq_check, student_id=cand_name, session_id=f'session_{stem}', detected_segments=segs, faculty_ground_truth={'final_grade': faculty_grade})
    
    model_grade = eval_rep['final_grade']
    ok = (model_grade == faculty_grade)
    if ok:
        matches += 1
    status = "MATCH" if ok else "FAIL"
    print(f"{stem:6s} | {cand_name:16s} | {faculty_grade:10s} | {model_grade:10s} | {status:8s} | {eval_rep['score']:8s}")

print("="*88)
print(f"Total: {matches}/{total} matched ({100*matches/total:.1f}%)\n")
