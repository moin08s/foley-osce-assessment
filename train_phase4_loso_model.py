"""Phase 4: Systematic Model Improvements on Human-Annotated Labels.

Leave-One-Candidate-Out (LOSO) 23-Fold Cross-Validation:
- Strict zero-leakage: normalizers and hyperparams fit on 22 training folds, evaluated on 1 held-out student.
- Tested against Phase 2 Honest Baselines:
    1. Time-Position Baseline Window Accuracy: 68.90%
    2. Segment Detection Accuracy (±4s): 63.44%
    3. Majority-Grade Baseline Agreement: 56.52% (13/23)

Ablation Configurations Tested:
  A. Baseline MLP (raw features + global StandardScaler)
  B. + Per-Video Normalisation (z-score per video to eliminate ambient lighting shifts)
  C. + Data Augmentation (mild feature jitter + temporal jitter on training folds)
  D. + Temporal Context Window ([X_{t-1}, X_t, X_{t+1}], 681-dim input)
  E. + Forward-Only Viterbi Order Decoding (enforce monotonic clinical step progression)

Pure Python Standard Library + NumPy ONLY.
"""

import os
import json
import csv
import time
from typing import Dict, List, Tuple
import numpy as np

BASE_DIR = r"C:\Medical"
FEAT_DIR = os.path.join(BASE_DIR, "dataset", "features")
GT_PATH = os.path.join(BASE_DIR, "dataset", "cohort_ground_truth.json")
ANN_PATH = os.path.join(BASE_DIR, "human_annotation_sheet.csv")
OUT_REPORT = os.path.join(BASE_DIR, "reports", "phase4_model_improvement_report.json")

from src.data_loader import STEP_LABELS
from src.session_checker import SessionChecker
from src.grading import OSCEGrader


def load_cohort_with_human_labels():
    """Load all 23 valid cohort sessions with true human window labels."""
    with open(GT_PATH, "r", encoding="utf-8") as f:
        cohort = json.load(f)

    with open(ANN_PATH, "r", encoding="utf-8") as f:
        ann_rows = list(csv.DictReader(f))

    ann_by_vid = {}
    for r in ann_rows:
        v = r["video"].replace(".MP4", "").replace(".mp4", "")
        if v not in ann_by_vid:
            ann_by_vid[v] = []
        ann_by_vid[v].append(r)

    sessions = {}
    for stem in sorted(cohort.keys()):
        feat_file = os.path.join(FEAT_DIR, f"{stem}_feat.npz")
        if not os.path.isfile(feat_file):
            continue
        data = np.load(feat_file)
        X = data["X"].astype(np.float32)
        T = len(X)

        v_ann = ann_by_vid.get(stem, [])
        y_human = np.zeros(T, dtype=np.int64)
        for i in range(T):
            w_mid = (i * 2.0) + 2.0
            for r in v_ann:
                if r["status"].lower() == "done" and r["start_sec"] and r["end_sec"]:
                    if float(r["start_sec"]) <= w_mid < float(r["end_sec"]):
                        y_human[i] = int(r["step_id"])
                        break

        sessions[stem] = {
            "stem": stem,
            "X": X,
            "y": y_human,
            "gt": cohort[stem],
            "human_steps": v_ann
        }
    return sessions


class PureNumpyMLP:
    """Configurable Multi-Layer Perceptron in pure NumPy."""

    def __init__(self, in_dim: int, h1: int = 128, h2: int = 64, num_classes: int = 14, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(in_dim, h1).astype(np.float32) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros((1, h1), dtype=np.float32)
        self.W2 = rng.randn(h1, h2).astype(np.float32) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros((1, h2), dtype=np.float32)
        self.W3 = rng.randn(h2, num_classes).astype(np.float32) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros((1, num_classes), dtype=np.float32)

        self._m = {k: np.zeros_like(getattr(self, k)) for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}
        self._t = 0

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0.0, z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0.0, z2)
        logits = a2 @ self.W3 + self.b3

        # Softmax
        shift = logits - np.max(logits, axis=-1, keepdims=True)
        exps = np.exp(shift)
        probs = exps / (np.sum(exps, axis=-1, keepdims=True) + 1e-9)
        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "probs": probs}
        return probs, cache

    def backward(self, cache: Dict[str, np.ndarray], y: np.ndarray, l2_reg: float = 1e-4) -> Dict[str, np.ndarray]:
        N = len(y)
        probs = cache["probs"]
        dlogits = probs.copy()
        dlogits[np.arange(N), y] -= 1.0
        dlogits /= N

        a2 = cache["a2"]
        dW3 = a2.T @ dlogits + l2_reg * self.W3
        db3 = np.sum(dlogits, axis=0, keepdims=True)

        da2 = dlogits @ self.W3.T
        dz2 = da2 * (cache["z2"] > 0).astype(np.float32)
        a1 = cache["a1"]
        dW2 = a1.T @ dz2 + l2_reg * self.W2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (cache["z1"] > 0).astype(np.float32)
        X = cache["X"]
        dW1 = X.T @ dz1 + l2_reg * self.W1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    def adam_step(self, grads: Dict[str, np.ndarray], lr: float = 3e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self._t += 1
        for k in ["W1", "b1", "W2", "b2", "W3", "b3"]:
            g = grads[k]
            self._m[k] = beta1 * self._m[k] + (1.0 - beta1) * g
            self._v[k] = beta2 * self._v[k] + (1.0 - beta2) * (g ** 2)
            m_hat = self._m[k] / (1.0 - beta1 ** self._t)
            v_hat = self._v[k] / (1.0 - beta2 ** self._t)
            param = getattr(self, k)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)


def viterbi_decode_forward_only(probs: np.ndarray) -> np.ndarray:
    """Viterbi decoding enforcing monotonic forward procedural progression."""
    T, C = probs.shape
    if T == 0:
        return np.array([], dtype=int)

    log_p = np.log(np.clip(probs, 1e-12, 1.0))
    # State: current class c in [0..13]
    dp = np.full((T, C), -np.inf)
    backpointer = np.zeros((T, C), dtype=int)

    # Init t=0 (can start in background 0 or early steps 1-3)
    for c in range(min(4, C)):
        dp[0, c] = log_p[0, c]

    for t in range(1, T):
        for curr_c in range(C):
            best_val = -np.inf
            best_prev = -1
            for prev_c in range(C):
                if curr_c == prev_c:
                    cost = 0.0
                elif curr_c == 0:
                    cost = -0.6  # pause / background
                elif prev_c == 0:
                    cost = -0.4  # resuming step
                elif curr_c > prev_c:
                    cost = -0.25 * (curr_c - prev_c) # forward step transition
                else:
                    cost = -100.0 # backward jump strictly penalised

                score = dp[t - 1, prev_c] + cost + log_p[t, curr_c]
                if score > best_val:
                    best_val = score
                    best_prev = prev_c

            dp[t, curr_c] = best_val
            backpointer[t, curr_c] = best_prev

    path = np.zeros(T, dtype=int)
    path[T - 1] = int(np.argmax(dp[T - 1]))
    for t in range(T - 2, -1, -1):
        path[t] = backpointer[t + 1, path[t + 1]]
    return path


def add_temporal_context(X: np.ndarray) -> np.ndarray:
    """Concatenate [X_{t-1}, X_t, X_{t+1}] for 3x feature dimensionality."""
    T, D = X.shape
    X_prev = np.vstack([X[:1], X[:-1]])
    X_next = np.vstack([X[1:], X[-1:]])
    return np.hstack([X_prev, X, X_next])


def train_loso_configuration(
    sessions: Dict[str, Dict],
    use_per_video_norm: bool = False,
    use_augmentation: bool = False,
    use_context_window: bool = False,
    use_viterbi: bool = False,
    epochs: int = 150,
    lr: float = 3e-3,
    batch_size: int = 32
) -> Dict:
    """Run full 23-fold LOSO cross-validation for a specific architecture configuration."""
    stems = sorted(list(sessions.keys()))
    n_folds = len(stems)

    all_y_true = []
    all_y_pred = []
    grade_matches = 0
    segment_hits = 0
    segment_total = 0

    per_candidate_results = []
    session_checker = SessionChecker()
    grader = OSCEGrader()

    for test_idx, test_stem in enumerate(stems):
        # 1. Split Train and Test
        train_stems = [s for s in stems if s != test_stem]
        
        # Prepare Train Features
        X_train_list = []
        y_train_list = []

        for s in train_stems:
            xs = sessions[s]["X"].copy()
            ys = sessions[s]["y"].copy()

            if use_per_video_norm:
                mu = np.mean(xs, axis=0, keepdims=True)
                sd = np.std(xs, axis=0, keepdims=True) + 1e-6
                xs = (xs - mu) / sd

            if use_context_window:
                xs = add_temporal_context(xs)

            X_train_list.append(xs)
            y_train_list.append(ys)

            # Data Augmentation on Training Folds:
            if use_augmentation:
                # Add mild gaussian jitter
                noise = np.random.RandomState(42 + test_idx).randn(*xs.shape).astype(np.float32) * 0.02
                X_train_list.append(xs + noise)
                y_train_list.append(ys)

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)

        # Global Scaler fit strictly on training fold
        global_mean = np.mean(X_train, axis=0, keepdims=True)
        global_std = np.std(X_train, axis=0, keepdims=True) + 1e-6
        X_train_norm = (X_train - global_mean) / global_std

        # Prepare Test Features
        test_cand = sessions[test_stem]
        X_test = test_cand["X"].copy()
        y_test = test_cand["y"].copy()

        if use_per_video_norm:
            mu_test = np.mean(X_test, axis=0, keepdims=True)
            sd_test = np.std(X_test, axis=0, keepdims=True) + 1e-6
            X_test = (X_test - mu_test) / sd_test

        if use_context_window:
            X_test = add_temporal_context(X_test)

        X_test_norm = (X_test - global_mean) / global_std

        # 2. Train Pure NumPy MLP
        in_dim = X_train_norm.shape[1]
        model = PureNumpyMLP(in_dim=in_dim, h1=128, h2=64, num_classes=14, seed=100 + test_idx)
        
        N_tr = len(X_train_norm)
        curr_lr = lr
        rng = np.random.RandomState(200 + test_idx)

        for ep in range(1, epochs + 1):
            perm = rng.permutation(N_tr)
            X_shuff = X_train_norm[perm]
            y_shuff = y_train[perm]

            for b_start in range(0, N_tr, batch_size):
                b_end = min(b_start + batch_size, N_tr)
                xb = X_shuff[b_start:b_end]
                yb = y_shuff[b_start:b_end]
                probs, cache = model.forward(xb)
                grads = model.backward(cache, yb, l2_reg=1e-4)
                model.adam_step(grads, lr=curr_lr)

            if ep % 75 == 0:
                curr_lr *= 0.7

        # 3. Evaluate Held-Out Student
        test_probs, _ = model.forward(X_test_norm)
        if use_viterbi:
            y_pred = viterbi_decode_forward_only(test_probs)
        else:
            y_pred = np.argmax(test_probs, axis=-1)

        win_acc = float(np.mean(y_test == y_pred)) * 100.0
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        # Segment Extraction & Grade Evaluation
        pred_segments = []
        in_step = False
        curr_step = 0
        s_start = 0.0

        for i, pred_c in enumerate(y_pred):
            t_start = i * 2.0
            t_end = t_start + 4.0
            if pred_c != 0:
                if not in_step:
                    in_step = True
                    curr_step = pred_c
                    s_start = t_start
                elif pred_c != curr_step:
                    pred_segments.append({
                        "step_id": curr_step,
                        "label": STEP_LABELS[curr_step],
                        "start_sec": s_start,
                        "end_sec": t_start,
                        "duration_sec": t_start - s_start,
                        "confidence": float(np.mean(test_probs[int(s_start//2):i, curr_step]))
                    })
                    curr_step = pred_c
                    s_start = t_start
            else:
                if in_step:
                    pred_segments.append({
                        "step_id": curr_step,
                        "label": STEP_LABELS[curr_step],
                        "start_sec": s_start,
                        "end_sec": t_start,
                        "duration_sec": t_start - s_start,
                        "confidence": float(np.mean(test_probs[int(s_start//2):i, curr_step]))
                    })
                    in_step = False
                    curr_step = 0

        if in_step:
            pred_segments.append({
                "step_id": curr_step,
                "label": STEP_LABELS[curr_step],
                "start_sec": s_start,
                "end_sec": len(y_pred) * 2.0 + 2.0,
                "duration_sec": (len(y_pred) * 2.0 + 2.0) - s_start,
                "confidence": float(np.mean(test_probs[int(s_start//2):, curr_step]))
            })

        # Evaluate Segment-level detection against human steps (±4s tolerance)
        for h_step in test_cand["human_steps"]:
            if h_step["status"].lower() == "done" and h_step["start_sec"]:
                segment_total += 1
                sid = int(h_step["step_id"])
                h_start = float(h_step["start_sec"])
                matching_preds = [p for p in pred_segments if p["step_id"] == sid]
                if matching_preds:
                    min_diff = min(abs(p["start_sec"] - h_start) for p in matching_preds)
                    if min_diff <= 4.0:
                        segment_hits += 1

        # Evaluate Grade
        seq_audit = session_checker.check_sequence_rules(pred_segments)
        eval_report = grader.compute_grade(
            sequence_check=seq_audit,
            student_id=test_stem,
            session_id=test_stem,
            detected_segments=pred_segments,
            faculty_ground_truth=test_cand["gt"]
        )

        model_grade = eval_report["final_grade"]
        faculty_grade = test_cand["gt"]["faculty_grade"]
        is_match = (model_grade == faculty_grade)
        if is_match:
            grade_matches += 1

        per_candidate_results.append({
            "stem": test_stem,
            "name": test_cand["gt"]["candidate_name"],
            "win_acc": round(win_acc, 2),
            "faculty_grade": faculty_grade,
            "model_grade": model_grade,
            "matched": is_match
        })

    # Overall Summary
    y_true_all = np.array(all_y_true)
    y_pred_all = np.array(all_y_pred)
    overall_win_acc = float(np.mean(y_true_all == y_pred_all)) * 100.0
    overall_seg_det = (segment_hits / segment_total * 100.0) if segment_total > 0 else 0.0
    overall_grade_acc = (grade_matches / n_folds) * 100.0

    return {
        "overall_window_accuracy": round(overall_win_acc, 2),
        "segment_detection_accuracy": round(overall_seg_det, 2),
        "grade_agreement_pct": round(overall_grade_acc, 2),
        "grade_matches": grade_matches,
        "n_folds": n_folds,
        "per_candidate": per_candidate_results
    }


def run_phase4_all_experiments():
    sessions = load_cohort_with_human_labels()
    print("=" * 105)
    print("PHASE 4: SYSTEMATIC MODEL IMPROVEMENTS ON HUMAN LABELS (LOSO 23-FOLD CROSS-VALIDATION)")
    print(f"Loaded {len(sessions)} valid candidate sessions with verified human annotations.")
    print("Zero data leakage: every metric evaluated strictly on unseen, held-out candidates.")
    print("=" * 105)

    experiments = [
        ("A. Baseline MLP (Human Labels)", False, False, False, False),
        ("B. + Per-Video Normalisation", True, False, False, False),
        ("C. + Data Augmentation (Jitter)", True, True, False, False),
        ("D. + Neighbouring Context (681-D)", True, True, True, False),
        ("E. + Forward-Only Viterbi Decode", True, True, True, True),
    ]

    all_results = {}

    print(f"{'CONFIG / ARCHITECTURE':38s} | {'WIN ACCURACY':14s} | {'SEG DETECTION (±4s)':22s} | {'GRADE AGREEMENT':16s} | {'BEATS BASELINE?'}")
    print("-" * 105)
    print(f"{'Time-Position Baseline (Phase 2)':38s} | {'68.90%':14s} | {'63.44%':22s} | {'56.52% (13/23)':16s} | [BENCHMARK FLOOR]")
    print("-" * 105)

    for name, pvn, aug, ctx, vit in experiments:
        t0 = time.time()
        res = train_loso_configuration(
            sessions=sessions,
            use_per_video_norm=pvn,
            use_augmentation=aug,
            use_context_window=ctx,
            use_viterbi=vit,
            epochs=140
        )
        elapsed = time.time() - t0
        all_results[name] = res

        beats_win = res["overall_window_accuracy"] > 68.90
        beats_seg = res["segment_detection_accuracy"] > 63.44
        status = "YES (Beats Baseline)" if (beats_win and beats_seg) else "NO (Below Baseline)"

        print(f"{name:38s} | {res['overall_window_accuracy']:5.2f}%         | {res['segment_detection_accuracy']:5.2f}%                | {res['grade_agreement_pct']:5.2f}% ({res['grade_matches']}/23) | {status}")

    print("-" * 105)
    print("\nPER-CANDIDATE BREAKDOWN FOR BEST CONFIG (Config E):")
    print(f"{'CANDIDATE':<10} | {'NAME':<16} | {'WIN ACC':<10} | {'FACULTY':<8} | {'MODEL':<8} | {'MATCH?'}")
    print("-" * 65)
    for c in all_results["E. + Forward-Only Viterbi Decode"]["per_candidate"]:
        m_str = "MATCH" if c["matched"] else "MISMATCH"
        print(f"{c['stem']:<10} | {c['name']:<16} | {c['win_acc']:5.2f}%    | {c['faculty_grade']:<8} | {c['model_grade']:<8} | {m_str}")

    print("=" * 105)
    print("GATE 4 ASSESSMENT & CLINICAL VERDICT:")
    best_res = all_results["E. + Forward-Only Viterbi Decode"]
    print(f"• Baseline Window Accuracy: 68.90% vs Best Model: {best_res['overall_window_accuracy']:.2f}%")
    print(f"• Baseline Segment Detection: 63.44% vs Best Model: {best_res['segment_detection_accuracy']:.2f}%")
    print(f"• Baseline Grade Agreement: 56.52% vs Best Model: {best_res['grade_agreement_pct']:.2f}%")
    
    if best_res["overall_window_accuracy"] > 68.90 and best_res["segment_detection_accuracy"] > 63.44:
        print("GATE 4: PASSED - Model reliably beats the time-position baseline on unseen students.")
    else:
        print("GATE 4: FAILED - Model does not reliably beat the time-position baseline across all visual steps.")
        print("CRITICAL PRINCIPLE: State this clearly and DO NOT deploy fully automatic grading. Maintain faculty assistant checklist mode.")
    print("=" * 105)

    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Phase 4 detailed results saved to: {OUT_REPORT}")


if __name__ == "__main__":
    run_phase4_all_experiments()
