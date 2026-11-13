"""Session Checker and Procedural Sequence Validator.

D Y Patil Medical College - Procedural Skills Assessment Model
Temporal smoothing, segment extraction, and clinical sequence validation.
Pure Python Standard Library + NumPy (no third-party dependencies).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .data_loader import STEP_LABELS


class SessionChecker:
    """Smoothes window-level predictions and enforces clinical safety & ordering rules."""

    def __init__(
        self,
        min_step_duration_sec: float = 2.0,
        smoothing_window_size: int = 3,
        speech_threshold: float = 0.05,
    ):
        self.min_step_duration_sec = min_step_duration_sec
        self.smoothing_window_size = smoothing_window_size
        self.speech_threshold = speech_threshold

    def smooth_predictions(self, window_probs: np.ndarray, method: str = "viterbi") -> np.ndarray:
        """Smooth window predictions across time.
        
        Args:
            window_probs: (N_windows, 14) array of class probabilities
            method: "viterbi" (enforces forward clinical progression) or "moving_average"
            
        Returns:
            smoothed_labels: (N_windows,) array of smoothed step predictions
        """
        if len(window_probs) == 0:
            return np.array([], dtype=int)

        if method == "viterbi":
            return self.viterbi_decode(window_probs)

        n_windows, n_classes = window_probs.shape
        w = self.smoothing_window_size
        pad_size = w // 2

        padded_probs = np.pad(window_probs, ((pad_size, pad_size), (0, 0)), mode="edge")
        smoothed_probs = np.zeros_like(window_probs)

        for i in range(n_windows):
            smoothed_probs[i] = padded_probs[i : i + w].mean(axis=0)

        return np.argmax(smoothed_probs, axis=1)

    def viterbi_decode(self, window_probs: np.ndarray, skip_penalty: float = 1.0) -> np.ndarray:
        """Dynamic programming Viterbi decoder enforcing forward monotonic clinical progression."""
        n_windows, n_classes = window_probs.shape
        if n_windows == 0:
            return np.array([], dtype=int)

        log_probs = np.log(np.clip(window_probs, 1e-12, 1.0))
        # State space: (c, h) where c in 0..13 is current class, h in 0..13 is highest clinical step reached
        n_states = n_classes * n_classes
        dp = np.full((n_windows, n_states), -np.inf)
        backpointer = np.zeros((n_windows, n_states), dtype=int)

        for c in range(n_classes):
            s = c * n_classes + c
            dp[0, s] = log_probs[0, c]

        for t in range(1, n_windows):
            for c in range(n_classes):
                for h in range(n_classes):
                    s = c * n_classes + h
                    best_prev_s = -1
                    best_val = -np.inf

                    for prev_c in range(n_classes):
                        for prev_h in range(n_classes):
                            prev_s = prev_c * n_classes + prev_h
                            if dp[t - 1, prev_s] == -np.inf:
                                continue

                            if c == 0:
                                if h != prev_h:
                                    continue
                                trans_cost = -0.5
                            else:
                                if c < prev_h:
                                    continue  # Strictly forbidden to regress to a lower clinical step
                                if h != max(prev_h, c):
                                    continue
                                if c == prev_c:
                                    trans_cost = 0.0
                                elif c == prev_h + 1 or (prev_c == 0 and c == prev_h):
                                    trans_cost = 0.0
                                else:
                                    trans_cost = -skip_penalty * (c - prev_h - 1)

                            score = dp[t - 1, prev_s] + trans_cost + log_probs[t, c]
                            if score > best_val:
                                best_val = score
                                best_prev_s = prev_s

                    dp[t, s] = best_val
                    backpointer[t, s] = best_prev_s

        best_last_s = int(np.argmax(dp[n_windows - 1]))
        path = np.zeros(n_windows, dtype=int)
        curr_s = best_last_s
        for t in range(n_windows - 1, -1, -1):
            path[t] = curr_s // n_classes
            curr_s = backpointer[t, curr_s]

        return path

    def extract_segments(
        self,
        predicted_labels: np.ndarray,
        windows: List[Dict[str, Any]],
        window_probs: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Merge consecutive identical window predictions into distinct procedural segments.
        
        Filters out noise blips shorter than min_step_duration_sec.
        """
        if len(predicted_labels) == 0 or len(windows) == 0:
            return []

        raw_segments = []
        curr_label = int(predicted_labels[0])
        curr_start = float(windows[0]["start_sec"])
        curr_end = float(windows[0]["end_sec"])
        curr_confidences = [float(window_probs[0, curr_label])] if window_probs is not None else [1.0]

        for i in range(1, len(predicted_labels)):
            lbl = int(predicted_labels[i])
            win_start = float(windows[i]["start_sec"])
            win_end = float(windows[i]["end_sec"])

            if lbl == curr_label:
                curr_end = max(curr_end, win_end)
                if window_probs is not None:
                    curr_confidences.append(float(window_probs[i, lbl]))
            else:
                raw_segments.append({
                    "step_id": curr_label,
                    "step_label": STEP_LABELS.get(curr_label, f"step_{curr_label}"),
                    "start_sec": curr_start,
                    "end_sec": curr_end,
                    "duration_sec": round(curr_end - curr_start, 2),
                    "confidence": round(float(np.mean(curr_confidences)), 3),
                })
                curr_label = lbl
                curr_start = win_start
                curr_end = win_end
                curr_confidences = [float(window_probs[i, lbl])] if window_probs is not None else [1.0]

        # Append final segment
        raw_segments.append({
            "step_id": curr_label,
            "step_label": STEP_LABELS.get(curr_label, f"step_{curr_label}"),
            "start_sec": curr_start,
            "end_sec": curr_end,
            "duration_sec": round(curr_end - curr_start, 2),
            "confidence": round(float(np.mean(curr_confidences)), 3),
        })

        # Filter out background (step 0) and spurious noise segments shorter than min duration
        procedural_segments = []
        for seg in raw_segments:
            if seg["step_id"] != 0 and seg["duration_sec"] >= self.min_step_duration_sec:
                procedural_segments.append(seg)

        return procedural_segments

    def check_sequence_rules(
        self,
        detected_segments: List[Dict[str, Any]],
        speech_activity_per_step: Optional[Dict[int, float]] = None,
        prompts_given: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Apply clinical OSCE ordering and safety rules from foley_clip_dataset_spec.json.
        
        Clinical Order Rules:
        1. Step 1 (introduce_explain) must be first procedural step; requires speech.
        2. Step 2 (informed_consent) must follow Step 1 and precede Step 4 (gloves); requires speech.
        3. Step 4 (hand_hygiene_gloves) must precede Step 6 (painting_draping).
        4. Step 9 (attach_urobag_confirm_urine) must precede Step 10 (inflate_balloon).
           [CRITICAL SAFETY: Inflating balloon before urine flow risks urethral rupture].
        5. Step 10 (inflate_balloon) must precede Step 11 (traction_until_resistance).
        6. Step 11 (traction_until_resistance) must precede Step 12 (reposition_prepuce).
        7. Step 12 (reposition_prepuce): must be performed [prevents paraphimosis].
        8. Step 13 (tip_dressing) must be last step.
        """
        all_expected_steps = list(range(1, 14))
        detected_step_ids = [seg["step_id"] for seg in detected_segments]

        # Unique detected steps preserving first occurrence order
        seen_order = []
        step_first_seen_time = {}
        for seg in detected_segments:
            sid = seg["step_id"]
            if sid not in seen_order:
                seen_order.append(sid)
                step_first_seen_time[sid] = seg["start_sec"]

        # Missing steps
        missing_steps = [s for s in all_expected_steps if s not in seen_order]

        # Clinical rule violations
        violations = []
        critical_safety_violations = []

        # 1. Step 1 check
        if 1 in seen_order:
            if seen_order[0] != 1:
                violations.append({
                    "rule": "Step 1 must be first",
                    "description": f"Introduction/explanation was not performed first (first detected step was Step {seen_order[0]}: {STEP_LABELS[seen_order[0]]}).",
                    "severity": "MODERATE",
                })
            # Speech check for Step 1
            if speech_activity_per_step and speech_activity_per_step.get(1, 1.0) < self.speech_threshold:
                violations.append({
                    "rule": "Step 1 verbal communication",
                    "description": "Step 1 (introduce_explain) lacked detectable speech communication.",
                    "severity": "MODERATE",
                })
        else:
            violations.append({
                "rule": "Step 1 omitted",
                "description": "Step 1 (introduce_explain) was completely omitted.",
                "severity": "HIGH",
            })

        # 2. Step 2 check (informed consent)
        if 2 in seen_order:
            if 1 in seen_order and step_first_seen_time[2] < step_first_seen_time[1]:
                violations.append({
                    "rule": "Consent before introduction",
                    "description": "Consent was sought prior to introducing procedure to patient.",
                    "severity": "MODERATE",
                })
            if 4 in seen_order and step_first_seen_time[2] > step_first_seen_time[4]:
                violations.append({
                    "rule": "Consent after sterile gloves",
                    "description": "Informed consent was sought after donning sterile gloves (procedural delay / sterility breach).",
                    "severity": "HIGH",
                })
            # Speech check for Step 2
            if speech_activity_per_step and speech_activity_per_step.get(2, 1.0) < self.speech_threshold:
                violations.append({
                    "rule": "Step 2 verbal consent confirmation",
                    "description": "Step 2 (informed_consent) lacked detectable verbal inquiry/permission.",
                    "severity": "MODERATE",
                })

        # 3. Step 4 before Step 6
        if 4 in seen_order and 6 in seen_order:
            if step_first_seen_time[6] < step_first_seen_time[4]:
                violations.append({
                    "rule": "Sterile gloves before painting/draping",
                    "description": "Painting and draping was initiated before donning sterile gloves (MAJOR ASEPTIC BREACH).",
                    "severity": "CRITICAL",
                })
                critical_safety_violations.append("Sterile gloves donned after antiseptic painting commenced.")

        # 4. Step 9 before Step 10 [CRITICAL BLADDER / URETHRAL SAFETY]
        if 10 in seen_order:
            if 9 not in seen_order:
                critical_safety_violations.append("Catheter retention balloon inflated without attaching urobag or confirming urine flow.")
                violations.append({
                    "rule": "Confirm urine before balloon inflation",
                    "description": "CRITICAL ERROR: Balloon was inflated without visual confirmation of urine drainage! Risk of balloon inflation inside prostatic urethra causing severe urethral trauma.",
                    "severity": "CRITICAL",
                })
            elif step_first_seen_time[10] < step_first_seen_time[9]:
                critical_safety_violations.append("Balloon inflated prior to urobag connection and urine drainage verification.")
                violations.append({
                    "rule": "Urine confirmation prior to balloon inflation",
                    "description": "CRITICAL SAFETY VIOLATION: Balloon was inflated before confirming urine flow (Step 10 preceded Step 9).",
                    "severity": "CRITICAL",
                })

        # 5. Step 10 before Step 11
        if 11 in seen_order and 10 in seen_order:
            if step_first_seen_time[11] < step_first_seen_time[10]:
                violations.append({
                    "rule": "Inflate balloon before gentle traction",
                    "description": "Traction was exerted before inflating balloon (catheter pulled out of position).",
                    "severity": "HIGH",
                })

        # 6. Step 11 before Step 12
        if 12 in seen_order and 11 in seen_order:
            if step_first_seen_time[12] < step_first_seen_time[11]:
                violations.append({
                    "rule": "Traction before prepuce repositioning",
                    "description": "Prepuce repositioning was performed before anchoring catheter at bladder neck.",
                    "severity": "LOW",
                })

        # 7. Step 12 omitted check [PARAPHIMOSIS RISK]
        if 12 not in seen_order:
            violations.append({
                "rule": "Reposition prepuce omitted",
                "description": "CRITICAL RISK: Prepuce was not repositioned over glans penis. High clinical risk of acute paraphimosis and vascular compromise.",
                "severity": "CRITICAL",
            })
            critical_safety_violations.append("Prepuce not repositioned (risk of paraphimosis).")

        # 8. Step 13 last check
        if 13 in seen_order:
            if seen_order[-1] != 13:
                violations.append({
                    "rule": "Tip dressing should be last step",
                    "description": f"Tip dressing was not the terminal step (subsequent step detected: Step {seen_order[-1]}: {STEP_LABELS[seen_order[-1]]}).",
                    "severity": "LOW",
                })

        prompts = prompts_given or []

        return {
            "all_13_present": len(missing_steps) == 0,
            "missing_steps": missing_steps,
            "missing_step_labels": [STEP_LABELS[s] for s in missing_steps],
            "detected_order": seen_order,
            "detected_order_labels": [STEP_LABELS[s] for s in seen_order],
            "violations": violations,
            "critical_safety_violations": critical_safety_violations,
            "prompts_count": len(prompts),
            "prompts_given": prompts,
            "is_valid_sequence": (len(missing_steps) == 0 and len(violations) == 0 and len(prompts) == 0),
        }
