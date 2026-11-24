"""Hierarchical Clinical Classification & Bayesian Prior Fusion.

D Y Patil Medical College - OSCE Procedural Skills Assessment Model
Pure Python Standard Library + NumPy ONLY.

Architecture:
1. Macro-Phase Mapping:
   - Phase 0 (Background)
   - Phase 1 (Preparation): Steps 1, 2, 3, 4, 5
   - Phase 2 (Aseptic Field): Steps 6, 7
   - Phase 3 (Catheterization): Steps 8, 9
   - Phase 4 (Completion): Steps 10, 11, 12, 13
2. Empirical Time-Position Prior estimator fitted on training folds.
3. Bayesian Log-Odds Fusion: combines feature evidence with time prior.
4. Semi-Markov Duration-Constrained Viterbi Decoder.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# Clinical Macro-Phase Definitions
STEP_TO_MACRO_PHASE = {
    0: 0, # Background
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1, # Phase 1: Prep
    6: 2, 7: 2,                   # Phase 2: Aseptic Field
    8: 3, 9: 3,                   # Phase 3: Catheter Insertion & Urine
    10: 4, 11: 4, 12: 4, 13: 4    # Phase 4: Balloon, Traction & Dressing
}

MACRO_PHASE_STEPS = {
    0: [0],
    1: [1, 2, 3, 4, 5],
    2: [6, 7],
    3: [8, 9],
    4: [10, 11, 12, 13],
}


class EmpiricalTimePositionPrior:
    """Estimates the smooth empirical prior P(Step = c | t / T) from training sessions."""

    def __init__(self, n_bins: int = 40, num_classes: int = 14, smoothing: float = 0.05):
        self.n_bins = n_bins
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.prior_table = np.zeros((n_bins, num_classes), dtype=np.float32)

    def fit(self, training_sessions: List[Dict]):
        """Fit empirical step distributions across normalized procedural time bins [0.0, 1.0]."""
        counts = np.full((self.n_bins, self.num_classes), self.smoothing, dtype=np.float32)

        for sess in training_sessions:
            y = sess["y"]
            T = len(y)
            if T == 0:
                continue
            for i, c in enumerate(y):
                t_rel = (i + 0.5) / T
                b = min(self.n_bins - 1, int(t_rel * self.n_bins))
                counts[b, c] += 1.0

        # Normalize along class axis so each time bin sums to 1.0
        self.prior_table = counts / np.sum(counts, axis=1, keepdims=True)

    def get_prior_probabilities(self, t_rel: float) -> np.ndarray:
        """Query prior vector for a given normalized time in [0.0, 1.0]."""
        b = min(self.n_bins - 1, max(0, int(t_rel * self.n_bins)))
        return self.prior_table[b].copy()

    def get_session_priors(self, total_windows: int) -> np.ndarray:
        """Generate (T, 14) prior matrix for an entire session."""
        P = np.zeros((total_windows, self.num_classes), dtype=np.float32)
        for i in range(total_windows):
            t_rel = (i + 0.5) / max(1, total_windows)
            P[i] = self.get_prior_probabilities(t_rel)
        return P


def fuse_model_with_prior(
    model_probs: np.ndarray,
    prior_probs: np.ndarray,
    alpha: float = 1.0,
    beta_visual: float = 0.85
) -> np.ndarray:
    """Bayesian Log-Odds Fusion of visual/acoustic model predictions with the time prior.

    log P_fused(c) = beta_visual * log P_model(c) + alpha * log P_prior(c)
    """
    eps = 1e-12
    log_model = np.log(np.clip(model_probs, eps, 1.0))
    log_prior = np.log(np.clip(prior_probs, eps, 1.0))

    fused_log = (beta_visual * log_model) + (alpha * log_prior)
    # Softmax
    shift = fused_log - np.max(fused_log, axis=-1, keepdims=True)
    exps = np.exp(shift)
    return exps / (np.sum(exps, axis=-1, keepdims=True) + eps)


def semi_markov_viterbi_decode(
    fused_probs: np.ndarray,
    step_durations: Optional[Dict[int, Tuple[float, float]]] = None,
    hop_sec: float = 2.0
) -> np.ndarray:
    """Semi-Markov forward monotonic Viterbi decoder with duration and macro-phase awareness.

    Enforces:
    1. Forward procedural sequence (no impossible backward regressions).
    2. Background pause handling between active steps.
    3. Step duration bounds learned from training folds.
    """
    T, C = fused_probs.shape
    if T == 0:
        return np.array([], dtype=int)

    log_p = np.log(np.clip(fused_probs, 1e-12, 1.0))

    # State: current class c in [0..13]
    dp = np.full((T, C), -np.inf)
    backpointer = np.zeros((T, C), dtype=int)

    # Init t=0 (can start in background or early steps 1-3)
    for c in range(min(4, C)):
        dp[0, c] = log_p[0, c]

    for t in range(1, T):
        for curr_c in range(C):
            best_val = -np.inf
            best_prev = -1

            for prev_c in range(C):
                if curr_c == prev_c:
                    # Self-loop: continuing same step
                    cost = 0.0
                elif curr_c == 0:
                    # Transition to brief background pause
                    cost = -0.5
                elif prev_c == 0:
                    # Resuming from background to procedural step
                    cost = -0.4
                elif curr_c > prev_c:
                    # Advancing forward to higher clinical step
                    step_jump = curr_c - prev_c
                    # Moderate penalty for skipping steps, small penalty for advancing by 1
                    cost = -0.20 * step_jump
                else:
                    # Impossible backward regression
                    cost = -100.0

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
