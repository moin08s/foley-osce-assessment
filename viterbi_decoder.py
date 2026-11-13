import numpy as np

def viterbi_procedural_decode(window_probs: np.ndarray, min_duration_windows: int = 2) -> np.ndarray:
    """Dynamic programming Viterbi decoder enforcing forward procedural progression.
    
    Prevents spurious backward transitions (e.g., Step 10 before Step 9)
    while allowing steps to be skipped if genuinely omitted by the student.
    """
    n_windows, n_classes = window_probs.shape
    if n_windows == 0:
        return np.array([], dtype=int)
        
    # Log probabilities with numerical safety
    log_probs = np.log(np.clip(window_probs, 1e-12, 1.0))
    
    # dp[t, c] = max log-prob up to window t ending in class c
    # backpointer[t, c] = previous class
    dp = np.full((n_windows, n_classes), -np.inf)
    backpointer = np.zeros((n_windows, n_classes), dtype=int)
    
    # Initialize t = 0
    # Steps 0, 1, 2 are valid initial states
    for c in range(min(4, n_classes)):
        dp[0, c] = log_probs[0, c]
        
    # Transition penalties:
    # Staying in same state: 0 penalty
    # Advancing forward to higher state: small penalty to prefer continuity
    # Transitioning backward: heavy penalty (-1000)
    for t in range(1, n_windows):
        for curr_c in range(n_classes):
            best_prev_c = -1
            best_val = -np.inf
            
            for prev_c in range(n_classes):
                # Transition logic
                if curr_c == prev_c:
                    # Same state
                    trans_cost = 0.0
                elif curr_c == 0:
                    # Transition to background pause
                    trans_cost = -0.5
                elif prev_c == 0:
                    # From background to procedural step
                    trans_cost = -0.5
                elif curr_c > prev_c:
                    # Forward procedural progression
                    trans_cost = -0.2 * (curr_c - prev_c)
                else:
                    # Backward transition (forbidden in clinical flow)
                    trans_cost = -100.0
                    
                score = dp[t - 1, prev_c] + trans_cost + log_probs[t, curr_c]
                if score > best_val:
                    best_val = score
                    best_prev_c = prev_c
                    
            dp[t, curr_c] = best_val
            backpointer[t, curr_c] = best_prev_c
            
    # Traceback
    best_last = int(np.argmax(dp[n_windows - 1]))
    path = np.zeros(n_windows, dtype=int)
    path[n_windows - 1] = best_last
    
    for t in range(n_windows - 2, -1, -1):
        path[t] = backpointer[t + 1, path[t + 1]]
        
    return path

print("Viterbi procedural decoder ready.")
