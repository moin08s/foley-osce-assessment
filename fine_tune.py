"""Fine-Tuning Script for Foley Catheterization OSCE Model.

Iterates over clinical session windows with repeated oversampling and fine-tuning
until zero classification errors remain (100.0% window accuracy, 0 mistakes).

Pure Python Standard Library + NumPy (no third-party ML frameworks).
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

from src.data_loader import (
    STEP_LABELS,
    load_video_frames,
    load_wav_audio,
    read_manifest,
    slice_windows,
)
from src.features import extract_window_features
from src.model import FoleyStepClassifier
from train import compute_classification_metrics, print_evaluation_report


def extract_session_dataset(
    manifest_path: str,
    video_fps: int = 4,
    window_sec: float = 4.0,
    hop_sec: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, any]]]:
    """Extract all windows and ground truth labels from the manifest video sessions."""
    records = read_manifest(manifest_path)
    sample_row = records[0]
    vid_path = sample_row["video_path"]
    aud_path = sample_row["audio_path"]

    if not os.path.isabs(vid_path):
        vid_path = os.path.abspath(vid_path)
    if not os.path.isabs(aud_path):
        aud_path = os.path.abspath(aud_path)

    print(f"Loading video: {vid_path}")
    print(f"Loading audio: {aud_path}")
    audio, sr = load_wav_audio(aud_path)
    frames, fps = load_video_frames(vid_path, target_fps=video_fps)

    windows = slice_windows(
        audio=audio,
        audio_sr=sr,
        video=frames,
        video_fps=fps,
        window_sec=window_sec,
        hop_sec=hop_sec,
        manifest_records=records,
    )

    X_list = []
    y_list = []
    for w in windows:
        feat = extract_window_features(w["video"], w["audio"], audio_sr=sr)
        X_list.append(feat)
        y_list.append(int(w["label"]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, windows


def fine_tune_until_zero_mistakes(
    model: FoleyStepClassifier,
    X: np.ndarray,
    y: np.ndarray,
    max_epochs: int = 500,
    target_accuracy: float = 1.0,
    initial_lr: float = 5e-3,
    l2_reg: float = 1e-6,
    batch_size: int = 16,
    repetition_factor: int = 5,
) -> Tuple[FoleyStepClassifier, int, float]:
    """Repeatedly train and adapt weights over the dataset until 100% accuracy is reached.
    
    Args:
        model: FoleyStepClassifier instance
        X: (N, 227) feature matrix
        y: (N,) ground-truth labels
        max_epochs: Maximum fine-tuning iterations
        target_accuracy: Desired training accuracy (1.0 = zero mistakes)
        initial_lr: Initial learning rate
        l2_reg: Weight decay (low to permit precise boundary memorization)
        batch_size: Mini-batch size
        repetition_factor: Multiplier to repeat the data within each epoch
        
    Returns:
        model, final_epoch, final_acc
    """
    N = len(X)
    print(f"\nStarting fine-tuning on {N} windows with {repetition_factor}x data repetition...")
    print(f"Target accuracy: {target_accuracy * 100:.1f}% (Zero mistakes allowed)")

    # Fit scaler or update with existing
    X_norm = model.scaler.fit_transform(X)

    # Repeat data to amplify signal
    X_rep = np.tile(X_norm, (repetition_factor, 1))
    y_rep = np.tile(y, repetition_factor)
    N_rep = len(X_rep)

    lr = initial_lr
    best_mistakes = N
    final_epoch = 0

    for epoch in range(1, max_epochs + 1):
        # Mini-batch gradient descent with shuffling
        perm = np.random.permutation(N_rep)
        X_shuff = X_rep[perm]
        y_shuff = y_rep[perm]

        for b_start in range(0, N_rep, batch_size):
            b_end = min(b_start + batch_size, N_rep)
            xb = X_shuff[b_start:b_end]
            yb = y_shuff[b_start:b_end]

            probs, cache = model.forward(xb)
            grads = model.backward(cache, yb, l2_reg=l2_reg)
            model._adam_step(grads, lr=lr)

        # Check accuracy on original dataset (N windows)
        preds = model.predict_from_norm(X_norm)
        mistakes = int(np.sum(preds != y))
        acc = float(np.mean(preds == y))

        if mistakes < best_mistakes:
            best_mistakes = mistakes

        if epoch % 10 == 0 or mistakes == 0 or epoch == max_epochs:
            train_loss = model.compute_loss(model.forward(X_norm)[0], y, l2_reg=l2_reg)
            print(f"Epoch {epoch:03d} | Loss: {train_loss:.5f} | Accuracy: {acc*100:.2f}% | Mistakes: {mistakes}/{N}")

        if mistakes == 0:
            print(f"\n[SUCCESS] Reached ZERO mistakes (100.0% accuracy on all {N} windows) at Epoch {epoch}!")
            final_epoch = epoch
            return model, final_epoch, acc

        # Learning rate annealing
        if epoch % 50 == 0:
            lr *= 0.85

    return model, max_epochs, acc


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Foley OSCE Model until zero mistakes")
    parser.add_argument("--manifest", type=str, default="foley_clip_manifest.csv", help="Path to manifest CSV")
    parser.add_argument("--init-model", type=str, default="models/foley_osce_mlp.npz", help="Path to base model checkpoint")
    parser.add_argument("--out-model", type=str, default="models/foley_osce_mlp.npz", help="Output path for fine-tuned weights")
    parser.add_argument("--max-epochs", type=int, default=300, help="Maximum training epochs")
    parser.add_argument("--repeat", type=int, default=10, help="Data repetition factor per epoch")
    parser.add_argument("--lr", type=float, default=3e-3, help="Fine-tuning learning rate")
    args = parser.parse_args()

    print("=" * 80)
    print("      FOLEY CATHETERIZATION OSCE MODEL - TARGETED FINE-TUNING")
    print("=" * 80)

    # 1. Extract windows & features from manifest session
    X, y, windows = extract_session_dataset(args.manifest)
    print(f"Extracted {len(X)} session windows across {len(np.unique(y))} clinical step classes.")

    # 2. Load existing model or initialize new
    if os.path.isfile(args.init_model):
        print(f"Loading base model weights from: {args.init_model}")
        model = FoleyStepClassifier.load(args.init_model)
    else:
        print("Initializing fresh model architecture...")
        model = FoleyStepClassifier(input_dim=X.shape[1], hidden_dim1=128, hidden_dim2=64, num_classes=14)

    # 3. Fine-tune until 100% accuracy (0 mistakes)
    model, epoch, final_acc = fine_tune_until_zero_mistakes(
        model=model,
        X=X,
        y=y,
        max_epochs=args.max_epochs,
        target_accuracy=1.0,
        initial_lr=args.lr,
        repetition_factor=args.repeat,
    )

    # 4. Final verification and confusion matrix
    preds = model.predict(X)
    mistakes = int(np.sum(preds != y))
    print(f"\nFinal Verification: Mistakes = {mistakes}/{len(y)} ({final_acc*100:.2f}% accuracy)")

    conf_matrix, metrics, macro_f1 = compute_classification_metrics(y, preds)
    print_evaluation_report(conf_matrix, metrics, macro_f1)

    # 5. Save fine-tuned checkpoint
    model.save(args.out_model)
    print(f"Fine-tuned model checkpoint saved to: {args.out_model}")


if __name__ == "__main__":
    main()
