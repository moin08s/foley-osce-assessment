"""Unit Tests for Male Foley's Catheterization OSCE Assessment System.

Tests all modules with synthetic dummy data:
- src.data_loader
- src.features
- src.model
- src.session_checker
- src.grading
"""

import os
import tempfile
import unittest
import wave

import numpy as np

from src.data_loader import (
    STEP_LABELS,
    label_window,
    load_wav_audio,
    slice_windows,
)
from src.features import (
    build_mel_filterbank,
    compute_log_mel_features,
    compute_short_time_energy,
    compute_spectral_flux,
    compute_stft,
    extract_audio_features,
    extract_video_features,
    extract_window_features,
)
from src.grading import OSCEGrader
from src.model import FoleyStepClassifier, StandardScaler
from src.session_checker import SessionChecker
from src.video_slicer import audit_step_defects


class TestDataLoader(unittest.TestCase):

    def test_audio_loader_and_slicing(self):
        # Generate a 4-second 16kHz mono sine wave in temporary WAV file
        sr = 16000
        duration = 4.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        samples = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        try:
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(samples.tobytes())

            loaded_audio, loaded_sr = load_wav_audio(wav_path)
            self.assertEqual(loaded_sr, 16000)
            self.assertAlmostEqual(len(loaded_audio) / loaded_sr, 4.0, places=2)
            self.assertTrue(-1.0 <= loaded_audio.min() <= loaded_audio.max() <= 1.0)

            # Test slicing
            dummy_video = np.zeros((16, 64, 64, 3), dtype=np.uint8)  # 4s at 4fps = 16 frames
            windows = slice_windows(loaded_audio, sr, dummy_video, 4.0, window_sec=2.0, hop_sec=1.0)
            self.assertGreater(len(windows), 0)
            self.assertEqual(windows[0]["audio"].shape[0], int(2.0 * sr))
            self.assertEqual(windows[0]["video"].shape, (8, 64, 64, 3))

        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    def test_label_window(self):
        manifest = [
            {"step_id": 1, "start_sec": 0.0, "end_sec": 10.0},
            {"step_id": 2, "start_sec": 10.0, "end_sec": 20.0},
        ]
        self.assertEqual(label_window(1.0, 5.0, manifest), 1)
        self.assertEqual(label_window(11.0, 15.0, manifest), 2)
        self.assertEqual(label_window(25.0, 28.0, manifest), 0)  # Background


class TestFeatures(unittest.TestCase):

    def test_video_features_shape(self):
        frames = np.random.randint(0, 255, size=(16, 64, 64, 3), dtype=np.uint8)
        feat = extract_video_features(frames)
        self.assertEqual(feat.shape, (181,))
        self.assertFalse(np.isnan(feat).any())

    def test_audio_features_shape(self):
        sr = 16000
        audio = np.random.randn(sr * 4).astype(np.float32) * 0.1
        feat = extract_audio_features(audio, sr=sr)
        self.assertEqual(feat.shape, (46,))
        self.assertFalse(np.isnan(feat).any())

    def test_fused_features_shape(self):
        frames = np.zeros((16, 64, 64, 3), dtype=np.uint8)
        audio = np.zeros(16000 * 4, dtype=np.float32)
        fused = extract_window_features(frames, audio, audio_sr=16000)
        self.assertEqual(fused.shape, (227,))


class TestModel(unittest.TestCase):

    def test_standard_scaler(self):
        X = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]], dtype=np.float32)
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
        np.testing.assert_allclose(X_norm.mean(axis=0), [0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(X_norm.std(axis=0), [1.0, 1.0], atol=1e-5)

    def test_model_training_and_serialization(self):
        X = np.random.randn(40, 227).astype(np.float32)
        y = np.random.randint(0, 14, size=(40,)).astype(np.int64)

        clf = FoleyStepClassifier(input_dim=227, hidden_dim1=32, hidden_dim2=16, num_classes=14)
        history = clf.fit(X, y, epochs=10, batch_size=8, lr=1e-2, verbose=False)
        self.assertLess(history["loss"][-1], history["loss"][0])

        preds = clf.predict(X)
        self.assertEqual(preds.shape, (40,))

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp_path = f.name
        try:
            clf.save(tmp_path)
            loaded_clf = FoleyStepClassifier.load(tmp_path)
            loaded_preds = loaded_clf.predict(X)
            np.testing.assert_array_equal(preds, loaded_preds)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestSessionCheckerAndGrading(unittest.TestCase):

    def setUp(self):
        self.checker = SessionChecker()
        self.grader = OSCEGrader()

    def test_perfect_session_grade_P(self):
        segments = [
            {"step_id": s, "step_label": STEP_LABELS[s], "start_sec": (s - 1) * 10.0, "end_sec": s * 10.0, "duration_sec": 10.0, "confidence": 0.98}
            for s in range(1, 14)
        ]
        speech = {1: 0.8, 2: 0.7}
        audit = self.checker.check_sequence_rules(segments, speech_activity_per_step=speech)
        report = self.grader.compute_grade(audit, "Student01", "sess01", segments)

        self.assertEqual(report["final_grade"], "P")
        self.assertTrue(report["all_13_present"])
        self.assertEqual(len(report["critical_safety_violations"]), 0)

    def test_prompted_step_grade_P_minus_1(self):
        segments = [
            {"step_id": s, "step_label": STEP_LABELS[s], "start_sec": (s - 1) * 10.0, "end_sec": s * 10.0, "duration_sec": 10.0, "confidence": 0.98}
            for s in range(1, 14)
        ]
        speech = {1: 0.8, 2: 0.7}
        # One prompted step (e.g. Step 12 reminded by faculty)
        audit = self.checker.check_sequence_rules(segments, speech_activity_per_step=speech, prompts_given=[12])
        report = self.grader.compute_grade(audit, "Student02", "sess02", segments)

        self.assertEqual(report["final_grade"], "P-1")

    def test_safety_violation_grade_R(self):
        # Step 10 (balloon inflate) before Step 9 (urine confirm)
        segments = [
            {"step_id": s if s not in (9, 10) else (10 if s == 9 else 9), "step_label": "step", "start_sec": (s - 1) * 10.0, "end_sec": s * 10.0, "duration_sec": 10.0, "confidence": 0.98}
            for s in range(1, 14)
        ]
        speech = {1: 0.8, 2: 0.7}
        audit = self.checker.check_sequence_rules(segments, speech_activity_per_step=speech)
        report = self.grader.compute_grade(audit, "Student03", "sess03", segments)

        self.assertEqual(report["final_grade"], "R")
        self.assertGreater(len(report["critical_safety_violations"]), 0)

    def test_video_slicer_defect_audit(self):
        # Normal step
        aud_safe = audit_step_defects(1, "introduce_explain", 8.0, list(range(1, 14)), speech_vad_ratio=0.2)
        self.assertFalse(aud_safe["has_defect"])

        # Balloon before urine flow defect
        aud_defect = audit_step_defects(10, "inflate_balloon", 8.0, [1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 11, 12, 13])
        self.assertTrue(aud_defect["has_defect"])
        self.assertEqual(aud_defect["defect_level"], "CRITICAL")


if __name__ == "__main__":
    unittest.main()
