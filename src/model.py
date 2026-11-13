"""Pure NumPy Neural Network Classifier for OSCE Assessment.

D Y Patil Medical College - Procedural Skills Assessment Model
Multi-Layer Perceptron (MLP) trained from scratch with Adam optimizer.
Strict constraint: Python Standard Library + NumPy ONLY.
Zero PyTorch, TensorFlow, or scikit-learn dependencies.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class StandardScaler:
    """Standardizes features by removing the mean and scaling to unit variance (pure NumPy)."""

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.mean: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean = np.mean(X, axis=0, keepdims=True)
        var = np.var(X, axis=0, keepdims=True)
        self.scale = np.sqrt(var) + self.epsilon
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("StandardScaler must be fitted before transforming data.")
        return (X - self.mean) / self.scale

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class FoleyStepClassifier:
    """3-Layer MLP Classifier for 14-way OSCE Step Classification (0=bg, 1-13=steps).
    
    Architecture:
        Input: D_in (default 227)
        Dense: 128 units + ReLU
        Dense: 64 units + ReLU
        Dense: 14 units + Numerically stable Softmax
    """

    def __init__(
        self,
        input_dim: int = 227,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        num_classes: int = 14,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.num_classes = num_classes
        self.scaler = StandardScaler()

        # He (Kaiming) normal initialization
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(input_dim, hidden_dim1).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim1), dtype=np.float32)

        self.W2 = rng.randn(hidden_dim1, hidden_dim2).astype(np.float32) * np.sqrt(2.0 / hidden_dim1)
        self.b2 = np.zeros((1, hidden_dim2), dtype=np.float32)

        self.W3 = rng.randn(hidden_dim2, num_classes).astype(np.float32) * np.sqrt(2.0 / hidden_dim2)
        self.b3 = np.zeros((1, num_classes), dtype=np.float32)

        # Adam optimizer state
        self._m = {}
        self._v = {}
        self._t = 0
        for name in ["W1", "b1", "W2", "b2", "W3", "b3"]:
            param = getattr(self, name)
            self._m[name] = np.zeros_like(param)
            self._v[name] = np.zeros_like(param)

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def _relu_grad(z: np.ndarray) -> np.ndarray:
        return (z > 0.0).astype(np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax along axis -1."""
        shift_logits = logits - np.max(logits, axis=-1, keepdims=True)
        exps = np.exp(shift_logits)
        return exps / (np.sum(exps, axis=-1, keepdims=True) + 1e-9)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Forward pass through 3-layer MLP.
        
        Args:
            X: Normalized inputs of shape (N, input_dim)
            
        Returns:
            probs: Class probabilities of shape (N, num_classes)
            cache: Intermediate activations for backpropagation
        """
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self._relu(Z1)

        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self._relu(Z2)

        Z3 = np.dot(A2, self.W3) + self.b3
        probs = self._softmax(Z3)

        cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2,
            "Z3": Z3,
            "probs": probs,
        }
        return probs, cache

    def compute_loss(self, probs: np.ndarray, y: np.ndarray, l2_reg: float = 1e-4) -> float:
        """Compute Categorical Cross-Entropy Loss with L2 regularization."""
        N = len(y)
        correct_logprobs = -np.log(probs[np.arange(N), y] + 1e-9)
        data_loss = float(np.mean(correct_logprobs))
        reg_loss = 0.5 * l2_reg * (
            float(np.sum(self.W1 ** 2)) +
            float(np.sum(self.W2 ** 2)) +
            float(np.sum(self.W3 ** 2))
        )
        return data_loss + reg_loss

    def backward(self, cache: Dict[str, np.ndarray], y: np.ndarray, l2_reg: float = 1e-4) -> Dict[str, np.ndarray]:
        """Compute gradients via backpropagation."""
        N = len(y)
        probs = cache["probs"].copy()
        probs[np.arange(N), y] -= 1.0
        dZ3 = probs / N  # (N, num_classes)

        dW3 = np.dot(cache["A2"].T, dZ3) + l2_reg * self.W3
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * self._relu_grad(cache["Z2"])
        dW2 = np.dot(cache["A1"].T, dZ2) + l2_reg * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self._relu_grad(cache["Z1"])
        dW1 = np.dot(cache["X"].T, dZ1) + l2_reg * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return {
            "W1": dW1, "b1": db1,
            "W2": dW2, "b2": db2,
            "W3": dW3, "b3": db3,
        }

    def _adam_step(
        self,
        grads: Dict[str, np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Single Adam optimization step."""
        self._t += 1
        for name in ["W1", "b1", "W2", "b2", "W3", "b3"]:
            g = grads[name]
            self._m[name] = beta1 * self._m[name] + (1.0 - beta1) * g
            self._v[name] = beta2 * self._v[name] + (1.0 - beta2) * (g ** 2)

            # Bias correction
            m_hat = self._m[name] / (1.0 - beta1 ** self._t)
            v_hat = self._v[name] / (1.0 - beta2 ** self._t)

            param = getattr(self, name)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, name, param)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        l2_reg: float = 1e-4,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train classifier with mini-batch Adam optimization."""
        # Fit scaler on training data
        X_norm = self.scaler.fit_transform(X)

        if val_data is not None:
            X_val, y_val = val_data
            X_val_norm = self.scaler.transform(X_val)

        N = len(X)
        history: Dict[str, List[float]] = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(N)
            X_shuffled = X_norm[indices]
            y_shuffled = y[indices]

            epoch_losses = []
            for start_idx in range(0, N, batch_size):
                end_idx = min(start_idx + batch_size, N)
                xb = X_shuffled[start_idx:end_idx]
                yb = y_shuffled[start_idx:end_idx]

                probs, cache = self.forward(xb)
                loss = self.compute_loss(probs, yb, l2_reg=l2_reg)
                epoch_losses.append(loss)

                grads = self.backward(cache, yb, l2_reg=l2_reg)
                self._adam_step(grads, lr=lr)

            mean_train_loss = float(np.mean(epoch_losses))
            train_preds = self.predict_from_norm(X_norm)
            train_acc = float(np.mean(train_preds == y))

            history["loss"].append(mean_train_loss)
            history["accuracy"].append(train_acc)

            if val_data is not None:
                val_probs, _ = self.forward(X_val_norm)
                val_loss = self.compute_loss(val_probs, y_val, l2_reg=l2_reg)
                val_preds = np.argmax(val_probs, axis=1)
                val_acc = float(np.mean(val_preds == y_val))
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)

                if verbose and (epoch % 10 == 0 or epoch == epochs):
                    print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {mean_train_loss:.4f} Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%")
            else:
                if verbose and (epoch % 10 == 0 or epoch == epochs):
                    print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {mean_train_loss:.4f} Acc: {train_acc*100:.1f}%")

        return history

    def predict_from_norm(self, X_norm: np.ndarray) -> np.ndarray:
        """Internal helper for argmax predictions on already normalized features."""
        probs, _ = self.forward(X_norm)
        return np.argmax(probs, axis=-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict 14-way class probability distribution for input feature matrix X."""
        X_norm = self.scaler.transform(X)
        probs, _ = self.forward(X_norm)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict most likely class label integer (0-13) for input feature matrix X."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=-1)

    def save(self, model_path: str) -> None:
        """Save model weights and feature scaler parameters to compressed .npz archive."""
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        np.savez_compressed(
            model_path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            W3=self.W3, b3=self.b3,
            scaler_mean=self.scaler.mean if self.scaler.mean is not None else np.zeros((1, self.input_dim)),
            scaler_scale=self.scaler.scale if self.scaler.scale is not None else np.ones((1, self.input_dim)),
            input_dim=self.input_dim,
            hidden_dim1=self.hidden_dim1,
            hidden_dim2=self.hidden_dim2,
            num_classes=self.num_classes,
        )

    @classmethod
    def load(cls, model_path: str) -> "FoleyStepClassifier":
        """Load trained model weights and feature scaler parameters from .npz file."""
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        data = np.load(model_path)
        input_dim = int(data["input_dim"])
        hidden_dim1 = int(data["hidden_dim1"])
        hidden_dim2 = int(data["hidden_dim2"])
        num_classes = int(data["num_classes"])

        model = cls(
            input_dim=input_dim,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            num_classes=num_classes,
        )
        model.W1 = data["W1"]
        model.b1 = data["b1"]
        model.W2 = data["W2"]
        model.b2 = data["b2"]
        model.W3 = data["W3"]
        model.b3 = data["b3"]
        model.scaler.mean = data["scaler_mean"]
        model.scaler.scale = data["scaler_scale"]
        return model
