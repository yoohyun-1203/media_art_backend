from __future__ import annotations

import numpy as np


def _clamp(value, low=-1.0, high=1.0):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0

    if not np.isfinite(numeric):
        numeric = 0.0
    return max(float(low), min(float(high), numeric))


def normalize_ser_prediction(prediction, arousal_hint: float = 0.0):
    prediction = prediction or {}
    label = prediction.get("label", "unknown")
    if label is None:
        label = "unknown"

    return {
        "valence": _clamp(prediction.get("valence", 0.0)),
        "arousal": _clamp(prediction.get("arousal", arousal_hint)),
        "confidence": _clamp(prediction.get("confidence", 0.0), 0.0, 1.0),
        "label": str(label),
    }


class RollingSerWindow:
    def __init__(self, rate: int = 16000, seconds: float = 1.0):
        self.rate = rate
        self.seconds = seconds
        self.target_samples = int(rate * seconds)
        self._buffer = np.empty(0, dtype=np.float32)

    def push(self, pcm: np.ndarray):
        samples = np.asarray(pcm)
        if samples.ndim > 1:
            samples = samples[:, 0]

        samples = samples.astype(np.float32, copy=False).reshape(-1)
        if samples.size and np.max(np.abs(samples)) > 1.5:
            samples = samples / 32768.0

        self._buffer = np.concatenate((self._buffer, samples))[-self.target_samples :]
        if self._buffer.shape[0] < self.target_samples:
            return None

        return self._buffer.copy()

    def add(self, data: np.ndarray):
        return self.push(data)


class LocalSerFallback:
    def predict(self, samples: np.ndarray, arousal_hint: float = 0.0):
        return {
            "valence": 0.0,
            "arousal": float(arousal_hint),
            "confidence": 0.0,
            "label": "unknown",
        }


class LocalSerModelAdapter:
    def __init__(self, load_model=None, fallback=None):
        self._load_model = load_model
        self._fallback = fallback or LocalSerFallback()
        self._model = None
        self._loaded = False
        self.last_error = None

    def _get_model(self):
        if self._load_model is None:
            return None
        if not self._loaded:
            self._model = self._load_model()
            self._loaded = True
        return self._model

    def predict(self, samples: np.ndarray, arousal_hint: float = 0.0):
        try:
            model = self._get_model()
            if model is None:
                return self._fallback.predict(samples, arousal_hint=arousal_hint)

            if hasattr(model, "predict"):
                try:
                    prediction = model.predict(samples, arousal_hint=arousal_hint)
                except TypeError:
                    prediction = model.predict(samples)
            else:
                try:
                    prediction = model(samples, arousal_hint=arousal_hint)
                except TypeError:
                    prediction = model(samples)

            self.last_error = None
            return normalize_ser_prediction(prediction, arousal_hint=arousal_hint)
        except Exception as exc:
            self.last_error = exc
            return self._fallback.predict(samples, arousal_hint=arousal_hint)


class LocalSerRuntime:
    def __init__(self, window=None, model=None, fallback=None, rate: int = 16000, seconds: float = 1.0):
        self.window = window or RollingSerWindow(rate=rate, seconds=seconds)
        self.fallback = fallback or LocalSerFallback()
        self.model = model or LocalSerModelAdapter(fallback=self.fallback)

    def process(self, pcm, arousal_hint: float = 0.0):
        try:
            samples = self.window.push(pcm)
        except (TypeError, ValueError):
            samples = None

        if samples is None:
            return self.fallback.predict(np.zeros(0, dtype=np.float32), arousal_hint=arousal_hint)

        return self.model.predict(samples, arousal_hint=arousal_hint)
