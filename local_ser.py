from __future__ import annotations

import numpy as np


SER_MODEL_RATE = 16000
DEFAULT_LABEL_VALENCE = {
    "ang": -0.8,
    "anger": -0.8,
    "angry": -0.8,
    "disgust": -0.7,
    "fear": -0.7,
    "sad": -0.7,
    "sadness": -0.7,
    "neutral": 0.0,
    "neu": 0.0,
    "calm": 0.2,
    "hap": 0.8,
    "happy": 0.8,
    "happiness": 0.8,
    "joy": 0.8,
    "surprise": 0.3,
}


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


class HuggingFaceAudioSerModel:
    def __init__(self, model_id: str, classifier=None, label_valence=None, input_rate: int = SER_MODEL_RATE):
        self.model_id = str(model_id or "").strip()
        self._classifier = classifier
        self.label_valence = {**DEFAULT_LABEL_VALENCE, **(label_valence or {})}
        self.input_rate = int(input_rate)

    def _get_classifier(self):
        if self._classifier is None:
            from transformers import pipeline

            self._classifier = pipeline(
                "audio-classification",
                model=self.model_id,
                top_k=1,
            )
        return self._classifier

    def predict(self, samples: np.ndarray, arousal_hint: float = 0.0):
        classifier = self._get_classifier()
        raw = classifier({
            "raw": np.asarray(samples, dtype=np.float32),
            "sampling_rate": self.input_rate,
        })
        if raw and isinstance(raw[0], list):
            raw = raw[0]
        top = raw[0] if raw else {}
        label = str(top.get("label", "unknown"))
        return {
            "valence": self.label_valence.get(label.strip().lower(), 0.0),
            "arousal": float(arousal_hint),
            "confidence": float(top.get("score", 0.0)),
            "label": label,
        }


class AudeeringDimensionalSerModel:
    """Direct arousal/valence regression model from audEERING's MSP-Dim model."""

    def __init__(self, model_id: str, input_rate: int = SER_MODEL_RATE, processor=None, model=None):
        self.model_id = str(model_id or "").strip()
        self.input_rate = int(input_rate)
        self._processor = processor
        self._model = model

    def _load(self):
        if self._processor is not None and self._model is not None:
            return self._processor, self._model

        import torch
        import torch.nn as nn
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import (
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )

        class RegressionHead(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

            def forward(self, features, **kwargs):
                x = self.dropout(features)
                x = self.dense(x)
                x = torch.tanh(x)
                x = self.dropout(x)
                return self.out_proj(x)

        class EmotionModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.wav2vec2 = Wav2Vec2Model(config)
                self.classifier = RegressionHead(config)
                self.init_weights()

            def forward(self, input_values):
                outputs = self.wav2vec2(input_values)
                hidden_states = outputs[0]
                hidden_states = torch.mean(hidden_states, dim=1)
                return hidden_states, self.classifier(hidden_states)

        self._processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self._model = EmotionModel.from_pretrained(self.model_id)
        self._model.eval()
        return self._processor, self._model

    def predict(self, samples: np.ndarray, arousal_hint: float = 0.0):
        import torch

        processor, model = self._load()
        processed = processor(np.asarray(samples, dtype=np.float32), sampling_rate=self.input_rate)
        input_values = torch.from_numpy(np.asarray(processed["input_values"][0], dtype=np.float32)).reshape(1, -1)
        with torch.no_grad():
            _embedding, logits = model(input_values)
        raw = logits.detach().cpu().numpy()[0]
        # audEERING outputs approximately 0..1; map to signed live range.
        arousal = _clamp((float(raw[0]) * 2.0) - 1.0)
        valence = _clamp((float(raw[2]) * 2.0) - 1.0)
        confidence = max(0.0, min(1.0, abs(valence)))
        return {
            "valence": valence,
            "arousal": arousal,
            "confidence": confidence,
            "label": "dimensional",
        }


def build_local_ser_model(model_id: str | None, input_rate: int = SER_MODEL_RATE, backend: str = "classification"):
    model_id = str(model_id or "").strip()
    if not model_id:
        return None
    if str(backend or "").strip().lower() == "audeering_dimensional":
        return LocalSerModelAdapter(load_model=lambda: AudeeringDimensionalSerModel(model_id, input_rate=input_rate))
    return LocalSerModelAdapter(load_model=lambda: HuggingFaceAudioSerModel(model_id, input_rate=input_rate))


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
