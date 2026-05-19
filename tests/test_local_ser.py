import builtins
import importlib
import sys
import unittest
from unittest import mock

import numpy as np

from local_ser import (
    AudeeringDimensionalSerModel,
    HuggingFaceAudioSerModel,
    LocalSerFallback,
    LocalSerModelAdapter,
    LocalSerRuntime,
    RollingSerWindow,
    build_local_ser_model,
    valence_for_label,
)


class RollingSerWindowTests(unittest.TestCase):
    def test_window_returns_none_until_enough_audio_is_available(self):
        window = RollingSerWindow(rate=16000, seconds=1.0)
        samples = np.zeros(1024, dtype=np.float32)

        result = window.push(samples)

        self.assertIsNone(result)

    def test_window_returns_audio_after_one_second(self):
        window = RollingSerWindow(rate=16000, seconds=1.0)
        samples = np.zeros(16000, dtype=np.float32)

        result = window.push(samples)

        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 16000)

    def test_window_uses_first_channel_for_multichannel_input(self):
        window = RollingSerWindow(rate=4, seconds=1.0)
        samples = np.array(
            [
                [0.1, 0.9],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.4, 0.6],
            ],
            dtype=np.float32,
        )

        result = window.push(samples)

        np.testing.assert_allclose(result, np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))

    def test_window_normalizes_int_like_pcm(self):
        window = RollingSerWindow(rate=4, seconds=1.0)
        samples = np.array([0, 16384, -32768, 32767], dtype=np.int16)

        result = window.push(samples)

        np.testing.assert_allclose(
            result,
            np.array([0.0, 0.5, -1.0, 32767 / 32768.0], dtype=np.float32),
        )

    def test_window_overflow_keeps_latest_audio(self):
        window = RollingSerWindow(rate=4, seconds=1.0)

        self.assertIsNone(window.push(np.array([0.1, 0.2], dtype=np.float32)))
        result = window.push(np.array([0.3, 0.4, 0.5], dtype=np.float32))

        np.testing.assert_allclose(result, np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32))

    def test_window_returns_copy_not_internal_buffer(self):
        window = RollingSerWindow(rate=4, seconds=1.0)

        first_result = window.push(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
        first_result[0] = 99.0
        second_result = window.push(np.array([], dtype=np.float32))

        np.testing.assert_allclose(
            second_result,
            np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        )


class LocalSerFallbackTests(unittest.TestCase):
    def test_predict_returns_unknown_with_arousal_hint(self):
        fallback = LocalSerFallback()

        result = fallback.predict(np.zeros(16000, dtype=np.float32), arousal_hint=0.25)

        self.assertEqual(
            result,
            {
                "valence": 0.0,
                "arousal": 0.25,
                "confidence": 0.0,
                "label": "unknown",
            },
        )


class LocalSerModelAdapterTests(unittest.TestCase):
    def test_adapter_does_not_load_model_until_prediction(self):
        load_calls = []

        class DummyModel:
            def predict(self, samples, arousal_hint=0.0):
                return {
                    "valence": 0.35,
                    "arousal": arousal_hint,
                    "confidence": 0.8,
                    "label": "calm",
                }

        def load_model():
            load_calls.append("loaded")
            return DummyModel()

        adapter = LocalSerModelAdapter(load_model=load_model)

        self.assertEqual(load_calls, [])

        result = adapter.predict(np.zeros(4, dtype=np.float32), arousal_hint=0.25)

        self.assertEqual(load_calls, ["loaded"])
        self.assertEqual(
            result,
            {
                "valence": 0.35,
                "arousal": 0.25,
                "confidence": 0.8,
                "label": "calm",
            },
        )

        adapter.predict(np.zeros(4, dtype=np.float32), arousal_hint=0.25)

        self.assertEqual(load_calls, ["loaded"])

    def test_adapter_uses_fallback_when_no_loader_is_configured(self):
        adapter = LocalSerModelAdapter(load_model=None)

        result = adapter.predict(np.zeros(4, dtype=np.float32), arousal_hint=-0.4)

        self.assertEqual(
            result,
            {
                "valence": 0.0,
                "arousal": -0.4,
                "confidence": 0.0,
                "label": "unknown",
            },
        )

    def test_import_does_not_require_heavy_optional_dependencies(self):
        blocked = {"sounddevice", "librosa", "google", "openai"}
        original_import = builtins.__import__

        def fail_for_blocked(name, *args, **kwargs):
            root = name.split(".", 1)[0]
            if root in blocked:
                raise ModuleNotFoundError(name)
            return original_import(name, *args, **kwargs)

        old_module = sys.modules.pop("local_ser", None)
        try:
            with mock.patch("builtins.__import__", side_effect=fail_for_blocked):
                module = importlib.import_module("local_ser")
            self.assertTrue(hasattr(module, "LocalSerModelAdapter"))
        finally:
            sys.modules.pop("local_ser", None)
            if old_module is not None:
                sys.modules["local_ser"] = old_module


class HuggingFaceAudioSerModelTests(unittest.TestCase):
    def test_predict_maps_emotion_label_to_valence(self):
        classifier = mock.Mock(return_value=[[{"label": "happiness", "score": 0.91}]])
        model = HuggingFaceAudioSerModel("demo/model", classifier=classifier)

        result = model.predict(np.zeros(16000, dtype=np.float32), arousal_hint=0.4)

        self.assertEqual(result["label"], "happiness")
        self.assertEqual(result["valence"], 0.8)
        self.assertEqual(result["arousal"], 0.4)
        self.assertEqual(result["confidence"], 0.91)
        classifier.assert_called_once()
        self.assertEqual(classifier.call_args.args[0]["sampling_rate"], 16000)

    def test_predict_maps_superb_short_labels_to_valence(self):
        classifier = mock.Mock(return_value=[[{"label": "ang", "score": 0.91}]])
        model = HuggingFaceAudioSerModel("demo/model", classifier=classifier)

        result = model.predict(np.zeros(16000, dtype=np.float32), arousal_hint=0.4)

        self.assertEqual(result["label"], "ang")
        self.assertEqual(result["valence"], -0.8)

    def test_sad_label_is_weaker_negative_when_arousal_is_high(self):
        self.assertEqual(valence_for_label("sad", arousal_hint=0.0), -0.7)
        self.assertEqual(valence_for_label("sad", arousal_hint=0.6), -0.2)

    def test_quiet_voice_softens_sad_and_tilts_neutral_positive(self):
        self.assertEqual(valence_for_label("sad", arousal_hint=-0.3), -0.15)
        self.assertEqual(valence_for_label("neu", arousal_hint=-0.3), 0.12)

    def test_predict_passes_actual_input_rate_for_pipeline_resampling(self):
        classifier = mock.Mock(return_value=[[{"label": "sadness", "score": 0.7}]])
        model = HuggingFaceAudioSerModel("demo/model", classifier=classifier, input_rate=48000)

        model.predict(np.zeros(48000, dtype=np.float32), arousal_hint=-0.2)

        payload = classifier.call_args.args[0]
        self.assertEqual(payload["sampling_rate"], 48000)
        self.assertEqual(payload["raw"].shape[0], 48000)

    def test_build_local_ser_model_is_disabled_without_model_id(self):
        self.assertIsNone(build_local_ser_model(""))

    def test_build_local_ser_model_returns_lazy_adapter_with_model_id(self):
        self.assertIsInstance(build_local_ser_model("demo/model"), LocalSerModelAdapter)


class AudeeringDimensionalSerModelTests(unittest.TestCase):
    def test_predict_maps_direct_regression_outputs_to_signed_range(self):
        class FakeProcessor:
            def __call__(self, samples, sampling_rate):
                return {"input_values": [samples]}

        class FakeTensor:
            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.array([[0.8, 0.5, 0.2]], dtype=np.float32)

        class FakeModel:
            def __call__(self, input_values):
                return None, FakeTensor()

        with mock.patch.dict(sys.modules, {"torch": mock.Mock(from_numpy=lambda value: mock.Mock(reshape=lambda *_args: value), no_grad=lambda: mock.MagicMock())}):
            model = AudeeringDimensionalSerModel(
                "demo/model",
                processor=FakeProcessor(),
                model=FakeModel(),
            )
            result = model.predict(np.zeros(16000, dtype=np.float32))

        self.assertAlmostEqual(result["arousal"], 0.6)
        self.assertAlmostEqual(result["valence"], -0.6)
        self.assertEqual(result["label"], "dimensional")

    def test_build_local_ser_model_supports_audeering_backend(self):
        self.assertIsInstance(
            build_local_ser_model("demo/model", backend="audeering_dimensional"),
            LocalSerModelAdapter,
        )


class LocalSerRuntimeTests(unittest.TestCase):
    def test_runtime_waits_for_one_second_window_then_calls_adapter(self):
        calls = []

        class RecordingAdapter:
            def predict(self, samples, arousal_hint=0.0):
                calls.append((samples.copy(), arousal_hint))
                return {
                    "valence": -0.5,
                    "arousal": 0.9,
                    "confidence": 0.7,
                    "label": "tense",
                }

        runtime = LocalSerRuntime(
            window=RollingSerWindow(rate=4, seconds=1.0),
            model=RecordingAdapter(),
        )

        first = runtime.process(np.array([0.1, 0.2], dtype=np.float32), arousal_hint=0.3)
        second = runtime.process(np.array([0.3, 0.4], dtype=np.float32), arousal_hint=0.3)

        self.assertEqual(first["label"], "unknown")
        self.assertEqual(first["confidence"], 0.0)
        self.assertEqual(len(calls), 1)
        np.testing.assert_allclose(
            calls[0][0],
            np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        )
        self.assertEqual(calls[0][1], 0.3)
        self.assertEqual(second["label"], "tense")
        self.assertEqual(second["confidence"], 0.7)


if __name__ == "__main__":
    unittest.main()
