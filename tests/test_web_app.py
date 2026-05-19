import os
import unittest
from unittest import mock

os.environ["MEDIA_ART_LOAD_DOTENV"] = "0"

import web_app


class RunTestOscTests(unittest.TestCase):
    def test_run_test_osc_sends_live_verification_channels(self):
        fake_result = {
            "ok": True,
            "transcript": "웹 테스트 감정 메시지",
            "valence": -0.2,
            "audio_arousal": 0.7,
            "td_valence": -0.7,
            "td_arousal": 0.7,
        }

        with mock.patch.object(web_app.backend, "analyze_text_result", return_value=dict(fake_result)), \
             mock.patch.object(web_app.backend, "estimate_valence_confidence", return_value=0.56), \
             mock.patch.object(web_app.backend, "send_live_osc") as send_live_osc, \
             mock.patch.object(web_app, "read_touchdesigner_state", return_value={}):
            result = web_app.run_test_osc()

        send_live_osc.assert_called_once_with(
            arousal_live=0.7,
            arousal_confidence=1.0,
            valence_target=-0.7,
            valence_confidence=0.56,
            text_final="웹 테스트 감정 메시지",
        )
        self.assertTrue(result["live_osc_sent"])
        self.assertEqual(
            result["live_osc"],
            {
                "arousal_live": 0.7,
                "arousal_confidence": 1.0,
                "valence_target": -0.7,
                "valence_confidence": 0.56,
            },
        )
        self.assertEqual(result["valence_confidence"], 0.56)


class LiveSignalCompositionTests(unittest.TestCase):
    def test_live_loop_can_send_arousal_with_neutral_valence_before_ser_result(self):
        signal = {"serial_prefix": "v", "valence": -0.25, "arousal": 0.65}

        with mock.patch.object(web_app, "compose_led_mood_signal", return_value=signal) as compose_led_mood_signal, \
             mock.patch.object(web_app.backend, "send_live_osc") as send_live_osc:
            result = web_app.send_composed_live_signal(
                arousal_live=0.8,
                arousal_confidence=0.9,
                latest_valence=0.0,
                latest_valence_confidence=0.0,
                ambient_valence=-0.2,
                ambient_arousal=-0.4,
                has_mic_activity=True,
            )

        compose_led_mood_signal.assert_called_once_with(
            arousal_live=0.8,
            arousal_confidence=0.9,
            latest_valence=0.0,
            latest_valence_confidence=0.0,
            ambient_valence=-0.2,
            ambient_arousal=-0.4,
            has_mic_activity=True,
        )
        send_live_osc.assert_called_once_with(
            arousal_live=0.65,
            arousal_confidence=0.9,
            valence_target=-0.25,
            valence_confidence=0.0,
        )
        self.assertIs(result, signal)

    def test_live_audio_chunk_sends_signal_without_stt_or_segment_analysis(self):
        features = {
            "arousal_live": 0.42,
            "arousal_confidence": 0.8,
            "rms": 0.05,
            "zcr": 0.1,
            "spectral_centroid": 1234.0,
        }
        signal = {"serial_prefix": "v", "valence": 0.0, "arousal": 0.42}

        with mock.patch.object(web_app.backend, "compute_live_audio_features", return_value=features) as compute_features, \
             mock.patch.object(web_app.live_valence_tracker, "update", return_value={"valence": 0.0, "confidence": 0.0, "event": "start", "committed": False}) as update_valence, \
             mock.patch.object(web_app.voice_baseline, "update", return_value={"rms_baseline": 0.02, "relative_level": 2.0, "relative_arousal": 0.2}) as update_baseline, \
             mock.patch.object(web_app, "update_visitor_memory", return_value={"valence": 0.0, "arousal": 0.0}), \
             mock.patch.object(web_app, "visitor_valence_bias", return_value=0.0), \
             mock.patch.object(web_app, "smooth_td_arousal", return_value=0.21) as smooth_td_arousal, \
             mock.patch.object(web_app, "send_composed_live_signal", return_value=signal) as send_composed_live_signal, \
             mock.patch.object(web_app.backend, "process_audio_result") as process_audio_result:
            result = web_app.process_live_audio_chunk("audio chunk", overflowed=True, now=123.5)

        compute_features.assert_called_once_with("audio chunk", rate=web_app.backend.RATE)
        update_baseline.assert_called_once_with(rms=0.05, arousal_live=0.42, has_signal=True)
        update_valence.assert_called_once_with(
            candidate_valence=0.0,
            candidate_confidence=0.0,
            has_signal=True,
            now=123.5,
        )
        smooth_td_arousal.assert_called_once_with(0.42)
        send_composed_live_signal.assert_called_once_with(
            arousal_live=0.21,
            arousal_confidence=0.8,
            latest_valence=0.0,
            latest_valence_confidence=0.0,
            ambient_valence=0.0,
            ambient_arousal=0.0,
            has_mic_activity=True,
        )
        process_audio_result.assert_not_called()
        self.assertEqual(result["timestamp"], 123.5)
        self.assertTrue(result["overflowed"])
        self.assertEqual(result["valence_target"], 0.0)
        self.assertEqual(result["valence_confidence"], 0.0)
        self.assertEqual(result["serial_prefix"], "v")
        self.assertIn("processing_ms", result)

    def test_live_audio_chunk_uses_ser_valence_without_replacing_feature_arousal(self):
        features = {
            "arousal_live": 0.42,
            "arousal_confidence": 0.8,
            "rms": 0.05,
            "zcr": 0.1,
            "spectral_centroid": 1234.0,
        }
        ser_result = {
            "valence": -0.6,
            "arousal": -0.2,
            "confidence": 0.75,
            "label": "tense",
        }
        signal = {"serial_prefix": "v", "valence": -0.6, "arousal": 0.42}

        fake_ser_runtime = mock.Mock()
        fake_ser_runtime.process.return_value = ser_result

        with mock.patch.object(web_app.backend, "compute_live_audio_features", return_value=features), \
             mock.patch.object(web_app, "local_ser_runtime", fake_ser_runtime), \
             mock.patch.object(web_app.live_valence_tracker, "update", return_value={"valence": -0.6, "confidence": 0.75, "event": "continue", "committed": False}), \
             mock.patch.object(web_app.voice_baseline, "update", return_value={"rms_baseline": 0.02, "relative_level": 2.0, "relative_arousal": 0.2}), \
             mock.patch.object(web_app, "update_visitor_memory", return_value={"valence": 0.0, "arousal": 0.0}), \
             mock.patch.object(web_app, "visitor_valence_bias", return_value=0.0), \
             mock.patch.object(web_app, "smooth_td_arousal", return_value=0.21), \
             mock.patch.object(web_app, "send_composed_live_signal", return_value=signal) as send_composed_live_signal, \
             mock.patch.object(web_app.backend, "process_audio_result") as process_audio_result:
            result = web_app.process_live_audio_chunk("audio chunk", overflowed=False, now=123.5)

        self.assertEqual(fake_ser_runtime.process.call_args.args[0], "audio chunk")
        self.assertAlmostEqual(fake_ser_runtime.process.call_args.kwargs["arousal_hint"], 0.288)
        send_composed_live_signal.assert_called_once_with(
            arousal_live=0.21,
            arousal_confidence=0.8,
            latest_valence=-0.6,
            latest_valence_confidence=0.75,
            ambient_valence=0.0,
            ambient_arousal=0.0,
            has_mic_activity=True,
        )
        process_audio_result.assert_not_called()
        self.assertEqual(result["arousal_live"], 0.42)
        self.assertEqual(result["valence_target"], -0.6)
        self.assertEqual(result["valence_confidence"], 0.75)
        self.assertEqual(result["ser_label"], "tense")
        self.assertEqual(result["ser_arousal"], -0.2)
        self.assertIn("processing_ms", result)

    def test_dual_live_audio_chunk_sends_left_and_right_arousal_separately(self):
        features = {
            "left_arousal_live": -0.3,
            "right_arousal_live": 0.7,
            "left_arousal_confidence": 0.2,
            "right_arousal_confidence": 0.8,
            "left_rms": 0.01,
            "right_rms": 0.08,
            "arousal_live": 0.7,
            "arousal_confidence": 0.8,
        }
        ser_result = {
            "valence": 0.25,
            "arousal": 0.1,
            "confidence": 0.6,
            "label": "engaged",
        }
        signal = {"serial_prefix": "v", "valence": 0.25, "arousal": 0.7}
        fake_ser_runtime = mock.Mock()
        fake_ser_runtime.process.return_value = ser_result

        with mock.patch.object(web_app.backend, "compute_dual_live_audio_features", return_value=features) as compute_features, \
             mock.patch.object(web_app, "local_ser_runtime", fake_ser_runtime), \
             mock.patch.object(web_app.live_valence_tracker, "update", return_value={"valence": 0.25, "confidence": 0.6, "event": "continue", "committed": False}), \
             mock.patch.object(web_app.voice_baseline, "update", return_value={"rms_baseline": 0.02, "relative_level": 2.0, "relative_arousal": 0.2}), \
             mock.patch.object(web_app, "update_visitor_memory", return_value={"valence": 0.0, "arousal": 0.0}), \
             mock.patch.object(web_app, "visitor_valence_bias", return_value=0.0), \
             mock.patch.object(web_app, "smooth_td_arousal", return_value=0.35), \
             mock.patch.object(web_app, "compose_led_mood_signal", return_value=signal), \
             mock.patch.object(web_app.backend, "send_live_osc") as send_live_osc:
            result = web_app.process_dual_live_audio_chunk("left audio", "right audio", overflowed=True, now=123.5)

        compute_features.assert_called_once_with("left audio", "right audio", rate=web_app.backend.RATE)
        self.assertEqual(fake_ser_runtime.process.call_args.args[0], "right audio")
        self.assertAlmostEqual(fake_ser_runtime.process.call_args.kwargs["arousal_hint"], 0.4)
        send_live_osc.assert_called_once_with(
            arousal_live=0.7,
            arousal_confidence=0.8,
            left_arousal_live=-0.3,
            right_arousal_live=0.7,
            left_arousal_confidence=0.2,
            right_arousal_confidence=0.8,
            valence_target=0.25,
            valence_confidence=0.6,
        )
        self.assertTrue(result["overflowed"])
        self.assertEqual(result["left_arousal_live"], -0.3)
        self.assertEqual(result["right_arousal_live"], 0.7)
        self.assertEqual(result["valence_target"], 0.25)
        self.assertIn("processing_ms", result)


class VirtualMicWebTests(unittest.TestCase):
    def test_run_virtual_mic_scenario_updates_latest_frame_and_result(self):
        fake_frame = web_app.virtual_mic_scenarios.VirtualMicFrame(
            time=0.0,
            left_arousal=0.4,
            right_arousal=-0.4,
            valence=0.2,
            left_confidence=0.7,
            right_confidence=0.3,
            valence_confidence=0.8,
        )
        fake_result = {"name": "left_soft_voice", "osc": {"ok": True}, "readback": {"ok": False}}

        def fake_runner(**kwargs):
            kwargs["on_frame"](fake_frame)
            return fake_result

        with mock.patch.object(web_app.virtual_mic_scenarios, "run_named_scenario", side_effect=fake_runner) as runner:
            result = web_app.run_virtual_mic_scenario("left_soft_voice", duration_scale=0.5, readback=True)

        runner.assert_called_once()
        self.assertIs(result, fake_result)
        state = web_app.get_virtual_mic()
        self.assertEqual(state["status"], "done")
        self.assertEqual(state["result"], fake_result)
        self.assertEqual(state["latest"]["left_arousal_live"], 0.4)
        self.assertEqual(state["latest"]["right_arousal_live"], -0.4)
        self.assertEqual(state["latest"]["valence_target"], 0.2)


class DebugConsoleTests(unittest.TestCase):
    def test_debug_osc_pattern_sends_whitelisted_live_signal(self):
        with mock.patch.object(web_app.backend, "send_live_osc") as send_live_osc:
            result = web_app.run_debug_osc_pattern("red_high")

        send_live_osc.assert_called_once_with(
            arousal_live=0.7,
            arousal_confidence=1.0,
            valence_target=-0.7,
            valence_confidence=1.0,
            text_final="debug:red_high",
        )
        self.assertEqual(result["pattern"], "red_high")
        self.assertEqual(result["payload"], "v,-0.700,0.700")

    def test_debug_osc_pattern_rejects_unknown_name(self):
        with self.assertRaises(ValueError):
            web_app.run_debug_osc_pattern("anything")

    def test_debug_serial_send_uses_v_payload_through_td_bridge(self):
        with mock.patch.object(web_app, "td_bridge_action", return_value={"ok": True}) as td_bridge_action:
            result = web_app.run_debug_serial_send(0.9, -1.4)

        td_bridge_action.assert_called_once_with(
            "serial_send",
            path="/project1/serial1",
            message="v,0.900,-1.000",
        )
        self.assertEqual(result["payload"], "v,0.900,-1.000")
        self.assertEqual(result["valence"], 0.9)
        self.assertEqual(result["arousal"], -1.0)

    def test_debug_snapshot_keeps_td_actions_independent(self):
        def fake_debug_call(label, callback):
            return {"ok": label != "td.audit", "label": label}

        with mock.patch.object(web_app, "debug_call", side_effect=fake_debug_call):
            snapshot = web_app.build_debug_snapshot()

        self.assertTrue(snapshot["ok"])
        self.assertIn("live", snapshot)
        self.assertIn("mic", snapshot)
        self.assertTrue(snapshot["td"]["ping"]["ok"])
        self.assertFalse(snapshot["td"]["audit"]["ok"])
        self.assertTrue(snapshot["td"]["oscin2"]["ok"])
        self.assertTrue(snapshot["td"]["serialRows"]["ok"])

    def test_audio_input_devices_filter_inputs_and_mark_selected(self):
        devices = [
            {"name": "Speaker", "max_input_channels": 0},
            {"name": "USB Mic", "max_input_channels": 2, "default_samplerate": 44100},
        ]

        with mock.patch.object(web_app.backend.sd, "query_devices", return_value=devices), \
             mock.patch.object(web_app.backend, "DEVICE", 1):
            result = web_app.list_audio_input_devices()

        self.assertEqual(len(result["devices"]), 1)
        self.assertEqual(result["devices"][0]["index"], 1)
        self.assertTrue(result["devices"][0]["selected"])


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        with web_app.evaluation_lock:
            web_app.evaluation_samples.clear()
        self._old_parent_memory_path = web_app.PARENT_MEMORY_PATH
        self._old_visitor_memory_path = web_app.VISITOR_MEMORY_PATH
        self._old_genome_path = web_app.GENOME_PATH
        web_app.PARENT_MEMORY_PATH = web_app.ROOT / "memory" / ".test_parent_memory.json"
        web_app.VISITOR_MEMORY_PATH = web_app.ROOT / "memory" / ".test_visitor_memory.json"
        web_app.GENOME_PATH = web_app.ROOT / "memory" / ".test_genome.json"
        if web_app.PARENT_MEMORY_PATH.exists():
            web_app.PARENT_MEMORY_PATH.unlink()
        if web_app.VISITOR_MEMORY_PATH.exists():
            web_app.VISITOR_MEMORY_PATH.unlink()
        if web_app.GENOME_PATH.exists():
            web_app.GENOME_PATH.unlink()
        web_app.visitor_valence_smoother.value = 0.0
        web_app.last_visitor_memory_at = 0.0

    def tearDown(self):
        if web_app.PARENT_MEMORY_PATH.exists():
            web_app.PARENT_MEMORY_PATH.unlink()
        if web_app.VISITOR_MEMORY_PATH.exists():
            web_app.VISITOR_MEMORY_PATH.unlink()
        if web_app.GENOME_PATH.exists():
            web_app.GENOME_PATH.unlink()
        web_app.PARENT_MEMORY_PATH = self._old_parent_memory_path
        web_app.VISITOR_MEMORY_PATH = self._old_visitor_memory_path
        web_app.GENOME_PATH = self._old_genome_path

    def test_record_evaluation_sample_compares_latest_prediction(self):
        with mock.patch.object(web_app, "get_live", return_value={"latest": {
            "ser_label": "hap",
            "valence_target": 0.8,
            "ser_confidence": 0.9,
            "arousal_live": 0.4,
        }}):
            sample = web_app.record_evaluation_sample("hap")

        self.assertTrue(sample["correct"])
        self.assertEqual(sample["predicted_label"], "hap")

    def test_evaluation_summary_reports_accuracy(self):
        with mock.patch.object(web_app, "get_live", return_value={"latest": {"ser_label": "sad"}}):
            web_app.record_evaluation_sample("sad")
            web_app.record_evaluation_sample("hap")

        summary = web_app.evaluation_summary()

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["accuracy"], 0.5)

    def test_record_parent_sample_persists_memory_file(self):
        with mock.patch.object(web_app, "get_live", return_value={"latest": {
            "ser_label": "ang",
            "valence_target": -0.8,
            "ser_confidence": 0.9,
            "arousal_live": 0.5,
        }}):
            sample = web_app.record_parent_sample("ang", speaker="yoohyun")

        summary = web_app.parent_memory_summary()

        self.assertEqual(sample["speaker"], "yoohyun")
        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["byLabel"]["ang"], 1)
        self.assertTrue(web_app.PARENT_MEMORY_PATH.exists())

    def test_parent_bias_learns_predicted_to_expected_direction(self):
        with mock.patch.object(web_app, "get_live", return_value={"latest": {
            "ser_label": "sad",
            "valence_target": 0.0,
            "ser_confidence": 0.9,
            "arousal_live": 0.2,
        }}):
            for _index in range(4):
                web_app.record_parent_sample("hap", speaker="team")

        biased = web_app.apply_parent_bias({"label": "sad", "valence": -0.7})

        self.assertGreater(biased["valence"], -0.7)
        self.assertGreater(biased["parent_bias_strength"], 0.0)

    def test_update_visitor_memory_accumulates_slow_mood(self):
        first = web_app.update_visitor_memory(
            {"label": "hap", "valence": 0.8, "confidence": 1.0},
            arousal_live=0.6,
            arousal_confidence=0.8,
            now=100.0,
        )
        second = web_app.update_visitor_memory(
            {"label": "hap", "valence": 0.8, "confidence": 1.0},
            arousal_live=0.6,
            arousal_confidence=0.8,
            now=101.0,
        )

        self.assertGreater(second["valence"], first["valence"])
        self.assertLess(second["valence"], 0.1)
        self.assertTrue(web_app.VISITOR_MEMORY_PATH.exists())

    def test_update_visitor_memory_ignores_low_confidence_or_too_frequent_chunks(self):
        web_app.update_visitor_memory(
            {"label": "hap", "valence": 0.8, "confidence": 0.3},
            arousal_live=0.6,
            arousal_confidence=0.8,
            now=100.0,
        )
        web_app.update_visitor_memory(
            {"label": "hap", "valence": 0.8, "confidence": 1.0},
            arousal_live=0.6,
            arousal_confidence=0.8,
            now=100.1,
        )
        web_app.update_visitor_memory(
            {"label": "sad", "valence": -0.7, "confidence": 1.0},
            arousal_live=0.6,
            arousal_confidence=0.8,
            now=100.2,
        )

        summary = web_app.visitor_memory_summary()

        self.assertEqual(summary["count"], 1)

    def test_evolve_genome_from_visitor_memory_changes_slowly(self):
        memory = {
            "samples": [
                {"valence": 0.8, "arousal": 0.7}
                for _index in range(12)
            ]
        }

        genome = web_app.evolve_genome_from_visitor_memory(memory)

        self.assertGreater(genome["visitor_bias_strength"], web_app.DEFAULT_GENOME["visitor_bias_strength"] - 0.01)
        self.assertLessEqual(genome["visitor_bias_strength"], 0.28)
        self.assertTrue(web_app.GENOME_PATH.exists())


if __name__ == "__main__":
    unittest.main()
