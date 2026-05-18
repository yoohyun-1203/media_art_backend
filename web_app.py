import json
import os
import threading
import time
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import audio_io
import main as backend
from live_signal import compose_led_mood_signal
from local_ser import LocalSerRuntime
from mood_meter import clamp_mood_value, mood_payload
from td_bridge_client import (
    OSC_IN_PATH,
    SERIAL_DAT_PATH,
    TD_BRIDGE_URL,
    read_touchdesigner_state,
    td_bridge_action,
    td_channels,
)
from tools import virtual_mic_scenarios


HOST = "127.0.0.1"
PORT = int(os.getenv("WEB_PORT", "8765"))
ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
DEBUG_OSC_PATTERNS = {
    "red_high": {"label": "red/high", "valence": -0.7, "arousal": 0.7},
    "yellow_high": {"label": "yellow/high", "valence": 0.7, "arousal": 0.7},
    "blue_low": {"label": "blue/low", "valence": -0.7, "arousal": -0.7},
    "green_low": {"label": "green/low", "valence": 0.7, "arousal": -0.7},
    "neutral": {"label": "neutral", "valence": 0.0, "arousal": 0.0},
}
local_ser_runtime = LocalSerRuntime(rate=backend.RATE, seconds=1.0)

job_lock = threading.Lock()
job_state = {
    "running": False,
    "status": "idle",
    "message": "대기 중",
    "result": None,
    "error": None,
}

live_lock = threading.Lock()
live_stop_event = threading.Event()
live_thread = None
live_state = {
    "running": False,
    "status": "idle",
    "message": "실시간 대기 중",
    "latest": None,
    "result": None,
    "error": None,
}


virtual_mic_lock = threading.Lock()
virtual_mic_state = {
    "running": False,
    "status": "idle",
    "message": "Virtual mic ready",
    "latest": None,
    "result": None,
    "error": None,
    "scenarios": virtual_mic_scenarios.scenario_catalog(),
    "arousalMirrorStrategy": virtual_mic_scenarios.AROUSAL_MIRROR_STRATEGY,
}


def set_job(**updates):
    with job_lock:
        job_state.update(updates)
        return dict(job_state)


def get_job():
    with job_lock:
        return dict(job_state)


def set_live(**updates):
    with live_lock:
        live_state.update(updates)
        return dict(live_state)


def get_live():
    with live_lock:
        return dict(live_state)


def set_virtual_mic(**updates):
    with virtual_mic_lock:
        virtual_mic_state.update(updates)
        return dict(virtual_mic_state)


def get_virtual_mic():
    with virtual_mic_lock:
        return dict(virtual_mic_state)


def run_virtual_mic_scenario(name, duration_scale=1.0, readback=False):
    def on_frame(frame):
        latest = virtual_mic_scenarios.frame_to_state(frame)
        set_virtual_mic(
            latest=latest,
            message=f"Running {name}: t={latest['time']:.2f}s",
        )

    try:
        set_virtual_mic(
            running=True,
            status="running",
            message=f"Running {name}",
            latest=None,
            result=None,
            error=None,
        )
        result = virtual_mic_scenarios.run_named_scenario(
            name=name,
            duration_scale=duration_scale,
            readback=readback,
            on_frame=on_frame,
        )
        set_virtual_mic(
            running=False,
            status="done",
            message=f"Finished {name}",
            result=result,
            error=None,
        )
        return result
    except Exception as exc:
        set_virtual_mic(
            running=False,
            status="error",
            message=f"Virtual mic failed: {exc}",
            error=str(exc),
        )
        raise


def run_debug_osc_pattern(name):
    if name not in DEBUG_OSC_PATTERNS:
        raise ValueError(f"unknown debug pattern: {name}")
    pattern = DEBUG_OSC_PATTERNS[name]
    valence, arousal, payload = mood_payload(pattern["valence"], pattern["arousal"])
    backend.send_live_osc(
        arousal_live=arousal,
        arousal_confidence=1.0,
        valence_target=valence,
        valence_confidence=1.0,
        text_final=f"debug:{name}",
    )
    return {
        "pattern": name,
        "label": pattern["label"],
        "valence": valence,
        "arousal": arousal,
        "payload": payload,
    }


def run_debug_serial_send(valence, arousal):
    safe_valence, safe_arousal, payload = mood_payload(valence, arousal)
    response = td_bridge_action(
        "serial_send",
        path=SERIAL_DAT_PATH,
        message=payload,
    )
    return {
        "valence": safe_valence,
        "arousal": safe_arousal,
        "payload": payload,
        "touchdesigner": response,
    }


def debug_call(label, callback):
    started = time.time()
    try:
        return {
            "ok": True,
            "label": label,
            "elapsedMs": round((time.time() - started) * 1000, 1),
            "result": callback(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "label": label,
            "elapsedMs": round((time.time() - started) * 1000, 1),
            "error": str(exc),
        }


def list_audio_input_devices():
    result = audio_io.list_audio_input_devices(selected_device=backend.DEVICE)
    result["liveInput"] = live_input_config()
    return result


def probe_audio_input_device(device=None, duration=0.5):
    return audio_io.probe_audio_input_device(device=device, duration=duration)


def using_dual_input_devices():
    return backend.LEFT_DEVICE is not None and backend.RIGHT_DEVICE is not None


def live_input_config():
    if using_dual_input_devices():
        return {
            "mode": "dual_devices",
            "leftDevice": backend.LEFT_DEVICE,
            "rightDevice": backend.RIGHT_DEVICE,
            "channelsPerDevice": 1,
            "rate": backend.RATE,
            "chunk": backend.CHUNK,
            "noiseGateDb": backend.NOISE_GATE_DB,
        }
    return {
        "mode": "single_device",
        "device": backend.DEVICE,
        "channels": backend.CHANNELS,
        "rate": backend.RATE,
        "chunk": backend.CHUNK,
        "noiseGateDb": backend.NOISE_GATE_DB,
    }


def build_debug_snapshot():
    return {
        "ok": True,
        "timestamp": time.time(),
        "backend": {
            "web": {"host": HOST, "port": PORT},
            "osc": {"ip": backend.OSC_IP, "port": backend.OSC_PORT},
            "touchdesignerBridge": TD_BRIDGE_URL,
            "liveInput": live_input_config(),
        },
        "live": get_live(),
        "job": get_job(),
        "virtualMic": get_virtual_mic(),
        "mic": debug_call("mic.devices", list_audio_input_devices),
        "td": {
            "ping": debug_call("td.ping", lambda: td_bridge_action("ping")),
            "audit": debug_call("td.audit", lambda: td_bridge_action("audit", path="/project1", maxDepth=1)),
            "oscin2": debug_call("td.oscin2", lambda: td_bridge_action("channels", path=OSC_IN_PATH)),
            "serialParams": debug_call("td.serial.params", lambda: td_bridge_action("params", path=SERIAL_DAT_PATH)),
            "serialRows": debug_call("td.serial.rows", lambda: td_bridge_action("dat_rows", path=SERIAL_DAT_PATH, maxRows=10)),
        },
        "patterns": DEBUG_OSC_PATTERNS,
    }


def analyze_worker():
    try:
        set_job(
            running=True,
            status="recording",
            message="마이크 입력을 기다리는 중입니다. 소리를 내면 녹음이 시작됩니다.",
            result=None,
            error=None,
        )
        filepath = backend.record_audio()
        if not filepath:
            set_job(
                running=False,
                status="error",
                message="녹음된 오디오가 없습니다.",
                error="no_audio_recorded",
            )
            return

        set_job(status="analyzing", message="OpenAI Whisper API와 Gemini API로 분석 중입니다.")
        result = backend.process_audio_result(filepath)

        if result.get("ok"):
            try:
                backend.manage_archive_limit(backend.ARCHIVE_DIR, max_files=20)
            except Exception:
                pass
            result["valence_confidence"] = backend.estimate_valence_confidence(
                result.get("transcript", ""),
                result.get("valence", 0.0),
            )
            result["touchdesigner"] = read_touchdesigner_state()
            set_job(
                running=False,
                status="done",
                message="TouchDesigner 전송까지 완료했습니다.",
                result=result,
                error=None,
            )
        else:
            set_job(
                running=False,
                status="error",
                message="분석에 실패했습니다.",
                result=result,
                error=result.get("error", "analysis_failed"),
            )
    except Exception as exc:
        set_job(
            running=False,
            status="error",
            message="실행 중 오류가 발생했습니다.",
            error=str(exc),
            result={"traceback": traceback.format_exc()},
        )


def run_test_osc():
    text = "웹 테스트 감정 메시지"
    result = backend.analyze_text_result(text, audio_arousal=0.7)
    result["valence_confidence"] = backend.estimate_valence_confidence(
        result.get("transcript", ""),
        result.get("valence", 0.0),
    )
    live_osc = {
        "arousal_live": float(result.get("td_arousal", result.get("audio_arousal", 0.0))),
        "arousal_confidence": 1.0,
        "valence_target": float(result.get("td_valence", result.get("valence", 0.0))),
        "valence_confidence": result["valence_confidence"],
    }
    backend.send_live_osc(
        **live_osc,
        text_final=result.get("transcript", ""),
    )
    result["live_osc"] = live_osc
    result["live_osc_sent"] = True
    result["touchdesigner"] = read_touchdesigner_state()
    return result


def send_composed_live_signal(
    arousal_live,
    arousal_confidence,
    latest_valence,
    latest_valence_confidence,
    ambient_valence,
    ambient_arousal,
    has_mic_activity,
):
    signal = compose_led_mood_signal(
        arousal_live=arousal_live,
        arousal_confidence=arousal_confidence,
        latest_valence=latest_valence,
        latest_valence_confidence=latest_valence_confidence,
        ambient_valence=ambient_valence,
        ambient_arousal=ambient_arousal,
        has_mic_activity=has_mic_activity,
    )
    backend.send_live_osc(
        arousal_live=signal["arousal"],
        arousal_confidence=arousal_confidence,
        valence_target=signal["valence"],
        valence_confidence=latest_valence_confidence,
    )
    return signal


def process_live_audio_chunk(data, overflowed=False, now=None):
    features = backend.compute_live_audio_features(data, rate=backend.RATE)
    arousal_confidence = float(features.get("arousal_confidence", 0.0))
    ser_result = local_ser_runtime.process(data, arousal_hint=features["arousal_live"])
    ser_confidence = float(ser_result.get("confidence", 0.0))
    signal = send_composed_live_signal(
        arousal_live=features["arousal_live"],
        arousal_confidence=arousal_confidence,
        latest_valence=float(ser_result.get("valence", 0.0)),
        latest_valence_confidence=ser_confidence,
        ambient_valence=0.0,
        ambient_arousal=0.0,
        has_mic_activity=arousal_confidence > 0.0,
    )
    return {
        **features,
        "timestamp": time.time() if now is None else now,
        "overflowed": bool(overflowed),
        "valence_target": signal["valence"],
        "valence_confidence": ser_confidence,
        "ser_arousal": float(ser_result.get("arousal", features["arousal_live"])),
        "ser_confidence": ser_confidence,
        "ser_label": str(ser_result.get("label", "unknown")),
        "serial_prefix": signal.get("serial_prefix", "v"),
    }


def process_dual_live_audio_chunk(left_data, right_data, overflowed=False, now=None):
    features = backend.compute_dual_live_audio_features(left_data, right_data, rate=backend.RATE)
    arousal_confidence = float(features.get("arousal_confidence", 0.0))
    primary_data = left_data
    if float(features["right_arousal_live"]) > float(features["left_arousal_live"]):
        primary_data = right_data
    ser_result = local_ser_runtime.process(primary_data, arousal_hint=features["arousal_live"])
    ser_confidence = float(ser_result.get("confidence", 0.0))
    signal = compose_led_mood_signal(
        arousal_live=features["arousal_live"],
        arousal_confidence=arousal_confidence,
        latest_valence=float(ser_result.get("valence", 0.0)),
        latest_valence_confidence=ser_confidence,
        ambient_valence=0.0,
        ambient_arousal=0.0,
        has_mic_activity=arousal_confidence > 0.0,
    )
    backend.send_live_osc(
        arousal_live=signal["arousal"],
        arousal_confidence=arousal_confidence,
        left_arousal_live=features["left_arousal_live"],
        right_arousal_live=features["right_arousal_live"],
        left_arousal_confidence=features["left_arousal_confidence"],
        right_arousal_confidence=features["right_arousal_confidence"],
        valence_target=signal["valence"],
        valence_confidence=ser_confidence,
    )
    return {
        **features,
        "timestamp": time.time() if now is None else now,
        "overflowed": bool(overflowed),
        "valence_target": signal["valence"],
        "valence_confidence": ser_confidence,
        "ser_arousal": float(ser_result.get("arousal", features["arousal_live"])),
        "ser_confidence": ser_confidence,
        "ser_label": str(ser_result.get("label", "unknown")),
        "serial_prefix": signal.get("serial_prefix", "v"),
        "live_input_mode": "dual_devices",
        "left_device": backend.LEFT_DEVICE,
        "right_device": backend.RIGHT_DEVICE,
    }


def live_worker():
    set_live(
        running=True,
        status="listening",
        message="실시간 입력을 듣는 중입니다.",
        latest=None,
        result=None,
        error=None,
    )

    stream = None
    streams = []

    try:
        if using_dual_input_devices():
            left_stream = backend.sd.InputStream(
                device=backend.LEFT_DEVICE,
                samplerate=backend.RATE,
                channels=1,
                dtype="int16",
                blocksize=backend.CHUNK,
            )
            right_stream = backend.sd.InputStream(
                device=backend.RIGHT_DEVICE,
                samplerate=backend.RATE,
                channels=1,
                dtype="int16",
                blocksize=backend.CHUNK,
            )
            streams = [left_stream, right_stream]
            for active_stream in streams:
                active_stream.start()

            while not live_stop_event.is_set():
                left_data, left_overflowed = left_stream.read(backend.CHUNK)
                right_data, right_overflowed = right_stream.read(backend.CHUNK)
                latest = process_dual_live_audio_chunk(
                    left_data,
                    right_data,
                    overflowed=left_overflowed or right_overflowed,
                )
                set_live(
                    latest=latest,
                    status="listening",
                    message="실시간 입력을 듣는 중입니다.",
                )
        else:
            stream = backend.sd.InputStream(
                device=backend.DEVICE,
                samplerate=backend.RATE,
                channels=backend.CHANNELS,
                dtype="int16",
                blocksize=backend.CHUNK,
            )
            stream.start()

            while not live_stop_event.is_set():
                data, overflowed = stream.read(backend.CHUNK)
                latest = process_live_audio_chunk(data, overflowed=overflowed)
                set_live(
                    latest=latest,
                    status="listening",
                    message="실시간 입력을 듣는 중입니다.",
                )

        set_live(running=False, status="stopped", message="실시간 정지됨")
    except Exception as exc:
        set_live(
            running=False,
            status="error",
            message="실시간 입력을 시작하지 못했습니다.",
            error=str(exc),
            result={"traceback": traceback.format_exc()},
        )
    finally:
        for active_stream in streams:
            try:
                active_stream.stop()
                active_stream.close()
            except Exception:
                pass
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass


def start_live():
    global live_thread
    state = get_live()
    if state.get("running"):
        return False, state
    live_stop_event.clear()
    live_thread = threading.Thread(target=live_worker, daemon=True)
    live_thread.start()
    return True, get_live()


def stop_live():
    live_stop_event.set()
    state = set_live(running=False, status="stopping", message="실시간 정지 요청을 보냈습니다.")
    return state


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def log_message(self, fmt, *args):
        print("[web]", fmt % args)

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        if self.path == "/api/status":
            self.send_json(get_job())
            return
        if self.path == "/api/live/status":
            self.send_json(get_live())
            return
        if self.path == "/api/virtual-mic/status":
            self.send_json(get_virtual_mic())
            return
        if self.path == "/api/virtual-mic/scenarios":
            self.send_json({
                "ok": True,
                "scenarios": virtual_mic_scenarios.scenario_catalog(),
                "arousalMirrorStrategy": virtual_mic_scenarios.AROUSAL_MIRROR_STRATEGY,
            })
            return
        if self.path == "/api/debug/snapshot":
            self.send_json(build_debug_snapshot())
            return
        if self.path == "/api/debug/audio-devices":
            try:
                self.send_json({"ok": True, "audio": list_audio_input_devices()})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=500)
            return
        if self.path == "/api/health":
            self.send_json({
                "ok": True,
                "backend": "media_art_backend",
                "osc": {"ip": backend.OSC_IP, "port": backend.OSC_PORT},
                "touchdesignerBridge": TD_BRIDGE_URL,
                "liveInput": live_input_config(),
                "live": get_live(),
                "virtualMic": get_virtual_mic(),
            })
            return
        super().do_GET()

    def do_POST(self):
        if self.path == "/api/start":
            state = get_job()
            if state.get("running"):
                self.send_json({"ok": False, "error": "already_running", "state": state}, status=409)
                return
            thread = threading.Thread(target=analyze_worker, daemon=True)
            thread.start()
            self.send_json({"ok": True, "state": get_job()})
            return

        if self.path == "/api/live/start":
            started, state = start_live()
            if not started:
                self.send_json({"ok": False, "error": "already_running", "state": state}, status=409)
                return
            self.send_json({"ok": True, "state": state})
            return

        if self.path == "/api/live/stop":
            self.send_json({"ok": True, "state": stop_live()})
            return

        if self.path == "/api/virtual-mic/run":
            try:
                body = self.read_json()
                name = body.get("scenario") or "silence_baseline"
                duration_scale = float(body.get("durationScale", 1.0))
                readback = bool(body.get("readback", False))
                state = get_virtual_mic()
                if state.get("running"):
                    self.send_json({"ok": False, "error": "already_running", "state": state}, status=409)
                    return
                thread = threading.Thread(
                    target=run_virtual_mic_scenario,
                    args=(name, duration_scale, readback),
                    daemon=True,
                )
                thread.start()
                self.send_json({"ok": True, "state": get_virtual_mic()})
            except KeyError as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=400)
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=500)
            return

        if self.path == "/api/test-osc":
            try:
                self.send_json({"ok": True, "result": run_test_osc()})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=500)
            return

        if self.path == "/api/debug/td-ping":
            try:
                self.send_json({"ok": True, "result": td_bridge_action("ping")})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=500)
            return

        if self.path == "/api/debug/td-audit":
            try:
                self.send_json({"ok": True, "result": td_bridge_action("audit", path="/project1", maxDepth=1)})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=500)
            return

        if self.path == "/api/debug/td-readback":
            try:
                self.send_json({"ok": True, "result": {
                    "oscin2": debug_call("td.oscin2", lambda: td_bridge_action("channels", path=OSC_IN_PATH)),
                    "serialParams": debug_call("td.serial.params", lambda: td_bridge_action("params", path=SERIAL_DAT_PATH)),
                    "serialRows": debug_call("td.serial.rows", lambda: td_bridge_action("dat_rows", path=SERIAL_DAT_PATH, maxRows=10)),
                }})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=500)
            return

        if self.path == "/api/debug/osc-pattern":
            try:
                body = self.read_json()
                self.send_json({"ok": True, "result": run_debug_osc_pattern(body.get("pattern", "neutral"))})
            except ValueError as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=400)
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=500)
            return

        if self.path == "/api/debug/serial-send":
            try:
                body = self.read_json()
                self.send_json({"ok": True, "result": run_debug_serial_send(body.get("valence", 0.0), body.get("arousal", 0.0))})
            except ValueError as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=400)
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=500)
            return

        if self.path == "/api/debug/audio-probe":
            try:
                body = self.read_json()
                self.send_json({"ok": True, "result": probe_audio_input_device(
                    device=body.get("device"),
                    duration=body.get("duration", 0.5),
                )})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=500)
            return

        self.send_json({"ok": False, "error": "not_found"}, status=404)


def main():
    if not WEB_ROOT.exists():
        raise RuntimeError(f"web directory not found: {WEB_ROOT}")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"웹 컨트롤러 실행 중: http://{HOST}:{PORT}")
    print("종료하려면 Ctrl+C")
    server.serve_forever()


if __name__ == "__main__":
    main()
