# OpenClaw Session Handoff - 2026-05-18

이 문서는 OpenClaw에서 이번 세션 작업을 이어받기 위한 요약이다. API 키 값, `.env` 내용, 개인 비밀값은 포함하지 않는다.

## 프로젝트 위치

- 작업 기준 폴더: `G:\다른 컴퓨터\노트북\Documents\New project\innerworld\Innerworld\media_art_backend`
- 관련 TouchDesigner 폴더: `G:\다른 컴퓨터\노트북\Documents\New project\innerworld\Innerworld\touchdesigner`
- Google Drive의 원 프로젝트: `New project`

## 전체 시스템 의도

이 프로젝트는 음성/텍스트 입력에서 감정 신호를 만들고 TouchDesigner와 Arduino LED로 보내는 미디어아트 백엔드다.

기본 데이터 흐름:

```text
마이크/텍스트 입력
-> Python backend
-> live audio arousal 또는 Whisper STT
-> Gemini valence 분석
-> OSC 127.0.0.1:5000
-> TouchDesigner
-> serial1
-> Arduino
-> WS2811 LED ring
```

실시간 LED 반응은 현재 STT가 아니라 RMS, ZCR, spectral centroid 기반의 빠른 audio feature로 arousal을 즉시 계산한다. Whisper/Gemini는 발화 구간 이후 감정 보정에 가깝다.

## 이번 세션에서 한 작업

### 1. 환경/비밀값 정리

- 바탕화면 `secret.txt`에서 Gemini API 키를 읽어 프로젝트 `.env`에 `GEMINI_API_KEY`로 추가했다.
- 기존 `OPENAI_API_KEY`는 유지했다.
- 키 값은 터미널 출력이나 문서에 노출하지 않았다.
- `test_gemini.py`에 하드코딩되어 있던 Gemini API 키를 제거하고 `.env` 기반으로 바꿨다.

### 2. Python backend 리팩토링

기존 `main.py`가 녹음, 오디오 분석, AI client, Gemini 감정 분석, OSC 전송, CLI까지 모두 담당하던 상태였다. 기능별 모듈로 분리했다.

새로 만든/정리한 파일:

- `config.py`: 환경변수와 기본 설정
- `ai_clients.py`: Gemini/OpenAI lazy client
- `audio_processing.py`: RMS, arousal, live audio feature 계산
- `audio_io.py`: 마이크 녹음, 오디오 디바이스 조회, wav archive 정리
- `emotion_analysis.py`: STT 이후 감정 분석 파이프라인
- `mood_meter.py`: 한국어 mood meter grid, 색상 매핑, `v,valence,arousal` payload
- `osc_sender.py`: OSC 전송
- `td_bridge_client.py`: TouchDesigner HTTP bridge client
- `main.py`: CLI/호환 wrapper 중심으로 축소
- `web_app.py`: TouchDesigner bridge와 audio probe 로직을 새 모듈로 이동

주요 결과:

- `main.py`: 약 487줄에서 96줄로 축소
- `web_app.py`: 일부 bridge/probe 로직 분리, 아직 531줄로 크다
- 기존 테스트가 기대하던 `main.get_gemini_client()`, `main.gemini_client`, `main.genai.Client` mock 방식과의 호환성은 유지했다.

### 3. 검증

검증에 사용한 Python:

```powershell
C:\Users\mini-twosea\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
```

`venv311`은 이전 노트북 경로인 `C:\Users\kksu1\...`를 가리켜 현재 PC에서는 바로 실행되지 않는다.

실행한 검증:

```powershell
python -m unittest discover -s tests -v
```

결과:

```text
Ran 52 tests
OK
```

문법 컴파일:

```powershell
python -m py_compile <source/test/tool python files>
```

결과:

```text
exit code 0
```

하드코딩 키 패턴 검색:

```powershell
Select-String -Pattern 'GEMINI_API_KEY\s*=|OPENAI_API_KEY\s*=|AIza|sk-'
```

결과:

```text
Python files: no matches
```

## 실시간 음성/STT 검증 결과

사용자 PC에 마이크가 없어 실제 마이크 end-to-end 검증은 하지 못했다.

대신 `wav_archive`에 있던 기존 짧은 wav 파일 5개로 STT/Gemini 지연시간을 측정했다.

관찰값:

- 기존 wav 길이: 대부분 약 0.6-1.15초
- live audio feature 계산: 대략 0.6-6.7ms
- Whisper STT latency: 약 0.8-5.2초
- Gemini latency: 보통 약 0.8-1.1초, 첫 cold call은 약 5.7초
- STT+Gemini 전체 후처리: 보통 약 1.8-3.7초, 첫 cold run은 약 10.9초

합성 오디오 chunk 기준 `process_live_audio_chunk()` 처리 속도:

```text
chunk audio size: 64ms
mean processing: 0.434ms
median processing: 0.392ms
p95 processing: 0.634ms
p99 processing: 1.911ms
max processing: 2.555ms
```

해석:

- LED arousal 반응용 live audio feature는 충분히 빠르다.
- 현재 구조는 "실시간 자막/실시간 STT"가 아니라 "실시간 arousal 반응 + 발화 구간 후 STT/Gemini 감정 보정"에 가깝다.

## TouchDesigner/Arduino 상태

이번 세션에서 TouchDesigner bridge ping을 시도했지만 실패했다.

```text
http://127.0.0.1:9988/td -> timeout
```

가능한 원인:

- TouchDesigner가 열려 있지 않음
- 해당 `.toe`에서 webserver가 켜져 있지 않음
- callbacks DAT가 연결되지 않음
- 실제 열려 있는 `.toe` 파일이 프로젝트 canonical 파일과 다름

주의:

- 문서상 최신 TouchDesigner 수정본이 `Documents\New project`가 아니라 `Downloads\Innerworld\Innerworld\Innerworld.23.toe` 쪽에 있을 수 있다는 기록이 있었다.
- 실제 수정 전 현재 열려 있는 `.toe` 파일과 저장 대상 파일을 반드시 확인해야 한다.

## Local SER 현재 상태

`local_ser.py`는 아직 실제 SER 모델이 아니다.

현재 구성:

- `RollingSerWindow`: 최근 1초짜리 PCM window를 만든다.
- `LocalSerFallback`: 실제 예측 없이 다음 값을 반환한다.

```python
{
    "valence": 0.0,
    "arousal": arousal_hint,
    "confidence": 0.0,
    "label": "unknown",
}
```

즉 현재 실시간 감정 반응은 Local SER 모델이 아니라 `compute_live_audio_features()`의 RMS/ZCR/spectral centroid 기반 arousal이다.

Local SER를 진짜로 만들려면:

1. `RollingSerWindow`로 1초 샘플을 만든다.
2. 로컬 모델이 `samples -> valence/arousal/confidence/label`을 반환하게 한다.
3. `web_app.process_live_audio_chunk()` 또는 별도 `live_runtime` 모듈에서 이 결과를 `compose_led_mood_signal()`에 넣는다.

## 추천 다음 작업

### 우선순위 1: 실제 Local SER 통합

목표:

- `LocalSerFallback`을 실제 모델 adapter로 교체하거나 병렬 클래스로 추가한다.
- 모델이 없을 때는 기존 fallback을 유지한다.

추천 인터페이스:

```python
class LocalSerModel:
    def predict(self, samples: np.ndarray, arousal_hint: float = 0.0) -> dict:
        return {
            "valence": float,
            "arousal": float,
            "confidence": float,
            "label": str,
        }
```

주의:

- 모델 로딩은 import time이 아니라 lazy load로 한다.
- 모델 의존성이 없어도 테스트와 web_app import가 깨지지 않아야 한다.
- `tests/test_local_ser.py`에 fallback과 model-disabled 상태 테스트를 먼저 추가한다.

### 우선순위 2: `web_app.py` 추가 분리

아직 `web_app.py`는 531줄로 크다.

추천 분리:

- `debug_tools.py`: debug OSC pattern, serial send, snapshot
- `live_runtime.py`: live thread, process_live_audio_chunk, start/stop state
- `web_routes.py` 또는 기존 `web_app.py`: HTTP handler만 유지

### 우선순위 3: STT latency 개선

현재 Whisper API는 짧은 녹음이 끝난 뒤 요청하는 구조라 실시간성이 낮다.

대안:

- rolling segment STT: 1-2초 단위로 비동기 요청
- streaming STT: OpenAI Realtime 또는 다른 streaming STT
- STT는 semantic correction 용도로만 쓰고 LED는 local feature/SER를 우선시

### 우선순위 4: TouchDesigner end-to-end 확인

필수 확인:

- 현재 열린 `.toe` 파일 경로
- `/project1/webserver1` 또는 bridge DAT 활성 상태
- `td_bridge_callbacks.py` 연결 여부
- `/project1/oscin2`, `/project1/serial1` 채널/파라미터
- `serial_send`로 `v,0.700,0.700` 전송 시 Arduino LED 반응

## OpenClaw에 붙여넣을 프롬프트

아래 프롬프트를 OpenClaw 새 세션에 그대로 넣으면 된다.

```text
너는 OpenClaw에서 Innerworld media_art_backend 작업을 이어받는 코딩 에이전트다.

먼저 다음 문서를 읽고 현재 상태를 파악해줘:
G:\다른 컴퓨터\노트북\Documents\New project\innerworld\Innerworld\media_art_backend\docs\openclaw-session-handoff-2026-05-18.md

작업 기준 폴더:
G:\다른 컴퓨터\노트북\Documents\New project\innerworld\Innerworld\media_art_backend

현재까지 완료된 작업:
- .env에 GEMINI_API_KEY가 추가되어 있고 OPENAI_API_KEY도 존재한다. 절대 키 값을 출력하지 마라.
- test_gemini.py의 하드코딩 Gemini API 키는 제거되었다.
- main.py는 config.py, ai_clients.py, audio_processing.py, audio_io.py, emotion_analysis.py, mood_meter.py, osc_sender.py로 리팩토링되었다.
- td_bridge_client.py가 추가되었고 web_app.py 일부 bridge/probe 로직이 분리되었다.
- 전체 unittest 52개와 py_compile이 통과했다.
- 현재 PC에는 마이크가 없어 실제 마이크 end-to-end 검증은 못 했다.
- TouchDesigner bridge http://127.0.0.1:9988/td ping은 timeout이었다.
- local_ser.py는 아직 실제 SER 모델이 아니라 RollingSerWindow + LocalSerFallback만 있다.

먼저 아래 검증부터 실행해 현재 상태를 확인해줘:
1. python -m unittest discover -s tests -v
2. Python 파일 전체에서 API 키 하드코딩 패턴 검색
3. TouchDesigner가 열려 있다면 td_bridge_client.td_bridge_action("ping") 확인

다음 목표는 실제 Local SER 통합 준비다.
요구사항:
- import time에 heavy dependency/model load를 하지 말 것
- sounddevice/librosa/google/openai 같은 의존성이 없어도 테스트 import가 깨지지 않게 할 것
- Local SER 모델이 없으면 기존 fallback 경로가 유지되어야 함
- RollingSerWindow로 최근 1초 샘플을 만들고, 모델 adapter가 valence/arousal/confidence/label을 반환하게 설계할 것
- web_app.process_live_audio_chunk 또는 새 live_runtime.py에 SER 결과를 연결하되, 기존 compute_live_audio_features 기반 arousal 반응은 유지할 것
- 먼저 tests/test_local_ser.py에 실패 테스트를 추가하고, 그 다음 구현할 것
- 구현 후 unittest 전체를 다시 통과시킬 것

추가로 가능한 작업:
- web_app.py를 debug_tools.py, live_runtime.py 등으로 더 분리
- TouchDesigner bridge가 살아나면 /project1/oscin2, /project1/serial1, serial_send를 end-to-end로 점검
- STT는 진짜 실시간 반응이 아니라 후처리이므로, streaming STT 또는 rolling segment STT는 별도 후속 작업으로 제안

주의:
- .env, .env.txt, secret.txt의 값은 절대 출력하지 마라.
- 기존 사용자 변경을 되돌리지 마라.
- 실제 열려 있는 .toe 파일 경로를 확인하기 전에는 TouchDesigner 파일 저장을 하지 마라.
```

