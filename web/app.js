const startButton = document.querySelector("#startButton");
const testButton = document.querySelector("#testButton");
const liveStartButton = document.querySelector("#liveStartButton");
const liveStopButton = document.querySelector("#liveStopButton");
const statusText = document.querySelector("#statusText");
const liveStatusText = document.querySelector("#liveStatusText");
const transcript = document.querySelector("#transcript");
const emotion = document.querySelector("#emotion");
const valence = document.querySelector("#valence");
const arousal = document.querySelector("#arousal");
const liveArousal = document.querySelector("#liveArousal");
const arousalConfidence = document.querySelector("#arousalConfidence");
const valenceConfidence = document.querySelector("#valenceConfidence");
const rgb = document.querySelector("#rgb");
const raw = document.querySelector("#raw");
const controllerPreviewSummary = document.querySelector("#controllerPreviewSummary");
const controllerValueSummary = document.querySelector("#controllerValueSummary");
const controllerPayload = document.querySelector("#controllerPayload");
const controllerLedPreview = document.querySelector("#controllerLedPreview");
const arduinoPreviewSummary = document.querySelector("#arduinoPreviewSummary");
const arduinoPayload = document.querySelector("#arduinoPayload");
const arduinoTransform = document.querySelector("#arduinoTransform");
const arduinoHardwarePreview = document.querySelector("#arduinoHardwarePreview");
const virtualScenarioSelect = document.querySelector("#virtualScenarioSelect");
const virtualReadbackToggle = document.querySelector("#virtualReadbackToggle");
const virtualRunButton = document.querySelector("#virtualRunButton");
const virtualStatusText = document.querySelector("#virtualStatusText");
const virtualExpectedText = document.querySelector("#virtualExpectedText");
const virtualSignalText = document.querySelector("#virtualSignalText");
const virtualRaw = document.querySelector("#virtualRaw");
const debugSnapshotButton = document.querySelector("#debugSnapshotButton");
const debugTdPingButton = document.querySelector("#debugTdPingButton");
const debugTdAuditButton = document.querySelector("#debugTdAuditButton");
const debugReadbackButton = document.querySelector("#debugReadbackButton");
const debugPatternSelect = document.querySelector("#debugPatternSelect");
const debugOscButton = document.querySelector("#debugOscButton");
const debugValenceInput = document.querySelector("#debugValenceInput");
const debugArousalInput = document.querySelector("#debugArousalInput");
const debugSerialButton = document.querySelector("#debugSerialButton");
const debugAudioDeviceSelect = document.querySelector("#debugAudioDeviceSelect");
const debugAudioProbeButton = document.querySelector("#debugAudioProbeButton");
const debugStatusText = document.querySelector("#debugStatusText");
const debugBridgeText = document.querySelector("#debugBridgeText");
const debugOscReadbackText = document.querySelector("#debugOscReadbackText");
const debugSerialText = document.querySelector("#debugSerialText");
const debugMicText = document.querySelector("#debugMicText");
const debugRaw = document.querySelector("#debugRaw");
const ledPreviewModel = window.InnerworldLedPreview;

let pollTimer = null;
let livePollTimer = null;
let virtualMicPollTimer = null;
let virtualMicScenarios = [];
let arduinoPreviewFrame = 0;
let latestLedPayload = null;

function setBusy(isBusy) {
  startButton.disabled = isBusy;
  testButton.disabled = isBusy;
}

function setLiveBusy(isRunning) {
  liveStartButton.disabled = isRunning;
  liveStopButton.disabled = !isRunning;
}

function fmt(value) {
  if (typeof value === "number") {
    return value.toFixed(3);
  }
  return value ?? "-";
}

function channelSummary(channels) {
  if (!channels) {
    return "-";
  }
  const pairs = [
    ["valence", channels["emotion/valence_target"] ?? channels["emotion/valence"]],
    ["arousal", channels["emotion/arousal_live"] ?? channels["emotion/arousal"]],
    ["a_conf", channels["emotion/arousal_confidence"]],
    ["v_conf", channels["emotion/valence_confidence"]],
  ];
  return pairs.map(([label, value]) => `${label} ${fmt(value)}`).join(" / ");
}

function debugResultText(entry) {
  if (!entry) {
    return "-";
  }
  if (!entry.ok) {
    return entry.error || "failed";
  }
  return `ok ${fmt(entry.elapsedMs)}ms`;
}

function setDebugBusy(isBusy) {
  for (const button of [
    debugSnapshotButton,
    debugTdPingButton,
    debugTdAuditButton,
    debugReadbackButton,
    debugOscButton,
    debugSerialButton,
    debugAudioProbeButton,
  ]) {
    if (button) {
      button.disabled = isBusy;
    }
  }
}

function renderAudioDevices(audio) {
  if (!debugAudioDeviceSelect || !audio?.devices) {
    return;
  }
  const previous = debugAudioDeviceSelect.value;
  debugAudioDeviceSelect.innerHTML = "";
  for (const device of audio.devices) {
    const option = document.createElement("option");
    option.value = String(device.index);
    option.textContent = `${device.index}: ${device.name}`;
    if (device.selected || String(device.index) === previous) {
      option.selected = true;
    }
    debugAudioDeviceSelect.appendChild(option);
  }
}

function renderDebugPayload(payload) {
  if (!payload) {
    return;
  }
  if (debugRaw) {
    debugRaw.textContent = JSON.stringify(payload, null, 2);
  }
  if (debugStatusText) {
    debugStatusText.textContent = payload.ok === false ? (payload.error || "Debug failed") : "Debug updated";
  }

  const snapshot = payload.td ? payload : payload.result;
  if (snapshot?.td) {
    if (debugBridgeText) {
      debugBridgeText.textContent = debugResultText(snapshot.td.ping);
    }
    if (debugOscReadbackText) {
      const channels = snapshot.td.oscin2?.result?.channels;
      debugOscReadbackText.textContent = snapshot.td.oscin2?.ok ? channelSummary(channels) : debugResultText(snapshot.td.oscin2);
    }
    if (debugSerialText) {
      const serialParams = snapshot.td.serialParams;
      const rows = snapshot.td.serialRows;
      const port = serialParams?.result?.params?.port?.value;
      const baud = serialParams?.result?.params?.baudrate?.value;
      debugSerialText.textContent = port ? `${port} / ${baud} / ${debugResultText(rows)}` : debugResultText(rows);
    }
    if (debugMicText) {
      const devices = snapshot.mic?.result?.devices || [];
      const selected = devices.find((device) => device.selected);
      debugMicText.textContent = selected ? `${selected.index}: ${selected.name}` : `${devices.length} input devices`;
    }
    renderAudioDevices(snapshot.mic?.result);
    return;
  }

  if (payload.result?.oscin2 || payload.result?.serialRows) {
    if (debugOscReadbackText) {
      const channels = payload.result.oscin2?.result?.channels;
      debugOscReadbackText.textContent = payload.result.oscin2?.ok ? channelSummary(channels) : debugResultText(payload.result.oscin2);
    }
    if (debugSerialText) {
      debugSerialText.textContent = debugResultText(payload.result.serialRows);
    }
    return;
  }

  if (payload.result?.payload) {
    if (debugSerialText) {
      debugSerialText.textContent = payload.result.payload;
    }
    renderLedPreviewFromState({
      latest: {
        valence_target: payload.result.valence,
        arousal_live: payload.result.arousal,
      },
    });
  }

  if (payload.result?.rms !== undefined) {
    if (debugMicText) {
      debugMicText.textContent = `device ${payload.result.device} / rms ${fmt(payload.result.rms)} / peak ${fmt(payload.result.peak)}`;
    }
  }
}

async function debugGet(path) {
  const response = await fetch(path);
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `${path} failed: ${response.status}`);
  }
  return payload;
}

async function debugPost(path, body = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `${path} failed: ${response.status}`);
  }
  return payload;
}

async function runDebugAction(label, action) {
  if (!debugStatusText) {
    return;
  }
  setDebugBusy(true);
  debugStatusText.textContent = label;
  try {
    const payload = await action();
    renderDebugPayload(payload);
  } catch (error) {
    debugStatusText.textContent = error.message;
    if (debugRaw) {
      debugRaw.textContent = JSON.stringify({ ok: false, error: error.message }, null, 2);
    }
  } finally {
    setDebugBusy(false);
  }
}

function tdValue(result, path, name) {
  return result?.touchdesigner?.[path]?.[name];
}

function buildControllerPreview() {
  if (!controllerLedPreview || !ledPreviewModel) {
    return;
  }

  const fragment = document.createDocumentFragment();
  for (let index = 0; index < ledPreviewModel.LED_COUNT; index += 1) {
    const led = document.createElement("span");
    led.className = "led-dot led-line-dot";
    led.dataset.index = String(index);
    fragment.appendChild(led);
  }
  controllerLedPreview.appendChild(fragment);
}

function buildArduinoHardwarePreview() {
  if (!arduinoHardwarePreview || !ledPreviewModel?.buildArduinoHardwareModel) {
    return;
  }

  const model = ledPreviewModel.buildArduinoHardwareModel("v,0,0");
  const fragment = document.createDocumentFragment();
  for (const line of model.lines) {
    const row = document.createElement("div");
    row.className = "arduino-line";
    row.dataset.pin = line.pin;

    const label = document.createElement("div");
    label.className = "arduino-line-label";
    const pin = document.createElement("strong");
    pin.textContent = line.pin;
    const length = document.createElement("span");
    length.textContent = `${line.physicalLength} LED`;
    label.append(pin, length);

    const dots = document.createElement("div");
    dots.className = "arduino-line-dots";
    dots.style.gridTemplateColumns = `repeat(${line.physicalLength}, var(--arduino-dot-size))`;
    for (const dot of line.dots) {
      const led = document.createElement("span");
      led.className = "led-dot arduino-dot";
      led.dataset.line = String(line.index);
      led.dataset.dot = String(dot.index);
      dots.appendChild(led);
    }

    row.append(label, dots);
    fragment.appendChild(row);
  }

  arduinoHardwarePreview.appendChild(fragment);
}

function renderArduinoHardwarePreview(parsed) {
  if (!arduinoHardwarePreview || !ledPreviewModel?.buildArduinoHardwareModel) {
    return;
  }

  const model = ledPreviewModel.buildArduinoHardwareModel(parsed, {
    frame: arduinoPreviewFrame,
  });

  if (arduinoPreviewSummary) {
    arduinoPreviewSummary.textContent = model.summary;
  }
  if (arduinoPayload) {
    arduinoPayload.textContent = `${model.payload} / L ${parsed.leftPayload || model.payload} / R ${parsed.rightPayload || model.payload}`;
  }
  if (arduinoTransform) {
    arduinoTransform.textContent = model.valueSummary;
  }

  for (const line of model.lines) {
    const row = arduinoHardwarePreview.querySelector(`.arduino-line[data-pin="${line.pin}"]`);
    if (!row) {
      continue;
    }
    row.dataset.side = line.side;
    row.dataset.activity = String(line.activity);
    const dots = row.querySelectorAll(".arduino-dot");
    line.dots.forEach((dotModel, index) => {
      const dot = dots[index];
      if (!dot) {
        return;
      }
      dot.style.backgroundColor = dotModel.active ? dotModel.color : "#202632";
      dot.style.boxShadow = dotModel.active ? `0 0 ${dotModel.glow}px ${dotModel.color}` : "none";
      dot.style.opacity = String(dotModel.opacity);
      dot.dataset.active = dotModel.active ? "true" : "false";
      dot.dataset.side = dotModel.side;
    });
  }
}

function renderLedPreview(parsed) {
  if (!controllerLedPreview || !ledPreviewModel) {
    return;
  }

  latestLedPayload = parsed;
  const model = ledPreviewModel.buildLedModel(parsed.valence, parsed);
  const dots = controllerLedPreview.querySelectorAll(".led-dot");
  model.leds.forEach((cell, index) => {
    const dot = dots[index];
    if (!dot) {
      return;
    }
    dot.style.backgroundColor = cell.active ? cell.color : "#202632";
    dot.style.boxShadow = cell.active ? `0 0 ${cell.glow}px ${cell.color}` : "none";
    dot.style.opacity = String(cell.opacity);
    dot.dataset.side = cell.side;
  });

  controllerPreviewSummary.textContent = model.summary;
  controllerValueSummary.textContent = model.valueSummary;
  controllerPayload.textContent = `${parsed.payload} / L ${parsed.leftPayload || parsed.payload} / R ${parsed.rightPayload || parsed.payload}`;
  renderArduinoHardwarePreview(parsed);
}

function renderLedPreviewFromState(state) {
  if (!ledPreviewModel) {
    return;
  }
  renderLedPreview(ledPreviewModel.payloadFromLiveState(state));
}

function selectedVirtualScenario() {
  return virtualMicScenarios.find((scenario) => scenario.name === virtualScenarioSelect?.value);
}

function renderVirtualExpected() {
  if (!virtualExpectedText) {
    return;
  }
  const scenario = selectedVirtualScenario();
  virtualExpectedText.textContent = scenario?.expectedBehavior || "-";
}

function renderVirtualMicState(state) {
  if (!virtualStatusText || !virtualRunButton) {
    return;
  }

  virtualStatusText.textContent = state.message || state.status || "Virtual mic ready";
  virtualRunButton.disabled = Boolean(state.running);

  if (state.latest) {
    const left = fmt(state.latest.left_arousal_live);
    const right = fmt(state.latest.right_arousal_live);
    const valenceTarget = fmt(state.latest.valence_target);
    virtualSignalText.textContent = `left ${left} / right ${right} / valence ${valenceTarget}`;
    renderLedPreviewFromState({ latest: state.latest });
  }

  if (state.result) {
    virtualRaw.textContent = JSON.stringify(state.result, null, 2);
  } else if (state.error) {
    virtualRaw.textContent = JSON.stringify(state, null, 2);
  }
}

async function getVirtualMicStatus() {
  const response = await fetch("/api/virtual-mic/status");
  if (!response.ok) {
    throw new Error(`virtual mic status failed: ${response.status}`);
  }
  return response.json();
}

async function pollVirtualMicStatus() {
  try {
    const state = await getVirtualMicStatus();
    renderVirtualMicState(state);
    if (!state.running && virtualMicPollTimer) {
      clearInterval(virtualMicPollTimer);
      virtualMicPollTimer = null;
    }
  } catch (error) {
    if (virtualStatusText) {
      virtualStatusText.textContent = error.message;
    }
    if (virtualRunButton) {
      virtualRunButton.disabled = false;
    }
  }
}

function ensureVirtualMicPoll() {
  if (!virtualMicPollTimer) {
    virtualMicPollTimer = setInterval(pollVirtualMicStatus, 250);
  }
}

async function loadVirtualMicScenarios() {
  if (!virtualScenarioSelect) {
    return;
  }

  const response = await fetch("/api/virtual-mic/scenarios");
  const payload = await response.json();
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || `scenario load failed: ${response.status}`);
  }

  virtualMicScenarios = payload.scenarios || [];
  virtualScenarioSelect.innerHTML = "";
  for (const scenario of virtualMicScenarios) {
    const option = document.createElement("option");
    option.value = scenario.name;
    option.textContent = scenario.name;
    virtualScenarioSelect.appendChild(option);
  }
  renderVirtualExpected();
}

async function startVirtualMicScenario() {
  if (!virtualScenarioSelect || !virtualRunButton) {
    return;
  }

  const scenario = virtualScenarioSelect.value || "silence_baseline";
  virtualRunButton.disabled = true;
  virtualStatusText.textContent = `Starting ${scenario}`;
  virtualRaw.textContent = "{}";
  const response = await fetch("/api/virtual-mic/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      scenario,
      durationScale: 1,
      readback: Boolean(virtualReadbackToggle?.checked),
    }),
  });
  const payload = await response.json();
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || `virtual mic failed: ${response.status}`);
  }
  renderVirtualMicState(payload.state);
  ensureVirtualMicPoll();
}

function renderResult(result) {
  if (!result) {
    return;
  }

  transcript.textContent = result.transcript || "분석된 문장이 없습니다.";
  emotion.textContent = result.emotion_word
    ? `${result.emotion_word} · ${result.color_name}`
    : "-";
  valence.textContent = fmt(result.td_valence);
  arousal.textContent = fmt(result.td_arousal);
  valenceConfidence.textContent = fmt(result.valence_confidence);

  const r = tdValue(result, "/project1/RGBs", "r");
  const g = tdValue(result, "/project1/RGBs", "g");
  const b = tdValue(result, "/project1/RGBs", "b");
  rgb.textContent = [r, g, b].some((item) => item !== undefined)
    ? `${fmt(r)}, ${fmt(g)}, ${fmt(b)}`
    : "-";

  raw.textContent = JSON.stringify(result, null, 2);
  renderLedPreviewFromState({ result });
}

function renderState(state) {
  statusText.textContent = state.message || state.status || "대기 중";
  document.body.dataset.status = state.status || "idle";
  setBusy(Boolean(state.running));

  if (state.result) {
    renderResult(state.result);
  }
  if (state.error && !state.result) {
    raw.textContent = JSON.stringify(state, null, 2);
  }
}

function renderLiveState(state) {
  liveStatusText.textContent = state.message || state.status || "실시간 대기 중";
  document.body.dataset.liveStatus = state.status || "idle";
  setLiveBusy(Boolean(state.running));

  if (state.latest) {
    liveArousal.textContent = fmt(state.latest.arousal_live);
    arousalConfidence.textContent = fmt(state.latest.arousal_confidence);
  }
  if (state.result) {
    renderResult(state.result);
  }
  renderLedPreviewFromState(state);
  if (state.error && !state.result) {
    raw.textContent = JSON.stringify(state, null, 2);
  }
}

async function getStatus() {
  const response = await fetch("/api/status");
  if (!response.ok) {
    throw new Error(`status failed: ${response.status}`);
  }
  return response.json();
}

async function getLiveStatus() {
  const response = await fetch("/api/live/status");
  if (!response.ok) {
    throw new Error(`live status failed: ${response.status}`);
  }
  return response.json();
}

async function pollStatus() {
  try {
    const state = await getStatus();
    renderState(state);
    if (!state.running && pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  } catch (error) {
    statusText.textContent = error.message;
    setBusy(false);
  }
}

async function pollLiveStatus() {
  try {
    const state = await getLiveStatus();
    renderLiveState(state);
    if (!state.running && livePollTimer) {
      clearInterval(livePollTimer);
      livePollTimer = null;
    }
  } catch (error) {
    liveStatusText.textContent = error.message;
    setLiveBusy(false);
  }
}

function ensureLivePoll() {
  if (!livePollTimer) {
    livePollTimer = setInterval(pollLiveStatus, 500);
  }
}

async function startAnalysis() {
  setBusy(true);
  statusText.textContent = "녹음 작업을 시작하는 중입니다.";
  const response = await fetch("/api/start", { method: "POST" });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `start failed: ${response.status}`);
  }
  renderState(payload.state);
  pollTimer = setInterval(pollStatus, 1000);
}

async function startLive() {
  setLiveBusy(true);
  liveStatusText.textContent = "실시간 모드를 시작하는 중입니다.";
  const response = await fetch("/api/live/start", { method: "POST" });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `live start failed: ${response.status}`);
  }
  renderLiveState(payload.state);
  ensureLivePoll();
}

async function stopLive() {
  liveStatusText.textContent = "실시간 모드를 정지하는 중입니다.";
  const response = await fetch("/api/live/stop", { method: "POST" });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `live stop failed: ${response.status}`);
  }
  renderLiveState(payload.state);
}

async function sendTest() {
  setBusy(true);
  statusText.textContent = "OSC 테스트를 전송하는 중입니다.";
  const response = await fetch("/api/test-osc", { method: "POST" });
  const payload = await response.json();
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || `test failed: ${response.status}`);
  }
  statusText.textContent = "OSC 테스트 전송 완료";
  renderResult(payload.result);
  setBusy(false);
}

startButton.addEventListener("click", () => {
  startAnalysis().catch((error) => {
    statusText.textContent = error.message;
    setBusy(false);
  });
});

liveStartButton.addEventListener("click", () => {
  startLive().catch((error) => {
    liveStatusText.textContent = error.message;
    setLiveBusy(false);
  });
});

liveStopButton.addEventListener("click", () => {
  stopLive().catch((error) => {
    liveStatusText.textContent = error.message;
  });
});

testButton.addEventListener("click", () => {
  sendTest().catch((error) => {
    statusText.textContent = error.message;
    setBusy(false);
  });
});

if (virtualScenarioSelect) {
  virtualScenarioSelect.addEventListener("change", renderVirtualExpected);
}

if (virtualRunButton) {
  virtualRunButton.addEventListener("click", () => {
    startVirtualMicScenario().catch((error) => {
      virtualStatusText.textContent = error.message;
      virtualRunButton.disabled = false;
    });
  });
}

if (debugSnapshotButton) {
  debugSnapshotButton.addEventListener("click", () => {
    runDebugAction("Snapshot running", () => debugGet("/api/debug/snapshot"));
  });
}

if (debugTdPingButton) {
  debugTdPingButton.addEventListener("click", () => {
    runDebugAction("TD ping running", () => debugPost("/api/debug/td-ping"));
  });
}

if (debugTdAuditButton) {
  debugTdAuditButton.addEventListener("click", () => {
    runDebugAction("TD audit running", () => debugPost("/api/debug/td-audit"));
  });
}

if (debugReadbackButton) {
  debugReadbackButton.addEventListener("click", () => {
    runDebugAction("Readback running", () => debugPost("/api/debug/td-readback"));
  });
}

if (debugOscButton) {
  debugOscButton.addEventListener("click", () => {
    runDebugAction("OSC pattern sending", () => debugPost("/api/debug/osc-pattern", {
      pattern: debugPatternSelect?.value || "neutral",
    }));
  });
}

if (debugSerialButton) {
  debugSerialButton.addEventListener("click", () => {
    runDebugAction("Serial payload sending", () => debugPost("/api/debug/serial-send", {
      valence: debugValenceInput?.value || 0,
      arousal: debugArousalInput?.value || 0,
    }));
  });
}

if (debugAudioProbeButton) {
  debugAudioProbeButton.addEventListener("click", () => {
    runDebugAction("Mic probe running", () => debugPost("/api/debug/audio-probe", {
      device: debugAudioDeviceSelect?.value || undefined,
      duration: 0.6,
    }));
  });
}

buildControllerPreview();
buildArduinoHardwarePreview();
renderLedPreviewFromState({});
setInterval(() => {
  if (!latestLedPayload) {
    return;
  }
  arduinoPreviewFrame = (arduinoPreviewFrame + 1) % ledPreviewModel.ARDUINO_SEND_STEPS;
  renderArduinoHardwarePreview(latestLedPayload);
}, 90);
loadVirtualMicScenarios()
  .then(pollVirtualMicStatus)
  .catch((error) => {
    if (virtualStatusText) {
      virtualStatusText.textContent = error.message;
    }
  });
pollStatus();
pollLiveStatus();
if (debugSnapshotButton) {
  debugGet("/api/debug/snapshot")
    .then(renderDebugPayload)
    .catch((error) => {
      debugStatusText.textContent = error.message;
    });
}
