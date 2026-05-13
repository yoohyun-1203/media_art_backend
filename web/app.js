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

let pollTimer = null;
let livePollTimer = null;

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

function tdValue(result, path, name) {
  return result?.touchdesigner?.[path]?.[name];
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

pollStatus();
pollLiveStatus();
