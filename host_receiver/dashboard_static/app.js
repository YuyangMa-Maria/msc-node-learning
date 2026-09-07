const state = {
  nodes: {},
  fusion: null,
  mode: "unknown",
  ble: false,
  marh: false,
  modelVersion: null,
  nodeStates: {},
  risks: [],
  events: 0,
};

function updateConnectivity(nodes) {
  state.nodeStates = nodes || {};
  ["VSN", "ASN"].forEach((node) => {
    const online = Boolean(state.nodeStates[node]?.online);
    const element = $(`${node.toLowerCase()}-online`);
    element.textContent = online ? "Online" : "Offline";
    element.classList.toggle("offline", !online);
  });
}

const $ = (id) => document.getElementById(id);
const fmt = (value, digits = 3) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "--";
const timeLabel = (iso) => iso ? new Date(iso).toLocaleTimeString([], {hour12: false}) : "--";
const humanise = (value) => String(value || "--").replaceAll("_", " ");

function setBadge(element, status) {
  element.textContent = status || "Waiting";
  element.className = `badge ${status || "neutral"}`;
}

function pulse(id) {
  const element = $(id);
  element.classList.add("active");
  setTimeout(() => element.classList.remove("active"), 900);
}

function setWorkflow(id, kind = "complete") {
  const element = $(id);
  element.classList.remove("active", "complete");
  element.classList.add(kind);
}

function resetWorkflow() {
  document.querySelectorAll(".workflow-step").forEach((step) => step.classList.remove("active", "complete"));
}

function updateNode(event) {
  const key = event.node.toLowerCase();
  state.nodes[event.node] = event;
  $(`${key}-risk`).textContent = fmt(event.risk_score);
  $(`${key}-confidence`).textContent = fmt(event.confidence);
  $(`${key}-risk-bar`).style.width = `${Math.max(0, Math.min(1, event.risk_score)) * 100}%`;
  setBadge($(`${key}-state`), event.status);
  $(`${key}-meta`).textContent = `Sequence ${event.sequence} at ${timeLabel(event.host_time_iso)}`;
  if (event.node === "VSN") {
    const total = event.latency_ms?.total;
    $("vsn-latency").textContent = total >= 0 ? `${fmt(total, 1)} ms` : "--";
    pulse("vsn-card");
    setWorkflow("step-vsn");
    setWorkflow("step-transfer", "active");
    $("vsn-link").classList.add("active");
  } else if (event.node === "ASN") {
    pulse("asn-card");
    setWorkflow("step-asn");
    setWorkflow("step-transfer");
    $("asn-link").classList.add("active");
  }
}

function updateFusion(event) {
  state.fusion = event;
  const fusion = event.fusion;
  const alert = event.alert || {};
  const severity = alert.severity || "info";
  const action = humanise(alert.action);
  $("fusion-risk").textContent = fmt(fusion.risk_score);
  $("risk-level").textContent = `L${fusion.risk_level}`;
  $("risk-level").style.background = fusion.risk_level >= 5 ? "#c92a2a" : fusion.risk_level >= 3 ? "#c77c02" : "#2b8a3e";
  $("risk-trend").textContent = fusion.risk_trend;
  $("active-nodes").textContent = fusion.active_nodes;
  $("alert-summary").className = `alert-summary ${severity}`;
  $("alert-severity").className = `alert-severity ${severity}`;
  $("alert-severity").textContent = severity.toUpperCase();
  $("alert-headline").textContent = alert.headline || "Regional risk update";
  $("alert-message").textContent = alert.message || "No warning text received.";
  $("alert-evidence").textContent = alert.evidence || "none";
  $("alert-action").textContent = action;

  const banner = $("alert-banner");
  banner.className = `alert-banner ${severity}${alert.attention_required ? "" : " hidden"}`;
  $("alert-banner-severity").className = `alert-severity ${severity}`;
  $("alert-banner-severity").textContent = severity.toUpperCase();
  $("alert-banner-headline").textContent = alert.headline || "Regional risk update";
  $("alert-banner-message").textContent = alert.message || "No warning text received.";
  $("alert-banner-evidence").textContent = alert.evidence || "none";
  $("alert-banner-action").textContent = action;

  const marh = event.marh || {};
  $("marh-recommendation").textContent = marh.recommended
    ? `Recommended: ${humanise(marh.reason)}`
    : "Not requested";
  state.modelVersion = event.model.version;
  $("model-version").textContent = `0x${Number(event.model.version).toString(16).padStart(4, "0")}`;
  state.risks.push({value: fusion.risk_score, level: fusion.risk_level});
  state.risks = state.risks.slice(-30);
  drawChart();
  pulse("fusion-card");
  pulse("receiver-card");
  setWorkflow("step-fusion");
  setWorkflow("step-alert");
  $("receiver-link").classList.add("active");
}

function drawChart() {
  const canvas = $("risk-chart");
  const scale = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 600;
  const height = canvas.clientHeight || 210;
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#d6dde3";
  ctx.lineWidth = 1;
  [0, .25, .5, .75, 1].forEach((value) => {
    const y = 12 + (1 - value) * (height - 32);
    ctx.beginPath(); ctx.moveTo(38, y); ctx.lineTo(width - 8, y); ctx.stroke();
    ctx.fillStyle = "#61707e"; ctx.font = "10px Segoe UI"; ctx.fillText(value.toFixed(2), 4, y + 3);
  });
  if (state.risks.length < 2) return;
  ctx.strokeStyle = "#0b7285";
  ctx.lineWidth = 2;
  ctx.beginPath();
  state.risks.forEach((point, index) => {
    const x = 38 + index * (width - 50) / Math.max(1, state.risks.length - 1);
    const y = 12 + (1 - point.value) * (height - 32);
    index === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function eventResult(event) {
  if (event.event === "node_output") return `risk ${fmt(event.risk_score)} / confidence ${fmt(event.confidence)}`;
  if (event.event === "fusion") return `L${event.fusion.risk_level} ${fmt(event.fusion.risk_score)} / ${event.alert.headline}`;
  if (event.event === "command_result") return `${event.success ? "PASS" : "FAIL"}: ${event.detail}`;
  if (event.event === "ble_state") return event.connected ? "connected" : `disconnected (${event.reason})`;
  if (event.event === "sampling_state") return `${event.mode}, ASN sync ${event.asn_synchronised}`;
  if (event.event === "marh_state") return event.active ? "active" : "inactive";
  return "received";
}

function appendEvent(event) {
  state.events += 1;
  $("event-count").textContent = `${state.events} events`;
  const row = document.createElement("tr");
  const context = event.node || event.mode || event.command || event.peer || "system";
  [timeLabel(event.host_time_iso), event.event, context, eventResult(event)].forEach((value) => {
    const cell = document.createElement("td");
    cell.textContent = value;
    row.appendChild(cell);
  });
  const log = $("event-log");
  log.prepend(row);
  while (log.children.length > 50) log.removeChild(log.lastChild);
}

function applyEvent(event, addToLog = true) {
  if (event.event === "node_output") updateNode(event);
  if (event.event === "fusion") updateFusion(event);
  if (event.event === "ble_state") {
    state.ble = Boolean(event.connected);
    $("ble-state").textContent = state.ble ? "Connected" : "Offline";
    updateConnectivity({...state.nodeStates, ASN: {...state.nodeStates.ASN, online: state.ble}});
  }
  if (event.event === "node_state") updateConnectivity(event.nodes);
  if (event.event === "marh_state") {
    state.marh = Boolean(event.active);
    $("marh-state").textContent = state.marh ? "MARH active" : "Receiver";
    $("marh-state").className = `badge ${state.marh ? "normal" : "neutral"}`;
  }
  if (event.event === "sampling_state") setMode(event.mode);
  if (event.event === "command_result") {
    $("last-command").textContent = `${event.command}: ${event.success ? "PASS" : "FAIL"}`;
    if (event.command?.startsWith("SAMPLE_") && event.success) setWorkflow("step-request");
  }
  if (addToLog) appendEvent(event);
}

function setMode(mode) {
  state.mode = mode || "automatic";
  $("mode-auto").classList.toggle("active", state.mode === "automatic");
  $("mode-manual").classList.toggle("active", state.mode === "manual");
  document.querySelectorAll("[data-command^='SAMPLE']").forEach((button) => {
    button.disabled = state.mode !== "manual";
  });
}

async function sendCommand(command) {
  if (command.startsWith("SAMPLE")) {
    resetWorkflow();
    setWorkflow("step-request", "active");
    $("last-command").textContent = `${command}: requested`;
  }
  const response = await fetch("/api/command", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({command}),
  });
  if (!response.ok) {
    const body = await response.json();
    $("last-command").textContent = `${command}: ${body.error || "rejected"}`;
  }
}

document.querySelectorAll("[data-command]").forEach((button) => {
  button.addEventListener("click", () => sendCommand(button.dataset.command));
});

async function initialise() {
  const response = await fetch("/api/state");
  const initial = await response.json();
  state.ble = initial.ble_connected;
  state.marh = initial.marh_active;
  updateConnectivity(initial.node_states || {});
  setMode(initial.sampling_mode);
  (initial.history || []).forEach((event) => applyEvent(event, event.event !== "node_state"));
  $("ble-state").textContent = state.ble ? "Connected" : "Offline";
  $("marh-state").textContent = state.marh ? "MARH active" : "Receiver";

  const stream = new EventSource("/api/events");
  stream.onopen = () => {
    $("stream-status").classList.remove("offline");
    $("stream-label").textContent = "Receiver online";
    sendCommand("NODE STATUS");
  };
  stream.onerror = () => {
    $("stream-status").classList.add("offline");
    $("stream-label").textContent = "Reconnecting";
  };
  stream.onmessage = (message) => {
    const event = JSON.parse(message.data);
    applyEvent(event, event.event !== "node_state");
  };
}

function updateClock() {
  $("clock").textContent = new Date().toLocaleTimeString([], {hour12: false});
}
updateClock();
setInterval(updateClock, 1000);
setInterval(() => sendCommand("NODE STATUS").catch(() => {}), 3000);
window.addEventListener("resize", drawChart);
initialise().catch((error) => {
  $("stream-label").textContent = error.message;
});
