/**
 * analyze.js — Analyze tab: temporal video + audio watermark detection.
 *
 * Sub-tabs: "an-video" (upload video → segment timeline + table + video player)
 *           "an-audio" (upload audio → segment timeline + table + audio player)
 *
 * mergeSegments() is defined in main.js (global scope).
 */

const $an = (id) => document.getElementById(id);

function escAnHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// ── Sub-tab switching ──────────────────────────────────────────────────────
function wireAnalyzeSubTabs() {
  document.querySelectorAll(".sub-tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.subtab;
      document.querySelectorAll(".sub-tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".sub-tab-content").forEach((c) => c.classList.add("hidden"));
      btn.classList.add("active");
      $an(tab).classList.remove("hidden");
    });
  });
}

// ── VIDEO ANALYSIS ─────────────────────────────────────────────────────────
let _anVideoObjectUrl = null;

function wireVideoDropZone() {
  const zone  = $an("an-drop-zone");
  const input = $an("an-file-input");
  zone.addEventListener("click", (e) => { if (!e.target.closest("label")) input.click(); });
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("dragover"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
    if (e.dataTransfer.files.length) setAnVideoFile(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", (e) => {
    if (e.target.files.length) setAnVideoFile(e.target.files[0]);
  });
}

function setAnVideoFile(file) {
  $an("an-file-name").textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`;
  const dt = new DataTransfer();
  dt.items.add(file);
  $an("an-file-input").files = dt.files;
  if (_anVideoObjectUrl) URL.revokeObjectURL(_anVideoObjectUrl);
  _anVideoObjectUrl = URL.createObjectURL(file);
}

async function handleVideoAnalyze() {
  const input  = $an("an-file-input");
  const status = $an("an-status");

  if (!input.files || !input.files.length) {
    status.textContent = "Vui lòng chọn file video.";
    status.className = "status-error";
    return;
  }

  const btn     = $an("an-btn-analyze");
  const spinner = $an("an-btn-spinner");
  btn.disabled = true;
  spinner.classList.remove("hidden");
  status.textContent = "Đang phân tích...";
  status.className = "status-info";
  $an("an-step-result").classList.add("hidden");

  try {
    const fd = new FormData();
    fd.append("video", input.files[0]);
    const resp = await fetch("/api/video/detect_temporal", { method: "POST", body: fd });
    if (!resp.ok) throw await resp.json().catch(() => ({}));
    const data = await resp.json();

    // Show video player with object URL
    const player = $an("an-video-player");
    if (_anVideoObjectUrl) {
      player.src = _anVideoObjectUrl;
      player.classList.remove("hidden");
    }

    const merged = mergeSegments(data.segments);
    renderVideoSummary(data);
    renderAnTimeline(data.segments, "an-timeline", getVideoSegColor);
    renderVideoTable(merged);
    $an("an-step-result").classList.remove("hidden");

    const detected = data.segments.filter((s) => s.detected).length;
    status.textContent =
      `Hoàn tất · ${data.total_duration_s}s · ${detected} / ${data.segments.length} phân đoạn có watermark`;
    status.className = "status-success";
  } catch (err) {
    status.textContent = `Lỗi: ${err.error || JSON.stringify(err)}`;
    status.className = "status-error";
  } finally {
    btn.disabled = false;
    spinner.classList.add("hidden");
  }
}

function renderVideoSummary(data) {
  const detected = data.segments.filter((s) => s.detected).length;
  $an("an-summary").textContent =
    `${data.total_duration_s}s · ${data.segments.length} phân đoạn · ${detected} phát hiện watermark`;
}

function getVideoSegColor(s) {
  if (s.detected) return "tl-green";
  if (s.bit_accuracy != null && s.bit_accuracy >= 0.5) return "tl-yellow";
  return "tl-red";
}

function renderVideoTable(segments) {
  const tbody  = $an("an-body");
  const player = $an("an-video-player");
  tbody.innerHTML = "";

  segments.forEach((s) => {
    const tr = document.createElement("tr");

    const statusHtml = s.detected
      ? `<span style="color:var(--success);font-weight:700">✓ Phát hiện</span>`
      : (s.bit_accuracy != null && s.bit_accuracy >= 0.5)
          ? `<span style="color:var(--warn);font-weight:700">~ Một phần</span>`
          : `<span style="color:var(--danger)">✗ Không</span>`;

    // Only show decoded text if ECC correctable
    const decoded = (s.correctable && s.decoded_text)
      ? `"${escAnHtml(s.decoded_text)}"`
      : `<span style="color:var(--text-muted)">—</span>`;

    const pct = s.bit_accuracy != null ? `${(s.bit_accuracy * 100).toFixed(1)}%` : "—";

    tr.style.cursor = "pointer";
    tr.innerHTML = `
      <td>${s.start_s}s – ${s.end_s}s</td>
      <td>${statusHtml}</td>
      <td>${decoded}</td>
      <td class="detail-only">${pct}</td>
    `;

    tr.addEventListener("click", () => {
      player.currentTime = s.start_s;
      player.play();
      tbody.querySelectorAll("tr").forEach((t) => t.classList.remove("selected-row"));
      tr.classList.add("selected-row");
      player.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });

    tbody.appendChild(tr);
  });
}

// ── AUDIO ANALYSIS ─────────────────────────────────────────────────────────
let _anAudioObjectUrl = null;

function wireAnAudioDropZone() {
  const zone  = $an("an-audio-drop-zone");
  const input = $an("an-audio-file-input");
  zone.addEventListener("click", (e) => { if (!e.target.closest("label")) input.click(); });
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("dragover"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
    if (e.dataTransfer.files.length) setAnAudioFile(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", (e) => {
    if (e.target.files.length) setAnAudioFile(e.target.files[0]);
  });
}

function setAnAudioFile(file) {
  $an("an-audio-file-name").textContent = `${file.name} (${(file.size / 1024).toFixed(0)} KB)`;
  const dt = new DataTransfer();
  dt.items.add(file);
  $an("an-audio-file-input").files = dt.files;
  if (_anAudioObjectUrl) URL.revokeObjectURL(_anAudioObjectUrl);
  _anAudioObjectUrl = URL.createObjectURL(file);
}

async function handleAudioAnalyze() {
  const input  = $an("an-audio-file-input");
  const status = $an("an-audio-status");

  if (!input.files || !input.files.length) {
    status.textContent = "Vui lòng chọn file audio.";
    status.className = "status-error";
    return;
  }

  const btn     = $an("an-audio-btn-analyze");
  const spinner = $an("an-audio-btn-spinner");
  btn.disabled = true;
  spinner.classList.remove("hidden");
  status.textContent = "Đang phân tích...";
  status.className = "status-info";
  $an("an-audio-step-result").classList.add("hidden");

  try {
    const fd = new FormData();
    fd.append("audio", input.files[0]);
    const resp = await fetch("/api/audio/detect_temporal", { method: "POST", body: fd });
    if (!resp.ok) throw await resp.json().catch(() => ({}));
    const data = await resp.json();

    // Show audio player
    const player = $an("an-audio-player");
    if (_anAudioObjectUrl) {
      player.src = _anAudioObjectUrl;
      player.classList.remove("hidden");
    }

    const merged = mergeSegments(data.segments);
    renderAudioSummary(data);
    renderAnTimeline(data.segments, "an-audio-timeline", getAudioSegColor);
    renderAudioTable(merged);
    $an("an-audio-step-result").classList.remove("hidden");

    const detected = data.segments.filter((s) => s.detected).length;
    status.textContent =
      `Hoàn tất · ${data.total_duration_s}s · ${detected} / ${data.segments.length} phân đoạn có watermark`;
    status.className = "status-success";
  } catch (err) {
    status.textContent = `Lỗi: ${err.error || JSON.stringify(err)}`;
    status.className = "status-error";
  } finally {
    btn.disabled = false;
    spinner.classList.add("hidden");
  }
}

function renderAudioSummary(data) {
  const detected = data.segments.filter((s) => s.detected).length;
  $an("an-audio-summary").textContent =
    `${data.total_duration_s}s · ${data.segments.length} phân đoạn · ${detected} phát hiện watermark`;
}

function getAudioSegColor(s) {
  const p = s.detection_prob || 0;
  if (p >= 0.7) return "tl-green";
  if (p >= 0.5) return "tl-yellow";
  return "tl-red";
}

function renderAudioTable(segments) {
  const tbody  = $an("an-audio-body");
  const player = $an("an-audio-player");
  tbody.innerHTML = "";

  segments.forEach((s) => {
    const tr   = document.createElement("tr");
    const prob = s.detection_prob || 0;

    const statusHtml = prob >= 0.7
      ? `<span style="color:var(--success);font-weight:700">✓ Phát hiện</span>`
      : prob >= 0.5
          ? `<span style="color:var(--warn);font-weight:700">~ Một phần</span>`
          : `<span style="color:var(--danger)">✗ Không</span>`;

    // Only show decoded text when actually detected
    const decoded = (s.detected && s.decoded_text)
      ? `"${escAnHtml(s.decoded_text)}"`
      : `<span style="color:var(--text-muted)">—</span>`;

    const pct = `${(prob * 100).toFixed(1)}%`;

    tr.style.cursor = "pointer";
    tr.innerHTML = `
      <td>${s.start_s}s – ${s.end_s}s</td>
      <td>${statusHtml}</td>
      <td>${decoded}</td>
      <td class="detail-only">${pct}</td>
    `;

    tr.addEventListener("click", () => {
      player.currentTime = s.start_s;
      player.play();
      tbody.querySelectorAll("tr").forEach((t) => t.classList.remove("selected-row"));
      tr.classList.add("selected-row");
      player.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });

    tbody.appendChild(tr);
  });
}

// ── Shared timeline renderer ───────────────────────────────────────────────
function renderAnTimeline(segments, containerId, colorFn) {
  const bar = $an(containerId);
  bar.innerHTML = "";
  segments.forEach((s) => {
    const div = document.createElement("div");
    div.className = "timeline-seg " + colorFn(s);
    div.title = `${s.start_s}s – ${s.end_s}s`;
    bar.appendChild(div);
  });
}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  wireAnalyzeSubTabs();
  wireVideoDropZone();
  wireAnAudioDropZone();
  $an("an-btn-analyze").addEventListener("click", handleVideoAnalyze);
  $an("an-audio-btn-analyze").addEventListener("click", handleAudioAnalyze);
});
