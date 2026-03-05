/**
 * main.js — VideoSeal Demo UI logic.
 *
 * State machine:
 *   sessionId    → set after successful embed
 *   originalBits → 256-bit array from embed response
 *   eccType      → "rs" | "bch"
 *   rsK          → current RS data bytes (k)
 *   bchT         → current BCH correction capability (t)
 */

// ── Configuration ─────────────────────────────────────────────────────────

const ATTACKS = [
  { key: "screen_recording",  label: "Quay màn hình" },
  { key: "upscale_downscale", label: "Upscale / Downscale" },
  { key: "brightness",        label: "Đổi độ sáng/tối" },
  { key: "h264",              label: "Re-encode H.264" },
  { key: "h265",              label: "Re-encode H.265" },
  { key: "crop",              label: "Crop 10–20%" },
  { key: "gaussian_noise",    label: "Thêm noise" },
  { key: "logo_overlay",      label: "Chèn logo" },
  { key: "bitrate_reduce",   label: "Giảm bitrate (1 Mbps)" },
];

const RS_TOTAL       = 32;
const PASS_THRESHOLD = 0.70;

// ── State ─────────────────────────────────────────────────────────────────
let sessionId    = null;
let originalBits = null;
let originalText = null;
let eccType      = "bch";  // fixed: BCH mode
let rsK          = 20;

// ── DOM Helpers ───────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

function show(id)  { $(id).classList.remove("hidden"); }
function hide(id)  { $(id).classList.add("hidden"); }

function setStatus(msg, level) {
  const el = $("embed-status");
  el.textContent = msg;
  el.className   = `status-${level}`;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function setLoading(btnTextId, spinnerId, isLoading) {
  const btn = document.querySelector(`#${spinnerId}`).closest("button");
  btn.disabled           = isLoading;
  $(spinnerId).classList.toggle("hidden", !isLoading);
}

// ── Initialization ────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  buildAttackCheckboxes();
  wireDropZone();
  wireTextLengthHint();
  initDetailMode();

  $("btn-embed").addEventListener("click", handleEmbed);
  $("btn-run-selected").addEventListener("click", () => handleAttacks(false));
  $("btn-run-all").addEventListener("click",      () => handleAttacks(true));
  $("upload-attack-input").addEventListener("change", handleUploadAttackVideo);
  $("btn-temporal-session").addEventListener("click", handleTemporalAnalysis);

  $("detail-toggle").addEventListener("click", handleDetailToggle);
  $("detail-pw-submit").addEventListener("click", submitDetailPassword);
  $("detail-pw-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") submitDetailPassword();
    if (e.key === "Escape") hide("detail-pw-form");
  });
});


// ── Text length hint ───────────────────────────────────────────────────────
const BCH_DATA_BYTES = 4;   // fixed BCH: 4 data bytes

function wireTextLengthHint() {
  const input    = $("wm-text");
  const maxBytes = BCH_DATA_BYTES;

  const update = () => {
    const bytes = new TextEncoder().encode(input.value).length;
    const hint  = $("text-byte-hint");
    if (bytes > maxBytes) {
      hint.textContent = `⚠ Text dài ${bytes} bytes, vượt quá ${maxBytes} bytes. Rút ngắn hoặc điều chỉnh tham số.`;
      hint.style.color = "#e74c3c";
    } else {
      hint.textContent = `${bytes} / ${maxBytes} bytes`;
      hint.style.color = "";
    }
  };

  input.removeEventListener("input", input._hintHandler);
  input._hintHandler = update;
  input.addEventListener("input", update);
  update();
}

// ── Drop Zone ─────────────────────────────────────────────────────────────
function wireDropZone() {
  const zone  = $("drop-zone");
  const input = $("file-input");

  zone.addEventListener("click", (e) => { if (!e.target.closest("label")) input.click(); });
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("dragover");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  });

  input.addEventListener("change", (e) => {
    if (e.target.files.length) setSelectedFile(e.target.files[0]);
  });
}

function setSelectedFile(file) {
  $("file-name").textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`;
  const dt = new DataTransfer();
  dt.items.add(file);
  $("file-input").files = dt.files;
}

// ── Attack Checkboxes ─────────────────────────────────────────────────────
function buildAttackCheckboxes() {
  const div = $("attack-checkboxes");
  ATTACKS.forEach(({ key, label }) => {
    const item = document.createElement("div");
    item.className = "attack-item";

    const cb    = document.createElement("input");
    cb.type     = "checkbox";
    cb.id       = `atk-${key}`;
    cb.value    = key;
    cb.checked  = true;

    const lbl   = document.createElement("label");
    lbl.htmlFor = `atk-${key}`;
    lbl.textContent = label;

    item.addEventListener("click", (e) => {
      if (e.target !== cb) cb.checked = !cb.checked;
    });

    item.appendChild(cb);
    item.appendChild(lbl);
    div.appendChild(item);
  });
}

// ── Embed Handler ─────────────────────────────────────────────────────────
async function handleEmbed() {
  const fileInput = $("file-input");
  const text      = $("wm-text").value.trim();

  eccType = "bch";
  rsK = parseInt($("rs-k")?.value) || 20;

  if (!fileInput.files || !fileInput.files.length) {
    setStatus("Vui lòng chọn file video.", "error");
    return;
  }
  if (!text) {
    setStatus("Vui lòng nhập văn bản watermark.", "error");
    return;
  }

  let maxBytes;
  if (eccType === "bch")       maxBytes = BCH_DATA_BYTES;
  else if (eccType === "ldpc") maxBytes = LDPC_DATA_BYTES;
  else                         maxBytes = rsK;

  const textBytes = new TextEncoder().encode(text).length;
  if (textBytes > maxBytes) {
    let paramHint;
    paramHint = `Chế độ này cho phép tối đa ${maxBytes} ký tự`;
    setStatus(`Text (${textBytes} bytes) vượt quá giới hạn. ${paramHint}.`, "error");
    return;
  }

  setLoading("btn-embed-text", "btn-embed-spinner", true);
  setStatus("Đang upload và xử lý...", "info");

  hide("step-videos");
  hide("step-bits");
  hide("step-attacks");
  hide("results-wrap");
  hide("attack-preview");

  try {
    const result = await apiEncode(
      fileInput.files[0],
      text,
      rsK,
      (pct) => setStatus(`Đang xử lý... ${pct}%`, "info"),
      eccType,
    );

    sessionId    = result.session_id;
    originalBits = result.bits_list;
    originalText = text;

    // Step 2: videos
    $("vid-original").src    = result.original_url;
    $("vid-watermarked").src = result.watermarked_url;
    $("video-meta").textContent =
      `${result.num_frames} frames · ${result.fps.toFixed(1)} fps · embed time: ${result.embed_time_s}s`;
    show("step-videos");

    // Step 3: bits
    const canvas = $("bits-canvas-embed");
    initAndRender(canvas, result.bits_list);

    if (result.ecc_type === "bch") {
      $("bits-meta").textContent =
        `Chế độ nâng cao · 4 ký tự · sửa ≤28 bit lỗi`;
    } else if (result.ecc_type === "ldpc") {
      $("bits-meta").textContent =
        `Chế độ tối ưu · ${result.max_text_bytes} ký tự`;
    } else {
      $("bits-meta").textContent =
        `Chế độ cơ bản · ${result.k} ký tự · sửa ≤${result.max_errors} lỗi`;
    }
    $("bits-hex").textContent = result.codeword_hex;
    show("step-bits");

    // Step 4: attacks
    show("step-attacks");

    // Step 5: temporal analysis
    show("step-temporal");

    setStatus(
      `Watermark nhúng thành công trong ${result.embed_time_s}s ✓`,
      "success",
    );
  } catch (err) {
    setStatus(`Lỗi: ${err.error || JSON.stringify(err)}`, "error");
  } finally {
    setLoading("btn-embed-text", "btn-embed-spinner", false);
  }
}

// ── Attack Handler ────────────────────────────────────────────────────────
async function handleAttacks(runAll) {
  if (!sessionId) {
    alert("Vui lòng embed watermark trước.");
    return;
  }

  let attacks;
  if (runAll) {
    attacks = ATTACKS.map((a) => a.key);
  } else {
    attacks = ATTACKS
      .filter(({ key }) => $(`atk-${key}`) && $(`atk-${key}`).checked)
      .map(({ key }) => key);
  }

  if (!attacks.length) {
    alert("Chọn ít nhất một attack.");
    return;
  }

  const selSpinner = $("btn-sel-spinner");
  const allSpinner = $("btn-all-spinner");
  $("btn-run-selected").disabled = true;
  $("btn-run-all").disabled      = true;
  selSpinner.classList.remove("hidden");
  allSpinner.classList.remove("hidden");

  hide("results-wrap");
  hide("attack-preview");

  try {
    const { results } = await apiRunAttacks(sessionId, attacks, rsK, eccType);
    renderResultsTable(results);
    show("results-wrap");
  } catch (err) {
    alert(`Lỗi attack: ${err.error || JSON.stringify(err)}`);
  } finally {
    $("btn-run-selected").disabled = false;
    $("btn-run-all").disabled      = false;
    selSpinner.classList.add("hidden");
    allSpinner.classList.add("hidden");
  }
}

// ── Results Table ─────────────────────────────────────────────────────────
function renderResultsTable(results) {
  const tbody = $("results-body");
  tbody.innerHTML = "";

  results.forEach((r) => {
    const tr = document.createElement("tr");
    tr.className = r.pass ? "row-pass" : "row-fail";

    const pct    = (r.bit_accuracy * 100).toFixed(1);
    const passed = r.pass ? "✓ Phát hiện" : "✗ Không phát hiện";

    // ECC info — unit depends on eccType
    let eccText;
    if (r.error) {
      eccText = `<span style="color:var(--danger)">Error: ${escapeHtml(r.error)}</span>`;
    } else if (r.correctable) {
      if (r.errors_corrected > 0) {
        const unit = (eccType === "bch" || eccType === "ldpc") ? "bit" : "byte";
        eccText = `<span style="color:var(--warn)">Sửa ${r.errors_corrected} ${unit} lỗi</span>`;
      } else {
        eccText = `<span style="color:var(--success)">Không lỗi</span>`;
      }
    } else {
      eccText = `<span style="color:var(--danger)">Không thể sửa</span>`;
    }

    // Decoded text — always show something; mark as raw when ECC failed
    let decoded;
    if (r.error) {
      decoded = `<em style="color:var(--text-muted)">—</em>`;
    } else if (!r.correctable) {
      decoded = `"${escapeHtml(r.decoded_text)}" <span style="color:var(--warn);font-size:.78rem">(raw)</span>`;
    } else {
      decoded = `"${escapeHtml(r.decoded_text)}"`;
    }

    // Accuracy bar
    const fillClass = r.pass ? "pass" : "fail";
    tr.innerHTML = `
      <td>${escapeHtml(r.display_name)}</td>
      <td class="detail-only">
        <div class="acc-bar">
          <div class="acc-track">
            <div class="acc-fill ${fillClass}" style="width:${pct}%"></div>
          </div>
          <span>${pct}%</span>
        </div>
      </td>
      <td>${passed}</td>
      <td>${decoded}</td>
      <td class="detail-only">${eccText}</td>
      <td>${r.process_time_s}s</td>
    `;

    tr.addEventListener("click", () => {
      tbody.querySelectorAll("tr").forEach((t) => t.classList.remove("selected-row"));
      tr.classList.add("selected-row");
      previewAttack(r);
    });

    tbody.appendChild(tr);
  });
}

// ── Append single row to results table (used by upload flow) ─────────────
function appendResultRow(r) {
  const tbody = $("results-body");

  // Remove any previous upload row so re-uploads don't pile up
  const existing = tbody.querySelector("[data-upload-row]");
  if (existing) existing.remove();

  const tr = document.createElement("tr");
  tr.className = r.pass ? "row-pass" : "row-fail";
  tr.dataset.uploadRow = "1";

  const pct    = (r.bit_accuracy * 100).toFixed(1);
  const passed = r.pass ? "✓ Phát hiện" : "✗ Không phát hiện";

  let eccText;
  if (r.error) {
    eccText = `<span style="color:var(--danger)">Error: ${escapeHtml(r.error)}</span>`;
  } else if (r.correctable) {
    const unit = (eccType === "bch" || eccType === "ldpc") ? "bit" : "byte";
    eccText = r.errors_corrected > 0
      ? `<span style="color:var(--warn)">Sửa ${r.errors_corrected} ${unit} lỗi</span>`
      : `<span style="color:var(--success)">Không lỗi</span>`;
  } else {
    eccText = `<span style="color:var(--danger)">Không thể sửa</span>`;
  }

  let decoded;
  if (r.error) {
    decoded = `<em style="color:var(--text-muted)">—</em>`;
  } else if (!r.correctable) {
    decoded = `"${escapeHtml(r.decoded_text)}" <span style="color:var(--warn);font-size:.78rem">(raw)</span>`;
  } else {
    decoded = `"${escapeHtml(r.decoded_text)}"`;
  }

  const fillClass = r.pass ? "pass" : "fail";
  tr.innerHTML = `
    <td>${escapeHtml(r.display_name)}</td>
    <td class="detail-only">
      <div class="acc-bar">
        <div class="acc-track">
          <div class="acc-fill ${fillClass}" style="width:${pct}%"></div>
        </div>
        <span>${pct}%</span>
      </div>
    </td>
    <td>${passed}</td>
    <td>${decoded}</td>
    <td class="detail-only">${eccText}</td>
    <td>${r.process_time_s}s</td>
  `;

  tr.addEventListener("click", () => {
    tbody.querySelectorAll("tr").forEach((t) => t.classList.remove("selected-row"));
    tr.classList.add("selected-row");
    previewAttack(r);
  });

  tbody.appendChild(tr);
}


// ── Upload External Attack Video ──────────────────────────────────────────
async function handleUploadAttackVideo(e) {
  const file = e.target.files[0];
  if (!file) return;
  if (!sessionId) {
    alert("Vui lòng nhúng watermark trước (Bước 2).");
    e.target.value = "";
    return;
  }

  const btnText    = $("upload-attack-btn-text");
  const spinner    = $("upload-attack-spinner");
  const statusDiv  = $("upload-attack-status");

  btnText.textContent = file.name;
  show("upload-attack-spinner");
  statusDiv.textContent = "Đang upload và phân tích...";
  statusDiv.className   = "upload-attack-status";

  try {
    const label  = `Upload: ${file.name}`;
    const result = await apiUploadAttackVideo(
      sessionId, file, label,
      (pct) => { btnText.textContent = `${pct}% — ${file.name}`; }
    );

    // Append to results table (or create it if not yet shown)
    show("results-wrap");
    appendResultRow(result);

    statusDiv.textContent = "✓ Phân tích xong — xem kết quả bên dưới";
    statusDiv.classList.add("upload-attack-status--ok");
    previewAttack(result);
  } catch (err) {
    statusDiv.textContent = `Lỗi: ${err.error || JSON.stringify(err)}`;
    statusDiv.classList.add("upload-attack-status--err");
  } finally {
    hide("upload-attack-spinner");
    btnText.textContent = "Chọn file video...";
    e.target.value = "";   // allow re-uploading same file
  }
}


// ── Attack Preview ────────────────────────────────────────────────────────
function previewAttack(r) {
  $("preview-title").textContent = r.display_name;

  const vid = $("vid-attacked");
  if (r.attacked_url) {
    vid.src = r.attacked_url;
    vid.load();
  }

  // Render original key (no error highlighting)
  const origCanvas = $("bits-canvas-orig-preview");
  if (originalBits) {
    initAndRender(origCanvas, originalBits);
  }

  // Render decoded key (errors highlighted red compared to original)
  const canvas = $("bits-canvas-attacked");
  if (r.bits_list && r.bits_list.length === 256) {
    initAndRender(canvas, r.bits_list, originalBits);
  }

  // Correct/incorrect badge next to "Key decode" label
  const badge   = $("decode-correct-badge");
  const isMatch = r.decoded_text !== null && r.decoded_text === originalText;
  if (r.error) {
    badge.innerHTML = `<span class="key-badge key-badge--error">✗ Lỗi</span>`;
  } else if (isMatch) {
    badge.innerHTML = `<span class="key-badge key-badge--ok">✓ Đúng</span>`;
  } else if (r.decoded_text !== null) {
    badge.innerHTML = `<span class="key-badge key-badge--warn">≠ Sai lệch</span>`;
  } else {
    badge.innerHTML = `<span class="key-badge key-badge--error">✗ Không decode được</span>`;
  }

  const pct     = (r.bit_accuracy * 100).toFixed(1);
  const errBits = originalBits && r.bits_list
    ? originalBits.filter((b, i) => b !== r.bits_list[i]).length
    : "?";

  // Text comparison rows
  const origTextHtml = originalText !== null
    ? `"${escapeHtml(originalText)}"`
    : `<em style="color:var(--text-muted)">—</em>`;

  let decodedTextHtml;
  if (r.error) {
    decodedTextHtml = `<span style="color:var(--danger)">Lỗi xử lý</span>`;
  } else if (!r.correctable) {
    // ECC failed — show raw decoded bytes with warning label
    decodedTextHtml = `"${escapeHtml(r.decoded_text)}" <span style="color:var(--warn);font-size:.78rem">(không khôi phục được)</span>`;
  } else {
    decodedTextHtml = `"${escapeHtml(r.decoded_text)}"`;
  }

  const textMatchIcon = (r.decoded_text !== null && r.decoded_text === originalText)
    ? `<span style="color:var(--success)">✓</span>`
    : `<span style="color:var(--danger)">✗</span>`;

  $("preview-stats").innerHTML =
    `<table class="key-compare-table">` +
    `<tr class="detail-only"><td class="kc-label">Bit accuracy</td>` +
        `<td><strong>${pct}%</strong> &nbsp;(${errBits} bit lỗi / 256)</td></tr>` +
    `<tr><td class="kc-label">Text gốc</td>` +
        `<td>${origTextHtml}</td></tr>` +
    `<tr><td class="kc-label">Text decode</td>` +
        `<td>${decodedTextHtml} &nbsp;${textMatchIcon}</td></tr>` +
    `</table>`;

  show("attack-preview");
  $("attack-preview").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Merge consecutive segments with same decoded key ─────────────────────
function mergeSegments(segments) {
  if (!segments.length) return [];

  // A segment is "decodable" if correctable (video) or detected (audio)
  const getKey = (s) => {
    const ok = (s.correctable !== false) && (s.detected !== false);
    return (ok && s.decoded_text) ? s.decoded_text : null;
  };

  const merged = [];
  let cur = { ...segments[0] };

  for (let i = 1; i < segments.length; i++) {
    const s = segments[i];
    if (getKey(cur) === getKey(s)) {
      cur.end_s = s.end_s;
      if (s.detected) cur.detected = true;
      if (s.bit_accuracy != null &&
          (cur.bit_accuracy == null || s.bit_accuracy > cur.bit_accuracy))
        cur.bit_accuracy = s.bit_accuracy;
      if (s.detection_prob != null &&
          (cur.detection_prob == null || s.detection_prob > cur.detection_prob))
        cur.detection_prob = s.detection_prob;
    } else {
      merged.push(cur);
      cur = { ...s };
    }
  }
  merged.push(cur);
  return merged;
}

// ── Detail Mode ───────────────────────────────────────────────────────────
function initDetailMode() {
  if (sessionStorage.getItem("detail_unlocked") === "1") {
    document.body.classList.add("detail-mode");
    $("detail-toggle").textContent = "🔓";
  }
}

function handleDetailToggle() {
  const body = document.body;
  if (body.classList.contains("detail-mode")) {
    body.classList.remove("detail-mode");
    sessionStorage.removeItem("detail_unlocked");
    $("detail-toggle").textContent = "🔒";
    hide("detail-pw-form");
  } else {
    const form = $("detail-pw-form");
    if (form.classList.contains("hidden")) {
      show("detail-pw-form");
      $("detail-pw-input").focus();
    } else {
      hide("detail-pw-form");
    }
  }
}

function submitDetailPassword() {
  if ($("detail-pw-input").value === "HDWM") {
    document.body.classList.add("detail-mode");
    sessionStorage.setItem("detail_unlocked", "1");
    $("detail-toggle").textContent = "🔓";
    hide("detail-pw-form");
    hide("detail-pw-error");
    $("detail-pw-input").value = "";
  } else {
    show("detail-pw-error");
    $("detail-pw-input").select();
  }
}

// ── Temporal Analysis ─────────────────────────────────────────────────────
async function handleTemporalAnalysis() {
  if (!sessionId) return;

  const btn     = $("btn-temporal-session");
  const spinner = $("btn-temporal-spinner");
  const status  = $("temporal-status");

  btn.disabled = true;
  spinner.classList.remove("hidden");
  status.textContent = "Đang phân tích...";
  status.className = "status-info";
  hide("temporal-result");

  try {
    const resp = await fetch("/api/video/detect_temporal", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ session_id: sessionId }),
    });
    if (!resp.ok) throw await resp.json().catch(() => ({}));
    const data = await resp.json();

    renderTimeline(data.segments, "temporal-timeline");
    renderTemporalTable(data.segments, "temporal-body");
    show("temporal-result");

    const detected = data.segments.filter((s) => s.detected).length;
    status.textContent =
      `${data.total_duration_s}s · ${data.segments.length} phân đoạn · ${detected} có watermark`;
    status.className = "status-success";
  } catch (err) {
    status.textContent = `Lỗi: ${err.error || JSON.stringify(err)}`;
    status.className = "status-error";
  } finally {
    btn.disabled = false;
    spinner.classList.add("hidden");
  }
}

function renderTimeline(segments, containerId) {
  const bar = $(containerId);
  bar.innerHTML = "";
  segments.forEach((s) => {
    const div = document.createElement("div");
    div.className = "timeline-seg " + getSegColor(s);
    div.title = `${s.start_s}s – ${s.end_s}s`;
    bar.appendChild(div);
  });
}

function getSegColor(s) {
  if (s.detected) return "tl-green";
  if (s.bit_accuracy != null && s.bit_accuracy >= 0.5) return "tl-yellow";
  return "tl-red";
}

function renderTemporalTable(segments, tbodyId) {
  const tbody = $(tbodyId);
  tbody.innerHTML = "";
  segments.forEach((s) => {
    const tr = document.createElement("tr");

    const statusHtml = s.detected
      ? `<span style="color:var(--success);font-weight:700">✓ Phát hiện</span>`
      : (s.bit_accuracy != null && s.bit_accuracy >= 0.5)
          ? `<span style="color:var(--warn);font-weight:700">~ Một phần</span>`
          : `<span style="color:var(--danger)">✗ Không</span>`;

    const decoded = (s.correctable && s.decoded_text)
      ? `"${escapeHtml(s.decoded_text)}"`
      : `<span style="color:var(--text-muted)">—</span>`;

    const pct = s.bit_accuracy != null
      ? `${(s.bit_accuracy * 100).toFixed(1)}%` : "—";

    tr.innerHTML = `
      <td>${s.start_s}s – ${s.end_s}s</td>
      <td>${statusHtml}</td>
      <td>${decoded}</td>
      <td class="detail-only">${pct}</td>
    `;
    tbody.appendChild(tr);
  });
}
