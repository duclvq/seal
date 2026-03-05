/**
 * audio.js — AudioSeal Demo UI logic (Audio tab).
 *
 * Handles the audio tab: upload → embed → compare → attacks.
 * AudioSeal uses a 16-bit message (2 ASCII bytes).
 */

// ── Audio attacks list ─────────────────────────────────────────────────────
const AUDIO_ATTACKS = [
  { key: "noise",      label: "Thêm noise ngẫu nhiên" },
  { key: "volume",     label: "Giảm âm lượng (×0.5)" },
  { key: "speed_down", label: "Chậm lại 10%" },
  { key: "speed_up",   label: "Nhanh hơn 10%" },
  { key: "lowpass",    label: "Lọc thấp 4 kHz" },
  { key: "mp3_64k",    label: "Nén MP3 64 kbps" },
  { key: "mp3_128k",   label: "Nén MP3 128 kbps" },
];

// ── State ──────────────────────────────────────────────────────────────────
let aSessionId   = null;
let aOrigBits    = null;   // 16-element array (embedded message)
let aOrigText    = null;

// ── DOM helpers ────────────────────────────────────────────────────────────
const $a = (id) => document.getElementById(id);

function aShow(id) { $a(id).classList.remove("hidden"); }
function aHide(id) { $a(id).classList.add("hidden"); }

function aSetStatus(msg, level) {
  const el = $a("a-embed-status");
  el.textContent = msg;
  el.className   = `status-${level}`;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// ── Bit bar renderer ───────────────────────────────────────────────────────
/**
 * Render 16 bits as colored squares inside a container div.
 * @param {HTMLElement} container
 * @param {number[]} bits       — 16 elements (0 or 1)
 * @param {number[]|null} refBits — if provided, mismatches shown in red
 */
function renderAudioBits(container, bits, refBits = null) {
  container.innerHTML = "";
  bits.forEach((bit, idx) => {
    const sq = document.createElement("span");
    sq.className = "audio-bit-sq";
    if (refBits !== null) {
      sq.classList.add(
        bit !== refBits[idx] ? "audio-bit-err" :
        bit === 1             ? "audio-bit-1"   : "audio-bit-0"
      );
    } else {
      sq.classList.add(bit === 1 ? "audio-bit-1" : "audio-bit-0");
    }
    container.appendChild(sq);
  });
}

// ── Drop zone ──────────────────────────────────────────────────────────────
function wireAudioDropZone() {
  const zone  = $a("a-drop-zone");
  const input = $a("a-file-input");

  input.addEventListener("click", (e) => e.stopPropagation());
  zone.addEventListener("click", (e) => {
    if (!e.target.closest("label")) input.click();
  });
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("dragover");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
    if (e.dataTransfer.files.length) setAudioFile(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", (e) => {
    if (e.target.files.length) setAudioFile(e.target.files[0]);
  });
}

function setAudioFile(file) {
  $a("a-file-name").textContent = `${file.name} (${(file.size / 1024).toFixed(0)} KB)`;
  const dt = new DataTransfer();
  dt.items.add(file);
  $a("a-file-input").files = dt.files;
}

// ── Attack checkboxes ──────────────────────────────────────────────────────
function buildAudioAttackCheckboxes() {
  const div = $a("a-attack-checkboxes");
  AUDIO_ATTACKS.forEach(({ key, label }) => {
    const item = document.createElement("div");
    item.className = "attack-item";

    const cb    = document.createElement("input");
    cb.type     = "checkbox";
    cb.id       = `a-atk-${key}`;
    cb.value    = key;
    cb.checked  = true;

    const lbl       = document.createElement("label");
    lbl.htmlFor     = `a-atk-${key}`;
    lbl.textContent = label;

    item.addEventListener("click", (e) => {
      if (e.target !== cb) cb.checked = !cb.checked;
    });

    item.appendChild(cb);
    item.appendChild(lbl);
    div.appendChild(item);
  });
}

// ── Embed handler ──────────────────────────────────────────────────────────
async function handleAudioEmbed() {
  const fileInput = $a("a-file-input");
  const text      = $a("a-wm-text").value;

  if (!fileInput.files || !fileInput.files.length) {
    aSetStatus("Vui lòng chọn file audio.", "error");
    return;
  }
  if (!text) {
    aSetStatus("Vui lòng nhập ký tự watermark.", "error");
    return;
  }

  const btn = $a("a-btn-embed");
  btn.disabled = true;
  $a("a-btn-embed-spinner").classList.remove("hidden");
  aSetStatus("Đang xử lý...", "info");

  aHide("a-step-compare");
  aHide("a-step-attacks");

  try {
    const fd = new FormData();
    fd.append("file", fileInput.files[0]);
    fd.append("text", text);

    const resp = await fetch("/api/audio/encode", { method: "POST", body: fd });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw err;
    }
    const result = await resp.json();

    aSessionId = result.session_id;
    aOrigBits  = result.bits_list;
    aOrigText  = result.original_text;

    // Step A2: audio comparison
    $a("a-aud-original").src    = result.original_url;
    $a("a-aud-watermarked").src = result.watermarked_url;
    $a("a-audio-meta").textContent =
      `Thời lượng: ${result.duration_s}s · Tần số: ${result.sample_rate} Hz · Thời gian: ${result.embed_time_s}s`;
    renderAudioBits($a("a-bits-embed"), result.bits_list);
    aShow("a-step-compare");

    // Step A3
    aShow("a-step-attacks");

    aSetStatus(`Watermark nhúng thành công (${result.embed_time_s}s) ✓`, "success");
  } catch (err) {
    aSetStatus(`Lỗi: ${err.error || JSON.stringify(err)}`, "error");
  } finally {
    btn.disabled = false;
    $a("a-btn-embed-spinner").classList.add("hidden");
  }
}

// ── Attack handler ─────────────────────────────────────────────────────────
async function handleAudioAttacks(runAll) {
  if (!aSessionId) {
    alert("Vui lòng embed watermark trước.");
    return;
  }

  const attacks = runAll
    ? AUDIO_ATTACKS.map((a) => a.key)
    : AUDIO_ATTACKS.filter(({ key }) => $a(`a-atk-${key}`)?.checked).map(({ key }) => key);

  if (!attacks.length) {
    alert("Chọn ít nhất một attack.");
    return;
  }

  ["a-btn-run-selected", "a-btn-run-all"].forEach((id) => {
    $a(id).disabled = true;
  });
  $a("a-btn-sel-spinner").classList.remove("hidden");
  $a("a-btn-all-spinner").classList.remove("hidden");
  aHide("a-results-wrap");
  aHide("a-attack-preview");

  try {
    const resp = await fetch("/api/audio/attacks/run", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ session_id: aSessionId, attacks }),
    });
    if (!resp.ok) throw await resp.json().catch(() => ({}));
    const { results } = await resp.json();
    renderAudioResultsTable(results);
    aShow("a-results-wrap");
  } catch (err) {
    alert(`Lỗi attack: ${err.error || JSON.stringify(err)}`);
  } finally {
    ["a-btn-run-selected", "a-btn-run-all"].forEach((id) => {
      $a(id).disabled = false;
    });
    $a("a-btn-sel-spinner").classList.add("hidden");
    $a("a-btn-all-spinner").classList.add("hidden");
  }
}

// ── Results table ──────────────────────────────────────────────────────────
function renderAudioResultsTable(results) {
  const tbody = $a("a-results-body");
  tbody.innerHTML = "";

  results.forEach((r) => {
    const tr = document.createElement("tr");
    tr.className = r.pass ? "row-pass" : "row-fail";

    const detPct = r.error ? "—" : `${(r.detection_prob * 100).toFixed(1)}%`;
    const bitPct = r.error || r.bit_accuracy == null ? "—"
                  : `${(r.bit_accuracy * 100).toFixed(1)}%`;
    const passed = r.pass ? "✓ Phát hiện" : "✗ Không phát hiện";

    // Key decode display
    let keyHtml;
    if (r.error) {
      keyHtml = `<span style="color:var(--danger)">Lỗi</span>`;
    } else {
      const match = r.decoded_text === aOrigText;
      keyHtml = `<code>${escHtml(r.decoded_text || "—")}</code> ` +
        (match ? `<span style="color:var(--success)">✓</span>`
               : `<span style="color:var(--danger)">✗</span>`);
    }

    tr.innerHTML = `
      <td>${escHtml(r.display_name || r.attack_key)}</td>
      <td style="color:${r.pass ? 'var(--success)' : 'var(--danger)'};font-weight:700">${passed}</td>
      <td class="detail-only">${detPct}</td>
      <td>${keyHtml}</td>
      <td>${r.process_time_s ?? "—"}s</td>
    `;

    tr.addEventListener("click", () => {
      tbody.querySelectorAll("tr").forEach((t) => t.classList.remove("selected-row"));
      tr.classList.add("selected-row");
      previewAudioAttack(r);
    });

    tbody.appendChild(tr);
  });
}

// ── Attack preview ─────────────────────────────────────────────────────────
function previewAudioAttack(r) {
  $a("a-preview-title").textContent = r.display_name || r.attack_key;

  const aud = $a("a-aud-attacked");
  if (r.attacked_url) {
    aud.src = r.attacked_url;
    aud.load();
  }

  // Bit bars
  if (aOrigBits)  renderAudioBits($a("a-bits-orig-preview"), aOrigBits);
  if (r.bits_list) renderAudioBits($a("a-bits-attacked"), r.bits_list, aOrigBits);

  // Badge
  const badge   = $a("a-decode-badge");
  const isMatch = r.decoded_text !== null && r.decoded_text === aOrigText;
  if (r.error) {
    badge.innerHTML = `<span class="key-badge key-badge--error">✗ Lỗi</span>`;
  } else if (isMatch) {
    badge.innerHTML = `<span class="key-badge key-badge--ok">✓ Đúng</span>`;
  } else {
    badge.innerHTML = `<span class="key-badge key-badge--warn">≠ Sai</span>`;
  }

  // Stats table
  const detPct = r.error ? "—" : `${(r.detection_prob * 100).toFixed(1)}%`;
  const bitPct = r.bit_accuracy != null ? `${(r.bit_accuracy * 100).toFixed(1)}%` : "—";
  const errBits = aOrigBits && r.bits_list
    ? aOrigBits.filter((b, i) => b !== r.bits_list[i]).length : "?";

  $a("a-preview-stats").innerHTML =
    `<table class="key-compare-table">` +
    `<tr><td class="kc-label">Xác suất phát hiện</td><td><strong>${detPct}</strong></td></tr>` +
    `<tr class="detail-only"><td class="kc-label">Bit accuracy</td>` +
        `<td><strong>${bitPct}</strong> &nbsp;(${errBits} bit lỗi / 16)</td></tr>` +
    `<tr><td class="kc-label">Text gốc</td><td>"${escHtml(aOrigText)}"</td></tr>` +
    `<tr><td class="kc-label">Text decode</td>` +
        `<td>"${escHtml(r.decoded_text || "")}" &nbsp;` +
        (isMatch ? `<span style="color:var(--success)">✓</span>`
                 : `<span style="color:var(--danger)">✗</span>`) +
        `</td></tr>` +
    `</table>`;

  aShow("a-attack-preview");
  $a("a-attack-preview").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Tab switching ──────────────────────────────────────────────────────────
function wireTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.add("hidden"));
      btn.classList.add("active");
      document.getElementById(`tab-${tab}`).classList.remove("hidden");
    });
  });
}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  wireTabs();
  wireAudioDropZone();
  buildAudioAttackCheckboxes();

  $a("a-btn-embed").addEventListener("click", handleAudioEmbed);
  $a("a-btn-run-selected").addEventListener("click", () => handleAudioAttacks(false));
  $a("a-btn-run-all").addEventListener("click",      () => handleAudioAttacks(true));
});
