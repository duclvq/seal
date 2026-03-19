"""
Monitor web UI — port 5002
Shows job stats, per-status lists, live refresh.
"""
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template_string

DB_PATH = os.getenv("DB_PATH", r"D:\vtv_setup\seal_1603\seal_1603\watcher_service\data\db\watermarks.db")
# Chạy trực tiếp (không Docker): set DB_PATH=<path> trước khi chạy

app = Flask(__name__)


def _conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def _query(sql, params=()):
    with _conn() as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(sql, params).fetchall()]


HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Monitor</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }
  header { background: #1a1d2e; border-bottom: 1px solid #2d3748; padding: 16px 24px;
           display: flex; align-items: center; justify-content: space-between; }
  header h1 { font-size: 1.2rem; font-weight: 600; color: #7c84ff; }
  .refresh-info { font-size: 0.78rem; color: #718096; }
  .container { max-width: 1400px; margin: 0 auto; padding: 24px; }

  /* Stats cards */
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin-bottom: 28px; }
  .card { background: #1a1d2e; border-radius: 10px; padding: 18px 20px; border: 1px solid #2d3748; }
  .card .label { font-size: 0.75rem; color: #718096; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 6px; }
  .card .value { font-size: 2rem; font-weight: 700; }
  .card.done .value   { color: #48bb78; }
  .card.processing .value { color: #63b3ed; }
  .card.pending .value { color: #ecc94b; }
  .card.error .value  { color: #fc8181; }
  .card.total .value  { color: #e2e8f0; }
  .card .sub  { font-size: 0.75rem; color: #718096; margin-top: 4px; }

  /* Tabs */
  .tabs { display: flex; gap: 4px; margin-bottom: 16px; }
  .tab { padding: 8px 18px; border-radius: 8px; cursor: pointer; font-size: 0.85rem;
         background: #1a1d2e; border: 1px solid #2d3748; color: #718096; transition: all .2s; }
  .tab.active, .tab:hover { background: #2d3748; color: #e2e8f0; }
  .tab.active { border-color: #7c84ff; color: #7c84ff; }

  /* Table */
  .table-wrap { background: #1a1d2e; border-radius: 10px; border: 1px solid #2d3748; overflow: hidden; }
  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  thead tr { background: #171923; }
  th { padding: 10px 14px; text-align: left; color: #718096; font-weight: 500;
       border-bottom: 1px solid #2d3748; white-space: nowrap; }
  td { padding: 9px 14px; border-bottom: 1px solid #1e2535; vertical-align: top; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #171923; }
  .fname { font-family: monospace; font-size: 0.8rem; max-width: 280px;
           overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }
  .badge.done       { background: #1c4532; color: #48bb78; }
  .badge.processing { background: #1a365d; color: #63b3ed; }
  .badge.pending    { background: #744210; color: #ecc94b; }
  .badge.error      { background: #3d1515; color: #fc8181; }
  .err-msg { color: #fc8181; font-size: 0.75rem; max-width: 300px; word-break: break-word; }
  .empty { padding: 40px; text-align: center; color: #4a5568; }
  .search-row { display: flex; gap: 12px; margin-bottom: 14px; align-items: center; }
  input[type=search] { flex: 1; background: #1a1d2e; border: 1px solid #2d3748; border-radius: 8px;
                       padding: 8px 14px; color: #e2e8f0; font-size: 0.85rem; outline: none; }
  input[type=search]:focus { border-color: #7c84ff; }
  .ts { color: #718096; font-size: 0.75rem; white-space: nowrap; }
  .wmkey { font-family: monospace; font-size: 0.8rem; color: #9f7aea; letter-spacing: .04em; }
  .wmkey-wrap { display: flex; align-items: center; gap: 6px; }
  .copy-btn { background: none; border: 1px solid #4a5568; border-radius: 4px; color: #718096;
              cursor: pointer; font-size: 0.7rem; padding: 1px 6px; line-height: 1.6; }
  .copy-btn:hover { border-color: #9f7aea; color: #9f7aea; }
  .del-btn { background: none; border: 1px solid #4a5568; border-radius: 4px; color: #718096;
             cursor: pointer; font-size: 0.7rem; padding: 1px 6px; line-height: 1.6; }
  .del-btn:hover { border-color: #fc8181; color: #fc8181; }
  .bulk-bar { display: flex; gap: 8px; align-items: center; margin-bottom: 14px; flex-wrap: wrap; }
  .bulk-btn { padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 0.78rem;
              border: 1px solid #2d3748; color: #718096; background: #1a1d2e; transition: all .2s; }
  .bulk-btn:hover { border-color: #fc8181; color: #fc8181; background: #1e1215; }
  .bulk-btn.confirm { border-color: #fc8181; color: #fc8181; background: #3d1515; }
  .toast { position: fixed; bottom: 24px; right: 24px; background: #2d3748; color: #e2e8f0;
           padding: 10px 20px; border-radius: 8px; font-size: 0.85rem; z-index: 999;
           opacity: 0; transition: opacity .3s; pointer-events: none; }
  .toast.show { opacity: 1; }
</style>
</head>
<body>
<header>
  <h1>🎬 Monitor</h1>
  <span class="refresh-info">Tự refresh mỗi 10 giây &nbsp;|&nbsp; <span id="last-update">—</span></span>
</header>
<div class="container">
  <div class="stats" id="stats-cards"></div>
  <div class="search-row">
    <input type="search" id="search" placeholder="Tìm tên file..." oninput="filterTable()">
  </div>
  <div class="bulk-bar" id="bulk-bar"></div>
  <div class="tabs" id="tabs"></div>
  <div class="table-wrap">
    <table id="job-table">
      <thead id="table-head"></thead>
      <tbody id="table-body"></tbody>
    </table>
  </div>
</div>

<div id="toast" class="toast"></div>
<script>
let allData = {};
let currentTab = 'all';

async function fetchData() {
  const r = await fetch('/api/jobs');
  allData = await r.json();
  render();
  document.getElementById('last-update').textContent = new Date().toLocaleTimeString('vi-VN');
}

function render() {
  renderStats();
  renderTabs();
  renderBulkBar();
  renderTable();
}

function renderStats() {
  const s = allData.stats || {};
  const cards = [
    { key: 'total',      label: 'Tổng',        cls: 'total'      },
    { key: 'done',       label: 'Hoàn thành',  cls: 'done'       },
    { key: 'processing', label: 'Đang xử lý',  cls: 'processing' },
    { key: 'pending',    label: 'Chờ',          cls: 'pending'    },
    { key: 'error',      label: 'Lỗi',          cls: 'error'      },
  ];
  document.getElementById('stats-cards').innerHTML = cards.map(c => `
    <div class="card ${c.cls}">
      <div class="label">${c.label}</div>
      <div class="value">${s[c.key] || 0}</div>
      ${c.key === 'done' && s.avg_time ? `<div class="sub">avg ${s.avg_time}s/video</div>` : ''}
    </div>`).join('');

  // Memory cards
  let memHtml = '';
  if (s.ram_pct !== undefined) {
    const ramCls = s.ram_pct > 85 ? 'error' : s.ram_pct > 70 ? 'pending' : 'done';
    memHtml += `<div class="card ${ramCls}">
      <div class="label">RAM</div>
      <div class="value" style="font-size:1.4rem">${s.ram_pct}%</div>
      <div class="sub">${s.ram_used_gb}/${s.ram_total_gb} GB</div>
    </div>`;
  }
  if (s.vram_pct !== undefined) {
    const vramCls = s.vram_pct > 85 ? 'error' : s.vram_pct > 70 ? 'pending' : 'done';
    memHtml += `<div class="card ${vramCls}">
      <div class="label">VRAM</div>
      <div class="value" style="font-size:1.4rem">${s.vram_pct}%</div>
      <div class="sub">${s.vram_reserved_gb}/${s.vram_total_gb} GB</div>
    </div>`;
  }
  document.getElementById('stats-cards').innerHTML += memHtml;
}

function renderTabs() {
  const tabs = ['all','done','processing','pending','error'];
  const labels = {all:'Tất cả', done:'Xong', processing:'Đang chạy', pending:'Chờ', error:'Lỗi'};
  const s = allData.stats || {};
  document.getElementById('tabs').innerHTML = tabs.map(t => `
    <div class="tab ${t===currentTab?'active':''}" onclick="setTab('${t}')">
      ${labels[t]} ${t!=='all' ? `(${s[t]||0})` : ''}
    </div>`).join('');
}

function setTab(tab) { currentTab = tab; renderTabs(); renderTable(); }

function filterTable() { renderTable(); }

function renderTable() {
  const jobs = (allData.jobs || []).filter(j => {
    if (currentTab !== 'all' && j.status !== currentTab) return false;
    const q = document.getElementById('search').value.toLowerCase();
    if (q && !j.filename.toLowerCase().includes(q)) return false;
    return true;
  });

  const cols = [
    {h:'#', f: j => j.id},
    {h:'File', f: j => `<span class="fname" title="${j.filename}">${j.filename}</span>`},
    {h:'Trạng thái', f: j => {
      let badge = `<span class="badge ${j.status}">${j.status}</span>`;
      if (j.status === 'processing' && j.total_frames > 0 && j.processed_frames > 0) {
        const pct = Math.min(100, Math.round(j.processed_frames / j.total_frames * 100));
        badge += `<div style="margin-top:4px;background:#2d3748;border-radius:4px;height:6px;width:80px">
          <div style="background:#63b3ed;height:100%;border-radius:4px;width:${pct}%"></div>
        </div><span style="font-size:0.7rem;color:#63b3ed">${pct}% (${j.processed_frames}/${j.total_frames})</span>`;
      } else if (j.status === 'done' && j.total_frames > 0) {
        badge += `<span style="font-size:0.7rem;color:#48bb78;margin-left:4px">${j.total_frames}f</span>`;
      }
      return badge;
    }},
    {h:'WM Key', f: j => j.watermark_text
      ? `<div class="wmkey-wrap"><span class="wmkey">${j.watermark_text}</span><button class="copy-btn" onclick="copyKey(event,'${j.watermark_text}')">copy</button></div>`
      : '<span style="color:#4a5568">—</span>'},
    {h:'FPS', f: j => j.fps ? j.fps.toFixed(1) : '—'},
    {h:'Thời gian xử lý', f: j => j.embed_time_s ? j.embed_time_s + 's' : '—'},
    {h:'Tạo lúc', f: j => `<span class="ts">${fmtTime(j.created_at)}</span>`},
    {h:'Xong lúc', f: j => `<span class="ts">${fmtTime(j.processed_at)}</span>`},
    {h:'Lỗi', f: j => j.error_msg ? `<span class="err-msg" title="${j.error_msg}">${j.error_msg.substring(0,80)}${j.error_msg.length>80?'…':''}</span>` : ''},
    {h:'', f: j => `<button class="del-btn" onclick="deleteJob(${j.id},'${j.filename.replace(/'/g,"\\'")}')">xóa</button>`},
  ];

  document.getElementById('table-head').innerHTML =
    '<tr>' + cols.map(c => `<th>${c.h}</th>`).join('') + '</tr>';

  if (jobs.length === 0) {
    document.getElementById('table-body').innerHTML =
      `<tr><td colspan="${cols.length}" class="empty">Không có dữ liệu</td></tr>`;
    return;
  }

  document.getElementById('table-body').innerHTML = jobs.map(j =>
    '<tr>' + cols.map(c => `<td>${c.f(j)}</td>`).join('') + '</tr>'
  ).join('');
}

function fmtTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleString('vi-VN', {dateStyle:'short', timeStyle:'short'});
}

function copyKey(e, key) {
  e.stopPropagation();
  navigator.clipboard.writeText(key).then(() => {
    const btn = e.target;
    btn.textContent = 'ok!';
    btn.style.color = '#48bb78';
    setTimeout(() => { btn.textContent = 'copy'; btn.style.color = ''; }, 1500);
  });
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3000);
}

async function deleteJob(id, fname) {
  if (!confirm(`Xóa job #${id} (${fname})?`)) return;
  const r = await fetch(`/api/delete_job/${id}`, {method:'POST'});
  const d = await r.json();
  showToast(d.msg);
  fetchData();
}

let bulkConfirm = {};
async function bulkDelete(status) {
  if (!bulkConfirm[status]) {
    bulkConfirm[status] = true;
    renderBulkBar();
    setTimeout(() => { bulkConfirm[status] = false; renderBulkBar(); }, 4000);
    return;
  }
  bulkConfirm[status] = false;
  const r = await fetch(`/api/delete_by_status/${status}`, {method:'POST'});
  const d = await r.json();
  showToast(d.msg);
  fetchData();
}

function renderBulkBar() {
  const s = allData.stats || {};
  const btns = [];
  for (const st of ['pending','processing','error']) {
    const cnt = s[st] || 0;
    if (cnt === 0) continue;
    const labels = {pending:'Chờ', processing:'Đang chạy', error:'Lỗi'};
    const isConfirm = bulkConfirm[st];
    btns.push(`<button class="bulk-btn ${isConfirm?'confirm':''}" onclick="bulkDelete('${st}')">
      ${isConfirm ? `⚠ Nhấn lần nữa để xóa ${cnt} ${labels[st]}` : `Xóa tất cả ${labels[st]} (${cnt})`}
    </button>`);
  }
  document.getElementById('bulk-bar').innerHTML = btns.length
    ? btns.join('')
    : '';
}

fetchData();
setInterval(fetchData, 10000);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/delete_job/<int:job_id>", methods=["POST"])
def api_delete_job(job_id):
    """Delete a single job by ID."""
    try:
        with _conn() as c:
            row = c.execute("SELECT status, filename FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return jsonify({"ok": False, "msg": f"Job #{job_id} không tồn tại"}), 404
            c.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        return jsonify({"ok": True, "msg": f"Đã xóa job #{job_id} ({row[1]})"})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


@app.route("/api/delete_by_status/<status>", methods=["POST"])
def api_delete_by_status(status):
    """Delete all jobs with given status."""
    allowed = ("pending", "processing", "error", "done")
    if status not in allowed:
        return jsonify({"ok": False, "msg": f"Status '{status}' không hợp lệ"}), 400
    try:
        with _conn() as c:
            n = c.execute("DELETE FROM jobs WHERE status=?", (status,)).rowcount
        return jsonify({"ok": True, "msg": f"Đã xóa {n} job(s) status='{status}'"})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


@app.route("/api/jobs")
def api_jobs():
    jobs = _query(
        "SELECT * FROM jobs ORDER BY id DESC LIMIT 500"
    )

    rows = _query(
        """SELECT status, COUNT(*) as cnt FROM jobs GROUP BY status"""
    )
    stats = {r["status"]: r["cnt"] for r in rows}
    stats["total"] = sum(stats.values())

    avg = _query(
        "SELECT ROUND(AVG(embed_time_s),1) as v FROM jobs WHERE status='done' AND embed_time_s IS NOT NULL"
    )
    stats["avg_time"] = avg[0]["v"] if avg else None

    # System memory info
    try:
        import psutil
        ram = psutil.virtual_memory()
        stats["ram_used_gb"] = round(ram.used / 1024**3, 1)
        stats["ram_total_gb"] = round(ram.total / 1024**3, 1)
        stats["ram_pct"] = ram.percent
    except ImportError:
        pass

    # GPU memory info
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            stats["vram_alloc_gb"] = round(alloc, 1)
            stats["vram_reserved_gb"] = round(reserved, 1)
            stats["vram_total_gb"] = round(total, 1)
            stats["vram_pct"] = round(reserved / total * 100, 0) if total > 0 else 0
    except Exception:
        pass

    return jsonify({"jobs": jobs, "stats": stats})


if __name__ == "__main__":
    port = int(os.getenv("MONITOR_PORT", "5002"))
    print(f"Monitor UI: http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
