"""
Monitor web UI -- port 5002
Shows job stats, per-status lists, live refresh.
"""
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template_string

DB_PATH = os.getenv("DB_PATH", r"D:\vtv_setup\seal_1603\seal_1603\watcher_service\data\db\watermarks.db")
INPUT_FOLDER = os.getenv("INPUT_FOLDER", "")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "")

app = Flask(__name__)


def _conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def _query(sql, params=()):
    with _conn() as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(sql, params).fetchall()]


HTML = r"""<!DOCTYPE html>
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
  .folder-bar { display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }
  .folder-card { background: #1a1d2e; border: 1px solid #2d3748; border-radius: 10px; padding: 12px 18px;
                 flex: 1; min-width: 280px; display: flex; align-items: center; gap: 10px; }
  .folder-card .icon { font-size: 1.3rem; }
  .folder-card .info { flex: 1; overflow: hidden; }
  .folder-card .lbl { font-size: 0.7rem; color: #718096; text-transform: uppercase; letter-spacing: .04em; }
  .folder-card .path { font-family: monospace; font-size: 0.8rem; color: #e2e8f0;
                       overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .folder-card .open-btn { background: none; border: 1px solid #4a5568; border-radius: 6px; color: #718096;
                           cursor: pointer; font-size: 0.75rem; padding: 4px 10px; white-space: nowrap; }
  .folder-card .open-btn:hover { border-color: #7c84ff; color: #7c84ff; }
  .fpath { font-family: monospace; font-size: 0.72rem; color: #718096; max-width: 200px;
           overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: pointer; }
  .fpath:hover { color: #7c84ff; text-decoration: underline; }
  .tabs { display: flex; gap: 4px; margin-bottom: 16px; }
  .tab { padding: 8px 18px; border-radius: 8px; cursor: pointer; font-size: 0.85rem;
         background: #1a1d2e; border: 1px solid #2d3748; color: #718096; transition: all .2s; }
  .tab.active, .tab:hover { background: #2d3748; color: #e2e8f0; }
  .tab.active { border-color: #7c84ff; color: #7c84ff; }
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
  <h1>Monitor</h1>
  <span class="refresh-info">Auto refresh 10s | <span id="last-update">-</span></span>
</header>
<div class="container">
  <div class="folder-bar" id="folder-bar"></div>
  <div class="stats" id="stats-cards"></div>
  <div class="search-row">
    <input type="search" id="search" placeholder="Search filename..." oninput="filterTable()">
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
var allData = {};
var currentTab = 'all';

async function fetchData() {
  var url = '/api/jobs';
  if (currentTab && currentTab !== 'all') url += '?status=' + currentTab;
  var r = await fetch(url);
  allData = await r.json();
  render();
  document.getElementById('last-update').textContent = new Date().toLocaleTimeString('vi-VN');
}

function render() {
  renderFolders();
  renderStats();
  renderTabs();
  renderBulkBar();
  renderTable();
}

function esc(s) { return s.split('\\').join('\\\\').split("'").join("\\'"); }
function h(s) { if (!s) return ''; var d=document.createElement('div'); d.textContent=s; return d.innerHTML; }

function renderFolders() {
  var inp = allData.input_folder || '';
  var out = allData.output_folder || '';
  if (!inp && !out) { document.getElementById('folder-bar').innerHTML = ''; return; }
  var html = '';
  if (inp) html += '<div class="folder-card"><span class="icon">&#128194;</span><div class="info"><div class="lbl">Input</div><div class="path" title="' + inp + '">' + inp + '</div></div><button class="open-btn" onclick="openFolder(\'' + esc(inp) + '\')">Open</button></div>';
  if (out) html += '<div class="folder-card"><span class="icon">&#128193;</span><div class="info"><div class="lbl">Output</div><div class="path" title="' + out + '">' + out + '</div></div><button class="open-btn" onclick="openFolder(\'' + esc(out) + '\')">Open</button></div>';
  document.getElementById('folder-bar').innerHTML = html;
}

function renderStats() {
  var s = allData.stats || {};
  var cards = [
    { key: 'total',      label: 'Total',       cls: 'total'      },
    { key: 'done',       label: 'Done',         cls: 'done'       },
    { key: 'processing', label: 'Processing',   cls: 'processing' },
    { key: 'pending',    label: 'Pending',       cls: 'pending'    },
    { key: 'error',      label: 'Error',         cls: 'error'      }
  ];
  var html = '';
  for (var i = 0; i < cards.length; i++) {
    var c = cards[i];
    html += '<div class="card ' + c.cls + '">';
    html += '<div class="label">' + c.label + '</div>';
    html += '<div class="value">' + (s[c.key] || 0) + '</div>';
    if (c.key === 'done' && s.avg_time) html += '<div class="sub">avg ' + s.avg_time + 's/video</div>';
    html += '</div>';
  }
  if (s.ram_pct !== undefined) {
    var ramCls = s.ram_pct > 85 ? 'error' : s.ram_pct > 70 ? 'pending' : 'done';
    html += '<div class="card ' + ramCls + '"><div class="label">RAM</div>';
    html += '<div class="value" style="font-size:1.4rem">' + s.ram_pct + '%</div>';
    html += '<div class="sub">' + s.ram_used_gb + '/' + s.ram_total_gb + ' GB</div></div>';
  }
  if (s.vram_pct !== undefined) {
    var vramCls = s.vram_pct > 85 ? 'error' : s.vram_pct > 70 ? 'pending' : 'done';
    html += '<div class="card ' + vramCls + '"><div class="label">VRAM</div>';
    html += '<div class="value" style="font-size:1.4rem">' + s.vram_pct + '%</div>';
    html += '<div class="sub">' + s.vram_reserved_gb + '/' + s.vram_total_gb + ' GB</div></div>';
  }
  document.getElementById('stats-cards').innerHTML = html;
}

function renderTabs() {
  var tabs = ['all','done','processing','pending','error'];
  var labels = {all:'All', done:'Done', processing:'Processing', pending:'Pending', error:'Error'};
  var s = allData.stats || {};
  var html = '';
  for (var i = 0; i < tabs.length; i++) {
    var t = tabs[i];
    html += '<div class="tab ' + (t===currentTab?'active':'') + '" onclick="setTab(\'' + t + '\')">';
    html += labels[t] + (t!=='all' ? ' (' + (s[t]||0) + ')' : '');
    html += '</div>';
  }
  document.getElementById('tabs').innerHTML = html;
}

function setTab(tab) { currentTab = tab; fetchData(); }
function filterTable() { renderTable(); }

function renderTable() {
  var jobs = (allData.jobs || []).filter(function(j) {
    if (currentTab !== 'all' && j.status !== currentTab) return false;
    var q = document.getElementById('search').value.toLowerCase();
    if (q && j.filename.toLowerCase().indexOf(q) === -1) return false;
    return true;
  });

  var colHeaders = ['#','File','Path','Status','WM Key','FPS','Time','Created','Done','Error',''];
  var headHtml = '<tr>';
  for (var i = 0; i < colHeaders.length; i++) headHtml += '<th>' + colHeaders[i] + '</th>';
  headHtml += '</tr>';
  document.getElementById('table-head').innerHTML = headHtml;

  if (jobs.length === 0) {
    document.getElementById('table-body').innerHTML = '<tr><td colspan="' + colHeaders.length + '" class="empty">No data</td></tr>';
    return;
  }

  var bodyHtml = '';
  for (var i = 0; i < jobs.length; i++) {
    try {
    var j = jobs[i];
    var fn = h(j.filename || '');
    bodyHtml += '<tr>';
    bodyHtml += '<td>' + j.id + '</td>';
    bodyHtml += '<td><span class="fname" title="' + fn + '">' + fn + '</span></td>';
    var p = j.output_path || j.input_path || '';
    if (p) {
      var dir = p.replace(/[\\/][^\\/]*$/, '');
      bodyHtml += '<td><span class="fpath" title="' + h(p) + '" onclick="openFolder(\'' + esc(dir) + '\')">' + h(p) + '</span></td>';
    } else {
      bodyHtml += '<td><span style="color:#4a5568">-</span></td>';
    }
    var badge = '<span class="badge ' + j.status + '">' + j.status + '</span>';
    if (j.status === 'processing' && j.total_frames > 0 && j.processed_frames > 0) {
      var pct = Math.min(100, Math.round(j.processed_frames / j.total_frames * 100));
      badge += '<div style="margin-top:4px;background:#2d3748;border-radius:4px;height:6px;width:80px">';
      badge += '<div style="background:#63b3ed;height:100%;border-radius:4px;width:' + pct + '%"></div></div>';
      badge += '<span style="font-size:0.7rem;color:#63b3ed">' + pct + '% (' + j.processed_frames + '/' + j.total_frames + ')</span>';
    } else if (j.status === 'done' && j.total_frames > 0) {
      badge += '<span style="font-size:0.7rem;color:#48bb78;margin-left:4px">' + j.total_frames + 'f</span>';
    }
    bodyHtml += '<td>' + badge + '</td>';
    if (j.watermark_text) {
      bodyHtml += '<td><div class="wmkey-wrap"><span class="wmkey">' + h(j.watermark_text) + '</span>';
      bodyHtml += '<button class="copy-btn" onclick="copyKey(event,\'' + esc(j.watermark_text) + '\')">copy</button></div></td>';
    } else {
      bodyHtml += '<td><span style="color:#4a5568">-</span></td>';
    }
    bodyHtml += '<td>' + (j.fps ? j.fps.toFixed(1) : '-') + '</td>';
    bodyHtml += '<td>' + (j.embed_time_s ? j.embed_time_s + 's' : '-') + '</td>';
    bodyHtml += '<td><span class="ts">' + fmtTime(j.created_at) + '</span></td>';
    bodyHtml += '<td><span class="ts">' + fmtTime(j.processed_at) + '</span></td>';
    if (j.error_msg) {
      var short = j.error_msg.length > 80 ? j.error_msg.substring(0,80) + '...' : j.error_msg;
      bodyHtml += '<td><span class="err-msg" title="' + h(j.error_msg) + '">' + h(short) + '</span></td>';
    } else {
      bodyHtml += '<td></td>';
    }
    bodyHtml += '<td><button class="del-btn" onclick="deleteJob(' + j.id + ',\'' + esc(j.filename || '') + '\')">del</button></td>';
    bodyHtml += '</tr>';
    } catch(e) { console.error('Row render error', j, e); }
  }
  document.getElementById('table-body').innerHTML = bodyHtml;
}

function fmtTime(iso) {
  if (!iso) return '-';
  var d = new Date(iso);
  return d.toLocaleString('vi-VN', {dateStyle:'short', timeStyle:'short'});
}

function copyKey(e, key) {
  e.stopPropagation();
  navigator.clipboard.writeText(key).then(function() {
    var btn = e.target;
    btn.textContent = 'ok!';
    btn.style.color = '#48bb78';
    setTimeout(function() { btn.textContent = 'copy'; btn.style.color = ''; }, 1500);
  });
}

function showToast(msg) {
  var t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(function() { t.classList.remove('show'); }, 3000);
}

async function deleteJob(id, fname) {
  if (!confirm('Delete job #' + id + ' (' + fname + ')?')) return;
  var r = await fetch('/api/delete_job/' + id, {method:'POST'});
  var d = await r.json();
  showToast(d.msg);
  fetchData();
}

var bulkConfirm = {};
async function bulkDelete(status) {
  if (!bulkConfirm[status]) {
    bulkConfirm[status] = true;
    renderBulkBar();
    setTimeout(function() { bulkConfirm[status] = false; renderBulkBar(); }, 4000);
    return;
  }
  bulkConfirm[status] = false;
  var r = await fetch('/api/delete_by_status/' + status, {method:'POST'});
  var d = await r.json();
  showToast(d.msg);
  fetchData();
}

function renderBulkBar() {
  var s = allData.stats || {};
  var btns = [];
  var sts = ['pending','processing','error'];
  var labels = {pending:'Pending', processing:'Processing', error:'Error'};
  for (var i = 0; i < sts.length; i++) {
    var st = sts[i];
    var cnt = s[st] || 0;
    if (cnt === 0) continue;
    if (bulkConfirm[st]) {
      btns.push('<button class="bulk-btn confirm" onclick="bulkDelete(\'' + st + '\')">Confirm delete ' + cnt + ' ' + labels[st] + '</button>');
    } else {
      btns.push('<button class="bulk-btn" onclick="bulkDelete(\'' + st + '\')">Delete all ' + labels[st] + ' (' + cnt + ')</button>');
    }
  }
  document.getElementById('bulk-bar').innerHTML = btns.join('');
}

async function openFolder(path) {
  await fetch('/api/open_folder', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({path: path})});
}

fetchData();
setInterval(fetchData, 10000);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/delete_job/<int:job_id>", methods=["POST"])
def api_delete_job(job_id):
    try:
        with _conn() as c:
            row = c.execute("SELECT status, filename FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return jsonify({"ok": False, "msg": f"Job #{job_id} not found"}), 404
            c.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        return jsonify({"ok": True, "msg": f"Deleted job #{job_id} ({row[1]})"})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


@app.route("/api/delete_by_status/<status>", methods=["POST"])
def api_delete_by_status(status):
    allowed = ("pending", "processing", "error", "done")
    if status not in allowed:
        return jsonify({"ok": False, "msg": f"Invalid status: {status}"}), 400
    try:
        with _conn() as c:
            n = c.execute("DELETE FROM jobs WHERE status=?", (status,)).rowcount
        return jsonify({"ok": True, "msg": f"Deleted {n} job(s) with status={status}"})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


@app.route("/api/open_folder", methods=["POST"])
def api_open_folder():
    import subprocess
    from flask import request
    data = request.get_json(silent=True) or {}
    folder = data.get("path", "")
    if not folder or not os.path.isdir(folder):
        return jsonify({"ok": False, "msg": f"Folder not found: {folder}"}), 400
    try:
        subprocess.Popen(["explorer", os.path.normpath(folder)])
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


@app.route("/api/jobs")
def api_jobs():
    # If a status filter is requested, use it; otherwise return latest 500
    from flask import request
    status_filter = request.args.get("status", "")
    if status_filter and status_filter != "all":
        jobs = _query(
            "SELECT * FROM jobs WHERE status=? ORDER BY id DESC LIMIT 500",
            (status_filter,),
        )
    else:
        jobs = _query("SELECT * FROM jobs ORDER BY id DESC LIMIT 500")

    rows = _query("SELECT status, COUNT(*) as cnt FROM jobs GROUP BY status")
    stats = {r["status"]: r["cnt"] for r in rows}
    stats["total"] = sum(stats.values())

    avg = _query(
        "SELECT ROUND(AVG(embed_time_s),1) as v FROM jobs WHERE status='done' AND embed_time_s IS NOT NULL"
    )
    stats["avg_time"] = avg[0]["v"] if avg else None

    try:
        import psutil
        ram = psutil.virtual_memory()
        stats["ram_used_gb"] = round(ram.used / 1024**3, 1)
        stats["ram_total_gb"] = round(ram.total / 1024**3, 1)
        stats["ram_pct"] = ram.percent
    except ImportError:
        pass

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

    return jsonify({"jobs": jobs, "stats": stats,
                    "input_folder": INPUT_FOLDER, "output_folder": OUTPUT_FOLDER})


if __name__ == "__main__":
    port = int(os.getenv("MONITOR_PORT", "5002"))
    print(f"Monitor UI: http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
