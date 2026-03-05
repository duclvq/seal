/**
 * api.js — fetch() wrappers for VideoSeal backend routes.
 */

/**
 * POST /api/encode
 *
 * @param {File}     file       - Video file
 * @param {string}   text       - Watermark text
 * @param {number}   k          - RS data bytes (used when eccType="rs")
 * @param {function} onProgress - Called with percentage 0–100
 * @param {string}   eccType    - "rs" | "bch"
 * @returns {Promise<object>}   - JSON response
 */
function apiEncode(file, text, k, onProgress, eccType = "rs") {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("video", file);
    form.append("text", text);
    form.append("ecc_type", eccType);
    form.append("k", String(k));

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/encode");

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) {
        // Upload is ~50% of total work; embedding takes the rest
        const pct = Math.round((e.loaded / e.total) * 50);
        onProgress(pct);
      }
    });

    xhr.addEventListener("load", () => {
      if (xhr.status === 200) {
        onProgress(100);
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          reject({ error: "Invalid JSON response from server" });
        }
      } else {
        try {
          reject(JSON.parse(xhr.responseText));
        } catch {
          reject({ error: `Server error ${xhr.status}` });
        }
      }
    });

    xhr.addEventListener("error",   () => reject({ error: "Network error" }));
    xhr.addEventListener("timeout", () => reject({ error: "Request timed out" }));

    xhr.timeout = 10 * 60 * 1000; // 10-minute timeout for large videos
    xhr.send(form);
  });
}

/**
 * POST /api/attacks/run
 *
 * @param {string}   sessionId
 * @param {string[]} attacks    - Attack keys or ["all"]
 * @param {number}   k          - RS data bytes for decoding
 * @param {string}   eccType    - "rs" | "bch"
 * @returns {Promise<object>}   - { results: [...] }
 */
async function apiRunAttacks(sessionId, attacks, k, eccType = "rs") {
  const resp = await fetch("/api/attacks/run", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ session_id: sessionId, attacks, k, ecc_type: eccType }),
  });

  const data = await resp.json();
  if (!resp.ok) throw data;
  return data;
}

/**
 * GET /api/rs_info?k=<k>
 *
 * @param {number} k
 * @returns {Promise<{k, ecc_bytes, max_byte_errors, max_text_bytes}>}
 */
async function apiRsInfo(k) {
  const resp = await fetch(`/api/rs_info?k=${k}`);
  return resp.json();
}

/**
 * POST /api/attacks/upload_video
 *
 * @param {string}   sessionId
 * @param {File}     file         - The externally captured video
 * @param {string}   attackLabel  - Display name shown in results
 * @param {function} onProgress   - Called with percentage 0–100
 * @returns {Promise<object>}     - Single result object (same shape as attacks/run entry)
 */
function apiUploadAttackVideo(sessionId, file, attackLabel, onProgress) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("session_id",   sessionId);
    form.append("video",        file);
    form.append("attack_label", attackLabel);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/attacks/upload_video");

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) {
        onProgress(Math.round((e.loaded / e.total) * 60));
      }
    });

    xhr.addEventListener("load", () => {
      if (xhr.status === 200) {
        onProgress(100);
        try { resolve(JSON.parse(xhr.responseText)); }
        catch { reject({ error: "Invalid JSON response" }); }
      } else {
        try { reject(JSON.parse(xhr.responseText)); }
        catch { reject({ error: `Server error ${xhr.status}` }); }
      }
    });

    xhr.addEventListener("error",   () => reject({ error: "Network error" }));
    xhr.addEventListener("timeout", () => reject({ error: "Request timed out" }));
    xhr.timeout = 10 * 60 * 1000;
    xhr.send(form);
  });
}


/**
 * GET /api/bch_info?t=<t>
 *
 * @param {number} t
 * @returns {Promise<{t, ecc_bytes, max_bit_errors, max_text_bytes, available}>}
 */
async function apiBchInfo(t) {
  const resp = await fetch(`/api/bch_info?t=${t}`);
  return resp.json();
}
