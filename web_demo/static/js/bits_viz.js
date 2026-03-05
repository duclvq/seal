/**
 * bits_viz.js — Render 256 watermark bits as a 32×8 colored grid on a canvas.
 *
 * Color scheme (no compare):
 *   bit=1 → blue  (#2980b9)
 *   bit=0 → light gray (#ecf0f1)
 *
 * Color scheme (with compare reference):
 *   match & bit=1 → blue  (#2980b9)
 *   match & bit=0 → gray  (#bdc3c7)
 *   mismatch      → red   (#e74c3c)
 */

const BITS_COLS = 32;
const BITS_ROWS = 8;

/**
 * @param {HTMLCanvasElement} canvas
 * @param {number[]} bits        - Array of 256 integers (0 or 1)
 * @param {number[]|null} refBits - Optional reference bits for diff coloring
 */
function renderBits(canvas, bits, refBits = null) {
  const ctx  = canvas.getContext("2d");
  const W    = canvas.width;
  const H    = canvas.height;
  const cw   = W / BITS_COLS;
  const ch   = H / BITS_ROWS;
  const pad  = 1.5;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#f0f2f5";
  ctx.fillRect(0, 0, W, H);

  for (let idx = 0; idx < Math.min(bits.length, 256); idx++) {
    const row = Math.floor(idx / BITS_COLS);
    const col = idx % BITS_COLS;
    const x   = col * cw;
    const y   = row * ch;
    const bit = bits[idx];

    if (refBits !== null) {
      if (bit !== refBits[idx]) {
        ctx.fillStyle = "#e74c3c";   // error → red
      } else if (bit === 1) {
        ctx.fillStyle = "#2980b9";   // correct 1 → blue
      } else {
        ctx.fillStyle = "#bdc3c7";   // correct 0 → gray
      }
    } else {
      ctx.fillStyle = bit === 1 ? "#2980b9" : "#ecf0f1";
    }

    ctx.fillRect(x + pad, y + pad, cw - 2 * pad, ch - 2 * pad);
  }
}

/**
 * Set canvas resolution and call renderBits.
 * The canvas width is fixed; height is derived from BITS_ROWS.
 */
function initAndRender(canvas, bits, refBits = null) {
  const W = 640;
  const H = Math.round(W / BITS_COLS * BITS_ROWS);
  canvas.width  = W;
  canvas.height = H;
  canvas.style.maxWidth = "100%";
  renderBits(canvas, bits, refBits);
}
