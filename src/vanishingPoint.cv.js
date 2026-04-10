// ═══════════════════════════════════════════════════════════════
// vanishingPoint.cv.js
// Pure computer-vision functions — no DOM, no React, fully testable.
// ═══════════════════════════════════════════════════════════════

// ── Constants ───────────────────────────────────────────────────

export const MAX_LINES = 120;
export const MAX_VP    = 3;

export const HOUGH = {
  rhoRes:    2,
  thetaRes:  Math.PI / 180,
  minVotes:  8,
  suppressR: 3,
  suppressT: 5,
};

const SOBEL_KX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
const SOBEL_KY = [-1, -2, -1,  0, 0, 0,  1, 2, 1];

// ── Edge detection ──────────────────────────────────────────────

/**
 * Convert raw RGBA pixel data to a grayscale Float32Array using
 * standard luminance coefficients (BT.601).
 */
export function toGrayscale(data, size) {
  const gray = new Float32Array(size);
  for (let i = 0; i < size; i++)
    gray[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
  return gray;
}

/**
 * Apply 3×3 Sobel kernels to a grayscale image.
 * Returns { mag, ang } typed arrays of length width*height.
 * Border pixels are left at 0.
 */
export function computeSobelGradients(gray, width, height) {
  const mag = new Float32Array(width * height);
  const ang = new Float32Array(width * height);
  for (let y = 1; y < height - 1; y++)
    for (let x = 1; x < width - 1; x++) {
      let gx = 0, gy = 0;
      for (let ky = -1; ky <= 1; ky++)
        for (let kx = -1; kx <= 1; kx++) {
          const v  = gray[(y + ky) * width + (x + kx)];
          const ki = (ky + 1) * 3 + (kx + 1);
          gx += SOBEL_KX[ki] * v;
          gy += SOBEL_KY[ki] * v;
        }
      mag[y * width + x] = Math.sqrt(gx * gx + gy * gy);
      ang[y * width + x] = Math.atan2(gy, gx);
    }
  return { mag, ang };
}

/** Convenience wrapper: grayscale → Sobel gradients from an ImageData-like object. */
export function sobelEdges(imageData, width, height) {
  return computeSobelGradients(toGrayscale(imageData.data, width * height), width, height);
}

/**
 * Build an orange-tinted ImageData for the edge magnitude map,
 * normalised so the brightest pixel maps to 255.
 */
export function renderEdgeMap(mag, width, height) {
  let maxMag = 0;
  for (let i = 0; i < mag.length; i++) maxMag = Math.max(maxMag, mag[i]);
  const out = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < mag.length; i++) {
    const v = maxMag > 0 ? Math.min(255, (mag[i] / maxMag) * 510) : 0;
    out[i * 4]     = v;
    out[i * 4 + 1] = Math.round(v * 0.8);
    out[i * 4 + 2] = 0;
    out[i * 4 + 3] = 255;
  }
  return { data: out, width, height }; // ImageData-compatible shape (avoids DOM dep)
}

// ── Hough transform ─────────────────────────────────────────────

/**
 * Vote into a (theta, rho) accumulator for every edge pixel
 * whose magnitude exceeds `threshold`.
 */
export function buildAccumulator(mag, width, height, threshold) {
  const { rhoRes, thetaRes } = HOUGH;
  const diag = Math.sqrt(width * width + height * height);
  const numT = Math.round(Math.PI / thetaRes);
  const numR = Math.round(2 * diag / rhoRes) + 1;
  const acc  = new Int32Array(numT * numR);
  const cosT = new Float32Array(numT);
  const sinT = new Float32Array(numT);

  for (let t = 0; t < numT; t++) { cosT[t] = Math.cos(t * thetaRes); sinT[t] = Math.sin(t * thetaRes); }

  for (let y = 1; y < height - 1; y++)
    for (let x = 1; x < width - 1; x++) {
      if (mag[y * width + x] < threshold) continue;
      for (let t = 0; t < numT; t++) {
        const ri = Math.round((x * cosT[t] + y * sinT[t]) / rhoRes + numR / 2);
        if (ri >= 0 && ri < numR) acc[t * numR + ri]++;
      }
    }
  return { acc, numT, numR };
}

/**
 * Extract local maxima from the accumulator via non-maximum suppression,
 * sorted by vote count descending, capped at MAX_LINES.
 */
export function extractPeaks({ acc, numT, numR }) {
  const { rhoRes, thetaRes, minVotes, suppressR: sr, suppressT: st } = HOUGH;
  const lines = [];
  for (let t = st; t < numT - st; t++)
    for (let r = sr; r < numR - sr; r++) {
      const v = acc[t * numR + r];
      if (v < minVotes) continue;
      let isMax = true;
      outer: for (let dt = -st; dt <= st; dt++)
        for (let dr = -sr; dr <= sr; dr++) {
          if (dt === 0 && dr === 0) continue;
          if (acc[(t + dt) * numR + (r + dr)] >= v) { isMax = false; break outer; }
        }
      if (isMax) lines.push({ theta: t * thetaRes, rho: (r - numR / 2) * rhoRes, votes: v });
    }
  return lines.sort((a, b) => b.votes - a.votes).slice(0, MAX_LINES);
}

/** Full pipeline: edge magnitude → detected lines. */
export function houghLines(mag, width, height, threshold) {
  return extractPeaks(buildAccumulator(mag, width, height, threshold));
}

// ── Vanishing point detection ────────────────────────────────────

/**
 * Compute the intersection of two lines in normal form (theta, rho).
 * Returns null when lines are parallel (|det| < 1e-6).
 */
export function lineIntersection(l1, l2) {
  const det = Math.cos(l1.theta) * Math.sin(l2.theta) - Math.sin(l1.theta) * Math.cos(l2.theta);
  if (Math.abs(det) < 1e-6) return null;
  return {
    x: (l1.rho * Math.sin(l2.theta) - l2.rho * Math.sin(l1.theta)) / det,
    y: (l2.rho * Math.cos(l1.theta) - l1.rho * Math.cos(l2.theta)) / det,
  };
}

/**
 * Collect all pairwise intersections of non-parallel lines
 * whose coordinates fall within ±margin of the origin.
 */
export function collectIntersections(lines, margin) {
  const pts = [];
  for (let i = 0; i < lines.length; i++)
    for (let j = i + 1; j < lines.length; j++) {
      const dTheta = Math.abs(lines[i].theta - lines[j].theta);
      if (dTheta < 0.08 || Math.abs(dTheta - Math.PI) < 0.08) continue;
      const pt = lineIntersection(lines[i], lines[j]);
      if (pt && Math.abs(pt.x) <= margin && Math.abs(pt.y) <= margin) pts.push(pt);
    }
  return pts;
}

/**
 * One mean-shift iteration: shift (cx, cy) towards the mean of all
 * points within bandwidth² of the current position.
 */
export function meanShiftStep(cx, cy, pts, bw2) {
  let sx = 0, sy = 0, w = 0;
  for (const p of pts) {
    if ((p.x - cx) ** 2 + (p.y - cy) ** 2 < bw2) { sx += p.x; sy += p.y; w++; }
  }
  return w > 0 ? { x: sx / w, y: sy / w } : { x: cx, y: cy };
}

/**
 * Count how many lines pass within `radius` of the point (cx, cy)
 * using the point-to-line distance in normal form.
 */
export function countSupportingLines(cx, cy, lines, radius) {
  return lines.filter(l =>
    Math.abs(cx * Math.cos(l.theta) + cy * Math.sin(l.theta) - l.rho) < radius
  ).length;
}

/**
 * Detect up to MAX_VP vanishing points by:
 *   1. Collecting all line-pair intersections
 *   2. Running mean-shift clustering (10 iterations per seed)
 *   3. Filtering clusters with fewer than minSupport supporting lines
 *   4. Merging clusters that are too close (< 2×bandwidth)
 */
export function clusterVanishingPoints(lines, width, height, minSupport = 4) {
  const margin    = Math.max(width, height) * 3;
  const bandwidth = Math.min(width, height) * 0.15;
  const bw2       = bandwidth ** 2;
  const pts       = collectIntersections(lines, margin);
  if (pts.length === 0) return [];

  const used     = new Uint8Array(pts.length);
  const clusters = [];

  for (let i = 0; i < pts.length; i++) {
    if (used[i]) continue;
    let { x: cx, y: cy } = pts[i];
    for (let s = 0; s < 10; s++) ({ x: cx, y: cy } = meanShiftStep(cx, cy, pts, bw2));

    const support  = countSupportingLines(cx, cy, lines, bandwidth * 0.5);
    if (support < minSupport) continue;

    const tooClose = clusters.some(p => (p.x - cx) ** 2 + (p.y - cy) ** 2 < bw2 * 4);
    if (!tooClose) {
      clusters.push({ x: cx, y: cy, support });
      for (let k = 0; k < pts.length; k++)
        if ((pts[k].x - cx) ** 2 + (pts[k].y - cy) ** 2 < bw2) used[k] = 1;
    }
  }
  return clusters.sort((a, b) => b.support - a.support).slice(0, MAX_VP);
}

/**
 * Return the subset of `lines` that pass within 8% of the smaller
 * image dimension of the vanishing point `vp`.
 */
export function getLinesForVP(lines, vp, width, height) {
  const radius = Math.min(width, height) * 0.08;
  return lines.filter(l =>
    Math.abs(vp.x * Math.cos(l.theta) + vp.y * Math.sin(l.theta) - l.rho) < radius
  );
}

/**
 * Clip a line (normal form) to the image rectangle.
 * Returns [p0, p1] endpoint pair or null if the line misses the canvas.
 */
export function lineToEndpoints({ theta, rho }, width, height) {
  const cosT = Math.cos(theta), sinT = Math.sin(theta);
  const cands = [];
  if (Math.abs(sinT) > 1e-6) {
    cands.push({ x: 0,     y: rho / sinT });
    cands.push({ x: width, y: (rho - width * cosT) / sinT });
  }
  if (Math.abs(cosT) > 1e-6) {
    cands.push({ x: rho / cosT,                   y: 0 });
    cands.push({ x: (rho - height * sinT) / cosT, y: height });
  }
  const valid = cands.filter(p => p.x >= -5 && p.x <= width + 5 && p.y >= -5 && p.y <= height + 5);
  return valid.length >= 2 ? [valid[0], valid[valid.length - 1]] : null;
}
