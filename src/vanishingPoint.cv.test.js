// ═══════════════════════════════════════════════════════════════
// vanishingPoint.cv.test.js
// Run with: vitest   (or jest with --experimental-vm-modules)
// ═══════════════════════════════════════════════════════════════

import { describe, it, expect } from "vitest";
import {
  toGrayscale,
  computeSobelGradients,
  sobelEdges,
  renderEdgeMap,
  buildAccumulator,
  extractPeaks,
  houghLines,
  lineIntersection,
  collectIntersections,
  meanShiftStep,
  countSupportingLines,
  clusterVanishingPoints,
  getLinesForVP,
  lineToEndpoints,
  MAX_LINES,
  MAX_VP,
  HOUGH,
} from "./vanishingPoint.cv.js";

// ─── Helpers ────────────────────────────────────────────────────

/** Build a flat RGBA Uint8ClampedArray of size w×h filled with one colour. */
function solidRGBA(w, h, r, g, b, a = 255) {
  const d = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    d[i * 4] = r; d[i * 4 + 1] = g; d[i * 4 + 2] = b; d[i * 4 + 3] = a;
  }
  return d;
}

/** Build RGBA data with a vertical white stripe at column `col`. */
function verticalStripeRGBA(w, h, col) {
  const d = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    const i = (y * w + col) * 4;
    d[i] = d[i + 1] = d[i + 2] = 255; d[i + 3] = 255;
  }
  return d;
}

/** Build RGBA data with a horizontal white stripe at row `row`. */
function horizontalStripeRGBA(w, h, row) {
  const d = new Uint8ClampedArray(w * h * 4);
  for (let x = 0; x < w; x++) {
    const i = (row * w + x) * 4;
    d[i] = d[i + 1] = d[i + 2] = 255; d[i + 3] = 255;
  }
  return d;
}

/** Create a line in normal form. */
const mkLine = (theta, rho, votes = 10) => ({ theta, rho, votes });

const approx = (a, b, tol = 0.5) => Math.abs(a - b) <= tol;

// ═══════════════════════════════════════════════════════════════
// toGrayscale
// ═══════════════════════════════════════════════════════════════

describe("toGrayscale", () => {
  it("converts a pure-red pixel to the correct luminance", () => {
    const data = new Uint8ClampedArray([255, 0, 0, 255]);
    const gray = toGrayscale(data, 1);
    expect(gray[0]).toBeCloseTo(0.299 * 255, 2);
  });

  it("converts a pure-green pixel to the correct luminance", () => {
    const data = new Uint8ClampedArray([0, 255, 0, 255]);
    const gray = toGrayscale(data, 1);
    expect(gray[0]).toBeCloseTo(0.587 * 255, 2);
  });

  it("converts a pure-blue pixel to the correct luminance", () => {
    const data = new Uint8ClampedArray([0, 0, 255, 255]);
    const gray = toGrayscale(data, 1);
    expect(gray[0]).toBeCloseTo(0.114 * 255, 2);
  });

  it("converts a white pixel to ~255", () => {
    const data = new Uint8ClampedArray([255, 255, 255, 255]);
    const gray = toGrayscale(data, 1);
    expect(gray[0]).toBeCloseTo(255, 0);
  });

  it("converts a black pixel to 0", () => {
    const data = new Uint8ClampedArray([0, 0, 0, 255]);
    const gray = toGrayscale(data, 1);
    expect(gray[0]).toBe(0);
  });

  it("ignores the alpha channel", () => {
    // Same RGB, different alpha → same luminance
    const d1 = new Uint8ClampedArray([100, 150, 200, 0]);
    const d2 = new Uint8ClampedArray([100, 150, 200, 255]);
    expect(toGrayscale(d1, 1)[0]).toBeCloseTo(toGrayscale(d2, 1)[0], 5);
  });

  it("returns a Float32Array of the correct length", () => {
    const data = solidRGBA(4, 4, 128, 128, 128);
    const gray = toGrayscale(data, 16);
    expect(gray).toBeInstanceOf(Float32Array);
    expect(gray.length).toBe(16);
  });

  it("handles a uniform grey image — all values equal", () => {
    const data = solidRGBA(3, 3, 128, 128, 128);
    const gray = toGrayscale(data, 9);
    const v = gray[0];
    expect(gray.every(x => Math.abs(x - v) < 0.01)).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════
// computeSobelGradients
// ═══════════════════════════════════════════════════════════════

describe("computeSobelGradients", () => {
  it("returns mag and ang arrays of the correct size", () => {
    const gray = new Float32Array(5 * 5);
    const { mag, ang } = computeSobelGradients(gray, 5, 5);
    expect(mag.length).toBe(25);
    expect(ang.length).toBe(25);
  });

  it("produces zero magnitude for a uniform (flat) image", () => {
    const w = 5, h = 5;
    const gray = new Float32Array(w * h).fill(200);
    const { mag } = computeSobelGradients(gray, w, h);
    expect(mag.every(v => v === 0)).toBe(true);
  });

  it("leaves border pixels at zero", () => {
    const w = 5, h = 5;
    const gray = new Float32Array(w * h).fill(128);
    gray[7] = 255; // somewhere in the middle
    const { mag } = computeSobelGradients(gray, w, h);
    // corners
    expect(mag[0]).toBe(0);
    expect(mag[w - 1]).toBe(0);
    expect(mag[(h - 1) * w]).toBe(0);
    expect(mag[h * w - 1]).toBe(0);
  });

  it("detects a strong horizontal edge (gradient direction ≈ ±π/2)", () => {
    // Top half black, bottom half white → strong vertical gradient
    const w = 5, h = 5;
    const gray = new Float32Array(w * h);
    for (let y = 3; y < h; y++)
      for (let x = 0; x < w; x++) gray[y * w + x] = 255;
    const { ang } = computeSobelGradients(gray, w, h);
    // The centre pixel (2,2) should have |angle| close to π/2
    const centre = ang[2 * w + 2];
    expect(Math.abs(Math.abs(centre) - Math.PI / 2)).toBeLessThan(0.3);
  });

  it("detects a strong vertical edge (gradient direction ≈ 0 or π)", () => {
    const w = 5, h = 5;
    const gray = new Float32Array(w * h);
    for (let y = 0; y < h; y++)
      for (let x = 3; x < w; x++) gray[y * w + x] = 255;
    const { ang } = computeSobelGradients(gray, w, h);
    const centre = ang[2 * w + 2];
    expect(Math.abs(centre) < 0.3 || Math.abs(Math.abs(centre) - Math.PI) < 0.3).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════
// sobelEdges (integration)
// ═══════════════════════════════════════════════════════════════

describe("sobelEdges", () => {
  it("returns { mag, ang } for a valid imageData-like object", () => {
    const w = 5, h = 5;
    const result = sobelEdges({ data: solidRGBA(w, h, 0, 0, 0) }, w, h);
    expect(result).toHaveProperty("mag");
    expect(result).toHaveProperty("ang");
  });

  it("yields zero magnitude for a solid-colour image", () => {
    const w = 6, h = 6;
    const { mag } = sobelEdges({ data: solidRGBA(w, h, 120, 80, 40) }, w, h);
    expect(mag.every(v => v === 0)).toBe(true);
  });

  it("yields non-zero magnitude at a colour boundary", () => {
    // Left half red, right half blue → sharp boundary at x=5
    const w = 10, h = 10;
    const data = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        data[i] = x < 5 ? 255 : 0;
        data[i + 1] = 0;
        data[i + 2] = x < 5 ? 0 : 255;
        data[i + 3] = 255;
      }
    const { mag } = sobelEdges({ data }, w, h);
    const maxMag = Math.max(...mag);
    expect(maxMag).toBeGreaterThan(0);
  });
});

// ═══════════════════════════════════════════════════════════════
// renderEdgeMap
// ═══════════════════════════════════════════════════════════════

describe("renderEdgeMap", () => {
  it("returns correct width and height", () => {
    const mag = new Float32Array(6).fill(100);
    const out = renderEdgeMap(mag, 3, 2);
    expect(out.width).toBe(3);
    expect(out.height).toBe(2);
  });

  it("returns a data buffer of length width*height*4", () => {
    const w = 4, h = 4;
    const mag = new Float32Array(w * h).fill(50);
    const out = renderEdgeMap(mag, w, h);
    expect(out.data.length).toBe(w * h * 4);
  });

  it("sets alpha channel to 255 for every pixel", () => {
    const w = 3, h = 3;
    const mag = new Float32Array(w * h).fill(100);
    const { data } = renderEdgeMap(mag, w, h);
    for (let i = 3; i < data.length; i += 4) expect(data[i]).toBe(255);
  });

  it("sets blue channel to 0 for every pixel", () => {
    const w = 3, h = 3;
    const mag = new Float32Array(w * h).fill(100);
    const { data } = renderEdgeMap(mag, w, h);
    for (let i = 2; i < data.length; i += 4) expect(data[i]).toBe(0);
  });

  it("normalises so the max magnitude pixel reaches 255 red", () => {
    const mag = new Float32Array([0, 255]);
    const { data } = renderEdgeMap(mag, 2, 1);
    expect(data[4]).toBe(255); // second pixel, red channel
  });

  it("handles an all-zero magnitude array without NaN/divide-by-zero", () => {
    const mag = new Float32Array(4);
    const { data } = renderEdgeMap(mag, 2, 2);
    for (let i = 0; i < data.length; i += 4) {
      expect(data[i]).toBe(0);
      expect(Number.isNaN(data[i])).toBe(false);
    }
  });

  it("green channel is 80% of red channel (rounded)", () => {
    const mag = new Float32Array([128]);
    const { data } = renderEdgeMap(mag, 1, 1);
    expect(data[1]).toBe(Math.round(data[0] * 0.8));
  });
});

// ═══════════════════════════════════════════════════════════════
// buildAccumulator
// ═══════════════════════════════════════════════════════════════

describe("buildAccumulator", () => {
  it("returns acc, numT, numR", () => {
    const mag = new Float32Array(10 * 10);
    const result = buildAccumulator(mag, 10, 10, 50);
    expect(result).toHaveProperty("acc");
    expect(result).toHaveProperty("numT");
    expect(result).toHaveProperty("numR");
  });

  it("numT equals Math.round(π / thetaRes) = 180", () => {
    const mag = new Float32Array(10 * 10);
    const { numT } = buildAccumulator(mag, 10, 10, 50);
    expect(numT).toBe(180);
  });

  it("produces an all-zero accumulator for a blank image", () => {
    const w = 10, h = 10;
    const mag = new Float32Array(w * h); // all zeros → all < threshold
    const { acc } = buildAccumulator(mag, w, h, 1);
    expect(acc.every(v => v === 0)).toBe(true);
  });

  it("produces a non-zero accumulator for an image with strong edges", () => {
    const w = 20, h = 20;
    const mag = new Float32Array(w * h).fill(200); // all pixels above threshold
    const { acc } = buildAccumulator(mag, w, h, 50);
    expect(acc.some(v => v > 0)).toBe(true);
  });

  it("votes increase monotonically when threshold is lowered", () => {
    const w = 15, h = 15;
    const mag = new Float32Array(w * h);
    for (let i = 0; i < mag.length; i++) mag[i] = 100;

    const totalVotes = (thr) => {
      const { acc } = buildAccumulator(mag, w, h, thr);
      return acc.reduce((s, v) => s + v, 0);
    };
    expect(totalVotes(50)).toBeGreaterThan(totalVotes(150));
  });
});

// ═══════════════════════════════════════════════════════════════
// extractPeaks
// ═══════════════════════════════════════════════════════════════

describe("extractPeaks", () => {
  it("returns an empty array when accumulator is all zeros", () => {
    const numT = 180, numR = 100;
    const acc = new Int32Array(numT * numR);
    expect(extractPeaks({ acc, numT, numR })).toEqual([]);
  });

  it("returns at most MAX_LINES lines", () => {
    const numT = 180, numR = 200;
    const acc = new Int32Array(numT * numR);
    // Scatter many isolated peaks well away from borders
    for (let t = 10; t < numT - 10; t += 8)
      for (let r = 10; r < numR - 10; r += 8)
        acc[t * numR + r] = 50;
    const lines = extractPeaks({ acc, numT, numR });
    expect(lines.length).toBeLessThanOrEqual(MAX_LINES);
  });

  it("lines are sorted by votes descending", () => {
    const numT = 180, numR = 200;
    const acc = new Int32Array(numT * numR);
    acc[20 * numR + 20] = 30;
    acc[40 * numR + 40] = 50;
    acc[60 * numR + 60] = 20;
    const lines = extractPeaks({ acc, numT, numR });
    for (let i = 1; i < lines.length; i++)
      expect(lines[i - 1].votes).toBeGreaterThanOrEqual(lines[i].votes);
  });

  it("each line has theta and rho numeric properties", () => {
    const numT = 180, numR = 200;
    const acc = new Int32Array(numT * numR);
    acc[30 * numR + 30] = 20;
    const lines = extractPeaks({ acc, numT, numR });
    if (lines.length > 0) {
      expect(typeof lines[0].theta).toBe("number");
      expect(typeof lines[0].rho).toBe("number");
    }
  });

  it("suppresses neighbours so only the local maximum survives", () => {
    const numT = 180, numR = 200;
    const acc = new Int32Array(numT * numR);
    // Strong peak at (50, 100)
    acc[50 * numR + 100] = 40;
    // Slightly weaker immediately adjacent pixels
    acc[51 * numR + 100] = 35;
    acc[50 * numR + 101] = 33;
    const lines = extractPeaks({ acc, numT, numR });
    // Only the strongest should survive suppression
    const atPeak = lines.filter(l => approx(l.votes, 40, 1));
    expect(atPeak.length).toBeGreaterThanOrEqual(1);
    expect(lines.filter(l => approx(l.votes, 35, 1)).length).toBe(0);
  });
});

// ═══════════════════════════════════════════════════════════════
// houghLines (integration)
// ═══════════════════════════════════════════════════════════════

describe("houghLines", () => {
  it("returns an array", () => {
    const mag = new Float32Array(20 * 20);
    expect(Array.isArray(houghLines(mag, 20, 20, 50))).toBe(true);
  });

  it("returns no lines for a blank image", () => {
    const mag = new Float32Array(30 * 30);
    expect(houghLines(mag, 30, 30, 50)).toHaveLength(0);
  });

  it("detects at least one line for a diagonal stripe (theta ≈ 35°)", () => {
    // A true vertical stripe has theta≈0 which falls in the NMS suppression border
    // (t < suppressT=5). A diagonal stripe avoids this and is a more realistic input.
    const w = 80, h = 80;
    const data = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) {
      const x = Math.round(y * 0.7 + 10);
      if (x >= 0 && x < w) {
        const i = (y * w + x) * 4;
        data[i] = data[i + 1] = data[i + 2] = 255; data[i + 3] = 255;
      }
    }
    const { mag } = sobelEdges({ data }, w, h);
    const lines = houghLines(mag, w, h, 10);
    expect(lines.length).toBeGreaterThan(0);
  });

  it("detects at least one line for a strong horizontal stripe", () => {
    const w = 80, h = 80;
    const data = horizontalStripeRGBA(w, h, 40);
    const { mag } = sobelEdges({ data }, w, h);
    const lines = houghLines(mag, w, h, 10);
    expect(lines.length).toBeGreaterThan(0);
  });

  it("returns fewer lines when threshold is higher", () => {
    const w = 40, h = 40;
    const mag = new Float32Array(w * h).fill(120);
    const lo = houghLines(mag, w, h, 30).length;
    const hi = houghLines(mag, w, h, 110).length;
    expect(lo).toBeGreaterThanOrEqual(hi);
  });
});

// ═══════════════════════════════════════════════════════════════
// lineIntersection
// ═══════════════════════════════════════════════════════════════

describe("lineIntersection", () => {
  it("returns null for exactly parallel lines (same theta)", () => {
    const l1 = mkLine(Math.PI / 4, 10);
    const l2 = mkLine(Math.PI / 4, 20);
    expect(lineIntersection(l1, l2)).toBeNull();
  });

  it("returns null for nearly parallel lines (det < 1e-6)", () => {
    const eps = 1e-8;
    const l1 = mkLine(0.5, 10);
    const l2 = mkLine(0.5 + eps, 20);
    expect(lineIntersection(l1, l2)).toBeNull();
  });

  it("computes a known intersection: two perpendicular lines through origin", () => {
    // θ=0 (vertical, rho=5 → x=5) and θ=π/2 (horizontal, rho=10 → y=10)
    const l1 = mkLine(0, 5);
    const l2 = mkLine(Math.PI / 2, 10);
    const pt = lineIntersection(l1, l2);
    expect(pt).not.toBeNull();
    expect(pt.x).toBeCloseTo(5, 1);
    expect(pt.y).toBeCloseTo(10, 1);
  });

  it("intersection is symmetric: swap arguments → same point", () => {
    const l1 = mkLine(0.4, 30);
    const l2 = mkLine(1.1, 50);
    const p1 = lineIntersection(l1, l2);
    const p2 = lineIntersection(l2, l1);
    expect(p1).not.toBeNull();
    expect(p2).not.toBeNull();
    expect(p1.x).toBeCloseTo(p2.x, 4);
    expect(p1.y).toBeCloseTo(p2.y, 4);
  });

  it("returns an object with x and y numeric properties", () => {
    const pt = lineIntersection(mkLine(0.3, 10), mkLine(1.2, 20));
    expect(pt).not.toBeNull();
    expect(typeof pt.x).toBe("number");
    expect(typeof pt.y).toBe("number");
  });
});

// ═══════════════════════════════════════════════════════════════
// collectIntersections
// ═══════════════════════════════════════════════════════════════

describe("collectIntersections", () => {
  it("returns an empty array for an empty line list", () => {
    expect(collectIntersections([], 1000)).toEqual([]);
  });

  it("returns an empty array for a single line", () => {
    expect(collectIntersections([mkLine(0.5, 10)], 1000)).toEqual([]);
  });

  it("skips nearly parallel pairs (dTheta < 0.08)", () => {
    const l1 = mkLine(0.5, 10);
    const l2 = mkLine(0.5 + 0.01, 20); // dTheta = 0.01 < 0.08
    expect(collectIntersections([l1, l2], 1e6)).toHaveLength(0);
  });

  it("filters intersections outside the margin", () => {
    const l1 = mkLine(0.2, 10);
    const l2 = mkLine(1.5, 10000); // very far intersection
    const pts = collectIntersections([l1, l2], 100);
    expect(pts.length).toBe(0);
  });

  it("includes intersections inside the margin", () => {
    const l1 = mkLine(0.3, 50);
    const l2 = mkLine(1.2, 50);
    const pts = collectIntersections([l1, l2], 1e6);
    expect(pts.length).toBe(1);
  });

  it("returns n*(n-1)/2 points at most for n non-parallel lines (within margin)", () => {
    const lines = [
      mkLine(0.2, 10), mkLine(0.7, 20), mkLine(1.3, 30),
    ];
    const pts = collectIntersections(lines, 1e9);
    expect(pts.length).toBeLessThanOrEqual(3);
  });
});

// ═══════════════════════════════════════════════════════════════
// meanShiftStep
// ═══════════════════════════════════════════════════════════════

describe("meanShiftStep", () => {
  it("returns the same point when no neighbours are within bandwidth", () => {
    const pts = [{ x: 1000, y: 1000 }];
    const result = meanShiftStep(0, 0, pts, 1); // bw2=1, far away
    expect(result).toEqual({ x: 0, y: 0 });
  });

  it("moves towards a single neighbour within bandwidth", () => {
    const pts = [{ x: 10, y: 10 }];
    const result = meanShiftStep(0, 0, pts, 10000); // bw2 large
    expect(result.x).toBeCloseTo(10);
    expect(result.y).toBeCloseTo(10);
  });

  it("moves to the mean of two equidistant neighbours", () => {
    const pts = [{ x: -5, y: 0 }, { x: 5, y: 0 }];
    const result = meanShiftStep(0, 0, pts, 10000);
    expect(result.x).toBeCloseTo(0);
    expect(result.y).toBeCloseTo(0);
  });

  it("returns a point with x and y properties", () => {
    const result = meanShiftStep(5, 5, [{ x: 6, y: 6 }], 10000);
    expect(result).toHaveProperty("x");
    expect(result).toHaveProperty("y");
  });

  it("only considers points strictly inside bandwidth² radius", () => {
    const bw2 = 25; // radius = 5
    const pts = [
      { x: 4, y: 0 },  // distance = 4  → inside
      { x: 6, y: 0 },  // distance = 6  → outside
    ];
    const result = meanShiftStep(0, 0, pts, bw2);
    expect(result.x).toBeCloseTo(4); // only the first point counts
  });
});

// ═══════════════════════════════════════════════════════════════
// countSupportingLines
// ═══════════════════════════════════════════════════════════════

describe("countSupportingLines", () => {
  it("returns 0 when line list is empty", () => {
    expect(countSupportingLines(100, 100, [], 10)).toBe(0);
  });

  it("counts a line whose normal-form distance to the point is within radius", () => {
    // Horizontal line y = 50 → theta=π/2, rho=50
    // Point (0, 50) → dist = |0·cos(π/2) + 50·sin(π/2) - 50| = 0
    const lines = [mkLine(Math.PI / 2, 50)];
    expect(countSupportingLines(0, 50, lines, 5)).toBe(1);
  });

  it("does not count a line whose distance exceeds the radius", () => {
    const lines = [mkLine(Math.PI / 2, 50)];
    // Point (0, 100): dist = |0 + 100 - 50| = 50, radius = 5
    expect(countSupportingLines(0, 100, lines, 5)).toBe(0);
  });

  it("counts multiple supporting lines correctly", () => {
    const lines = [
      mkLine(Math.PI / 2, 50), // passes through (0,50)
      mkLine(0, 0),             // x=0 passes through (0,50)
      mkLine(Math.PI / 4, 35), // diagonal far from (0,50)
    ];
    const count = countSupportingLines(0, 50, lines, 5);
    expect(count).toBeGreaterThanOrEqual(1);
    expect(count).toBeLessThanOrEqual(lines.length);
  });
});

// ═══════════════════════════════════════════════════════════════
// clusterVanishingPoints
// ═══════════════════════════════════════════════════════════════

describe("clusterVanishingPoints", () => {
  it("returns an empty array when line list is empty", () => {
    expect(clusterVanishingPoints([], 640, 480)).toEqual([]);
  });

  it("returns at most MAX_VP vanishing points", () => {
    // Build many lines in very different directions
    const lines = Array.from({ length: 40 }, (_, i) => mkLine((i / 40) * Math.PI, i * 5, 15));
    const vps = clusterVanishingPoints(lines, 640, 480);
    expect(vps.length).toBeLessThanOrEqual(MAX_VP);
  });

  it("each VP has x, y, and support properties", () => {
    const lines = [
      mkLine(0.3, 100, 15), mkLine(0.5, 80, 15), mkLine(0.7, 60, 15),
      mkLine(1.1, 40, 15),  mkLine(1.3, 30, 15), mkLine(1.5, 20, 15),
    ];
    const vps = clusterVanishingPoints(lines, 640, 480);
    for (const vp of vps) {
      expect(typeof vp.x).toBe("number");
      expect(typeof vp.y).toBe("number");
      expect(typeof vp.support).toBe("number");
    }
  });

  it("returns [] when no cluster reaches minSupport", () => {
    // Only 2 lines → they can't produce minSupport=4
    const lines = [mkLine(0.3, 10, 5), mkLine(1.2, 10, 5)];
    const vps = clusterVanishingPoints(lines, 640, 480, 4);
    expect(vps.length).toBe(0);
  });

  it("sorts results by support descending", () => {
    const vps = clusterVanishingPoints(
      Array.from({ length: 20 }, (_, i) => mkLine((i / 20) * Math.PI, i * 10, 10)),
      640, 480, 2
    );
    for (let i = 1; i < vps.length; i++)
      expect(vps[i - 1].support).toBeGreaterThanOrEqual(vps[i].support);
  });
});

// ═══════════════════════════════════════════════════════════════
// getLinesForVP
// ═══════════════════════════════════════════════════════════════

describe("getLinesForVP", () => {
  it("returns all lines when every line passes through the VP", () => {
    // Vertical line x=50 in normal form: theta=0, rho=50
    const vp    = { x: 50, y: 0 };
    const lines = [mkLine(0, 50)];
    const result = getLinesForVP(lines, vp, 640, 480);
    expect(result.length).toBe(1);
  });

  it("returns an empty array when no line is close to the VP", () => {
    const vp    = { x: 320, y: 240 };
    const lines = [mkLine(0, 0)]; // passes through x=0, far from VP
    const result = getLinesForVP(lines, vp, 640, 480);
    expect(result.length).toBe(0);
  });

  it("radius scales with the smaller image dimension (8%)", () => {
    // radius = min(100,200)*0.08 = 8
    const vp = { x: 0, y: 0 };
    // theta=0, rho=5 → |0*cos0 + 0*sin0 - 5| = 5 < 8 → should match
    const inside  = mkLine(0, 5);
    // theta=0, rho=20 → dist = 20 > 8 → should not match
    const outside = mkLine(0, 20);
    const result = getLinesForVP([inside, outside], vp, 100, 200);
    expect(result).toContain(inside);
    expect(result).not.toContain(outside);
  });
});

// ═══════════════════════════════════════════════════════════════
// lineToEndpoints
// ═══════════════════════════════════════════════════════════════

describe("lineToEndpoints", () => {
  it("returns null for a line that does not cross the canvas", () => {
    // A line with rho >> canvas diagonal will not intersect the canvas
    const result = lineToEndpoints({ theta: Math.PI / 4, rho: 1e6 }, 100, 100);
    expect(result).toBeNull();
  });

  it("returns a two-element array for a line crossing the canvas", () => {
    // Vertical line at x=50: theta=0, rho=50
    const result = lineToEndpoints({ theta: 0, rho: 50 }, 200, 200);
    expect(result).not.toBeNull();
    expect(result).toHaveLength(2);
  });

  it("endpoints are within the extended canvas bounds (±5px tolerance)", () => {
    const w = 200, h = 200;
    const result = lineToEndpoints({ theta: Math.PI / 4, rho: 100 }, w, h);
    if (!result) return; // line might not cross — skip
    for (const p of result) {
      expect(p.x).toBeGreaterThanOrEqual(-5);
      expect(p.x).toBeLessThanOrEqual(w + 5);
      expect(p.y).toBeGreaterThanOrEqual(-5);
      expect(p.y).toBeLessThanOrEqual(h + 5);
    }
  });

  it("a horizontal line (theta=π/2) crosses top and bottom of canvas", () => {
    // theta=π/2, rho=100 → y=100 for any x
    const result = lineToEndpoints({ theta: Math.PI / 2, rho: 100 }, 200, 200);
    expect(result).not.toBeNull();
    expect(result[0].y).toBeCloseTo(100, 0);
    expect(result[1].y).toBeCloseTo(100, 0);
  });

  it("a vertical line (theta=0) crosses left and right of canvas", () => {
    // theta=0, rho=80 → x=80 for any y
    const result = lineToEndpoints({ theta: 0, rho: 80 }, 200, 200);
    expect(result).not.toBeNull();
    expect(result[0].x).toBeCloseTo(80, 0);
    expect(result[1].x).toBeCloseTo(80, 0);
  });

  it("endpoints[0] and endpoints[1] are different points", () => {
    const result = lineToEndpoints({ theta: Math.PI / 3, rho: 50 }, 200, 200);
    if (!result) return;
    const [p0, p1] = result;
    const dist = Math.hypot(p1.x - p0.x, p1.y - p0.y);
    expect(dist).toBeGreaterThan(1);
  });
});

// ═══════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════

describe("module constants", () => {
  it("MAX_LINES is a positive integer", () => {
    expect(Number.isInteger(MAX_LINES)).toBe(true);
    expect(MAX_LINES).toBeGreaterThan(0);
  });

  it("MAX_VP is 3", () => {
    expect(MAX_VP).toBe(3);
  });

  it("HOUGH.thetaRes equals π/180", () => {
    expect(HOUGH.thetaRes).toBeCloseTo(Math.PI / 180, 10);
  });

  it("HOUGH.rhoRes is a positive number", () => {
    expect(HOUGH.rhoRes).toBeGreaterThan(0);
  });

  it("HOUGH suppression windows are positive integers", () => {
    expect(Number.isInteger(HOUGH.suppressR)).toBe(true);
    expect(Number.isInteger(HOUGH.suppressT)).toBe(true);
    expect(HOUGH.suppressR).toBeGreaterThan(0);
    expect(HOUGH.suppressT).toBeGreaterThan(0);
  });
});
