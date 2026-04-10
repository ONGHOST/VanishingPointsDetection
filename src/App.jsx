// ═══════════════════════════════════════════════════════════════
// vanishingPoint.app.jsx  —  UI layer only
// All CV logic lives in vanishingPoint.cv.js
// ═══════════════════════════════════════════════════════════════

import { useState, useRef, useEffect, useCallback } from "react";
import {
  sobelEdges,
  renderEdgeMap,
  houghLines,
  clusterVanishingPoints,
  getLinesForVP,
  lineToEndpoints,
  MAX_VP,
} from "./vanishingPoint.cv.js";

// ── Design constants ─────────────────────────────────────────────

const VP_COLORS    = ["#00f5d4", "#f72585", "#ffd60a"];
const MAX_IMAGE_DIM = 640;
const mono          = { fontFamily: "'Courier New', 'Lucida Console', monospace" };
const T             = {
  xs:   { fontSize: 8,  ...mono },
  sm:   { fontSize: 9,  ...mono },
  base: { fontSize: 10, ...mono },
  md:   { fontSize: 12, ...mono },
  lg:   { fontSize: 15, ...mono },
};

// ═══════════════════════════════════════════════════════════════
// Canvas rendering (DOM-dependent, not tested)
// ═══════════════════════════════════════════════════════════════

function drawVanishingPoints(ctx, vps, lines, width, height) {
  vps.forEach((vp, i) => {
    const color   = VP_COLORS[i % VP_COLORS.length];
    const vpLines = getLinesForVP(lines, vp, width, height).slice(0, 20);
    const px      = Math.max(12, Math.min(width  - 12, vp.x));
    const py      = Math.max(12, Math.min(height - 12, vp.y));

    ctx.strokeStyle = color; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.55;
    vpLines.forEach(l => {
      const pts = lineToEndpoints(l, width, height);
      if (!pts) return;
      ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y); ctx.lineTo(pts[1].x, pts[1].y); ctx.stroke();
    });

    ctx.globalAlpha = 1;
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(px - 14, py); ctx.lineTo(px + 14, py); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(px, py - 14); ctx.lineTo(px, py + 14); ctx.stroke();
    ctx.beginPath(); ctx.arc(px, py, 9, 0, Math.PI * 2);
    ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.stroke();
    ctx.fillStyle = color + "33"; ctx.fill();
    ctx.beginPath(); ctx.arc(px, py, 18, 0, Math.PI * 2);
    ctx.strokeStyle = color + "55"; ctx.lineWidth = 1; ctx.stroke();
  });
}

// ═══════════════════════════════════════════════════════════════
// HOOK — useVanishingPoint
// ═══════════════════════════════════════════════════════════════

function useVanishingPoint(threshold) {
  const canvasRef  = useRef(null);
  const edgeRef    = useRef(null);
  const overlayRef = useRef(null);
  const [status, setStatus] = useState("idle");
  const [result, setResult] = useState(null);

  const analyze = useCallback((src) => {
    setStatus("processing"); setResult(null);
    const img = new window.Image();
    img.onload = () => {
      let w = img.width, h = img.height;
      const s = MAX_IMAGE_DIM / Math.max(w, h);
      if (s < 1) { w = Math.round(w * s); h = Math.round(h * s); }

      // Draw source image
      const srcCanvas = canvasRef.current;
      srcCanvas.width = w; srcCanvas.height = h;
      const srcCtx = srcCanvas.getContext("2d");
      srcCtx.drawImage(img, 0, 0, w, h);

      // Edge detection
      const { mag } = sobelEdges(srcCtx.getImageData(0, 0, w, h), w, h);

      // Edge visualisation
      const ec = edgeRef.current;
      ec.width = w; ec.height = h;
      const edgeMap = renderEdgeMap(mag, w, h);
      ec.getContext("2d").putImageData(new ImageData(edgeMap.data, w, h), 0, 0);

      // Line and VP detection
      const lines = houghLines(mag, w, h, threshold);
      const vps   = clusterVanishingPoints(lines, w, h);

      // Overlay
      const oc = overlayRef.current;
      oc.width = w; oc.height = h;
      drawVanishingPoints(oc.getContext("2d"), vps, lines, w, h);

      setResult({ vps, lineCount: lines.length, width: w, height: h });
      setStatus("done");
    };
    img.src = src;
  }, [threshold]);

  const reset = useCallback(() => { setStatus("idle"); setResult(null); }, []);

  return { canvasRef, edgeRef, overlayRef, status, result, analyze, reset };
}

// ═══════════════════════════════════════════════════════════════
// HOOK — useImageLoader
// ═══════════════════════════════════════════════════════════════

function useImageLoader(onLoad) {
  const [imageSrc, setImageSrc] = useState(null);
  const fileRef = useRef(null);

  const readFile = (file) => {
    if (!file?.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = (e) => { setImageSrc(e.target.result); onLoad(e.target.result); };
    reader.readAsDataURL(file);
  };

  return {
    imageSrc,
    fileRef,
    clearImage: ()  => setImageSrc(null),
    openPicker: ()  => fileRef.current?.click(),
    handleFile: (e) => readFile(e.target.files?.[0]),
    handleDrop: (e) => { e.preventDefault(); readFile(e.dataTransfer.files?.[0]); },
  };
}

// ═══════════════════════════════════════════════════════════════
// UI — Primitives
// ═══════════════════════════════════════════════════════════════

const SectionLabel = ({ children }) => (
  <div style={{ ...T.sm, color: "#444", letterSpacing: "0.2em", marginBottom: 10 }}>{children}</div>
);

const DataRow = ({ label, value, color = "#00f5d4" }) => (
  <div style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: "1px solid #1a1a2a" }}>
    <span style={{ ...T.sm, color: "#556", letterSpacing: "0.08em" }}>{label}</span>
    <span style={{ ...T.sm, color }}>{value}</span>
  </div>
);

const Toggle = ({ checked, onChange, label }) => (
  <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
    <div onClick={onChange} style={{ width: 28, height: 16, borderRadius: 8, position: "relative", transition: "all 0.2s", background: checked ? "#00f5d444" : "#1a1a2a", border: `1px solid ${checked ? "#00f5d4" : "#333"}` }}>
      <div style={{ width: 10, height: 10, borderRadius: "50%", position: "absolute", top: 2, left: checked ? 14 : 2, transition: "left 0.2s", background: checked ? "#00f5d4" : "#444" }} />
    </div>
    <span style={{ ...T.sm, color: "#667", letterSpacing: "0.1em" }}>{label}</span>
  </label>
);

const GhostButton = ({ onClick, accent, children }) => (
  <button onClick={onClick} style={{ ...T.sm, background: "none", cursor: "pointer", border: `1px solid ${accent ? "#00f5d444" : "#2a2a3a"}`, color: accent ? "#00f5d4" : "#667", letterSpacing: accent ? "0.2em" : "0.15em", padding: accent ? "8px 18px" : "5px 12px" }}>
    {children}
  </button>
);

// ═══════════════════════════════════════════════════════════════
// UI — SVG icons
// ═══════════════════════════════════════════════════════════════

const PerspectiveLogo = () => (
  <div style={{ width: 36, height: 36, border: "2px solid #00f5d4", display: "flex", alignItems: "center", justifyContent: "center", position: "relative" }}>
    <div style={{ width: 14, height: 14, background: "#00f5d4", transform: "rotate(45deg)" }} />
    <div style={{ position: "absolute", width: 36, height: 1, background: "#00f5d444", top: "50%" }} />
    <div style={{ position: "absolute", width: 1, height: 36, background: "#00f5d444", left: "50%" }} />
  </div>
);

const PerspectiveIcon = () => (
  <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
    <line x1="24" y1="24" x2="4"  y2="4"  stroke="#00f5d4" strokeWidth="1.5" opacity="0.6" />
    <line x1="24" y1="24" x2="44" y2="4"  stroke="#f72585" strokeWidth="1.5" opacity="0.6" />
    <line x1="24" y1="24" x2="24" y2="4"  stroke="#ffd60a" strokeWidth="1.5" opacity="0.6" />
    <line x1="24" y1="24" x2="4"  y2="44" stroke="#00f5d4" strokeWidth="1.5" opacity="0.3" />
    <line x1="24" y1="24" x2="44" y2="44" stroke="#f72585" strokeWidth="1.5" opacity="0.3" />
    <circle cx="24" cy="24" r="4" fill="#00f5d4" opacity="0.9" />
    <circle cx="24" cy="24" r="8" stroke="#00f5d4" strokeWidth="1" opacity="0.3" />
  </svg>
);

// ═══════════════════════════════════════════════════════════════
// UI — Header
// ═══════════════════════════════════════════════════════════════

const STATUS_COLOR = { idle: "#333", processing: "#ffd60a", done: "#00f5d4" };

const Header = ({ status }) => (
  <div style={{ borderBottom: "1px solid #1e1e2e", padding: "18px 32px", display: "flex", alignItems: "center", justifyContent: "space-between", background: "#0d0d1a" }}>
    <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
      <PerspectiveLogo />
      <div>
        <div style={{ ...T.lg, letterSpacing: "0.18em", color: "#00f5d4", fontWeight: 700 }}>VANISH.PT</div>
        <div style={{ ...T.sm, color: "#555", letterSpacing: "0.25em" }}>PERSPECTIVE ANALYSIS ENGINE</div>
      </div>
    </div>
    <div style={{ display: "flex", gap: 20, alignItems: "center" }}>
      <div style={{ ...T.base, color: "#444", letterSpacing: "0.1em" }}>HOUGH TRANSFORM · EDGE DETECTION · POINT CLUSTERING</div>
      <div style={{ width: 8, height: 8, borderRadius: "50%", background: STATUS_COLOR[status] ?? "#333", boxShadow: status === "done" ? "0 0 8px #00f5d4" : "none" }} />
    </div>
  </div>
);

// ═══════════════════════════════════════════════════════════════
// UI — Sidebar sections
// ═══════════════════════════════════════════════════════════════

const ALGORITHM_STEPS = [
  ["01", "Sobel Edge Detection"],
  ["02", "Hough Transform"],
  ["03", "Line Clustering"],
  ["04", "VP Extraction"],
];

const AlgorithmSteps = () => (
  <div>
    <SectionLabel>ALGORITHM</SectionLabel>
    {ALGORITHM_STEPS.map(([n, label]) => (
      <div key={n} style={{ display: "flex", gap: 10, alignItems: "center", padding: "7px 0", borderBottom: "1px solid #1a1a2a" }}>
        <span style={{ ...T.sm, color: "#00f5d4", opacity: 0.6 }}>{n}</span>
        <span style={{ ...T.base, color: "#778", letterSpacing: "0.05em" }}>{label}</span>
      </div>
    ))}
  </div>
);

const Controls = ({ threshold, setThreshold, showEdges, setShowEdges }) => (
  <div>
    <SectionLabel>CONTROLS</SectionLabel>
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ ...T.sm, color: "#667", letterSpacing: "0.1em" }}>EDGE THRESHOLD</span>
        <span style={{ ...T.sm, color: "#00f5d4" }}>{threshold}</span>
      </div>
      <input type="range" min={30} max={180} value={threshold} onChange={e => setThreshold(+e.target.value)}
        style={{ width: "100%", accentColor: "#00f5d4", background: "transparent" }} />
    </div>
    <Toggle checked={showEdges} onChange={() => setShowEdges(v => !v)} label="SHOW EDGES" />
  </div>
);

const VPCard = ({ vp, index }) => {
  const color = VP_COLORS[index % VP_COLORS.length];
  return (
    <div style={{ padding: 8, marginBottom: 6, border: `1px solid ${color}33`, background: color + "08" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
        <div style={{ width: 6, height: 6, background: color, borderRadius: "50%" }} />
        <span style={{ ...T.sm, color, letterSpacing: "0.1em" }}>VP {index + 1}</span>
      </div>
      <div style={{ ...T.xs, color: "#556", lineHeight: 1.8 }}>
        X: {Math.round(vp.x)} / Y: {Math.round(vp.y)}<br />SUPPORT: {vp.support} lines
      </div>
    </div>
  );
};

const ResultStats = ({ result }) => !result ? null : (
  <div>
    <SectionLabel>RESULTS</SectionLabel>
    <DataRow label="LINES FOUND" value={result.lineCount} />
    <DataRow label="VANISH PTS"  value={result.vps.length} />
    <DataRow label="IMAGE"       value={`${result.width}×${result.height}`} />
    <div style={{ marginTop: 16 }}>
      {result.vps.map((vp, i) => <VPCard key={i} vp={vp} index={i} />)}
    </div>
  </div>
);

const Sidebar = ({ threshold, setThreshold, showEdges, setShowEdges, result }) => (
  <div style={{ width: 220, background: "#0d0d1a", borderRight: "1px solid #1e1e2e", padding: "24px 18px", flexShrink: 0, display: "flex", flexDirection: "column", gap: 24 }}>
    <AlgorithmSteps />
    <Controls threshold={threshold} setThreshold={setThreshold} showEdges={showEdges} setShowEdges={setShowEdges} />
    <ResultStats result={result} />
  </div>
);

// ═══════════════════════════════════════════════════════════════
// UI — Main panels
// ═══════════════════════════════════════════════════════════════

const DropZone = ({ onDrop, onClick }) => (
  <div onDrop={onDrop} onDragOver={e => e.preventDefault()} onClick={onClick} style={{ flex: 1, border: "1px dashed #2a2a40", cursor: "pointer", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 20, background: "repeating-linear-gradient(45deg, transparent, transparent 20px, #0d0d1a 20px, #0d0d1a 21px)" }}>
    <div style={{ width: 80, height: 80, border: "2px solid #2a2a40", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <PerspectiveIcon />
    </div>
    <div style={{ textAlign: "center" }}>
      <div style={{ ...T.md, color: "#00f5d4", letterSpacing: "0.2em", marginBottom: 8 }}>DROP IMAGE HERE</div>
      <div style={{ ...T.sm, color: "#445", letterSpacing: "0.15em" }}>OR CLICK TO SELECT · JPG / PNG / WEBP</div>
    </div>
    <div style={{ ...T.sm, color: "#334", letterSpacing: "0.1em", textAlign: "center", maxWidth: 300 }}>
      Works best on architectural photos, corridors, roads, and interiors with strong perspective lines
    </div>
  </div>
);

const CanvasViewer = ({ canvasRef, edgeRef, overlayRef, showEdges }) => (
  <div style={{ position: "relative", display: "inline-block", maxWidth: "100%" }}>
    <canvas ref={canvasRef} style={{ display: "block", maxWidth: "100%", opacity: showEdges ? 0 : 1, position: showEdges ? "absolute" : "relative" }} />
    <canvas ref={edgeRef}   style={{ display: "block", maxWidth: "100%", opacity: showEdges ? 1 : 0, position: showEdges ? "relative" : "absolute", top: 0, left: 0 }} />
    <canvas ref={overlayRef} style={{ position: "absolute", top: 0, left: 0, maxWidth: "100%", pointerEvents: "none" }} />
  </div>
);

const ImagePanel = ({ status, result, showEdges, canvasRef, edgeRef, overlayRef, onClear, onLoad }) => (
  <div style={{ flex: 1 }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
      <div style={{ ...T.sm, color: "#445", letterSpacing: "0.2em" }}>
        {status === "processing" ? "⟳ ANALYZING..." : "ANALYSIS COMPLETE"}
      </div>
      <GhostButton onClick={onClear}>CLEAR</GhostButton>
    </div>
    <CanvasViewer canvasRef={canvasRef} edgeRef={edgeRef} overlayRef={overlayRef} showEdges={showEdges} />
    {result?.vps.length === 0 && (
      <div style={{ marginTop: 16, padding: "12px 16px", border: "1px solid #ffd60a33", background: "#ffd60a08", ...T.base, color: "#ffd60a", letterSpacing: "0.1em" }}>
        ⚠ No vanishing points detected. Try lowering the threshold or use an image with stronger perspective lines.
      </div>
    )}
    <div style={{ marginTop: 16 }}>
      <GhostButton onClick={onLoad} accent>LOAD NEW IMAGE</GhostButton>
    </div>
  </div>
);

// ═══════════════════════════════════════════════════════════════
// ROOT COMPONENT
// ═══════════════════════════════════════════════════════════════

export default function VanishingPointApp() {
  const [threshold, setThreshold] = useState(80);
  const [showEdges, setShowEdges] = useState(false);

  const { canvasRef, edgeRef, overlayRef, status, result, analyze, reset } = useVanishingPoint(threshold);
  const { imageSrc, fileRef, handleFile, handleDrop, openPicker, clearImage } = useImageLoader(analyze);

  useEffect(() => { if (imageSrc) analyze(imageSrc); }, [threshold]);

  const handleClear = () => { clearImage(); reset(); };

  return (
    <div style={{ minHeight: "100vh", background: "#0a0a0f", color: "#e8e8e8", ...mono, overflowX: "hidden" }}>
      <Header status={status} />
      <div style={{ display: "flex", minHeight: "calc(100vh - 73px)" }}>
        <Sidebar threshold={threshold} setThreshold={setThreshold} showEdges={showEdges} setShowEdges={setShowEdges} result={result} />
        <div style={{ flex: 1, padding: "28px 32px", display: "flex", flexDirection: "column", gap: 20 }}>
          {!imageSrc
            ? <DropZone onDrop={handleDrop} onClick={openPicker} />
            : <ImagePanel status={status} result={result} showEdges={showEdges} canvasRef={canvasRef} edgeRef={edgeRef} overlayRef={overlayRef} onClear={handleClear} onLoad={openPicker} />
          }
        </div>
      </div>
      <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }} onChange={handleFile} />
      <style>{`* { box-sizing: border-box; } input[type=range] { cursor: pointer; height: 4px; } canvas { image-rendering: pixelated; }`}</style>
    </div>
  );
}
