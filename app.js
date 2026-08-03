"use strict";

/* ------------------------------------------------------------------
 * Browser-Based Video Content Detection
 * ------------------------------------------------------------------
 * Pipeline:
 *   1. Load video (file or URL) into a <video> element.
 *   2. Seek through the video at a FIXED frame interval (uniform sampling).
 *   3. Draw each frame to an offscreen <canvas> (downscaled).
 *   4. Run NSFW.js MobileNetV2Mid on each frame via TensorFlow.js (WebGL).
 *   5. Aggregate per-frame scores -> { contains_inappropriate_content, confidence }
 *
 * Everything runs locally. No network calls beyond the initial model
 * asset download (an asset, not an inference API).
 * ------------------------------------------------------------------ */

// ---- Config ------------------------------------------------------------
const CFG = {
  framesPerSecond: 2,       // temporal resolution; raise for denser coverage
  classifyEdge: 224,        // MobileNetV2Mid input size (library resizes internally)
  maxFrames: 120,           // compute budget cap
  frameThreshold: 0.5,      // per-class threshold for a frame to count as inappropriate
  flagFrameFraction: 0.15,  // flag if this fraction of frames (or peak) exceeds threshold
  flagPeakConfidence: 0.8,
  inappropriateClasses: ["Porn", "Sexy"],
  includeHentai: true,

  // Calibrated scoring: a logistic (sigmoid) over a blended raw score.
  // Defaults are hand-picks — fit gain/bias against a labeled validation set
  // (see calibrateSigmoid) before production.
  calibration: {
    gain: 6,   // steepness of the decision boundary
    bias: 0.5, // raw score at which P(positive) = 0.5
  },
  calibratedFlagThreshold: 0.5, // flag when calibrated probability >= this
};

// ---- DOM ---------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const fileInput = $("fileInput");
const fileDrop = $("fileDrop");
const urlInput = $("urlInput");
const urlLoadBtn = $("urlLoadBtn");
const video = $("video");
const videoWrap = $("videoWrap");
const blurOverlay = $("blurOverlay");
const liveBadge = $("liveBadge");
const videoCard = $("videoCard");
const detectBtn = $("detectBtn");
const blurToggle = $("blurToggle");
const statusCard = $("statusCard");
const statusText = $("statusText");
const progressBar = $("progressBar");
const progressPct = $("progressPct");
const verdictCard = $("verdictCard");
const verdictTag = $("verdictTag");
const verdictConf = $("verdictConf");
const perfEl = $("perfCard");
const perfGrid = $("perfGrid");
const modelRing = $("modelRing");
const ringLabel = $("ringLabel");
const resultJson = $("resultJson");
const frameChart = $("frameChart");
const frameList = $("frameList");

// ---- State -------------------------------------------------------------
let model = null;
let modelLoading = null;
let currentVideoURL = null;   // canonical URL the user typed (for messages)
let currentVideoSrc = null;   // actual src assigned to <video> (url or blob:) — revoke if blob

// ---- Tabs --------------------------------------------------------------
document.querySelectorAll(".seg-item").forEach((t) =>
  t.addEventListener("click", () => {
    document.querySelectorAll(".seg-item").forEach((x) => {
      x.classList.remove("active");
      x.setAttribute("aria-selected", "false");
    });
    document.querySelectorAll(".panel").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    t.setAttribute("aria-selected", "true");
    $("panel-" + t.dataset.tab).classList.add("active");
  })
);

// ---- Model loading (lazy, async) --------------------------------------
// Backend priority: WebGPU (fastest) -> WebGL -> WASM -> CPU (always works).
// Backends are registered by the <script> tags in index.html.
//
// On mobile, the WebGL/WebGPU backend can silently miscompute the
// MobileNetV2Mid graph model (texture-precision / missing-op issues): the
// conv stack collapses, so the final dense layer emits only its bias terms.
// Because Neutral carries the highest class bias, EVERY frame scores as
// Neutral with ~0 probability elsewhere — the "all neutral 0" failure. The
// output becomes independent of the input, so we detect this by running the
// model on two distinct synthetic images and rejecting any backend whose
// output doesn't change with the input. We then fall through to WASM/CPU.

const WASM_PATH =
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/";

function checkModelBundles() {
  const w = window;
  const modelJson = w.model;
  const shard1 = w.group1_shard1of2;
  const shard2 = w.group1_shard2of2;
  if (!modelJson || !shard1 || !shard2) {
    throw new Error(
      "Model bundles missing. Hard-reload (Ctrl+Shift+R) to bust cache so " +
        "index.html loads the three mobilenet_v2_mid <script> tags " +
        "(model.min.js, group1-shard1of2.min.js, group1-shard2of2.min.js)."
    );
  }
  if (modelJson.format !== "graph-model") {
    throw new Error(
      "Wrong model format (got " + (modelJson.format || "layers") +
        "). Hard-reload (Ctrl+Shift+R) to load the mobilenet_v2_mid graph bundle."
    );
  }
}

// Two deliberately different synthetic images. A healthy backend yields
// different class distributions for them; a dead backend (output is just the
// final-layer biases, independent of input) yields identical outputs.
function syntheticCanvas(variant) {
  const c = document.createElement("canvas");
  c.width = c.height = CFG.classifyEdge;
  const cx = c.getContext("2d");
  if (variant === 0) {
    cx.fillStyle = "#000";
    cx.fillRect(0, 0, c.width, c.height);
  } else {
    const img = cx.createImageData(c.width, c.height);
    const d = img.data;
    let s = 0x5bd1e995;
    const rand = () => {
      s = (Math.imul(s ^ (s >>> 15), 0x2c1b3c6d)) ^ (s >>> 13);
      return (s >>> 0) / 4294967296;
    };
    for (let i = 0; i < d.length; i += 4) {
      d[i] = rand() * 255;
      d[i + 1] = rand() * 255;
      d[i + 2] = rand() * 255;
      d[i + 3] = 255;
    }
    cx.putImageData(img, 0, 0);
  }
  return c;
}

// Run the model on two distinct inputs and reject the backend if the output
// is non-finite OR identical for both inputs (the signature of a broken
// backend whose output no longer depends on the input).
async function backendIsHealthy(m) {
  try {
    const a = await m.classify(syntheticCanvas(0));
    const b = await m.classify(syntheticCanvas(1));
    const va = a.map((p) => p.probability);
    const vb = b.map((p) => p.probability);
    if (!va.length || va.length !== vb.length) return false;
    for (let i = 0; i < va.length; i++) {
      if (!Number.isFinite(va[i]) || !Number.isFinite(vb[i])) return false;
      if (Math.abs(va[i] - vb[i]) > 1e-4) return true; // input-dependent => alive
    }
    return false; // outputs identical across inputs => dead backend
  } catch (e) {
    return false;
  }
}

async function getModel() {
  if (model) return model;
  if (modelLoading) return modelLoading;
  setStatus("Loading model…");
  setRing("loading", "…");
  const tLoad = performance.now();
  modelLoading = (async () => {
    checkModelBundles();
    const order = ["webgpu", "webgl", "wasm", "cpu"];
    for (const name of order) {
      try {
        if (!tf.findBackend(name)) continue;
        if (name === "wasm" && tf.wasm) tf.wasm.setWasmPaths(WASM_PATH);
        const ok = await tf.setBackend(name);
        if (!ok) continue;
        await tf.ready();
        setStatus(`Loading model (${name})…`);
        const m = await nsfwjs.load("MobileNetV2Mid", { type: "graph" });
        if (await backendIsHealthy(m)) {
          model = m;
          return m;
        }
        // Backend produced input-independent (dead) output — dispose and
        // try the next backend. Common on mobile WebGL/WebGPU.
        try { m.dispose(); } catch (e) { /* ignore */ }
        setStatus(`Backend ${name} failed validation, retrying…`);
      } catch (e) { /* backend unavailable or load failed — try next */ }
    }
    // Absolute fallback: CPU always works (just slow).
    await tf.setBackend("cpu");
    await tf.ready();
    setStatus("Loading model (cpu)…");
    const m = await nsfwjs.load("MobileNetV2Mid", { type: "graph" });
    model = m;
    return m;
  })();
  modelLoading.then(() => {
    const s = (performance.now() - tLoad) / 1000;
    setRing("done", s < 1 ? `${Math.round(s * 1000)}ms` : `${s.toFixed(1)}s`);
  }).catch(() => setRing("done", "!"));
  return modelLoading;
}

function setRing(state, text) {
  if (!modelRing) return;
  modelRing.classList.toggle("loading", state === "loading");
  modelRing.classList.toggle("done", state === "done");
  if (text != null && ringLabel) ringLabel.textContent = text;
}

// Pre-warm the model in the background so detection starts faster.
getModel().then(() => setStatus("Ready. Load a video to begin.")).catch(report);

// ---- Video loading -----------------------------------------------------
fileDrop.addEventListener("dragover", (e) => e.preventDefault());
fileDrop.addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});
urlLoadBtn.addEventListener("click", () => {
  if (urlInput.value.trim()) loadURL(urlInput.value.trim());
});
// A known-good, CORS-enabled test video so users can verify the pipeline.
$("sampleBtn").addEventListener("click", () => {
  const SAMPLE =
    "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4";
  urlInput.value = SAMPLE;
  loadURL(SAMPLE);
});

function loadFile(file) {
  revokeCurrent();
  currentVideoURL = URL.createObjectURL(file);
  currentVideoSrc = currentVideoURL;
  video.removeAttribute("crossOrigin"); // blob: is same-origin
  attachVideo(currentVideoURL);
}

function loadURL(url) {
  revokeCurrent();
  currentVideoURL = url;
  currentVideoSrc = null;
  video.crossOrigin = "anonymous"; // needed to read pixels; host must send CORS
  attachVideo(url);
}

function attachVideo(src) {
  verdictCard.classList.add("hidden");
  perfEl.classList.add("hidden");
  video.src = src;
  videoCard.classList.remove("hidden");
  detectBtn.disabled = true;
  setStatus("Loading video…");
  video.onloadeddata = () => {
    detectBtn.disabled = false;
    setStatus(`Loaded: ${video.videoWidth}x${video.videoHeight}, ${video.duration.toFixed(1)}s. Ready to detect.`);
  };
  video.onerror = () => report(new Error(describeVideoError(video, currentVideoURL)));
}

// Map the opaque MediaError code to an actionable message.
function describeVideoError(vid, src) {
  const err = vid.error;
  const isRemote = src && !src.startsWith("blob:");
  const base = "Could not load video";
  if (!err) return base + ".";
  switch (err.code) {
    case 1: return `${base}: loading was aborted.`;
    case 2: // MEDIA_ERR_NETWORK — very commonly a 403 / hotlink block
      return isRemote
        ? `${base}: network error (often a 403 / hotlink block). ${corsHint()}`
        : `${base}: network error while reading the file.`;
    case 3: // MEDIA_ERR_DECODE
      return `${base}: the file is corrupt or uses a codec this browser can't decode.`;
    case 4: // MEDIA_ERR_SRC_NOT_SUPPORTED
      return isRemote
        ? `${base}: source not supported. The URL may be invalid, the host may be blocking it (403/hotlink), or the format is unsupported. ${corsHint()}`
        : `${base}: format not supported by this browser.`;
    default:
      return `${base} (error code ${err.code}).`;
  }
}

function corsHint() {
  return "This host may not allow cross-origin access (CORS). Try a different link, or use File Upload.";
}

// Surface detection-time errors (drawImage/getImageData/classify) clearly,
// especially the tainted-canvas SecurityError from missing CORS headers.
function describeDetectionError(e) {
  const msg = (e && (e.message || e.name)) || String(e);
  if (/security/i.test(msg) || /tainted|cors|cross-origin/i.test(msg)) {
    return `Can't read pixels from this video — the host didn't grant cross-origin access (CORS). The video plays, but we can't scan it. Use File Upload instead.`;
  }
  if (/duration/i.test(msg)) return msg;
  return msg;
}

function revokeCurrent() {
  if (currentVideoSrc && currentVideoSrc.startsWith("blob:")) URL.revokeObjectURL(currentVideoSrc);
  currentVideoSrc = null;
  currentVideoURL = null;
}

// ---- Detection ---------------------------------------------------------
detectBtn.addEventListener("click", runDetection);

async function runDetection() {
  detectBtn.disabled = true;
  statusCard.classList.remove("hidden");
  verdictCard.classList.add("hidden");
  perfEl.classList.add("hidden");
  try {
    const m = await getModel();
    const results = await analyzeVideo(video, m);
    showResults(results);
  } catch (e) {
    report(new Error(describeDetectionError(e)));
  } finally {
    detectBtn.disabled = false;
  }
}

/**
 * Single-pass uniform sampling.
 *
 * Seek at a fixed interval (1 / framesPerSecond), classify each frame, then
 * aggregate per-frame scores into a verdict. No adaptive refinement.
 *
 * NOTE: content BETWEEN two samples (e.g. a flash between 1.0s and 1.5s) can
 * be missed — a fundamental (Nyquist) sampling limit. Raise framesPerSecond
 * for denser coverage at the cost of more compute. True per-frame access via
 * WebCodecs is the real fix for sub-sample transients (see README).
 */
async function analyzeVideo(vid, m) {
  const duration = vid.duration;
  if (!isFinite(duration) || duration <= 0) throw new Error("Invalid video duration.");

  const interval = 1 / CFG.framesPerSecond;
  const totalPlanned = Math.min(CFG.maxFrames, Math.floor(duration / interval));

  const cls = document.createElement("canvas");
  const cctx = cls.getContext("2d", { willReadFrequently: true });
  const edge = CFG.classifyEdge;
  const scale = Math.min(edge / vid.videoWidth, edge / vid.videoHeight, 1);
  cls.width = Math.max(1, Math.round(vid.videoWidth * scale));
  cls.height = Math.max(1, Math.round(vid.videoHeight * scale));

  const frames = [];
  const t0 = performance.now();
  const backend = tf.getBackend();

  // ---- Single pass: uniform sampling (MODEL inference) -------------------
  for (let i = 0; i < totalPlanned; i++) {
    const time = Math.min(duration - 0.01, i * interval);
    await seekTo(vid, time);
    const rec = await classifyFrame(vid, m, cls, cctx, time);
    frames.push(rec);
    setProgress(Math.round(((i + 1) / totalPlanned) * 100));
    setStatus(`Scanning ${i + 1}/${totalPlanned} (${rec.topClass} · score ${(rec.inappropriateScore * 100).toFixed(0)}%)`);
    await nextTick(); // keep the UI responsive (important on mobile)
  }

  // ---- Aggregate ---------------------------------------------------------
  frames.sort((a, b) => a.time - b.time);
  let peak = 0;
  let flaggedCount = 0;
  for (const f of frames) {
    if (f.inappropriateScore > peak) peak = f.inappropriateScore;
    if (f.inappropriate) flaggedCount++;
  }
  const coarseCount = frames.length;
  const flaggedCoarse = flaggedCount;

  // Calibrated scoring: blend peak (intensity) and persistence (breadth) into
  // a single raw score, then squash through a calibrated sigmoid to get a
  // probability. See calibratedConfidence + calibrateSigmoid.
  const { rawConfidence, confidence } = calibratedConfidence({ peak, flaggedCoarse, coarseCount });

  // Decision policy: flag when calibrated probability clears threshold.
  // Legacy peak/fraction thresholds remain as a fallback signal.
  const fraction = coarseCount ? flaggedCoarse / coarseCount : 0;
  const flaggedByFraction = fraction >= CFG.flagFrameFraction;
  const flaggedByPeak = peak >= CFG.flagPeakConfidence;
  const contains = confidence >= CFG.calibratedFlagThreshold || flaggedByFraction || flaggedByPeak;

  const elapsed = (performance.now() - t0) / 1000;
  const out = frames.map((f, idx) => ({
    index: idx,
    time: f.time,
    class: f.topClass,
    probability: f.inappropriateScore, // decision metric = combined inappropriate mass
    topProb: f.topProb,
    scores: f.scores,
    inappropriate: f.inappropriate,
  }));

  return {
    result: { contains_inappropriate_content: contains, confidence },
    meta: {
      frames_analyzed: frames.length,
      flagged_frames: flaggedCount,
      flag_fraction: +fraction.toFixed(3),
      peak_confidence: +peak.toFixed(3),
      raw_confidence: +rawConfidence.toFixed(3),
      frames_per_second: CFG.framesPerSecond,
      sampling: "uniform",
      scoring: "calibrated sigmoid (peak×0.5 + persistence×0.5)",
      calibration: { gain: CFG.calibration.gain, bias: CFG.calibration.bias, threshold: CFG.calibratedFlagThreshold },
      thresholds: "calibrated — fit sigmoid gain/bias against a labeled validation set (see calibrateSigmoid)",
      video_duration_s: +duration.toFixed(2),
      processing_time_s: +elapsed.toFixed(2),
      throughput_fps: +(frames.length / elapsed).toFixed(2),
      backend,
      resolution_in: `${vid.videoWidth}x${vid.videoHeight}`,
      model: "nsfwjs v4 (MobileNetV2Mid, graph, 224)",
    },
    frames: out,
  };
}

// Score a frame by combined probability mass on inappropriate classes — more
// sensitive to borderline frames than top-1 (e.g. 40% Porn + 35% Sexy -> 0.75).
async function classifyFrame(vid, m, cls, cctx, time) {
  cctx.drawImage(vid, 0, 0, cls.width, cls.height);
  const preds = await m.classify(cls); // all 5 classes
  const scores = {};
  let top = preds[0];
  for (const p of preds) {
    scores[p.className] = +p.probability.toFixed(4);
    if (p.probability > top.probability) top = p;
  }
  const inappropriateScore = +inappropriateMass(scores).toFixed(4);
  const inappropriate = inappropriateScore >= CFG.frameThreshold;
  return {
    time: +time.toFixed(2),
    topClass: top.className,
    topProb: +top.probability.toFixed(4),
    scores,
    inappropriateScore,
    inappropriate,
  };
}

// Sum of probabilities over inappropriate classes (Porn + Sexy + [Hentai]).
function inappropriateMass(scores) {
  const set = CFG.inappropriateClasses.slice();
  if (CFG.includeHentai) set.push("Hentai");
  let s = 0;
  for (const c of set) s += scores[c] || 0;
  return s;
}

// ---- Calibrated scoring --------------------------------------------------
// Numerically stable logistic sigmoid. Defaults come from CFG.calibration;
// pass overrides to evaluate a candidate fit.
function sigmoid(x, gain = CFG.calibration.gain, bias = CFG.calibration.bias) {
  const z = gain * (x - bias);
  if (z >= 0) {
    const ez = Math.exp(-z);
    return 1 / (1 + ez);
  }
  const ez = Math.exp(z);
  return ez / (1 + ez);
}

// Blend per-frame signals into a raw score, then map to a probability via the
// calibrated sigmoid.
//   rawConfidence = peak*0.5 + persistence*0.5   where persistence = min(1, posRatio*2)
// `coarseCount`/`flaggedCoarse` use coarse (uniform) frames so refined regions
// — if/when adaptive sampling is added — don't skew breadth.
function calibratedConfidenceRaw({ peak, flaggedCoarse, coarseCount }) {
  const n = Math.max(1, coarseCount);
  const positiveRatio = flaggedCoarse / n;
  const persistence = Math.min(1, positiveRatio * 2);
  const rawConfidence = peak * 0.5 + persistence * 0.5;
  return { rawConfidence: +rawConfidence.toFixed(4) };
}

function calibratedConfidence(input) {
  const { rawConfidence } = calibratedConfidenceRaw(input);
  const confidence = +sigmoid(rawConfidence).toFixed(4);
  return { rawConfidence, confidence };
}

// Fit the sigmoid's gain/bias to labeled data by maximum likelihood (logistic
// regression via gradient descent). This is Platt scaling — run offline on a
// labeled video set, then paste the returned { gain, bias } into CFG.calibration.
function calibrateSigmoid(samples, opts = {}) {
  const {
    gain = 4,
    bias = 0.5,
    lr = 0.5,
    iterations = 4000,
    l2 = 1e-3, // ridge on gain (keeps boundary from over-sharpening)
  } = opts;

  let g = gain, b = bias;
  for (let it = 0; it < iterations; it++) {
    let gg = 0, gb = 0;
    for (const s of samples) {
      const { rawConfidence: raw } = calibratedConfidenceRaw(s);
      const p = sigmoid(raw, g, b);
      const y = s.label;
      gg += (p - y) * (raw - b);
      gb += (y - p) * g;
    }
    gg = gg / samples.length + l2 * g;
    gb = gb / samples.length;
    g -= lr * gg;
    b -= lr * gb;
    if (g <= 0) g = 1e-3; // keep gain positive
  }
  let loss = 0;
  for (const s of samples) {
    const { rawConfidence: raw } = calibratedConfidenceRaw(s);
    const p = Math.min(1 - 1e-7, Math.max(1e-7, sigmoid(raw, g, b)));
    loss += -(s.label * Math.log(p) + (1 - s.label) * Math.log(1 - p));
  }
  return { gain: +g.toFixed(4), bias: +b.toFixed(4), loss: +(loss / samples.length).toFixed(4), iterations };
}

function seekTo(vid, time) {
  return new Promise((resolve, reject) => {
    let done = false;
    // The `seeked` event fires when the seek completes internally, but on
    // mobile browsers (esp. iOS Safari) the video element's displayed frame
    // is not yet repainted by the compositor. Drawing to canvas at that
    // moment captures a stale/black frame, which the classifier scores as
    // Neutral ~100% -> every frame reads "neutral 0". Wait for a real paint
    // cycle before resolving so drawImage() reads the actual seeked frame.
    const finish = () => { if (done) return; done = true; cleanup(); resolve(); };
    const onSeeked = () => {
      if (done) return;
      if (typeof vid.requestVideoFrameCallback === "function") {
        // Fires when a new frame has been presented to the compositor.
        vid.requestVideoFrameCallback(finish);
        // Fallback in case rVFC never fires (paused/seeked on some UAs).
        setTimeout(finish, 60);
      } else {
        // Two rAFs guarantee the compositor has painted the new frame.
        requestAnimationFrame(() => requestAnimationFrame(finish));
      }
    };
    const onError = () => { if (done) return; done = true; cleanup(); reject(new Error("Seek failed")); };
    const cleanup = () => {
      vid.removeEventListener("seeked", onSeeked);
      vid.removeEventListener("error", onError);
    };
    vid.addEventListener("seeked", onSeeked);
    vid.addEventListener("error", onError);
    vid.currentTime = time;
  });
}

function nextTick() {
  return new Promise((r) => setTimeout(r, 0));
}

// ---- Results UI --------------------------------------------------------
function showResults({ result, meta, frames }) {
  const flagged = result.contains_inappropriate_content;
  verdictCard.classList.remove("hidden");
  verdictTag.classList.remove("safe", "flag");
  verdictTag.classList.add(flagged ? "flag" : "safe");
  verdictTag.textContent = flagged ? "Flagged" : "Appears safe";
  verdictConf.textContent = `${Math.round(result.confidence * 100)}%`;

  resultJson.textContent = JSON.stringify(result, null, 2);

  renderPerf(meta);

  frameChart.innerHTML = "";
  const maxP = Math.max(...frames.map((f) => f.probability), 0.001);
  frames.forEach((f) => {
    const bar = document.createElement("div");
    let cls2 = "bar";
    if (f.inappropriate) cls2 += " flag";
    bar.className = cls2;
    bar.style.height = `${Math.round((f.probability / maxP) * 100)}%`;
    const sc = f.scores || {};
    bar.title = `${f.time}s — top:${f.class} score:${(f.probability * 100).toFixed(0)}%\n` +
      `Drawings ${(sc.Drawings * 100 || 0).toFixed(0)}%  Hentai ${(sc.Hentai * 100 || 0).toFixed(0)}%  ` +
      `Neutral ${(sc.Neutral * 100 || 0).toFixed(0)}%  Porn ${(sc.Porn * 100 || 0).toFixed(0)}%  Sexy ${(sc.Sexy * 100 || 0).toFixed(0)}%`;
    frameChart.appendChild(bar);
  });

  frameList.innerHTML = frames
    .map((f) => {
      const sc = f.scores || {};
      const detail = `P${(sc.Porn * 100 || 0).toFixed(0)} S${(sc.Sexy * 100 || 0).toFixed(0)} H${(sc.Hentai * 100 || 0).toFixed(0)}`;
      return `<div class="frow"><span>${f.time}s — ${f.class}</span>` +
        `<span>${(f.probability * 100).toFixed(0)}% ${detail}${f.inappropriate ? " ⚑" : ""}</span></div>`;
    })
    .join("");

  applyLiveResponse(flagged, frames);
  setStatus(flagged ? "Done — flagged." : "Done — appears safe.");
}

// Render per-run timing as compact stat chips. Numbers come from meta (real
// measurements), not placeholders.
function renderPerf(meta) {
  if (!perfEl || !perfGrid) return;
  perfEl.classList.remove("hidden");
  const cells = [
    { l: "Total",    v: fmtTime(meta.processing_time_s) },
    { l: "Throughput", v: meta.throughput_fps != null ? `${meta.throughput_fps} fps` : "—" },
    { l: "Frames",   v: `${meta.frames_analyzed ?? "—"}` },
    { l: "Flagged",  v: meta.flagged_frames != null ? `${meta.flagged_frames}` : "—" },
    { l: "Peak",      v: meta.peak_confidence != null ? `${Math.round(meta.peak_confidence * 100)}%` : "—" },
    { l: "Backend",  v: meta.backend || "—" },
  ];
  perfGrid.innerHTML = cells
    .map((c) => `<div class="perf-stat"><div class="v">${c.v}</div><div class="l">${c.l}</div></div>`)
    .join("");
}

function fmtTime(s) {
  if (s == null || !isFinite(s)) return "—";
  if (s < 1) return `${Math.round(s * 1000)} ms`;
  return `${(+s).toFixed(2)} s`;
}

function applyLiveResponse(flagged, frames) {  if (!blurToggle.checked || !flagged) {
    blurOverlay.classList.add("hidden");
    liveBadge.classList.add("hidden");
    return;
  }
  const first = frames.find((f) => f.inappropriate);
  if (first) seekTo(video, first.time);
  blurOverlay.classList.remove("hidden");
  liveBadge.classList.remove("hidden");
  liveBadge.textContent = "Content restricted — blurred";
}

// ---- Helpers -----------------------------------------------------------
function setStatus(msg) {
  statusText.textContent = msg;
}
function setProgress(pct) {
  progressBar.style.transform = `scaleX(${Math.max(0, Math.min(100, pct)) / 100})`;
  progressPct.textContent = pct + "%";
}
function report(e) {
  console.error(e);
  setStatus("Error: " + e.message);
}
