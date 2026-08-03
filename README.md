# Browser-Based Video Content Detection

Determines whether a video contains inappropriate visual content — **entirely
in the browser**. No video data leaves the device; inference runs on-device via
TensorFlow.js.

```json
{ "contains_inappropriate_content": true, "confidence": 0.87 }
```

---

## How it works

1. Load a video (file upload or URL) into a muted `<video>` element.
2. Seek through it at a fixed interval (default 2 fps), up to `maxFrames` (120).
3. Draw each frame to an offscreen `<canvas>` (downscaled to 224px long edge).
4. Classify each frame with **NSFW.js** (`MobileNetV2Mid`) on TensorFlow.js.
5. Score each frame by the combined probability mass on `Porn + Sexy + Hentai`.
6. Aggregate into a calibrated sigmoid confidence and a flag decision.

Each frame yields to the event loop so the UI stays responsive (important on
mobile). The backend priority is **WebGPU → WebGL → WASM → CPU** with automatic
fallback. All thresholds live in the `CFG` object at the top of `app.js`.

### Aggregation

A frame's **inappropriate score** is the sum of probabilities on the
inappropriate classes (`Porn + Sexy + Hentai`) — more sensitive than top-1, so
borderline frames aren't rounded away. The video-level decision combines:

- **Peak confidence** — the max inappropriate score across all frames (≥ 0.8).
- **Flag fraction** — fraction of frames scoring ≥ `frameThreshold` (0.5),
  flagged when ≥ `flagFrameFraction` (0.15).
- **Calibrated probability** — peak (intensity, weight 0.5) and persistence
  (breadth, weight 0.5) blended and squashed through a logistic sigmoid. Flags
  when the calibrated probability clears `calibratedFlagThreshold` (0.5).

The sigmoid `gain`/`bias` in `CFG.calibration` are hand-picked defaults — fit
them against a labeled validation set using the included `calibrateSigmoid()`
helper before production use.

---

## Run locally

The app is static but needs to be served over http(s) (fetch/CORS don't work
from `file://`):

```bash
# Python
python -m http.server 8080

# or Node (no install)
npx serve -l 8080
```

Open <http://localhost:8080>. Choose **Upload** (drag/drop or browse) or
**Link** (must send CORS headers), press **Detect content**, and watch the
per-frame progress, verdict, chart, and performance stats.

> The first run downloads the TensorFlow.js + NSFW.js bundles (~6.4 MB total,
> cached afterwards). These are static model *assets* fetched once — inference
> always runs locally. To be fully air-gapped, host the bundles yourself and
> point the `<script src>` attributes in `index.html` at your own copies.

---

## Deploy

This is a static site — no build step. Deploy the folder to any static host:

- **Vercel** — import the repo; it's auto-detected as a static site (see
  `vercel.json`). No build or install commands needed.
- **Netlify / GitHub Pages / Cloudflare Pages** — point the publish directory
  at the repo root.

---

## Files

```
index.html      # UI: file/URL input, video player, results, chart, perf stats
styles.css      # styling
app.js          # pipeline: load → seek → rasterize → classify → aggregate
overview/       # longer-form write-up of the approach (self-contained HTML)
vercel.json     # static-deploy config for Vercel
```

---

## Limitations

- **CORS for URL input** — the remote host must send `Access-Control-Allow-Origin`
  or the canvas is tainted and classification throws. File uploads always work.
- **Uniform sampling blind spot** — content *between* two samples (e.g. a flash
  at 1.3s between 1.0s and 1.5s) can be missed. Raise `framesPerSecond` for
  denser coverage at the cost of more compute. True per-frame access via
  WebCodecs `VideoDecoder` is the production upgrade path.
- **Visual only** — audio and context are ignored; the model is image-based.
- **Long videos** — capped at `maxFrames`; long videos are sub-sampled.
- **Codec support** — only formats the browser can decode (usually MP4/H.264,
  WebM) are usable.
