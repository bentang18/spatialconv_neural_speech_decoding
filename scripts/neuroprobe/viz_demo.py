"""Self-contained interactive page from the reduced condition means.

One HTML file, no network, data inlined as JSON. It shows the same three things the static
figures show, but lets the viewer drive the comparison that matters: pick a task, pick a
depth, and watch whether six brains move together or not.

The payload is deliberately the SAME arrays the static figures and the printed numbers come
from -- projected once here, into a basis fit exactly as figure_tasks fits it -- so the page
cannot drift away from the reported result.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from scripts.neuroprobe.viz_common import (
    board_cs_auroc, load_all, pca_basis, shared_lobes, to_rgb,
)
from scripts.neuroprobe.viz_figures import (
    CONTRAST, _corr, _proj_origin, _traj, align_loso, collect, peak_settle, retrieval,
)

HTML = """<title>Cross-subject structure in a self-supervised iEEG encoder</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0 auto; max-width: 1100px; padding: 24px; line-height: 1.5; }
  h1 { font-size: 1.4rem; margin: 0 0 4px; }
  .sub { opacity: .7; font-size: .9rem; margin-bottom: 20px; }
  .row { display: flex; flex-wrap: wrap; gap: 18px; align-items: flex-start; }
  .panel { border: 1px solid rgba(128,128,128,.35); border-radius: 8px; padding: 12px;
           flex: 1 1 420px; min-width: 320px; }
  .ctl { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; margin-bottom: 14px; }
  label { font-size: .85rem; }
  select, input[type=range] { vertical-align: middle; }
  canvas { width: 100%; height: auto; display: block; image-rendering: pixelated; }
  .subj { display: inline-flex; align-items: center; gap: 4px; font-size: .8rem;
          margin-right: 8px; }
  .swatch { width: 11px; height: 11px; border-radius: 2px; display: inline-block; }
  .stat { font-variant-numeric: tabular-nums; font-size: .85rem; }
  .stat b { font-size: 1.05rem; }
  /* the 4-tap grid is 12 numeric columns wide, so it scrolls inside its own box rather than
     forcing the page to scroll sideways on a narrow screen */
  .tblwrap { overflow-x: auto; }
  table { border-collapse: collapse; font-size: .82rem; width: 100%; }
  th, td { text-align: right; padding: 3px 6px; border-bottom: 1px solid rgba(128,128,128,.2);
           white-space: nowrap; }
  th:first-child, td:first-child { text-align: left; }
  .selcol { background: rgba(128,128,160,.16); }
  th.gap, td.gap { border-bottom: none; padding: 0 5px; }
  .note { font-size: .8rem; opacity: .75; margin-top: 10px; }
</style>
<h1>Six brains, one trajectory</h1>
<div class="sub">Trial-averaged high-gamma responses from __NSESS__ BrainTreebank sessions
(__NSUBJ__ subjects), passed through a self-supervised encoder trained without any labels.
Each line is one session's <b>class-1 minus class-0</b> response tracing through a shared
PCA space. <b>Nothing fit here can rotate one subject onto another.</b> Each session gets a
per-feature calibration (2·C numbers, fit label-free over all its own windows, so it cannot
invent a class contrast — and in a contrast the mean term cancels outright), then one mean
subtraction and one scalar rescale. All of it is diagonal; rotation is what alignment would
need and none of these can do it. The 3-PC basis is shared, fit on all sessions pooled with
no subject labels. Two controls: <code>frame_brightness</code> stays at r&nbsp;≈&nbsp;0 in
this same basis at every depth, and the <b>LOSO</b> column refits the basis with <i>both</i>
subjects of each scored pair held out, so the number cannot be the basis's doing.
<br><b>Origin: __ORIGIN__.</b> __ORIGINWHY__
<br><span style="opacity:.75">Scope: electrodes are pooled to a lobe mean, so within-lobe
spatial structure is gone and this shows temporal, not spatial, correspondence. The shared
lobe is a single lobe for this cohort, and "the same lobe" is not the same tissue across
subjects with different coverage.</span></div>

<div class="ctl">
  <label>task <select id="task"></select></label>
  <label>depth <select id="tap"></select></label>
  <label>time <input id="t" type="range" min="0" value="0" style="width:200px"></label>
  <button id="play" title="play / pause">&#9654;</button>
  <label>speed <select id="speed"></select></label>
  <span class="stat" id="tlab"></span>
  <span class="note">t = 0 is the word onset</span>
</div>
<div class="ctl" id="subjects"></div>

<div class="row">
  <div class="panel">
    <canvas id="traj" width="900" height="700"></canvas>
    <div class="stat" id="score"></div>
    <div class="note">PC1 vs PC2 of one basis fit across every task at this depth, so no
    task gets a flattering projection. Amplitude is normalized per session (shared across
    tasks), because the score is a correlation and a correlation ignores scale.</div>
  </div>
  <div class="panel">
    <canvas id="rgb" width="900" height="700"></canvas>
    <div class="note">The DINOv3 view: each row is one region of one subject, time runs
    left to right, and colour is the first three PCs of the 256-d channel axis. Subjects do
    not share rows -- these brains have no anatomy in common beyond the temporal lobe -- but
    they share the colour basis, so matching colours mean matching representations.
    The dashed rule is the event at t=0; everything to its left is pre-stimulus. The solid
    rule is the player's current frame.</div>
  </div>
</div>

<div class="panel" style="margin-top:18px">
  <div class="tblwrap"><table id="tbl"></table></div>
  <div class="note"><b>r</b> is agreement of the plotted 3-PC trajectories;
  <b>LOSO</b> is the same number with the basis refit without either scored subject.
  <b>Retrieval</b> takes one subject's response at time t and picks the nearest of another
  subject's timepoints (chance 1/T) &mdash; basis-free, so it never touches the PCA.
  <b>Decoding AUROC</b> is the ridge readout from the board run on this same checkpoint,
  trained on other subjects and tested on held-out ones; CS cells are per-subject and this
  is their mean. <b>peak s</b> is when the contrast is largest and <b>settle</b> is where it
  ends up as a fraction of that peak &mdash; low means the response returns, high means it
  stays up.</div>
</div>

<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const taskSel = $("task"), tapSel = $("tap"), tSlider = $("t");
const playBtn = $("play"), speedSel = $("speed");
// 1x = real time, i.e. the window plays back in its own duration. Read off the time axis
// rather than passed in, so the 1 s and 2 s pages need no separate knob.
const HZ = 1 / (D.times[1] - D.times[0]);
let hidden = new Set();

for (const t of D.tasks) taskSel.add(new Option(t, t));
for (const t of D.taps) tapSel.add(new Option(t, t));
tapSel.value = D.taps[D.taps.length - 1];

$("subjects").innerHTML = D.sessions.map(s =>
  `<span class="subj"><input type="checkbox" checked data-k="${s.key}">
   <span class="swatch" style="background:${s.color}"></span>${s.key}</span>`).join("");
$("subjects").onchange = e => {
  const k = e.target.dataset.k;
  if (e.target.checked) hidden.delete(k); else hidden.add(k);
  draw();
};

function cur() { return D.data[tapSel.value][taskSel.value]; }

function drawTraj() {
  const c = $("traj"), g = c.getContext("2d");
  const W = c.width, H = c.height, pad = 46;
  g.clearRect(0, 0, W, H);
  const lim = D.lim[tapSel.value];
  const sx = v => pad + (v + lim) / (2 * lim) * (W - 2 * pad);
  const sy = v => H - pad - (v + lim) / (2 * lim) * (H - 2 * pad);
  g.strokeStyle = "rgba(128,128,128,.35)"; g.lineWidth = 1;
  g.beginPath(); g.moveTo(sx(-lim), sy(0)); g.lineTo(sx(lim), sy(0));
  g.moveTo(sx(0), sy(-lim)); g.lineTo(sx(0), sy(lim)); g.stroke();
  g.fillStyle = "gray"; g.font = "13px sans-serif";
  g.fillText("PC1", W - pad - 26, sy(0) - 8); g.fillText("PC2", sx(0) + 8, pad + 4);
  const ti = +tSlider.value;
  for (const s of D.sessions) {
    if (hidden.has(s.key)) continue;
    const p = cur().traj[s.key];
    if (!p) continue;
    g.strokeStyle = s.color; g.lineWidth = 2; g.globalAlpha = .85;
    g.beginPath();
    p.forEach((q, i) => i ? g.lineTo(sx(q[0]), sy(q[1])) : g.moveTo(sx(q[0]), sy(q[1])));
    g.stroke();
    g.globalAlpha = 1; g.fillStyle = s.color;
    g.beginPath(); g.arc(sx(p[ti][0]), sy(p[ti][1]), 6, 0, 7); g.fill();
  }
}

function drawRgb() {
  const c = $("rgb"), g = c.getContext("2d");
  const W = c.width, H = c.height;
  g.clearRect(0, 0, W, H);
  const vis = D.sessions.filter(s => !hidden.has(s.key) && cur().rgb[s.key]);
  if (!vis.length) return;
  const rows = vis.reduce((a, s) => a + cur().rgb[s.key].length, 0);
  const lab = 92, rh = Math.min(26, (H - 10) / Math.max(rows, 1));
  let y = 4;
  const T = D.nframes;
  const cw = (W - lab - 10) / T;
  g.font = "11px sans-serif";
  for (const s of vis) {
    const img = cur().rgb[s.key];
    for (let r = 0; r < img.length; r++) {
      for (let t = 0; t < T; t++) {
        const px = img[r][t];
        g.fillStyle = `rgb(${px[0]},${px[1]},${px[2]})`;
        g.fillRect(lab + t * cw, y, Math.ceil(cw), Math.ceil(rh));
      }
      g.fillStyle = "gray";
      g.fillText(`${s.key} ${D.lobes[s.key][r]}`.slice(0, 18), 2, y + rh * .75);
      y += rh;
    }
    y += 3;
  }
  // The event, fixed. Without it the strips are 2 s of colour with no anchor, and the eye
  // cannot tell a pre-stimulus band from a response -- which is the whole reading of the
  // panel once the origin is a pre-stimulus baseline. Dashed, so it never reads as the
  // draggable cursor; two-tone, because the strips are full-saturation RGB and a plain
  // white rule vanishes wherever a PC happens to be bright.
  if (D.t0 !== null) {
    const x0 = lab + D.t0 * cw;
    g.beginPath(); g.moveTo(x0, 0); g.lineTo(x0, y);
    g.setLineDash([]); g.strokeStyle = "rgba(0,0,0,.75)"; g.lineWidth = 2.5; g.stroke();
    g.setLineDash([5, 4]); g.strokeStyle = "white"; g.lineWidth = 1; g.stroke();
    g.setLineDash([]);
  }
  const ti = +tSlider.value;
  g.strokeStyle = "white"; g.lineWidth = 1.5;
  g.beginPath(); g.moveTo(lab + ti * cw, 0); g.lineTo(lab + ti * cw, y); g.stroke();
}

function drawTable() {
  // Every tap, not just the selected one. A single-tap table makes the depth ladder invisible
  // unless the reader clicks through four times and remembers the numbers; the whole claim
  // here is the TREND across taps, so the trend has to be on screen at once. The selected
  // tap's column is highlighted so the dropdown still tells you where you are.
  const taps = D.taps;
  const cell = (v, sel, dp) =>
    `<td${sel ? ' class="selcol"' : ""}>${v === null || v === undefined || !isFinite(v)
       ? "&ndash;" : (dp === 0 ? v : v.toFixed(dp))}</td>`;
  const rows = D.tasks.map(t => {
    const hl = t === taskSel.value ? ' style="font-weight:600"' : "";
    const a = taps.map(p => cell(D.data[p][t].align, p === tapSel.value, 2)).join("");
    const lo = taps.map(p => cell(D.data[p][t].loso, p === tapSel.value, 2)).join("");
    const r1 = taps.map(p => cell(D.retrieval[p][t].top1, p === tapSel.value, 3)).join("");
    const de = taps.map(p => cell(D.decode[p][t], p === tapSel.value, 4)).join("");
    const sh = D.data[tapSel.value][t].shape || {};
    return `<tr${hl}><td>${t}</td>${a}<td class="gap"></td>${lo}` +
           `<td class="gap"></td>${r1}<td class="gap"></td>${de}` +
           `<td class="gap"></td>` +
           cell(sh.peak_s, true, 2) + cell(sh.settle_frac, true, 2) + `</tr>`;
  }).join("");
  const head = taps.map(p =>
    `<th class="${p === tapSel.value ? "selcol" : ""}">${p}</th>`).join("");
  $("tbl").innerHTML =
    `<tr><th></th><th colspan="${taps.length}">cross-subject r (3-PC, pooled basis)</th>` +
    `<th class="gap"></th><th colspan="${taps.length}">same r, LOSO basis</th>` +
    `<th class="gap"></th><th colspan="${taps.length}">retrieval top-1` +
    ` (chance ${D.chance.toFixed(3)})</th>` +
    `<th class="gap"></th><th colspan="${taps.length}">cross-subject decoding AUROC` +
    ` (chance 0.5)</th>` +
    `<th class="gap"></th><th colspan="2">shape @ ${tapSel.value}</th></tr>` +
    `<tr><th>task</th>${head}<th class="gap"></th>${head}` +
    `<th class="gap"></th>${head}<th class="gap"></th>${head}` +
    `<th class="gap"></th>` +
    `<th class="selcol">peak s</th><th class="selcol">settle</th></tr>${rows}`;
}

function draw() {
  tSlider.max = D.nframes - 1;
  $("tlab").textContent = `${D.times[+tSlider.value].toFixed(2)} s`;
  const dec = D.decode[tapSel.value][taskSel.value];
  $("score").innerHTML = `cross-subject r = <b>${cur().align.toFixed(2)}</b>` +
    ` &nbsp;·&nbsp; retrieval top-1 = <b>` +
    `${D.retrieval[tapSel.value][taskSel.value].top1.toFixed(3)}</b>` +
    ` (chance ${D.chance.toFixed(3)})` +
    (dec == null ? "" :
      ` &nbsp;·&nbsp; cross-subject decoding AUROC = <b>${dec.toFixed(4)}</b>` +
      ` (chance 0.5)`);
  drawTraj(); drawRgb(); drawTable();
}
for (const v of [0.1, 0.25, 0.5, 1, 2]) speedSel.add(new Option(`${v}×`, v));
speedSel.value = "0.5";

// Accumulate elapsed time and redraw only when the frame index actually moves. At 0.1x a
// frame lasts ~300 ms, so redrawing per animation frame would be ~20 wasted canvas repaints
// per step.
let raf = null, last = 0, acc = 0;
function tick(ts) {
  if (last) {
    acc += (ts - last) / 1000 * HZ * (+speedSel.value);
    if (acc >= 1) {
      const adv = Math.floor(acc);
      acc -= adv;
      tSlider.value = (+tSlider.value + adv) % D.nframes;
      draw();
    }
  }
  last = ts;
  raf = requestAnimationFrame(tick);
}
function setPlay(on) {
  if (on === (raf !== null)) return;
  if (on) {
    last = 0; acc = 0;
    raf = requestAnimationFrame(tick);
    playBtn.innerHTML = "&#10074;&#10074;";
  } else {
    cancelAnimationFrame(raf); raf = null;
    playBtn.innerHTML = "&#9654;";
  }
}
playBtn.onclick = () => setPlay(raf === null);
taskSel.onchange = tapSel.onchange = draw;
tSlider.oninput = () => { setPlay(false); draw(); };
draw();
</script>
"""

COLORS = {1: "#e6194b", 2: "#3cb44b", 3: "#4363d8",
          4: "#f58231", 7: "#911eb4", 10: "#008080"}


def build(sessions, lobes, taps, tasks, hz: float, offset: float,
          *, n_pre: int | None = None, decode: dict | None = None) -> dict:
    data: dict = {}
    lim: dict = {}
    retr: dict = {}
    rgb_lobes: dict = {}
    n_frames = 0
    for tap in taps:
        loso = align_loso(sessions, lobes, tap, tasks, n_pre=n_pre)
        shape = peak_settle(sessions, lobes, tap, tasks, hz, offset, n_pre=n_pre)
        per_task = {t: collect(sessions, tap, t, CONTRAST, "all", lobes, centered=True,
                               n_pre=n_pre)
                    for t in tasks}
        per_task = {t: v for t, v in per_task.items() if v}
        # per-session scale shared across tasks -- identical to figure_tasks, so the page
        # and the figure cannot disagree
        scale: dict = {}
        for v in per_task.values():
            for s, m in v:
                scale[s.key] = scale.get(s.key, 0.0) + float((m ** 2).sum())
        scale = {k: float(np.sqrt(x)) for k, x in scale.items()}
        per_task = {t: [(s, m / scale[s.key]) for s, m in v if scale.get(s.key, 0) > 0]
                    for t, v in per_task.items()}
        # Loud, not skipped. A tap that silently vanishes takes the depth ladder with it,
        # and enc0 IS the control -- a page missing it looks like a result rather than a
        # missing shard.
        assert per_task, f"tap {tap} carries none of {list(tasks)}"
        stack = np.concatenate([m.reshape(-1, m.shape[-1])
                                for v in per_task.values() for _, m in v], axis=0)
        comps, mu, _ = pca_basis(stack, k=3)
        mu = _proj_origin(mu, n_pre)
        lim[tap] = float(max(np.abs(_traj(m, comps, mu)[:, :2]).max()
                             for v in per_task.values() for _, m in v)) * 1.05

        data[tap] = {}
        for t, v in per_task.items():
            traj = {s.key: _traj(m, comps, mu) for s, m in v}
            rs = [_corr(traj[v[a][0].key].ravel(), traj[v[b][0].key].ravel())
                  for a in range(len(v)) for b in range(a + 1, len(v))
                  if v[a][0].subject_id != v[b][0].subject_id]
            lo = loso.get(t, float("nan"))
            data[tap][t] = {
                "traj": {k: np.round(p[:, :2], 4).tolist() for k, p in traj.items()},
                "align": float(np.nanmean(rs)) if rs else 0.0,
                # JSON has no NaN; the page renders null as an em dash rather than "NaN"
                "loso": None if not np.isfinite(lo) else round(float(lo), 4),
                "shape": shape.get(t, {}),
                "rgb": _rgb_panels(sessions, tap, t, rgb_lobes, n_pre=n_pre),
            }
            n_frames = next(iter(traj.values())).shape[0]
        retr[tap] = {}
        for t in tasks:
            r = retrieval(sessions, lobes, tap, t, n_pre=n_pre)
            retr[tap][t] = {"top1": round(r.get("top1", 0.0), 4),
                            "median_rank": r.get("median_rank", 0)}
    # Advertise only tasks that produced a trajectory at EVERY tap. Asking for a task the
    # reduction does not carry used to leave it in the dropdown with no data behind it, and
    # selecting it threw on `cur().align` -- a blank page for what is really a missing shard.
    present = [t for t in tasks if all(t in data[tap] for tap in data)]
    assert present, f"none of {list(tasks)} survived at every tap"
    i0 = round(-offset * hz)
    t0 = int(i0) if 0 < i0 < n_frames else None
    return {
        "tasks": present, "taps": list(taps), "nframes": int(n_frames),
        "times": [round(offset + i / hz, 4) for i in range(n_frames)],
        "chance": 1.0 / max(n_frames, 1), "n_pre": n_pre or 0,
        # The event itself, as a frame index. None unless the window actually LEADS it:
        # at offset=0 the event is the left edge and a rule on the border marks nothing.
        "t0": t0,
        "lim": lim, "data": data, "retrieval": retr, "lobes": rgb_lobes,
        # cross-subject DECODING accuracy, from the board run on this same checkpoint.
        # The geometry and the accuracy are different claims; showing only the first
        # invites "pretty, but does it decode?" and the answer is already computed.
        "decode": {tap: {t: (decode or {}).get(t, {}).get(tap) for t in present}
                   for tap in taps},
        "sessions": [{"key": s.key, "color": COLORS.get(s.subject_id, "#888")}
                     for s in sessions],
    }


def _rgb_panels(sessions, tap: str, task: str, rgb_lobes: dict,
                *, n_pre: int | None = None) -> dict:
    """PC-RGB per session, one shared stretch across every panel (the DINOv3 recipe)."""
    from scripts.neuroprobe.viz_common import center_per_session
    from scripts.neuroprobe.viz_figures import _cond_matrix
    per = []
    for s in sessions:
        own = sorted({lb for lb in s.lobes if lb != "unknown"})
        m = _cond_matrix(s, tap, task, CONTRAST, "all", own)
        if m is not None:
            per.append((s, own, m))
            rgb_lobes[s.key] = own
    if not per:
        return {}
    cen = center_per_session([m for _, _, m in per], n_pre=n_pre)
    stack = np.concatenate([m.reshape(-1, m.shape[-1]) for m in cen], axis=0)
    comps, mu, _ = pca_basis(stack, k=3)
    proj = [((m.reshape(-1, m.shape[-1]) - mu) @ comps.T).reshape(m.shape[0], m.shape[1], 3)
            for m in cen]
    flat = to_rgb(np.concatenate([p.reshape(-1, 3) for p in proj], axis=0))
    out, off = {}, 0
    for (s, _, _), p in zip(per, proj):
        n = p.shape[0] * p.shape[1]
        img = (flat[off:off + n].reshape(p.shape) * 255).astype(np.uint8)
        out[s.key] = img.tolist()
        off += n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--red-dir", required=True)
    ap.add_argument("--out", default="results/viz_crosssubject/demo.html")
    ap.add_argument("--taps", default="enc0,enc3,enc6,enc12")
    ap.add_argument("--tasks", default="onset,speech,delta_volume,word_index,"
                                       "word_part_speech,frame_brightness")
    ap.add_argument("--hz", type=float, default=32.0)
    # Required, not defaulted: it is only a time-axis label, so a wrong value mislabels every
    # frame and nothing crashes. The 2 s page shipped with the 1 s default and read +0.5 s off.
    ap.add_argument("--offset", type=float, required=True,
                    help="seconds of the window before the event (negative if it leads)")
    # Baseline reference; see viz_common.center_per_session. 0 keeps the window time-average
    # as the origin, which is all the 1 s window can do. At 2 s pass 16 (0.5 s x 32 Hz).
    ap.add_argument("--n-pre", type=int, default=0,
                    help="pre-stimulus frames to baseline against; 0 = time-mean origin")
    # Board results for the SAME checkpoint. Optional, because the page still makes its
    # point without it, but a mismatched run would be worse than none -- hence explicit.
    ap.add_argument("--board-json", default=None,
                    help="board results JSON; adds cross-subject decoding AUROC")
    args = ap.parse_args()
    n_pre = args.n_pre or None
    assert not n_pre or args.offset < 0, \
        f"--n-pre {args.n_pre} needs a window that leads onset, got --offset {args.offset}"

    sessions = load_all(args.red_dir)
    taps = [t for t in args.taps.split(",") if t and any(t in s.shapes for s in sessions)]
    tasks = [t for t in args.tasks.split(",") if t]
    lobes = shared_lobes(sessions)
    assert lobes, "no lobe shared by every subject"
    decode = board_cs_auroc(args.board_json) if args.board_json else None
    payload = build(sessions, lobes, taps, tasks, args.hz, args.offset, n_pre=n_pre,
                    decode=decode)
    subj = len({s.subject_id for s in sessions})
    if n_pre:
        origin = (f"the {args.n_pre} pre-stimulus frames "
                  f"({args.offset:+.2f} to {args.offset + args.n_pre / args.hz:+.2f} s)")
        why = ("Zero means <i>no class difference before the word</i>, so a response that "
               "rises and stays up is drawn as failing to return — which is exactly how "
               "<code>onset</code> (a word after silence, over by the end) differs from "
               "<code>speech</code> (a word inside ongoing talk, the difference persists). "
               "For <code>speech</code> the classes already differ before t=0, so this "
               "re-references to the state at word onset rather than removing that offset.")
    else:
        origin = "the window's own time-average"
        why = ("This window has no pre-stimulus frames, so the origin is the average of the "
               "response itself. Sustained and transient contrasts therefore both read as "
               "closed loops; use the 2 s page to tell them apart.")
    html = (HTML.replace("__DATA__", json.dumps(payload))
            .replace("__NSESS__", str(len(sessions)))
            .replace("__ORIGINWHY__", why)
            .replace("__ORIGIN__", origin)
            .replace("__NSUBJ__", str(subj)))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(html)
    print(f"[write] {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB), "
          f"taps {taps}, {len(sessions)} sessions, lobes {lobes}")


if __name__ == "__main__":
    main()
