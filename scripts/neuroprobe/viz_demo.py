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

from scripts.neuroprobe.viz_common import load_all, pca_basis, shared_lobes, to_rgb
from scripts.neuroprobe.viz_figures import (
    CONTRAST, _corr, _traj, collect, retrieval,
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
  table { border-collapse: collapse; font-size: .82rem; width: 100%; }
  th, td { text-align: right; padding: 3px 8px; border-bottom: 1px solid rgba(128,128,128,.2); }
  th:first-child, td:first-child { text-align: left; }
  .note { font-size: .8rem; opacity: .75; margin-top: 10px; }
</style>
<h1>Six brains, one trajectory</h1>
<div class="sub">Trial-averaged high-gamma responses from __NSESS__ BrainTreebank sessions
(__NSUBJ__ subjects), passed through a self-supervised encoder trained without any labels.
Each line is one session's <b>class-1 minus class-0</b> response tracing through a shared
PCA space. Nothing here is fit to align subjects.</div>

<div class="ctl">
  <label>task <select id="task"></select></label>
  <label>depth <select id="tap"></select></label>
  <label>time <input id="t" type="range" min="0" value="0" style="width:200px"></label>
  <span class="stat" id="tlab"></span>
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
    they share the colour basis, so matching colours mean matching representations.</div>
  </div>
</div>

<div class="panel" style="margin-top:18px">
  <table id="tbl"></table>
  <div class="note">Retrieval: take one subject's response at time t, and among another
  subject's timepoints pick the nearest. Chance is 1/T. This is the cross-subject claim in
  its most direct form.</div>
</div>

<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const taskSel = $("task"), tapSel = $("tap"), tSlider = $("t");
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
  const ti = +tSlider.value;
  g.strokeStyle = "white"; g.lineWidth = 1.5;
  g.beginPath(); g.moveTo(lab + ti * cw, 0); g.lineTo(lab + ti * cw, y); g.stroke();
}

function drawTable() {
  const rows = D.tasks.map(t => {
    const r = D.retrieval[tapSel.value][t];
    const q = D.data[tapSel.value][t].align;
    const hl = t === taskSel.value ? ' style="font-weight:600"' : "";
    return `<tr${hl}><td>${t}</td><td>${q.toFixed(2)}</td>` +
           `<td>${r.top1.toFixed(3)}</td><td>${r.median_rank}</td></tr>`;
  }).join("");
  $("tbl").innerHTML =
    `<tr><th>task</th><th>cross-subject r</th><th>retrieval top-1` +
    ` (chance ${D.chance.toFixed(3)})</th><th>median rank</th></tr>${rows}`;
}

function draw() {
  tSlider.max = D.nframes - 1;
  $("tlab").textContent = `${D.times[+tSlider.value].toFixed(2)} s`;
  $("score").innerHTML = `cross-subject r = <b>${cur().align.toFixed(2)}</b>` +
    ` &nbsp;·&nbsp; retrieval top-1 = <b>` +
    `${D.retrieval[tapSel.value][taskSel.value].top1.toFixed(3)}</b>` +
    ` (chance ${D.chance.toFixed(3)})`;
  drawTraj(); drawRgb(); drawTable();
}
taskSel.onchange = tapSel.onchange = tSlider.oninput = draw;
draw();
</script>
"""

COLORS = {1: "#e6194b", 2: "#3cb44b", 3: "#4363d8",
          4: "#f58231", 7: "#911eb4", 10: "#008080"}


def build(sessions, lobes, taps, tasks, hz: float, offset: float) -> dict:
    data: dict = {}
    lim: dict = {}
    retr: dict = {}
    rgb_lobes: dict = {}
    n_frames = 0
    for tap in taps:
        per_task = {t: collect(sessions, tap, t, CONTRAST, "all", lobes, centered=True)
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
        stack = np.concatenate([m.reshape(-1, m.shape[-1])
                                for v in per_task.values() for _, m in v], axis=0)
        comps, mu, _ = pca_basis(stack, k=3)
        lim[tap] = float(max(np.abs(_traj(m, comps, mu)[:, :2]).max()
                             for v in per_task.values() for _, m in v)) * 1.05

        data[tap] = {}
        for t, v in per_task.items():
            traj = {s.key: _traj(m, comps, mu) for s, m in v}
            rs = [_corr(traj[v[a][0].key].ravel(), traj[v[b][0].key].ravel())
                  for a in range(len(v)) for b in range(a + 1, len(v))
                  if v[a][0].subject_id != v[b][0].subject_id]
            data[tap][t] = {
                "traj": {k: np.round(p[:, :2], 4).tolist() for k, p in traj.items()},
                "align": float(np.nanmean(rs)) if rs else 0.0,
                "rgb": _rgb_panels(sessions, tap, t, rgb_lobes),
            }
            n_frames = next(iter(traj.values())).shape[0]
        retr[tap] = {}
        for t in tasks:
            r = retrieval(sessions, lobes, tap, t)
            retr[tap][t] = {"top1": round(r.get("top1", 0.0), 4),
                            "median_rank": r.get("median_rank", 0)}
    return {
        "tasks": list(tasks), "taps": list(taps), "nframes": int(n_frames),
        "times": [round(offset + i / hz, 4) for i in range(n_frames)],
        "chance": 1.0 / max(n_frames, 1),
        "lim": lim, "data": data, "retrieval": retr, "lobes": rgb_lobes,
        "sessions": [{"key": s.key, "color": COLORS.get(s.subject_id, "#888")}
                     for s in sessions],
    }


def _rgb_panels(sessions, tap: str, task: str, rgb_lobes: dict) -> dict:
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
    cen = center_per_session([m for _, _, m in per])
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
    ap.add_argument("--offset", type=float, default=0.0,
                    help="seconds of the window before the event (negative if it leads)")
    args = ap.parse_args()

    sessions = load_all(args.red_dir)
    taps = [t for t in args.taps.split(",") if t and any(t in s.shapes for s in sessions)]
    tasks = [t for t in args.tasks.split(",") if t]
    lobes = shared_lobes(sessions)
    assert lobes, "no lobe shared by every subject"
    payload = build(sessions, lobes, taps, tasks, args.hz, args.offset)
    subj = len({s.subject_id for s in sessions})
    html = (HTML.replace("__DATA__", json.dumps(payload))
            .replace("__NSESS__", str(len(sessions)))
            .replace("__NSUBJ__", str(subj)))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(html)
    print(f"[write] {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB), "
          f"taps {taps}, {len(sessions)} sessions, lobes {lobes}")


if __name__ == "__main__":
    main()
