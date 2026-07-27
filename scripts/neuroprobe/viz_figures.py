"""Cross-subject encoder figures from the reduced condition means.

Figure A  trajectory of the trial-averaged response through a SHARED 3-PC space, one line
          per session. Two panels on purpose: without per-session centering (identity is
          the dominant structure and subjects sit apart) and with it (do they MOVE the same
          way). Those are different questions and the honest answer needs both.

Figure B  the DINOv3 panel. Tokens are (region x time), PCA runs over the CHANNEL axis, the
          first 3 components are painted as RGB. Panels are NOT row-matched across subjects
          -- the Lite cohort shares one lobe, so a matched grid does not exist -- and they
          do not need to be: DINOv3 paints different images through one basis and lets
          corresponding parts land on the same colour. Same construction here.

Quant     cross-subject similarity against the within-subject split-half ceiling, for every
          tap. The ceiling is the point: a raw correlation means nothing without knowing
          what the same subject's own two halves score.

enc0 is the control throughout. It is the untrained |STFT| frontend, gets byte-identical
treatment, and is constant across checkpoints -- so any enc12-minus-enc0 gap is the encoder
and not the pipeline.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from scripts.neuroprobe.viz_common import (  # noqa: E402
    center_per_session, load_all, pca_basis, session_matrix, shared_lobes, to_rgb,
)

SUBJ_COLORS = {1: "#e6194b", 2: "#3cb44b", 3: "#4363d8",
               4: "#f58231", 7: "#911eb4", 10: "#008080"}


def _flat(m: np.ndarray) -> np.ndarray:
    return m.reshape(-1)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else np.nan


def collect(sessions, tap: str, task: str, cls: int, half: str, lobes, *, centered: bool):
    """(session, matrix) pairs for one condition, optionally identity-centered."""
    out = []
    for s in sessions:
        m = session_matrix(s, tap, task, cls, half, lobes)
        if m is not None:
            out.append((s, m))
    if centered and out:
        mats = center_per_session([m for _, m in out])
        out = [(s, m) for (s, _), m in zip(out, mats)]
    return out


def figure_a(sessions, lobes, tap: str, task: str, out_path: str) -> dict:
    """3-D trajectory in a shared PC space, uncentered vs identity-centered."""
    fig = plt.figure(figsize=(13, 6))
    info = {}
    for col, centered in enumerate((False, True)):
        pairs = {c: collect(sessions, tap, task, c, "all", lobes, centered=centered)
                 for c in (0, 1)}
        stack = np.concatenate([m.reshape(-1, m.shape[-1])
                                for c in (0, 1) for _, m in pairs[c]], axis=0)
        comps, mu, evr = pca_basis(stack, k=3)
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        for c in (0, 1):
            for s, m in pairs[c]:
                p = (m.reshape(-1, m.shape[-1]) - mu) @ comps.T
                p = p.reshape(m.shape[0], m.shape[1], 3).mean(axis=0)  # average lobes -> (T,3)
                ax.plot(p[:, 0], p[:, 1], p[:, 2],
                        color=SUBJ_COLORS.get(s.subject_id, "#888"),
                        ls="-" if c == 1 else ":", lw=1.6, alpha=0.9)
                ax.scatter(*p[0], color=SUBJ_COLORS.get(s.subject_id, "#888"),
                           s=22, marker="o")
        title = "identity-centered" if centered else "raw (identity included)"
        ax.set_title(f"{tap} · {task} · {title}\nEVR {evr.sum():.2f}", fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        info["evr_centered" if centered else "evr_raw"] = [float(v) for v in evr]
    handles = [Line2D([], [], color=c, lw=2, label=f"S{s}")
               for s, c in SUBJ_COLORS.items()]
    handles += [Line2D([], [], color="k", ls="-", label=f"{task}=1"),
                Line2D([], [], color="k", ls=":", label=f"{task}=0")]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=8)
    fig.suptitle(f"Cross-subject trajectory · lobes={lobes} · circle = window start", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return info


def figure_b(sessions, tap: str, task: str, cls: int, out_path: str) -> dict:
    """Per-subject PC-RGB panels: rows are that subject's own lobes, columns are time."""
    per = []
    for s in sessions:
        lobes = sorted({lb for lb in s.lobes if lb != "unknown"})
        m = session_matrix(s, tap, task, cls, "all", lobes)
        if m is not None:
            per.append((s, lobes, m))
    assert per, "no session produced a panel"
    centered = center_per_session([m for _, _, m in per])
    stack = np.concatenate([m.reshape(-1, m.shape[-1]) for m in centered], axis=0)
    comps, mu, evr = pca_basis(stack, k=3)
    proj = [((m.reshape(-1, m.shape[-1]) - mu) @ comps.T).reshape(m.shape[0], m.shape[1], 3)
            for m in centered]
    rgb = to_rgb(np.concatenate([p.reshape(-1, 3) for p in proj], axis=0))
    off, rgbs = 0, []
    for p in proj:
        n = p.shape[0] * p.shape[1]
        rgbs.append(rgb[off:off + n].reshape(p.shape))
        off += n

    n = len(per)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.5 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, ((s, lobes, _), img) in enumerate(zip(per, rgbs)):
        ax = axes[i // ncol][i % ncol]
        ax.axis("on")
        ax.imshow(img, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(lobes)))
        ax.set_yticklabels(lobes, fontsize=6)
        ax.set_xlabel("time →", fontsize=7)
        ax.set_title(s.key, fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle(f"PC-RGB · {tap} · {task}={cls} · one shared 3-PC channel basis "
                 f"(EVR {evr.sum():.2f}) · rows are each subject's own lobes", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return {"evr": [float(v) for v in evr], "n_panels": n}


def quantify(sessions, lobes, tap: str, task: str, cls: int) -> dict:
    """Cross-subject similarity vs the within-session split-half ceiling."""
    grand = collect(sessions, tap, task, cls, "all", lobes, centered=True)
    h0 = {s.key: m for s, m in collect(sessions, tap, task, cls, "h0", lobes, centered=True)}
    h1 = {s.key: m for s, m in collect(sessions, tap, task, cls, "h1", lobes, centered=True)}

    ceiling = [_corr(_flat(h0[s.key]), _flat(h1[s.key]))
               for s, _ in grand if s.key in h0 and s.key in h1]
    cross, within_subj = [], []
    for i in range(len(grand)):
        for j in range(i + 1, len(grand)):
            si, mi = grand[i]
            sj, mj = grand[j]
            r = _corr(_flat(mi), _flat(mj))
            (within_subj if si.subject_id == sj.subject_id else cross).append(r)

    def _m(v):
        return float(np.nanmean(v)) if v else float("nan")

    c, cl = _m(cross), _m(ceiling)
    return {
        "tap": tap, "task": task, "class": cls, "n_sessions": len(grand),
        "cross_subject_r": c, "within_subject_diff_session_r": _m(within_subj),
        "split_half_ceiling_r": cl,
        "normalized": float(c / cl) if cl and np.isfinite(cl) and cl > 0 else float("nan"),
        "n_cross_pairs": len(cross),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--red-dir", required=True)
    ap.add_argument("--out-dir", default="results/viz_crosssubject")
    ap.add_argument("--task", default="onset")
    ap.add_argument("--taps", default="enc0,enc3,enc6,enc12")
    ap.add_argument("--tasks-quant", default="onset,speech,delta_volume,word_index,"
                                             "word_part_speech,frame_brightness")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sessions = load_all(args.red_dir)
    taps = [t for t in args.taps.split(",") if t]
    taps = [t for t in taps if any(t in s.shapes for s in sessions)]
    lobes = shared_lobes(sessions)
    print(f"[load] {len(sessions)} sessions, subjects "
          f"{sorted({s.subject_id for s in sessions})}, taps {taps}")
    print(f"[check] shared lobes across ALL subjects: {lobes}")
    assert lobes, "no lobe is shared by every subject — Figure A has no common axis"

    report: dict = {"sessions": [s.key for s in sessions], "shared_lobes": lobes,
                    "taps": taps, "figures": {}, "quant": []}

    for tap in taps:
        p = os.path.join(args.out_dir, f"figA_trajectory_{tap}_{args.task}.png")
        report["figures"][f"A/{tap}"] = figure_a(sessions, lobes, tap, args.task, p)
        print(f"[fig] {p}")
        p = os.path.join(args.out_dir, f"figB_pcrgb_{tap}_{args.task}.png")
        report["figures"][f"B/{tap}"] = figure_b(sessions, tap, args.task, 1, p)
        print(f"[fig] {p}")

    for task in [t for t in args.tasks_quant.split(",") if t]:
        for tap in taps:
            for cls in (0, 1):
                try:
                    q = quantify(sessions, lobes, tap, task, cls)
                except (KeyError, AssertionError):
                    continue
                report["quant"].append(q)
                print(f"[quant] {tap:6s} {task:16s} c{cls}  cross={q['cross_subject_r']:+.4f} "
                      f"ceiling={q['split_half_ceiling_r']:+.4f} "
                      f"norm={q['normalized']:+.4f}")

    dst = os.path.join(args.out_dir, "report.json")
    with open(dst, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[write] {dst}")


if __name__ == "__main__":
    main()
