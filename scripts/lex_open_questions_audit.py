"""Lexical open-questions audit (Q2, Q3, Q5, Q8).

Q2: S78 phoneme coverage — read phoneme codes straight from the .fif event_id;
    check whether /æ/ (AE) and /ʊ/ (UH) are present in S78 vs other lexical pts.
Q3: S78/S81 older-filename load — confirm the .fif loads via mne and through
    the v14 loader. The prior memory note about "older entity convention" refers
    to raw ieeg; derivatives .fif names are consistent across all 16.
Q5: Stim→response delay distribution per patient from events.tsv.
Q8: Zac's CCA decoder .h5 shapes (decoder weights, channel/time dim, labels).

Writes reports/lex_open_questions_2026_04_18/*.json
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd

mne.set_log_level("ERROR")

BIDS = Path(
    "/Users/bentang/Documents/Code/speech/BIDS_1.0_Lexical_µECoG"
    "/BIDS_1.0_Lexical_µECoG/BIDS"
)
OUT = Path("/Users/bentang/Documents/Code/speech/reports/lex_open_questions_2026_04_18")
OUT.mkdir(parents=True, exist_ok=True)

PATIENTS_FULL = [
    "S41", "S45", "S47", "S51", "S53", "S55", "S56", "S63",
    "S67", "S71", "S73", "S74", "S75", "S76", "S78", "S81",
]


def ieeg_events_tsv(pt: str) -> Path | None:
    """Prefer whichever candidate has stimulus/response-style trial_type."""
    base = BIDS / f"sub-{pt}/ieeg"
    candidates = [
        base / f"sub-{pt}_task-lexical_events.tsv",
        base / f"sub-{pt}_task-lexical_acq-01_run-01_events.tsv",
    ]
    # Take the one whose trial_type column contains stimulus/response labels
    for p in candidates:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, sep="\t", nrows=4)
            tt = df["trial_type"].astype(str)
            if any(x.startswith(("stim", "resp")) for x in tt.tolist()):
                return p
        except Exception:  # noqa: BLE001
            continue
    # Fall back to first existing
    for p in candidates:
        if p.exists():
            return p
    return None


def phonemelevel_fif(pt: str, desc: str) -> Path:
    return (
        BIDS / f"derivatives/epoch(phonemeLevel)(CAR)/sub-{pt}/epoch(band)(power)"
        / f"sub-{pt}_task-lexical_desc-{desc}_highgamma.fif"
    )


# ---------- Q5 — stim→resp delay ----------
def q5_delays() -> dict:
    out = {}
    for pt in PATIENTS_FULL:
        tsv = ieeg_events_tsv(pt)
        if tsv is None:
            out[pt] = {"error": "events.tsv missing"}
            continue
        df = pd.read_csv(tsv, sep="\t")
        df.columns = [c.lstrip("\ufeff") for c in df.columns]
        if "trial" not in df.columns:
            out[pt] = {"error": "events.tsv missing 'trial' column (legacy schema)",
                       "tsv": str(tsv)}
            continue
        # trial_type schema varies: "stimulus/word/low" (new) vs "stimulus" (old)
        tt = df["trial_type"].astype(str)
        is_stim = tt.str.startswith("stimulus")
        is_resp = tt.str.startswith("response")
        stim = df.loc[is_stim].sort_values("trial").set_index("trial")["onset"]
        resp = df.loc[is_resp].sort_values("trial").set_index("trial")["onset"]
        common = stim.index.intersection(resp.index)
        delays = (resp.loc[common].values - stim.loc[common].values).astype(float)
        out[pt] = {
            "n_trials": int(len(df) // 2),
            "n_stim": int(is_stim.sum()),
            "n_resp": int(is_resp.sum()),
            "delay_median_s": float(np.median(delays)) if len(delays) else None,
            "delay_mean_s": float(np.mean(delays)) if len(delays) else None,
            "delay_std_s": float(np.std(delays)) if len(delays) else None,
            "delay_p05_s": float(np.percentile(delays, 5)) if len(delays) else None,
            "delay_p95_s": float(np.percentile(delays, 95)) if len(delays) else None,
            "schema_variant": (
                "stim+mod" if any("/" in x for x in tt.head(2).tolist()) else "stim-only"
            ),
        }
    return out


# ---------- Q2/Q3 — phoneme coverage + .fif load sanity ----------
def q2q3_phoneme_inventory() -> dict:
    out = {}
    for pt in PATIENTS_FULL:
        fif = phonemelevel_fif(pt, "productionZscore")
        rec = {"fif_exists": fif.exists(), "fif_path": str(fif)}
        if not fif.exists():
            out[pt] = rec
            continue
        try:
            ep = mne.read_epochs(str(fif), preload=False, verbose="ERROR")
        except Exception as e:  # noqa: BLE001
            rec["load_error"] = repr(e)
            out[pt] = rec
            continue
        event_ids = dict(ep.event_id)
        codes = ep.events[:, 2]
        # Count per-event occurrences
        counts = {k: int((codes == v).sum()) for k, v in event_ids.items()}
        rec.update(
            n_phon=int(len(ep)),
            n_ch=int(len(ep.ch_names)),
            sfreq=float(ep.info["sfreq"]),
            tmin=float(ep.tmin),
            tmax=float(ep.tmax),
            n_unique_phonemes=len(event_ids),
            phonemes_sorted=sorted(event_ids.keys()),
            counts=counts,
        )
        out[pt] = rec
    return out


# ---------- Q8 — Zac's CCA decoder .h5 ----------
def _describe_h5(p: Path) -> dict:
    info: dict = {"path": str(p), "size_bytes": p.stat().st_size}
    groups = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            groups.append({
                "name": name,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
            })

    with h5py.File(p, "r") as h:
        h.visititems(visit)
        info["attrs"] = {k: (v.tolist() if hasattr(v, "tolist") else str(v))
                         for k, v in h.attrs.items()}
    info["datasets"] = groups
    return info


def q8_cca_decoders() -> dict:
    out = {}
    for pt in ["S41", "S73", "S76"]:
        root = BIDS / f"derivatives/decoding/sub-{pt}"
        if not root.exists():
            out[pt] = {"error": f"no decoder dir for {pt}"}
            continue
        files = sorted(root.rglob("*.h5"))
        out[pt] = {
            "n_files": len(files),
            "files": [_describe_h5(f) for f in files[:4]],  # first few
        }
    return out


def main() -> None:
    report = {
        "q2q3_phoneme_inventory": q2q3_phoneme_inventory(),
        "q5_stim_response_delay": q5_delays(),
        "q8_cca_decoders": q8_cca_decoders(),
    }
    (OUT / "audit.json").write_text(json.dumps(report, indent=2))
    # Summary print
    print("=== Q2/Q3 phoneme inventory ===")
    for pt, r in report["q2q3_phoneme_inventory"].items():
        if r.get("fif_exists"):
            print(f"  {pt}: n_phon={r['n_phon']} n_ch={r['n_ch']} "
                  f"unique_phonemes={r['n_unique_phonemes']} "
                  f"sfreq={r['sfreq']:.1f}")
        else:
            print(f"  {pt}: .fif MISSING")
    # Phoneme-set comparison: does S78 share the same set?
    inv = report["q2q3_phoneme_inventory"]
    base = set(inv["S41"]["phonemes_sorted"]) if inv["S41"].get("phonemes_sorted") else set()
    for pt in PATIENTS_FULL:
        r = inv[pt]
        if not r.get("phonemes_sorted"):
            continue
        s = set(r["phonemes_sorted"])
        missing = sorted(base - s)
        extra = sorted(s - base)
        if missing or extra:
            print(f"  {pt} vs S41: missing={missing} extra={extra}")
    print("\n=== Q5 stim→response delay (s) ===")
    for pt, r in report["q5_stim_response_delay"].items():
        if r.get("delay_median_s") is not None:
            print(f"  {pt}: n={r['n_stim']} median={r['delay_median_s']:.3f} "
                  f"mean={r['delay_mean_s']:.3f} std={r['delay_std_s']:.3f} "
                  f"p5={r['delay_p05_s']:.3f} p95={r['delay_p95_s']:.3f} "
                  f"schema={r['schema_variant']}")
        else:
            print(f"  {pt}: {r.get('error', 'no delays')}")
    print("\n=== Q8 Zac CCA decoders ===")
    for pt, r in report["q8_cca_decoders"].items():
        print(f"  {pt}: {r.get('n_files', 0)} files")
        for f in r.get("files", []):
            print(f"    {Path(f['path']).parent.name}/{Path(f['path']).name} "
                  f"({f['size_bytes']:,}b)")
            for d in f["datasets"][:5]:
                print(f"       {d['name']}  {d['shape']}  {d['dtype']}")
    print(f"\nwrote {OUT/'audit.json'}")


if __name__ == "__main__":
    main()
