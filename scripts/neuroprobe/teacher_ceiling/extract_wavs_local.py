"""Extract 16 kHz mono WAV from each BT movie rip (laptop-side, one-time).

Output schema:
    <out-dir>/<slug>.wav   (16 kHz mono PCM16)

The Whisper teacher extraction on DCC reads these by slug. Slugs follow the
BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING vocabulary; the rip-to-slug map
mirrors `reports/bt_alignment/cut_coverage.json` + per-film YTS/RAR/AV1
filenames.

Usage:
    .venv/bin/python scripts/neuroprobe/teacher_ceiling/extract_wavs_local.py \\
        --movies-dir movies \\
        --out-dir audio/bt_16k
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


# Slug -> first matching glob pattern under movies/.
# Patterns are forgiving; first hit wins. If your local rip names diverge,
# pass --override-map <slug>=<path>.
SLUG_TO_PATTERN = {
    "ant-man":                  "Ant-Man*",
    "aquaman":                  "Aquaman*",
    "avengers-infinity-war":    "Avengers*Infinity*",
    "black-panther":            "Black*Panther*",
    "cars-2":                   "Cars*2*",
    "coraline":                 "Coraline*",
    "fantastic-mr-fox":         "Fantastic*Mr*Fox*",
    "guardians-of-the-galaxy":  "Guardians*Galaxy*2014*",
    "guardians-of-the-galaxy-2": "Guardians*Galaxy*Vol*2*",
    "incredibles":              "*Incredibles*",
    "lotr-1":                   "*Fellowship*",
    "lotr-2":                   "*Two*Towers*",
    "megamind":                 "Megamind*",
    "shrek-the-third":          "Shrek*Third*",
    "spider-man-3-homecoming":  "Spider-Man*Homecoming*",
    "spider-man-far-from-home": "Spider-Man*Far*From*Home*",
    "the-martian":              "*Martian*",
    "thor-ragnarok":            "Thor*Ragnarok*",
    "toy-story":                "Toy*Story*",
    "venom":                    "Venom*",
}


def find_rip(movies_dir: Path, pattern: str) -> Path | None:
    matches = []
    for p in movies_dir.iterdir():
        if p.is_dir():
            sub = list(p.glob("*.mkv")) + list(p.glob("*.mp4")) + list(p.glob("*.avi"))
            matches.extend(sub)
        elif p.is_file():
            if any(p.name.lower().endswith(ext) for ext in (".mkv", ".mp4", ".avi")):
                matches.append(p)
    # Skip obvious samples/trailers/extras
    matches = [m for m in matches if "sample" not in m.name.lower()
               and "trailer" not in m.name.lower()
               and m.stat().st_size > 50 * 1024 * 1024]  # >50MB
    pat_lower = pattern.lower().replace("*", " ").split()
    candidates = []
    for m in matches:
        name_lower = str(m.parent.name).lower() + " " + m.name.lower()
        if all(token in name_lower for token in pat_lower):
            candidates.append(m)
    if not candidates:
        return None
    # Prefer the largest file (full-length feature, not a sample/trailer)
    return max(candidates, key=lambda p: p.stat().st_size)


def extract_one(slug: str, rip: Path, out_dir: Path, force: bool = False) -> dict:
    out = out_dir / f"{slug}.wav"
    if out.exists() and not force:
        return {"slug": slug, "rip": str(rip), "out": str(out), "status": "exists"}
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(rip),
        "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        "-loglevel", "warning", str(out),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return {"slug": slug, "rip": str(rip), "status": "failed",
                "stderr_tail": (res.stderr or "")[-500:]}
    return {"slug": slug, "rip": str(rip), "out": str(out), "status": "ok",
            "size_mb": round(out.stat().st_size / 1e6, 2)}


def main() -> None:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not on PATH")

    overrides = {}
    if args.override_map:
        for entry in args.override_map:
            slug, _, path = entry.partition("=")
            overrides[slug] = Path(path)

    results = []
    for slug, pat in SLUG_TO_PATTERN.items():
        if slug in overrides:
            rip = overrides[slug]
        else:
            rip = find_rip(args.movies_dir, pat)
        if rip is None or not rip.exists():
            results.append({"slug": slug, "status": "rip-not-found"})
            print(f"[wav] {slug}: NOT FOUND", flush=True)
            continue
        r = extract_one(slug, rip, args.out_dir, force=args.force)
        results.append(r)
        print(f"[wav] {slug}: {r['status']} ({r.get('size_mb', 'n/a')} MB)", flush=True)

    n_ok = sum(1 for r in results if r["status"] in ("ok", "exists"))
    print(f"\n[wav] {n_ok}/{len(SLUG_TO_PATTERN)} WAVs ready in {args.out_dir}", flush=True)
    import json
    (args.out_dir / "_extract_log.json").write_text(json.dumps(results, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--movies-dir", type=Path, default=Path("movies"))
    p.add_argument("--out-dir", type=Path, default=Path("audio/bt_16k"))
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--override-map", action="append", default=None,
        help="slug=path overrides; repeatable",
    )
    return p.parse_args()


if __name__ == "__main__":
    main()
