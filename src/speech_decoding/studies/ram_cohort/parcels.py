"""RAM ``ind.region`` -> v3 DKT ``parcel_fn`` (ingestion contract item 2).

RAM ships native FreeSurfer DK aparc per contact directly in ``electrodes.tsv``
(the hemi-agnostic ``ind.region`` base + a separate ``hemisphere`` column), so
localization is a TSV READ — no volumetric pipeline (contrast Cogan's #97). This
maps each voltage-order contact to a ``V14_DKT_PARCEL_LABELS`` id, mirroring
``dispatch_v3.make_bt_parcel_fn``'s seam but sourcing anatomy from
``electrodes.tsv`` instead of ``depth-wm.csv``.

DK->DKT policy (Ben decision #5). The model vocab is DKT-31 (drops
``bankssts``/``frontalpole``/``temporalpole``); RAM ``ind.region`` is DK aparc-34.
``dk_policy`` decides those 3 dropped bases:

  * ``"sentinel"`` (default, honest) -> the reserved unknown id; the contact's
    voltage is still pooled, it just carries no anatomy tag.
  * ``"neighbor"`` (the DKT protocol's own reassignment) -> ``bankssts`` and
    ``temporalpole`` -> ``superiortemporal``, ``frontalpole`` -> ``superiorfrontal``.

The flag makes BOTH ready; the default commits nothing. ``ind.region`` = ``n/a``
(RAM's ~10% unlocalized) or an out-of-vocab base always falls to the sentinel.

Anatomy is brain-constant, so a subject's parcels are resolved from that subject's
BEST-localized ``electrodes.tsv`` across all its sessions/tasks (``ind.region`` is
identical per implant; the per-session ``space-*`` sidecars differ only in coords,
which we discard — MNI is BANNED). ``sidecar_root`` is globbed recursively for
``*sub-<rid>*_electrodes.tsv``, so it works on the flat recon sidecar dir AND on
the pulled BIDS tree unchanged.
"""

from __future__ import annotations

import csv
import glob
import os
import typing as tp

# electrodes.tsv hemisphere -> FreeSurfer hemi prefix.
_HEMI: dict[str, str] = {"L": "lh", "R": "rh", "lh": "lh", "rh": "rh"}
# The DKT protocol reassigns each dropped DK base's vertices to this neighbour.
_DKT_NEIGHBOR: dict[str, str] = {
    "bankssts": "superiortemporal",
    "temporalpole": "superiortemporal",
    "frontalpole": "superiorfrontal",
}
_DKT_DROPPED: frozenset[str] = frozenset(_DKT_NEIGHBOR)
_NA_REGIONS: frozenset[str] = frozenset({"", "n/a", "na", "unknown", "none"})
_DK_POLICIES: frozenset[str] = frozenset({"sentinel", "neighbor"})


def read_electrodes_regions(path: str) -> dict[str, tuple[str, str]]:
    """``electrodes.tsv`` -> ``{contact_name: (hemisphere_raw, ind_region_lower)}``.

    Reads only the three columns we consume (``name``/``hemisphere``/``ind.region``);
    the MNI coords (``x``/``y``/``z``/``tal.*``) are never touched. ``utf-8-sig`` to
    swallow a BOM if present.
    """
    out: dict[str, tuple[str, str]] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name:
                continue
            hemi = (row.get("hemisphere") or "").strip()
            region = (row.get("ind.region") or "").strip().lower()
            out[name] = (hemi, region)
    return out


def region_to_parcel_id(
    hemi_raw: str,
    base_raw: str,
    label_to_id: dict[str, int],
    unknown_id: int,
    dk_policy: str,
) -> int:
    """One contact's ``(hemisphere, ind.region)`` -> DKT parcel id (or sentinel)."""
    base = base_raw.strip().lower()
    if base in _NA_REGIONS:
        return unknown_id
    hemi = _HEMI.get(hemi_raw.strip())
    if hemi is None:
        return unknown_id
    if base in _DKT_DROPPED:
        if dk_policy == "sentinel":
            return unknown_id
        base = _DKT_NEIGHBOR[base]  # dk_policy == "neighbor" (validated by caller)
    return label_to_id.get(f"ctx-{hemi}-{base}", unknown_id)


def resolve_parcel_ids(
    regions: dict[str, tuple[str, str]],
    labels: tp.Sequence[str],
    label_to_id: dict[str, int],
    unknown_id: int,
    dk_policy: str,
) -> tuple[list[int], int]:
    """Map voltage-order ``labels`` -> parcel ids; return ``(ids, n_in_vocab)``.

    A cache label with NO ``electrodes.tsv`` row falls to the sentinel (RAM is ~90%
    localized — unlike BT, a missing row is expected, not an error). Coverage is
    enforced by the caller's floor, not by raising here.
    """
    ids: list[int] = []
    n_valid = 0
    for lab in labels:
        hb = regions.get(lab)
        pid = (
            unknown_id
            if hb is None
            else region_to_parcel_id(hb[0], hb[1], label_to_id, unknown_id, dk_policy)
        )
        if pid != unknown_id:
            n_valid += 1
        ids.append(pid)
    return ids, n_valid


def _localized_count(regions: dict[str, tuple[str, str]]) -> int:
    return sum(1 for _, base in regions.values() if base not in _NA_REGIONS)


def best_regions_for_rid(sidecar_root: str, rid: str) -> dict[str, tuple[str, str]]:
    """Pick this subject's most-localized ``electrodes.tsv`` (anatomy is constant).

    Globs ``sidecar_root`` recursively for ``*sub-<rid>*_electrodes.tsv`` and
    returns the region map of the file with the most non-``n/a`` ``ind.region``
    rows. Raises if the subject has no ``electrodes.tsv`` at all.
    """
    hits = sorted(
        glob.glob(os.path.join(sidecar_root, "**", f"*sub-{rid}*_electrodes.tsv"),
                  recursive=True)
    )
    if not hits:
        raise FileNotFoundError(
            f"no electrodes.tsv for subject rid={rid!r} under {sidecar_root}"
        )
    best: dict[str, tuple[str, str]] = {}
    best_n = -1
    for path in hits:
        regions = read_electrodes_regions(path)
        n = _localized_count(regions)
        if n > best_n:
            best, best_n = regions, n
    return best


def make_ram_parcel_fn(
    sidecar_root: str,
    id_map: dict[str, int],
    *,
    dk_policy: str = "sentinel",
    atlas: str = "dkt",
    min_valid_fraction: float = 0.5,
) -> tp.Callable[[int, int, tp.Sequence[str]], "object"]:
    """RAM ``parcel_fn`` factory (mirrors ``make_bt_parcel_fn``'s seam).

    ``id_map`` is the persisted ``rid -> global_subject_id`` map
    (``ram_subject_id_map.json``); it is inverted so the dispatch-supplied
    ``subject_id`` resolves back to the R-id whose ``electrodes.tsv`` we read.
    ``dk_policy`` is Ben decision #5 (default honest ``"sentinel"``). Prints a
    per-subject coverage ``[check]`` and enforces the floor (S9-class guard: a
    namespace drift that silently zeroes a subject's routing fails LOUD).
    """
    if dk_policy not in _DK_POLICIES:
        raise ValueError(f"dk_policy {dk_policy!r} not in {sorted(_DK_POLICIES)}")
    from speech_decoding.studies.braintreebank.anatomy import atlas_spec
    import torch

    _lcol, plabels = atlas_spec(atlas)
    label_to_id = {lab: i for i, lab in enumerate(plabels)}
    unknown_id = len(plabels)
    gid_to_rid = {int(v): k for k, v in id_map.items()}

    def parcel_fn(subject_id: int, trial_id: int, labels: tp.Sequence[str]):
        rid = gid_to_rid[int(subject_id)]
        regions = best_regions_for_rid(sidecar_root, rid)
        ids, n_valid = resolve_parcel_ids(
            regions, labels, label_to_id, unknown_id, dk_policy
        )
        frac = n_valid / len(labels) if labels else 0.0
        status = "OK" if frac >= min_valid_fraction else "VIOLATED"
        print(
            f"[check] RAM parcel coverage subj {subject_id} ({rid}) "
            f"policy={dk_policy}: {n_valid}/{len(labels)} = {frac:.3f} {status}"
        )
        if frac < min_valid_fraction:
            raise ValueError(
                f"subject {subject_id} ({rid}): only {n_valid}/{len(labels)} "
                f"({frac:.3f}) contacts mapped to an in-vocab DKT parcel — below the "
                f"{min_valid_fraction:.2f} floor. The electrodes.tsv name namespace "
                "likely diverged from the voltage order (S9-class); its parcel-pool "
                "routing would be silently sentinel-zeroed."
            )
        return torch.tensor(ids, dtype=torch.long)

    return parcel_fn
