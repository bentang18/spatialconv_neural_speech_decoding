"""B02 cross-corpus WRS sampler primitive tests.

Locks the algorithmic core of B02:

* Per-corpus vb-eh aggregation matches `hours × n_electrodes × valid_bins`.
* Per-row weights honour the macro 50/50 (SWEC / broadband) split + the
  within-broadband ``vb_eh ** α=0.5`` share.
* ``WeightedRandomSampler`` consumes the weights with
  ``replacement=True`` (B02 lock).
* Empirical draw frequencies converge on the target macro split + within-
  broadband shares (probabilistic, large num_samples).
* Sister α / macro overrides land in the produced shares.

Out of scope (covered by the multi-corpus loader build):
* Manifest discovery + chain construction.
* StatefulDataLoader integration + mid-epoch resume.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from speech_decoding.experiments.wrs_sampler import (
    BROADBAND_CORPORA,
    DEFAULT_ALPHA,
    SWEC_CORPUS,
    ManifestRow,
    build_wrs_sampler,
    compute_per_row_weights,
    compute_vb_eh_per_corpus,
)


def _rows_for(corpus: str, *, n: int, n_electrodes: int, hours: float, valid_bins: int):
    return [
        ManifestRow(
            corpus=corpus,
            n_electrodes=n_electrodes,
            hours=hours,
            valid_bins=valid_bins,
        )
        for _ in range(n)
    ]


def _balanced_4_corpus_manifest():
    """Four-corpus toy manifest sized so the macro split is the load-bearing
    constraint (not the within-broadband shape)."""
    rows = []
    rows += _rows_for(SWEC_CORPUS, n=1000, n_electrodes=128, hours=1.0, valid_bins=22)
    rows += _rows_for("ajile12", n=200, n_electrodes=200, hours=1.0, valid_bins=21)
    rows += _rows_for("d_cohort", n=80, n_electrodes=128, hours=1.0, valid_bins=30)
    rows += _rows_for("braintreebank", n=10, n_electrodes=128, hours=1.0, valid_bins=30)
    return rows


def test_manifest_row_vb_eh_formula() -> None:
    row = ManifestRow(
        corpus="braintreebank", n_electrodes=128, hours=2.5, valid_bins=30,
    )
    # 2.5 × 128 × 30 = 9600
    assert row.vb_eh() == pytest.approx(2.5 * 128 * 30)


def test_compute_vb_eh_per_corpus_aggregates_by_corpus() -> None:
    rows = _rows_for("ajile12", n=3, n_electrodes=100, hours=2.0, valid_bins=21)
    rows += _rows_for("braintreebank", n=2, n_electrodes=128, hours=1.0, valid_bins=30)
    totals = compute_vb_eh_per_corpus(rows)
    assert totals["ajile12"] == pytest.approx(3 * 100 * 2.0 * 21)
    assert totals["braintreebank"] == pytest.approx(2 * 128 * 1.0 * 30)
    assert SWEC_CORPUS not in totals  # no SWEC rows → key absent


def test_weights_macro_50_50_split_swec_vs_broadband() -> None:
    """Each draw lands in SWEC 50% / broadband 50% under the B02 default."""
    rows = _balanced_4_corpus_manifest()
    weights = compute_per_row_weights(rows)

    swec_mass = 0.0
    broad_mass = 0.0
    for row, w in zip(rows, weights):
        if row.corpus == SWEC_CORPUS:
            swec_mass += w
        else:
            broad_mass += w
    # Per-row weights are normalised in expectation; sum within each macro
    # group should land on the macro share.
    assert swec_mass == pytest.approx(0.5, rel=1e-6)
    assert broad_mass == pytest.approx(0.5, rel=1e-6)


def test_weights_within_broadband_alpha_0_5_share() -> None:
    """Inside the broadband group, share ∝ vb_eh ** 0.5."""
    rows = _balanced_4_corpus_manifest()
    weights = compute_per_row_weights(rows, alpha=0.5)

    # Per-row totals per corpus (each row in a corpus has the SAME weight,
    # so corpus_share = n_rows × per_row_weight).
    corpus_mass: dict[str, float] = {}
    for row, w in zip(rows, weights):
        corpus_mass[row.corpus] = corpus_mass.get(row.corpus, 0.0) + w

    # Expected broadband shares: vb_eh = (200 × 1 × 200 × 21) = ~840k for
    # AJILE12, ~307k for D, ~38k for BT.
    vb_eh = compute_vb_eh_per_corpus(rows)
    broad_total = sum(
        vb_eh[c] ** 0.5 for c in BROADBAND_CORPORA if c in vb_eh
    )
    for c in BROADBAND_CORPORA:
        if c not in corpus_mass:
            continue
        expected_within = vb_eh[c] ** 0.5 / broad_total
        # The macro share for broadband is 0.5, so corpus mass ≈ 0.5 ×
        # within_share.
        assert corpus_mass[c] == pytest.approx(0.5 * expected_within, rel=1e-6)


def test_weights_alpha_sister_override() -> None:
    """`R-sampler-pure-h` (α=1.0) shifts the within-broadband shape towards
    pure vb_eh-proportional sampling (AJILE12's mass grows even more)."""
    rows = _balanced_4_corpus_manifest()
    w_alpha_default = compute_per_row_weights(rows, alpha=DEFAULT_ALPHA)
    w_alpha_one = compute_per_row_weights(rows, alpha=1.0)

    ajile_mass_default = sum(
        w for r, w in zip(rows, w_alpha_default) if r.corpus == "ajile12"
    )
    ajile_mass_one = sum(
        w for r, w in zip(rows, w_alpha_one) if r.corpus == "ajile12"
    )
    assert ajile_mass_one > ajile_mass_default, (
        "α=1.0 must amplify the largest broadband corpus's share over α=0.5"
    )


def test_weights_macro_sister_override() -> None:
    """`R-sampler-60-40` tilts the macro split toward broadband."""
    rows = _balanced_4_corpus_manifest()
    w = compute_per_row_weights(
        rows, macro_split={"swec": 0.4, "broadband": 0.6},
    )
    swec_mass = sum(_w for r, _w in zip(rows, w) if r.corpus == SWEC_CORPUS)
    broad_mass = sum(_w for r, _w in zip(rows, w) if r.corpus != SWEC_CORPUS)
    assert swec_mass == pytest.approx(0.4, rel=1e-6)
    assert broad_mass == pytest.approx(0.6, rel=1e-6)


def test_weights_swec_absent_collapses_into_broadband() -> None:
    """No SWEC rows → the SWEC macro share folds into broadband."""
    rows = _rows_for("ajile12", n=10, n_electrodes=100, hours=1.0, valid_bins=21)
    rows += _rows_for("braintreebank", n=5, n_electrodes=128, hours=1.0, valid_bins=30)
    w = compute_per_row_weights(rows)
    swec_mass = sum(_w for r, _w in zip(rows, w) if r.corpus == SWEC_CORPUS)
    broad_mass = sum(_w for r, _w in zip(rows, w) if r.corpus != SWEC_CORPUS)
    assert swec_mass == 0.0
    assert broad_mass == pytest.approx(1.0, rel=1e-6)


def test_weights_broadband_absent_collapses_into_swec() -> None:
    rows = _rows_for(SWEC_CORPUS, n=10, n_electrodes=128, hours=1.0, valid_bins=22)
    w = compute_per_row_weights(rows)
    assert float(np.sum(w)) == pytest.approx(1.0, rel=1e-6)


def test_weights_rejects_negative_alpha() -> None:
    rows = _balanced_4_corpus_manifest()
    with pytest.raises(ValueError, match="alpha"):
        compute_per_row_weights(rows, alpha=-0.1)


def test_weights_rejects_negative_macro_share() -> None:
    rows = _balanced_4_corpus_manifest()
    with pytest.raises(ValueError, match="macro_split"):
        compute_per_row_weights(rows, macro_split={"swec": -0.1, "broadband": 1.1})


def test_weights_rejects_zero_macro_split() -> None:
    rows = _balanced_4_corpus_manifest()
    with pytest.raises(ValueError, match="at least one positive"):
        compute_per_row_weights(rows, macro_split={"swec": 0.0, "broadband": 0.0})


def test_weights_rejects_empty_rows() -> None:
    with pytest.raises(ValueError, match="rows"):
        compute_per_row_weights([])


def test_build_wrs_sampler_returns_replacement_true_sampler() -> None:
    rows = _balanced_4_corpus_manifest()
    weights = compute_per_row_weights(rows)
    sampler = build_wrs_sampler(weights, num_samples=128)
    # WeightedRandomSampler stores replacement directly.
    assert sampler.replacement is True
    assert sampler.num_samples == 128


def test_build_wrs_sampler_uses_provided_generator() -> None:
    """Per-rank deterministic worker RNG is supplied via ``generator``."""
    rows = _balanced_4_corpus_manifest()
    weights = compute_per_row_weights(rows)
    g = torch.Generator().manual_seed(123)
    sampler = build_wrs_sampler(weights, num_samples=8, generator=g)
    assert sampler.generator is g


def test_empirical_macro_share_converges_with_replacement_draws() -> None:
    """Replacement draws over the WRS should converge on the macro 50/50
    split for a large enough sample."""
    rows = _balanced_4_corpus_manifest()
    weights = compute_per_row_weights(rows)
    g = torch.Generator().manual_seed(7)
    sampler = build_wrs_sampler(weights, num_samples=20_000, generator=g)

    is_swec = np.array([r.corpus == SWEC_CORPUS for r in rows])
    draws = list(sampler)
    drawn = is_swec[draws]
    swec_frac = float(drawn.mean())
    # 20k draws → SE ≈ √(0.25 / 20000) ≈ 0.0035; 3σ ≈ 0.011. Use a wider
    # 0.03 band for CI slack.
    assert 0.47 <= swec_frac <= 0.53, (
        f"empirical SWEC share {swec_frac:.4f} outside [0.47, 0.53]"
    )


def test_build_wrs_sampler_rejects_negative_weights() -> None:
    weights = np.array([0.5, -0.1, 0.3], dtype=np.float64)
    with pytest.raises(ValueError, match="non-negative"):
        build_wrs_sampler(weights, num_samples=1)


def test_build_wrs_sampler_rejects_zero_weights_sum() -> None:
    weights = np.zeros(5, dtype=np.float64)
    with pytest.raises(ValueError, match="zero"):
        build_wrs_sampler(weights, num_samples=1)


def test_build_wrs_sampler_rejects_non_1d_weights() -> None:
    weights = np.ones((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="1-d"):
        build_wrs_sampler(weights, num_samples=1)


def test_build_wrs_sampler_rejects_non_positive_num_samples() -> None:
    weights = np.ones(5, dtype=np.float64)
    with pytest.raises(ValueError, match="num_samples"):
        build_wrs_sampler(weights, num_samples=0)


def test_build_stateful_dataloader_raises_when_torchdata_missing() -> None:
    """Lazy import: when torchdata is absent (dev checkout), the factory
    raises with the explicit install hint instead of a bare ImportError."""
    pytest.importorskip("torch")
    try:
        import torchdata.stateful_dataloader  # noqa: F401
    except ImportError:
        from speech_decoding.experiments.wrs_sampler import (
            build_stateful_dataloader,
        )
        rows = _balanced_4_corpus_manifest()
        weights = compute_per_row_weights(rows)
        sampler = build_wrs_sampler(weights, num_samples=4)
        with pytest.raises(RuntimeError, match="torchdata"):
            build_stateful_dataloader(
                dataset=[None] * len(rows), sampler=sampler, batch_size=2,
            )
    else:
        # torchdata is installed in this env — only validate that the
        # factory builds without raising (we don't iterate the loader to
        # avoid coupling to the dataset surface).
        from speech_decoding.experiments.wrs_sampler import (
            build_stateful_dataloader,
        )
        from torch.utils.data import Dataset

        class _Dummy(Dataset):
            def __init__(self, n: int) -> None:
                self.n = n

            def __len__(self) -> int:
                return self.n

            def __getitem__(self, idx: int) -> int:
                return idx

        rows = _balanced_4_corpus_manifest()
        weights = compute_per_row_weights(rows)
        sampler = build_wrs_sampler(weights, num_samples=4)
        loader = build_stateful_dataloader(
            dataset=_Dummy(len(rows)), sampler=sampler, batch_size=2,
            num_workers=0,
        )
        assert loader is not None
