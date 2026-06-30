"""Tests for the results-ledger CSV writer."""

from __future__ import annotations

from speech_decoding.experiments.pretrain_probe_csv import (
    FIELDS,
    ResultRow,
    append_results,
    read_results,
)


def _row(task: str, auroc: float, **kw) -> ResultRow:
    base = dict(
        stamp="2026-06-30T00:00:00", ckpt="raw_371", readout="ridge", tap="raw_371",
        eval_mode="CrossSubject", task=task, split="test", auroc=auroc, n=6,
    )
    base.update(kw)
    return ResultRow(**base)


def test_append_writes_header_once_and_accumulates(tmp_path):
    p = str(tmp_path / "results.csv")
    append_results([_row("onset", 0.81)], p)
    append_results([_row("speech", 0.78), _row("volume", 0.55)], p)
    rows = read_results(p)
    assert [r["task"] for r in rows] == ["onset", "speech", "volume"]
    assert rows[0]["auroc"] == "0.81"
    # header appears exactly once
    with open(p) as f:
        assert f.read().count(",".join(FIELDS)) == 1


def test_roundtrip_fields(tmp_path):
    p = str(tmp_path / "r.csv")
    append_results([_row("pitch", 0.62, lam="1.0", notes="anchor=(2,1)")], p)
    (r,) = read_results(p)
    assert set(r) == set(FIELDS)
    assert r["lam"] == "1.0" and r["notes"] == "anchor=(2,1)"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
