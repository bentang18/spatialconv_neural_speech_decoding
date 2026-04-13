"""LOPO orchestration."""
from __future__ import annotations

import logging

import numpy as np
from scipy.stats import wilcoxon

from speech_decoding.data.bids_dataset import BIDSDataset
from speech_decoding.training.adaptor import adapt_stage2
from speech_decoding.training.lopo_trainer import train_stage1

logger = logging.getLogger(__name__)


def run_lopo(
    all_datasets: dict[str, BIDSDataset],
    config: dict,
    seeds: list[int],
    device: str = "cpu",
    baseline_pers: dict[str, float] | None = None,
) -> dict:
    """Run LOPO over all patients and seeds."""
    patient_ids = sorted(all_datasets)
    per_patient: dict[str, list[dict]] = {pid: [] for pid in patient_ids}

    for target_pid in patient_ids:
        logger.info("LOPO target: %s", target_pid)
        source_datasets = {
            pid: ds for pid, ds in all_datasets.items() if pid != target_pid
        }

        for seed in seeds:
            checkpoint = train_stage1(source_datasets, config, seed=seed, device=device)
            result = adapt_stage2(
                checkpoint,
                all_datasets[target_pid],
                source_datasets,
                config,
                seed=seed,
                device=device,
            )
            per_patient[target_pid].append(result)

    patient_pers = {
        pid: np.mean([r["per_mean"] for r in per_patient[pid]]).item()
        for pid in patient_ids
    }
    all_pers = list(patient_pers.values())
    summary = {
        "per_patient": per_patient,
        "patient_pers": patient_pers,
        "population_per_mean": np.mean(all_pers).item(),
        "population_per_std": np.std(all_pers).item(),
    }

    if baseline_pers is not None:
        paired_lopo = [patient_pers[pid] for pid in patient_ids]
        paired_base = [baseline_pers[pid] for pid in patient_ids]
        stat, p = wilcoxon(paired_lopo, paired_base, alternative="less")
        summary["wilcoxon_stat"] = float(stat)
        summary["wilcoxon_p"] = float(p)

    return summary
