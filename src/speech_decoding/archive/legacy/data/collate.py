"""Multi-patient collation for DataLoader.

Groups samples by patient_id since different patients have different
grid sizes and can't be naively stacked into a single tensor.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def collate_by_patient(
    samples: list[tuple[np.ndarray, list[int], str]],
) -> dict[str, tuple[torch.Tensor, list[list[int]], str]]:
    """Group samples by patient and stack into tensors.

    Args:
        samples: List of (grid_data, ctc_label, patient_id) from BIDSDataset.

    Returns:
        Dict mapping patient_id → (grid_tensor, labels_list, patient_id)
        where grid_tensor is (n_trials, H, W, T).
    """
    grouped: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for x, y, pid in samples:
        grouped[pid][0].append(x)
        grouped[pid][1].append(y)

    result = {}
    for pid, (xs, ys) in grouped.items():
        tensor = torch.from_numpy(np.stack(xs))
        result[pid] = (tensor, ys, pid)
    return result
