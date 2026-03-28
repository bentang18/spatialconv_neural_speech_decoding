"""Abstract base class for synthetic data generators."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Generator(ABC):
    def __init__(self, grid_h: int, grid_w: int, T: int):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.T = T

    @abstractmethod
    def generate(self, seed: int | None = None) -> np.ndarray:
        """Generate one sequence. Returns (H, W, T) float32 array."""
        ...
