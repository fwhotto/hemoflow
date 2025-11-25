"""Data models for preprocessor pipeline stages."""

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np


@dataclass
class VoxelizationResult:
    """Result of vessel voxelization stage."""
    volume: np.ndarray
    scale: Tuple[float, float, float]
    shift: Tuple[float, float, float]
    domain_size: Tuple[int, int, int]
    bbox: Tuple[float, float, float, float, float, float]
    dx: float


@dataclass
class OpeningData:
    """Opening information extracted from centerline."""
    radius_tangent_list: List[Tuple[float, np.ndarray, np.ndarray]]
    cut_list: np.ndarray


@dataclass
class GeometryResult:
    """Final geometry with labeled openings."""
    volume: np.ndarray
    opening_index: List[int]
    opening_radius: List[float]
    opening_normalized_q_ratio: List[float]
    opening_center: List[np.ndarray]
    opening_normal: List[np.ndarray]


@dataclass
class StentResult:
    """Stent voxelization with resistance coefficients."""
    volume: np.ndarray
    linear: np.ndarray
    quadratic: np.ndarray
