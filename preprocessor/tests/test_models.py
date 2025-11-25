"""Tests for data models."""

import numpy as np
import pytest
from preprocessor.models import (
    VoxelizationResult,
    OpeningData,
    GeometryResult,
    StentResult
)


class TestVoxelizationResult:
    """Tests for VoxelizationResult dataclass."""

    def test_creation(self):
        volume = np.zeros((10, 10, 10), dtype=bool)
        result = VoxelizationResult(
            volume=volume,
            scale=(1.0, 1.0, 1.0),
            shift=(0.0, 0.0, 0.0),
            domain_size=(10, 10, 10),
            bbox=(0, 10, 0, 10, 0, 10),
            dx=0.1
        )
        assert result.volume.shape == (10, 10, 10)
        assert result.dx == 0.1
        assert result.scale == (1.0, 1.0, 1.0)


class TestOpeningData:
    """Tests for OpeningData dataclass."""

    def test_creation(self):
        radius_tangent_list = [
            (1.0, np.array([0, 0, 0]), np.array([1, 0, 0])),
            (0.5, np.array([10, 10, 10]), np.array([0, 1, 0]))
        ]
        cut_list = np.array([0, 1])
        result = OpeningData(
            radius_tangent_list=radius_tangent_list,
            cut_list=cut_list
        )
        assert len(result.radius_tangent_list) == 2
        assert len(result.cut_list) == 2


class TestGeometryResult:
    """Tests for GeometryResult dataclass."""

    def test_creation(self):
        volume = np.zeros((10, 10, 10), dtype=np.int16)
        result = GeometryResult(
            volume=volume,
            opening_index=[10, 11],
            opening_radius=[1.0, 0.5],
            opening_normalized_q_ratio=[0.8, 0.2],
            opening_center=[np.array([0, 0, 0]), np.array([10, 10, 10])],
            opening_normal=[np.array([1, 0, 0]), np.array([0, 1, 0])]
        )
        assert result.volume.shape == (10, 10, 10)
        assert len(result.opening_index) == 2
        assert len(result.opening_radius) == 2


class TestStentResult:
    """Tests for StentResult dataclass."""

    def test_creation(self):
        volume = np.zeros((10, 10, 10), dtype=bool)
        linear = np.ones((10, 10, 10), dtype=np.float32)
        quadratic = np.ones((10, 10, 10), dtype=np.float32)
        result = StentResult(
            volume=volume,
            linear=linear,
            quadratic=quadratic
        )
        assert result.volume.shape == (10, 10, 10)
        assert result.linear.shape == (10, 10, 10)
        assert result.quadratic.shape == (10, 10, 10)
