"""Pytest fixtures and configuration for HemoFlow preprocessor tests."""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_voxel_volume():
    """Create a sample voxel volume for testing."""
    # Create a simple 10x10x10 volume with fluid in the center
    volume = np.zeros((10, 10, 10), dtype=bool)
    volume[3:7, 3:7, 3:7] = True
    return volume


@pytest.fixture
def sample_domain_size():
    """Sample domain size tuple."""
    return (100, 100, 100)


@pytest.fixture
def sample_scale_shift():
    """Sample scale and shift tuples."""
    scale = (10.0, 10.0, 10.0)
    shift = (0.0, 0.0, 0.0)
    return scale, shift


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for testing."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
