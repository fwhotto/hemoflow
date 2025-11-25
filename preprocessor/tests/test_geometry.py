"""Tests for geometry utility functions."""

import numpy as np
import pytest
from src.geometry import (
    inRange,
    inRange3D,
    select_face_from_normal,
    select_closest_face,
    check_corner_proximity,
    scaleAndShiftData
)


class TestInRange:
    """Tests for inRange function."""

    def test_within_range(self):
        assert inRange(5.0, 5.5, 1.0) is True

    def test_outside_range(self):
        assert inRange(5.0, 7.0, 1.0) is False

    def test_exact_boundary(self):
        # Uses < not <=, so exact boundary is False
        assert inRange(5.0, 6.0, 1.0) is False
        # But slightly inside is True
        assert inRange(5.0, 5.9, 1.0) is True


class TestInRange3D:
    """Tests for inRange3D function."""

    def test_all_within_range(self):
        value3D = np.array([5.0, 5.0, 5.0])
        rangeValue3D = np.array([5.5, 5.5, 5.5])
        assert inRange3D(value3D, rangeValue3D, 1.0) is True

    def test_one_outside_range(self):
        value3D = np.array([5.0, 5.0, 5.0])
        rangeValue3D = np.array([5.5, 5.5, 7.0])
        assert inRange3D(value3D, rangeValue3D, 1.0) is False


class TestSelectFaceFromNormal:
    """Tests for select_face_from_normal function."""

    def test_positive_x_tangent(self):
        tangent = np.array([1.0, 0.0, 0.0])
        assert select_face_from_normal(tangent) == 0  # X- face

    def test_negative_x_tangent(self):
        tangent = np.array([-1.0, 0.0, 0.0])
        assert select_face_from_normal(tangent) == 1  # X+ face

    def test_positive_y_tangent(self):
        tangent = np.array([0.0, 1.0, 0.0])
        assert select_face_from_normal(tangent) == 2  # Y- face

    def test_positive_z_tangent(self):
        tangent = np.array([0.0, 0.0, 1.0])
        assert select_face_from_normal(tangent) == 4  # Z- face


class TestSelectClosestFace:
    """Tests for select_closest_face function."""

    def test_closest_to_x_minus(self):
        pos = np.array([2.0, 50.0, 50.0])
        domain_size = (100, 100, 100)
        assert select_closest_face(pos, domain_size) == 0

    def test_closest_to_x_plus(self):
        pos = np.array([98.0, 50.0, 50.0])
        domain_size = (100, 100, 100)
        assert select_closest_face(pos, domain_size) == 1

    def test_closest_to_y_minus(self):
        pos = np.array([50.0, 2.0, 50.0])
        domain_size = (100, 100, 100)
        assert select_closest_face(pos, domain_size) == 2


class TestCheckCornerProximity:
    """Tests for check_corner_proximity function."""

    def test_not_at_corner(self):
        pos = np.array([50.0, 50.0, 50.0])
        domain_size = (100, 100, 100)
        result = check_corner_proximity(pos, domain_size, 5.0)
        assert result['is_corner'] is False

    def test_at_corner(self):
        pos = np.array([2.0, 2.0, 50.0])
        domain_size = (100, 100, 100)
        result = check_corner_proximity(pos, domain_size, 5.0)
        assert result['is_corner'] is True
        assert result['num_close'] == 2


class TestScaleAndShiftData:
    """Tests for scaleAndShiftData function."""

    def test_scale_and_shift(self):
        points = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        scale = (2.0, 2.0, 2.0)
        shift = (1.0, 1.0, 1.0)
        result = scaleAndShiftData(points, scale, shift)
        expected = [[4.0, 6.0, 8.0], [10.0, 12.0, 14.0]]
        np.testing.assert_array_almost_equal(result, expected)
