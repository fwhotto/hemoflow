"""Tests for geometry utility functions."""

import numpy as np
from preprocessor.geometry import (
    inRange,
    inRange3D,
    select_face_from_normal,
    select_closest_face,
    check_corner_proximity,
    scaleAndShiftData,
    apply_boundary_cuts
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


class TestApplyBoundaryCuts:
    """Tests for apply_boundary_cuts function."""

    def test_no_cuts(self):
        """Test with empty cut list - should return copy of original."""
        volume = np.ones((10, 10, 10), dtype=int)
        cut_list = np.array([])
        result = apply_boundary_cuts(volume, cut_list, cut_width=1)
        assert result.shape == (10, 10, 10)
        np.testing.assert_array_equal(result, volume)

    def test_cut_face_0_x_minus(self):
        """Test cutting X- face (face 0)."""
        volume = np.ones((10, 10, 10), dtype=int)
        cut_list = np.array([0])
        result = apply_boundary_cuts(volume, cut_list, cut_width=1)
        assert result.shape == (9, 10, 10)

    def test_cut_face_1_x_plus(self):
        """Test cutting X+ face (face 1)."""
        volume = np.ones((10, 10, 10), dtype=int)
        cut_list = np.array([1])
        result = apply_boundary_cuts(volume, cut_list, cut_width=1)
        assert result.shape == (9, 10, 10)

    def test_cut_face_2_y_minus(self):
        """Test cutting Y- face (face 2)."""
        volume = np.ones((10, 10, 10), dtype=int)
        cut_list = np.array([2])
        result = apply_boundary_cuts(volume, cut_list, cut_width=1)
        assert result.shape == (10, 9, 10)

    def test_cut_face_3_y_plus(self):
        """Test cutting Y+ face (face 3)."""
        volume = np.ones((10, 10, 10), dtype=int)
        cut_list = np.array([3])
        result = apply_boundary_cuts(volume, cut_list, cut_width=1)
        assert result.shape == (10, 9, 10)

    def test_cut_face_4_z_minus(self):
        """Test cutting Z- face (face 4)."""
        volume = np.ones((10, 10, 10), dtype=int)
        cut_list = np.array([4])
        result = apply_boundary_cuts(volume, cut_list, cut_width=1)
        assert result.shape == (10, 10, 9)

    def test_cut_face_5_z_plus(self):
        """Test cutting Z+ face (face 5)."""
        volume = np.ones((10, 10, 10), dtype=int)
        cut_list = np.array([5])
        result = apply_boundary_cuts(volume, cut_list, cut_width=1)
        assert result.shape == (10, 10, 9)

    def test_cut_multiple_faces(self):
        """Test cutting multiple faces simultaneously."""
        volume = np.ones((10, 10, 10), dtype=int)
        cut_list = np.array([0, 2, 4])  # X-, Y-, Z-
        result = apply_boundary_cuts(volume, cut_list, cut_width=1)
        assert result.shape == (9, 9, 9)

    def test_cut_all_faces(self):
        """Test cutting all 6 faces."""
        volume = np.ones((10, 10, 10), dtype=int)
        cut_list = np.array([0, 1, 2, 3, 4, 5])
        result = apply_boundary_cuts(volume, cut_list, cut_width=1)
        assert result.shape == (8, 8, 8)

    def test_cut_width_2(self):
        """Test with cut_width=2."""
        volume = np.ones((10, 10, 10), dtype=int)
        cut_list = np.array([0])
        result = apply_boundary_cuts(volume, cut_list, cut_width=2)
        assert result.shape == (8, 10, 10)

    def test_cut_width_3(self):
        """Test with cut_width=3."""
        volume = np.ones((10, 10, 10), dtype=int)
        cut_list = np.array([0, 1])  # Both X faces
        result = apply_boundary_cuts(volume, cut_list, cut_width=3)
        assert result.shape == (4, 10, 10)

    def test_sequential_cutting_order(self):
        """Test that cuts are applied sequentially (shape changes affect subsequent cuts)."""
        # Create volume with unique values to track what's removed
        volume = np.arange(1000).reshape((10, 10, 10))
        cut_list = np.array([0, 1])  # Cut X- then X+
        result = apply_boundary_cuts(volume, cut_list, cut_width=1)

        # After cutting X- (remove first slice), then X+ (remove last slice of remaining)
        # Final shape should be (8, 10, 10)
        assert result.shape == (8, 10, 10)

        # Verify the correct slices remain
        # X- cut removes slice 0, X+ cut removes last slice of what's left
        # So we should have original slices 1-8
        expected = volume[1:9, :, :]
        np.testing.assert_array_equal(result, expected)

    def test_original_volume_unchanged(self):
        """Test that original volume is not modified (function creates copy)."""
        volume = np.ones((10, 10, 10), dtype=int)
        original_shape = volume.shape
        cut_list = np.array([0, 1, 2, 3, 4, 5])
        result = apply_boundary_cuts(volume, cut_list, cut_width=1)

        # Original should be unchanged
        assert volume.shape == original_shape
        assert result.shape != volume.shape
