"""
Geometry rotation utilities for aligning openings with bounding box faces.

This module provides functions to rotate STL and VTP files so that the inlet
opening aligns with an axis-aligned bounding box face, which is required by
the HemoFlow preprocessor for proper opening detection.
"""

import numpy as np
from stl import mesh
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import logging
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def calculate_rotation_matrix(source_vector: np.ndarray, target_vector: np.ndarray) -> np.ndarray:
    """
    Calculate rotation matrix to align source_vector with target_vector.

    Uses Rodrigues' rotation formula to compute the rotation matrix that
    rotates source_vector to align with target_vector.

    Args:
        source_vector: Vector to rotate from (will be normalized)
        target_vector: Vector to rotate to (will be normalized)

    Returns:
        3x3 rotation matrix
    """
    # Normalize vectors
    source = source_vector / np.linalg.norm(source_vector)
    target = target_vector / np.linalg.norm(target_vector)

    # Check if vectors are already aligned
    if np.allclose(source, target):
        return np.eye(3)

    # Check if vectors are opposite
    if np.allclose(source, -target):
        # Find perpendicular vector and rotate 180 degrees
        perp = np.array([1, 0, 0]) if abs(source[0]) < 0.9 else np.array([0, 1, 0])
        perp = perp - np.dot(perp, source) * source
        perp = perp / np.linalg.norm(perp)
        return 2 * np.outer(perp, perp) - np.eye(3)

    # Rodrigues' rotation formula
    v = np.cross(source, target)
    c = np.dot(source, target)
    s = np.linalg.norm(v)

    # Skew-symmetric cross-product matrix
    v_cross = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    rotation_matrix = np.eye(3) + v_cross + np.dot(v_cross, v_cross) * ((1 - c) / (s * s))

    return rotation_matrix


def parse_target_axis(axis_string: str) -> np.ndarray:
    """
    Parse axis string (e.g., '+x', '-y', '+z') to unit vector.

    Args:
        axis_string: String like '+x', '-x', '+y', '-y', '+z', '-z'

    Returns:
        Unit vector along specified axis

    Raises:
        ValueError: If axis_string format is invalid
    """
    axis_string = axis_string.lower().strip()

    if len(axis_string) != 2 or axis_string[0] not in ['+', '-'] or axis_string[1] not in ['x', 'y', 'z']:
        raise ValueError(f"Invalid axis string '{axis_string}'. Expected format: '+x', '-y', etc.")

    sign = 1 if axis_string[0] == '+' else -1
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[axis_string[1]]

    vector = np.zeros(3)
    vector[axis_idx] = sign

    return vector


def rotate_stl(stl_path: str, rotation_matrix: np.ndarray, output_path: str) -> None:
    """
    Rotate STL file and save to output path.

    Args:
        stl_path: Path to input STL file
        rotation_matrix: 3x3 rotation matrix
        output_path: Path to save rotated STL
    """
    logger.info(f"Rotating STL: {stl_path}")

    # Load STL
    stl_mesh = mesh.Mesh.from_file(stl_path)

    # Rotate all vertices
    for i in range(len(stl_mesh.vectors)):
        for j in range(3):  # 3 vertices per triangle
            stl_mesh.vectors[i][j] = np.dot(rotation_matrix, stl_mesh.vectors[i][j])

    # Recalculate normals
    stl_mesh.update_normals()

    # Save rotated mesh
    stl_mesh.save(output_path)
    logger.info(f"Saved rotated STL to: {output_path}")


def rotate_vtp(vtp_path: str, rotation_matrix: np.ndarray, output_path: str) -> None:
    """
    Rotate VTP centerline file and save to output path.

    Args:
        vtp_path: Path to input VTP file
        rotation_matrix: 3x3 rotation matrix
        output_path: Path to save rotated VTP
    """
    logger.info(f"Rotating VTP: {vtp_path}")

    # Read VTP file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    polydata = reader.GetOutput()

    # Get points
    points = polydata.GetPoints()
    points_array = vtk_to_numpy(points.GetData())

    # Rotate points
    rotated_points = np.dot(points_array, rotation_matrix.T)

    # Update points
    new_points = vtk.vtkPoints()
    new_points.SetData(numpy_to_vtk(rotated_points))
    polydata.SetPoints(new_points)

    # Rotate point data arrays (e.g., tangent vectors)
    point_data = polydata.GetPointData()
    for i in range(point_data.GetNumberOfArrays()):
        array = point_data.GetArray(i)
        array_name = array.GetName()

        # Check if array is 3-component (vector data)
        if array.GetNumberOfComponents() == 3:
            logger.debug(f"Rotating point data array: {array_name}")
            array_np = vtk_to_numpy(array)
            rotated_array = np.dot(array_np, rotation_matrix.T)
            new_vtk_array = numpy_to_vtk(rotated_array)
            new_vtk_array.SetName(array_name)
            point_data.RemoveArray(array_name)
            point_data.AddArray(new_vtk_array)

    # Write rotated VTP
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    writer.SetDataModeToAscii()  # Use ASCII mode for better compatibility
    writer.Write()

    logger.info(f"Saved rotated VTP to: {output_path}")


def translate_stl(stl_path: str, translation: np.ndarray, output_path: str) -> None:
    """
    Translate STL file and save to output path.

    Args:
        stl_path: Path to input STL file
        translation: 3D translation vector
        output_path: Path to save translated STL
    """
    logger.info(f"Translating STL by {translation}")

    # Load STL
    stl_mesh = mesh.Mesh.from_file(stl_path)

    # Translate all vertices
    for i in range(len(stl_mesh.vectors)):
        for j in range(3):  # 3 vertices per triangle
            stl_mesh.vectors[i][j] += translation

    # Save translated mesh
    stl_mesh.save(output_path)
    logger.info(f"Saved translated STL to: {output_path}")


def translate_vtp(vtp_path: str, translation: np.ndarray, output_path: str) -> None:
    """
    Translate VTP centerline file and save to output path.

    Args:
        vtp_path: Path to input VTP file
        translation: 3D translation vector
        output_path: Path to save translated VTP
    """
    logger.info(f"Translating VTP by {translation}")

    # Read VTP file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    polydata = reader.GetOutput()

    # Check if polydata is valid
    if polydata is None or polydata.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Failed to read VTP file or file is empty: {vtp_path}")

    # Get points
    points = polydata.GetPoints()
    if points is None:
        raise RuntimeError(f"No points found in VTP file: {vtp_path}")

    points_array = vtk_to_numpy(points.GetData())

    # Translate points
    translated_points = points_array + translation

    # Update points
    new_points = vtk.vtkPoints()
    new_points.SetData(numpy_to_vtk(translated_points))
    polydata.SetPoints(new_points)

    # Write translated VTP
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    writer.SetDataModeToAscii()  # Use ASCII mode for better compatibility
    writer.Write()

    logger.info(f"Saved translated VTP to: {output_path}")


def get_stl_bounds(stl_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get bounding box of STL file.

    Args:
        stl_path: Path to STL file

    Returns:
        Tuple of (min_coords, max_coords) as 3D numpy arrays
    """
    stl_mesh = mesh.Mesh.from_file(stl_path)

    # Get all vertices
    vertices = stl_mesh.vectors.reshape(-1, 3)

    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)

    return min_coords, max_coords


def get_centerline_opening(vtp_path: str, opening_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get opening position and normal from centerline VTP file.

    Args:
        vtp_path: Path to VTP centerline file
        opening_index: Which opening to extract (0 = first line start, 1+ = line ends)

    Returns:
        Tuple of (position, normal) as 3D numpy arrays
    """
    # Read VTP file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    polydata = reader.GetOutput()

    # Get lines
    lines = polydata.GetLines()
    lines.InitTraversal()

    points = polydata.GetPoints()

    # Get tangent data if available
    tangent_array = polydata.GetPointData().GetArray("Tangent")

    line_idx = 0
    id_list = vtk.vtkIdList()

    while lines.GetNextCell(id_list):
        num_points = id_list.GetNumberOfIds()

        if opening_index == 0 and line_idx == 0:
            # First opening: start of first line
            point_id = id_list.GetId(0)
            position = np.array(points.GetPoint(point_id))

            if tangent_array:
                normal = np.array(tangent_array.GetTuple(point_id))
            else:
                # Approximate normal from first two points
                next_point = np.array(points.GetPoint(id_list.GetId(1)))
                normal = position - next_point
                normal = normal / np.linalg.norm(normal)

            return position, normal

        elif line_idx + 1 == opening_index:
            # Other openings: end of line
            point_id = id_list.GetId(num_points - 1)
            position = np.array(points.GetPoint(point_id))

            if tangent_array:
                normal = np.array(tangent_array.GetTuple(point_id))
            else:
                # Approximate normal from last two points
                prev_point = np.array(points.GetPoint(id_list.GetId(num_points - 2)))
                normal = position - prev_point
                normal = normal / np.linalg.norm(normal)

            return position, normal

        line_idx += 1

    raise ValueError(f"Opening index {opening_index} not found in centerline (only {line_idx} lines)")


def rotate_geometry_to_align_inlet(
    stl_path: str,
    centerline_path: str,
    target_axis: str,
    output_stl: str,
    output_centerline: str,
    inlet_index: int = 0,
    position_at_boundary: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate geometry so that inlet aligns with target axis and optionally
    position at bounding box boundary.

    Args:
        stl_path: Path to input STL file
        centerline_path: Path to input VTP centerline file
        target_axis: Target axis string (e.g., '-x', '+y', '-z')
        output_stl: Path to save rotated STL
        output_centerline: Path to save rotated centerline
        inlet_index: Index of inlet opening in centerline (default: 0)
        position_at_boundary: If True, translate so inlet is at bounding box face

    Returns:
        Tuple of (rotation_matrix, translation_vector)
    """
    logger.info("=" * 80)
    logger.info("Starting geometry rotation and alignment")
    logger.info("=" * 80)

    # Parse target axis - this specifies which BOUNDARY FACE the inlet should be on
    boundary_face_vector = parse_target_axis(target_axis)
    logger.debug(f"Target boundary face: {target_axis} -> {boundary_face_vector}")

    # The inlet normal should point INTO the domain, which is OPPOSITE to the boundary face
    # E.g., if inlet is on -X face (minimum X), normal should point in +X direction (into domain)
    target_normal_direction = -boundary_face_vector
    logger.debug(f"Target inlet normal direction (into domain): {target_normal_direction}")

    # Get inlet position and normal from centerline
    inlet_pos, inlet_normal = get_centerline_opening(centerline_path, inlet_index)
    logger.debug(f"Inlet position: {inlet_pos}")
    logger.debug(f"Inlet normal (from centerline): {inlet_normal}")

    # Calculate rotation matrix to align inlet normal with target direction
    # inlet_normal points INTO vessel, -inlet_normal points OUT of vessel (into domain)
    # We want to align this with target_normal_direction (into domain)
    rotation_matrix = calculate_rotation_matrix(-inlet_normal, target_normal_direction)
    logger.debug(f"Rotation matrix:\n{rotation_matrix}")

    # Rotate STL and centerline
    temp_stl = str(Path(output_stl).with_suffix('.temp.stl'))
    temp_centerline = str(Path(output_centerline).with_suffix('.temp.vtp'))

    rotate_stl(stl_path, rotation_matrix, temp_stl)
    rotate_vtp(centerline_path, rotation_matrix, temp_centerline)

    translation = np.zeros(3)

    if position_at_boundary:
        # Get rotated inlet position
        rotated_inlet_pos = np.dot(rotation_matrix, inlet_pos)
        logger.debug(f"Rotated inlet position: {rotated_inlet_pos}")

        # Get bounds of rotated geometry
        min_coords, max_coords = get_stl_bounds(temp_stl)
        logger.debug(f"Rotated geometry bounds: min={min_coords}, max={max_coords}")

        # Determine which boundary face to align to based on boundary face vector
        axis_idx = np.argmax(np.abs(boundary_face_vector))

        if boundary_face_vector[axis_idx] > 0:
            # Positive face (+X, +Y, +Z): align to max boundary
            target_coord = max_coords[axis_idx]
        else:
            # Negative face (-X, -Y, -Z): align to min boundary
            target_coord = min_coords[axis_idx]

        # Calculate translation to move inlet to boundary
        translation[axis_idx] = target_coord - rotated_inlet_pos[axis_idx]
        logger.debug(f"Translation vector: {translation}")

        # Apply translation
        translate_stl(temp_stl, translation, output_stl)
        translate_vtp(temp_centerline, translation, output_centerline)

        # Clean up temp files
        Path(temp_stl).unlink()
        Path(temp_centerline).unlink()
    else:
        # No translation needed, rename temp files
        Path(temp_stl).rename(output_stl)
        Path(temp_centerline).rename(output_centerline)

    logger.info("=" * 80)
    logger.info("Geometry rotation and alignment complete")
    logger.info(f"Output STL: {output_stl}")
    logger.info(f"Output centerline: {output_centerline}")
    logger.info("=" * 80)

    return rotation_matrix, translation


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 5:
        print("Usage: python rotate_geometry.py <stl_file> <vtp_file> <target_axis> <output_prefix>")
        print("Example: python rotate_geometry.py vessel.stl centerline.vtp -x rotated_")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    stl_file = sys.argv[1]
    vtp_file = sys.argv[2]
    target_axis = sys.argv[3]
    output_prefix = sys.argv[4]

    output_stl = output_prefix + Path(stl_file).name
    output_vtp = output_prefix + Path(vtp_file).name

    rotate_geometry_to_align_inlet(
        stl_file,
        vtp_file,
        target_axis,
        output_stl,
        output_vtp
    )
