"""Geometry utility functions for boundary detection and coordinate transformations."""

import logging
from typing import Tuple, List, Optional
import numpy as np


def inRange(value: float, rangeValue: float, distance: float) -> bool:
    """Check if value is within distance of rangeValue.

    Args:
        value: Value to check
        rangeValue: Reference value
        distance: Maximum allowed distance

    Returns:
        True if within range
    """
    if np.abs(rangeValue - value) < distance:
        return True
    return False


def inRange3D(value3D: np.ndarray, rangeValue3D: np.ndarray, distance: float) -> bool:
    """Check if 3D point is within distance of reference point.

    Args:
        value3D: 3D point to check
        rangeValue3D: Reference 3D point
        distance: Maximum allowed distance per dimension

    Returns:
        True if within range in all dimensions
    """
    isInRange = True
    for i in range(3):
        isInRange = (isInRange and inRange(value3D[i], rangeValue3D[i], distance))

    return isInRange


def select_face_from_normal(tangent: np.ndarray) -> int:
    """Select boundary face based on tangent/normal vector.

    The tangent vector points along the centerline (into the vessel at outlets),
    so we select the face OPPOSITE to the tangent direction.

    Args:
        tangent: Normalized tangent vector along centerline direction

    Returns:
        Face index [0-5] -> [X-, X+, Y-, Y+, Z-, Z+]
    """
    abs_tangent = np.abs(tangent)
    dominant_axis = np.argmax(abs_tangent)

    # Tangent points INTO vessel, so opening is on OPPOSITE face
    # Negative tangent -> opening on positive face, positive tangent -> opening on negative face
    if tangent[dominant_axis] > 0:
        return dominant_axis * 2      # X-, Y-, or Z- (opposite of positive tangent)
    else:
        return dominant_axis * 2 + 1  # X+, Y+, or Z+ (opposite of negative tangent)


def select_closest_face(pos: np.ndarray, domain_size: Tuple[int, int, int]) -> int:
    """Select boundary face based on closest distance.

    Args:
        pos: Position in voxel space (x, y, z)
        domain_size: Domain dimensions (x, y, z)

    Returns:
        Face index [0-5] -> [X-, X+, Y-, Y+, Z-, Z+]
    """
    face_distances = [
        pos[0],                    # Distance to X- (face 0)
        domain_size[0] - pos[0],   # Distance to X+ (face 1)
        pos[1],                    # Distance to Y- (face 2)
        domain_size[1] - pos[1],   # Distance to Y+ (face 3)
        pos[2],                    # Distance to Z- (face 4)
        domain_size[2] - pos[2],   # Distance to Z+ (face 5)
    ]
    return int(np.argmin(face_distances))


def check_corner_proximity(pos: np.ndarray,
                           domain_size: Tuple[int, int, int],
                           threshold: float) -> dict:
    """Check if opening is near a corner or edge.

    Args:
        pos: Position in voxel space
        domain_size: Domain dimensions
        threshold: Distance threshold for "close to boundary"

    Returns:
        Dictionary with corner detection info
    """
    face_distances = [
        pos[0], domain_size[0] - pos[0],
        pos[1], domain_size[1] - pos[1],
        pos[2], domain_size[2] - pos[2],
    ]
    close_faces = [i for i, d in enumerate(face_distances) if d < threshold]

    return {
        'is_corner': len(close_faces) > 1,
        'close_faces': close_faces,
        'num_close': len(close_faces),
        'min_distance': min(face_distances)
    }


def get_opening_face(center: np.ndarray, volume_shape: Tuple[int, int, int], threshold: float = 2.0) -> Optional[str]:
    """Determine which boundary face an opening is on.

    Args:
        center: Opening center coordinates (x, y, z)
        volume_shape: Shape of volume (nx, ny, nz)
        threshold: Distance threshold from boundary (in voxels)

    Returns:
        Face identifier ('X-', 'X+', 'Y-', 'Y+', 'Z-', 'Z+') or None
    """
    nx, ny, nz = volume_shape
    x, y, z = center

    faces = []
    if x < threshold:
        faces.append('X-')
    if x > nx - threshold:
        faces.append('X+')
    if y < threshold:
        faces.append('Y-')
    if y > ny - threshold:
        faces.append('Y+')
    if z < threshold:
        faces.append('Z-')
    if z > nz - threshold:
        faces.append('Z+')

    # Return the closest face if multiple
    if len(faces) == 1:
        return faces[0]
    elif len(faces) > 1:
        # Calculate distances to each face
        distances = []
        for face in faces:
            if face == 'X-':
                distances.append(x)
            elif face == 'X+':
                distances.append(nx - x)
            elif face == 'Y-':
                distances.append(y)
            elif face == 'Y+':
                distances.append(ny - y)
            elif face == 'Z-':
                distances.append(z)
            elif face == 'Z+':
                distances.append(nz - z)
        return faces[np.argmin(distances)]
    return None


def generateCutList(voxelDomainSize: Tuple[int, int, int],
                    radiusTangentVoxelList: List,
                    distance: int,
                    use_normals: bool = True) -> np.ndarray:
    """Generate list of domain boundaries to cut for openings.

    Uses tangent vectors to intelligently select ONE face per opening,
    avoiding corner/edge detection issues.

    Args:
        voxelDomainSize: Size of voxelized domain (x, y, z)
        radiusTangentVoxelList: List of (radius, position, tangent) tuples in voxel space
        distance: Distance threshold for validation
        use_normals: If True, use tangent vectors; else use closest face

    Returns:
        Array of boundary indices to cut [0-5] -> [X-, X+, Y-, Y+, Z-, Z+]
    """
    sidesToCut = np.zeros(6)
    face_names = ['X-', 'X+', 'Y-', 'Y+', 'Z-', 'Z+']

    logging.debug(f"Generating cutlist for voxel domain size: {voxelDomainSize}")
    logging.debug(f"  Using {'normal vectors' if use_normals else 'closest face'} for face selection")

    for idx, o in enumerate(radiusTangentVoxelList):
        radius, pos, tangent = o
        logging.info(f"  Opening {idx}: position={pos}, tangent={tangent}, |tangent|={np.linalg.norm(tangent):.3f}")

        # Select face using hybrid method
        if use_normals and np.linalg.norm(tangent) > 0.1:
            # Try normal-based selection
            normal_face = select_face_from_normal(tangent)
            closest_face = select_closest_face(pos, voxelDomainSize)

            # Calculate distance to normal-selected face
            axis = normal_face // 2
            if normal_face % 2 == 0:  # Negative face
                dist_to_normal_face = pos[axis]
            else:  # Positive face
                dist_to_normal_face = voxelDomainSize[axis] - pos[axis]

            # Use normal-based if opening is reasonably close to that face
            # Otherwise use closest face (more reliable for corner cases)
            if dist_to_normal_face < distance * 3:
                selected_face = normal_face
                method = "normal"
            else:
                selected_face = closest_face
                method = "closest (normal face too far)"
                logging.debug(
                    f"  Opening {idx}: Normal suggests face {normal_face} but it's {dist_to_normal_face:.1f} voxels away. "
                    f"Using closest face {closest_face} instead."
                )
        else:
            selected_face = select_closest_face(pos, voxelDomainSize)
            method = "closest"
            if use_normals and np.linalg.norm(tangent) <= 0.1:
                logging.warning(
                    f"  Opening {idx}: Tangent vector too small (|tangent|={np.linalg.norm(tangent):.3f}), "
                    f"falling back to closest face"
                )

        # Validate opening position
        corner_info = check_corner_proximity(pos, voxelDomainSize, distance)

        if corner_info['is_corner']:
            close_face_names = [face_names[i] for i in corner_info['close_faces']]
            logging.warning(
                f"  Opening {idx} at {pos} is near corner/edge "
                f"(close to {corner_info['num_close']} faces: {close_face_names}). "
                f"Selected face {face_names[selected_face]} using {method} method."
            )

        # Calculate distance to selected face
        axis = selected_face // 2
        if selected_face % 2 == 0:  # Negative face
            dist_to_face = pos[axis]
        else:  # Positive face
            dist_to_face = voxelDomainSize[axis] - pos[axis]

        if dist_to_face > distance * 1.5:
            logging.warning(
                f"  Opening {idx} is {dist_to_face:.1f} voxels from selected face {face_names[selected_face]}, "
                f"but distance threshold is {distance}. Consider increasing distance parameter."
            )

        sidesToCut[selected_face] = 1
        logging.debug(f"    -> Selected face: {face_names[selected_face]} (index {selected_face})")

    cut_faces = np.where(sidesToCut == 1)[0]
    cut_face_names = [face_names[i] for i in cut_faces]
    logging.info(f"  Faces to cut: {cut_face_names} (indices: {cut_faces})")

    return cut_faces


def scaleAndShiftData(points: List, scale: Tuple, shift: Tuple) -> List:
    """Transform points from physical space to voxel space.

    Args:
        points: List of 3D points
        scale: Scale factors (x, y, z)
        shift: Shift offsets (x, y, z)

    Returns:
        Transformed points
    """
    for i in range(len(points)):
        pts = points[i]
        for j in range(3):
            pts[j] = (pts[j] + shift[j]) * scale[j]
        points[i] = pts
    return points


def apply_boundary_cuts(volume: np.ndarray,
                       cut_list: np.ndarray,
                       cut_width: int = 1) -> np.ndarray:
    """Apply boundary layer cuts for opening creation.

    Cuts specified layers from domain boundaries.
    Face indices: 0=X-, 1=X+, 2=Y-, 3=Y+, 4=Z-, 5=Z+

    Note: Cuts are applied sequentially, so shape changes affect subsequent cuts.

    Args:
        volume: Input volume array
        cut_list: Array of face indices to cut
        cut_width: Number of layers to cut from each face

    Returns:
        Volume with boundary layers removed
    """
    result = volume.copy()

    if 0 in cut_list:
        result = result[cut_width:, :, :]
    if 1 in cut_list:
        result = result[:-cut_width, :, :]
    if 2 in cut_list:
        result = result[:, cut_width:, :]
    if 3 in cut_list:
        result = result[:, :-cut_width, :]
    if 4 in cut_list:
        result = result[:, :, cut_width:]
    if 5 in cut_list:
        result = result[:, :, :-cut_width]

    return result
