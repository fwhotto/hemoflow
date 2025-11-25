"""Module for outputting rescaled surface meshes in SI units.

This module handles saving transformed STL meshes that match the coordinate
space of the voxelized geometry. The meshes are:
- Shifted so the bounding box minimum is at the origin
- Scaled from millimeters to meters using si_factor

The resulting meshes are in the same coordinate system as the simulation
(origin at min corner, units in meters).
"""

import logging
import numpy as np
from pathlib import Path
from stl import mesh
from typing import Tuple


def get_mesh_bounds(stl_mesh: mesh.Mesh) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the bounding box of an STL mesh.

    Args:
        stl_mesh: The STL mesh object

    Returns:
        Tuple of (mins, maxs) where each is a 3D numpy array
    """
    # Extract all vertices from the mesh
    vertices = stl_mesh.vectors.reshape(-1, 3)

    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)

    return mins, maxs


def transform_and_save_stl(
    input_stl_path: str,
    output_stl_path: str,
    si_factor: float,
    bbox_min: np.ndarray = None
) -> None:
    """Transform STL mesh to SI units and save.

    Applies the following transformations:
    1. Shift: Translate so bounding box minimum is at origin
    2. Scale: Convert from mm to meters using si_factor

    The resulting mesh matches the coordinate system of the voxelized geometry.

    Args:
        input_stl_path: Path to input STL file (in mm, possibly rotated)
        output_stl_path: Path to output STL file (in meters)
        si_factor: Conversion factor from mm to meters (typically 0.001)
        bbox_min: Optional pre-calculated bounding box minimum. If None,
                  will be calculated from the input mesh.
    """
    # Load the STL mesh
    logging.info(f"  Loading STL: {input_stl_path}")
    stl_mesh = mesh.Mesh.from_file(input_stl_path)

    # Calculate bounding box if not provided
    if bbox_min is None:
        bbox_min, _ = get_mesh_bounds(stl_mesh)
        logging.debug(f"    Calculated bounding box min: {bbox_min}")

    # Apply transformations to all vertices
    # STL mesh has shape [n_triangles, 3, 3] (3 vertices per triangle, 3 coords per vertex)
    for i in range(len(stl_mesh.vectors)):
        for j in range(3):  # 3 vertices per triangle
            # Shift: Move bounding box minimum to origin
            stl_mesh.vectors[i][j] -= bbox_min

            # Scale: Convert mm to meters
            stl_mesh.vectors[i][j] *= si_factor

    # Recalculate normals after transformation
    stl_mesh.update_normals()

    # Save as binary STL
    output_path = Path(output_stl_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"  Saving rescaled STL: {output_stl_path}")
    stl_mesh.save(output_stl_path)

    # Log transformation details
    new_min, new_max = get_mesh_bounds(stl_mesh)
    logging.debug(f"    Output bounds: min={new_min}, max={new_max}")
    logging.debug(f"    Output size: {new_max - new_min} meters")


def save_rescaled_meshes(
    config,
    vessel_stl_path: str = None,
    coil_stl_path: str = None
) -> None:
    """Save rescaled meshes for vessel and/or coil.

    This function should be called after apply_geometry_rotation() so that
    the meshes have already been rotated/translated if configured.

    Args:
        config: PreprocessorConfig object with output_meshes settings
        vessel_stl_path: Path to vessel STL (uses config.geometry_stl if None)
        coil_stl_path: Path to coil STL (uses config.coil_stl if None)
    """
    if not config.output_meshes.enabled:
        return

    logging.info("="*60)
    logging.info("Outputting rescaled surface meshes in SI units")
    logging.info("="*60)

    # Use config paths if not explicitly provided
    if vessel_stl_path is None:
        vessel_stl_path = str(config.resolve_path(config.geometry_stl))
    if coil_stl_path is None and config.has_coil:
        coil_stl_path = str(config.resolve_path(config.coil_stl))

    output_dir = config.resolve_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate bounding box from vessel (both vessel and coil should use the same bbox)
    # This ensures they're in the same coordinate space
    vessel_mesh = mesh.Mesh.from_file(vessel_stl_path)
    bbox_min, bbox_max = get_mesh_bounds(vessel_mesh)

    logging.info(f"  Bounding box (mm): min={bbox_min}, max={bbox_max}")
    logging.info(f"  SI factor: {config.si_factor}")
    logging.info(f"  Output directory: {output_dir}")

    # Save vessel mesh
    if config.output_meshes.vasculature:
        output_vessel = output_dir / f"{config.output_base_name}vessel_SI.stl"
        logging.info(f"Saving rescaled vessel mesh:")
        transform_and_save_stl(
            vessel_stl_path,
            str(output_vessel),
            config.si_factor,
            bbox_min
        )

    # Save coil mesh (if present and requested)
    if config.has_coil and config.output_meshes.coil and coil_stl_path:
        output_coil = output_dir / f"{config.output_base_name}coil_SI.stl"
        logging.info(f"Saving rescaled coil mesh:")
        transform_and_save_stl(
            coil_stl_path,
            str(output_coil),
            config.si_factor,
            bbox_min  # Use vessel bbox to ensure same coordinate space
        )

    logging.info("="*60)
    logging.info("")
