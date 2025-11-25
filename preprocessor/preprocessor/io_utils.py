"""I/O utilities for saving geometry and debug visualization."""

import logging
from pathlib import Path
from typing import Optional
import numpy as np

# Type imports
from .config import PreprocessorConfig
from .models import VoxelizationResult, GeometryResult, StentResult

# PyVista import for debug visualization
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    logging.warning("PyVista not available - debug visualizations will be skipped")


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the preprocessor.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    logging.basicConfig(
        level=numeric_level,
        format='%(levelname)s: %(message)s'
    )


def save_debug_file(config: PreprocessorConfig, name: str, data: np.ndarray) -> None:
    """Save debug output file if enabled.

    Args:
        config: Preprocessor configuration
        name: Debug output name (e.g., 'fluid_only', 'geometry')
        data: Numpy array to save
    """
    if not config.debug.should_save(name):
        return

    try:
        import nrrd
    except ImportError:
        logging.warning("nrrd module not available, skipping debug output")
        return

    output_path = config.resolve_path(config.debug.output_dir) / f"{config.output_base_name}{name}.nrrd"
    logging.debug(f"Saving debug file: {output_path}")
    nrrd.write(str(output_path), data)


def save_geometry(config: PreprocessorConfig,
                  geometry_result: GeometryResult,
                  voxel_result: VoxelizationResult,
                  stent_result: Optional[StentResult] = None) -> None:
    """Save final geometry to compressed NPZ file.

    Args:
        config: Preprocessor configuration
        geometry_result: Final geometry with openings
        voxel_result: Voxelization result (for dx)
        stent_result: Optional stent result
    """
    logging.info("Saving final output")

    output_path = config.resolve_path(config.output_dir) / f"{config.output_base_name}c.npz"
    logging.info(f"  File: {output_path}")

    # Common data for all outputs
    common_data = {
        'geometryFlag': geometry_result.volume,
        'dx': np.array([voxel_result.dx]).astype(np.double, copy=False),
        'openingIndex': np.array(geometry_result.opening_index).astype(np.short, copy=False),
        'openingRadius': np.array(geometry_result.opening_radius).astype(np.double, copy=False),
        'openingNormalizedQRatio': np.array(geometry_result.opening_normalized_q_ratio).astype(np.double, copy=False),
        'openingCenter': np.array(geometry_result.opening_center).astype(np.double, copy=False),
        'openingNormal': np.array(geometry_result.opening_normal).astype(np.double, copy=False),
    }

    # Add stent data if available
    if stent_result:
        common_data['stent'] = stent_result.volume.astype(np.short, copy=False)
        common_data['linear'] = stent_result.linear.astype(np.int32, copy=False)
        common_data['quadratic'] = stent_result.quadratic.astype(np.int32, copy=False)

    np.savez_compressed(str(output_path), **common_data)


def visualize_geometry_debug(config: PreprocessorConfig, title: str, **meshes) -> None:
    """Create interactive PyVista visualization for debugging geometry transformations.

    Args:
        config: Preprocessor configuration
        title: Window title describing the current step
        **meshes: Named meshes to visualize. Can be:
            - STL file path (str): Loads and displays mesh
            - PyVista mesh object: Displays directly
            - NumPy boolean array: Converts voxels to mesh
            - Tuple (voxel_array, dx, shift): Voxels with spacing

    Example:
        visualize_geometry_debug(config, "After Rotation",
                                vessel="vessel_rotated.stl",
                                coil="coil_rotated.stl")
    """
    if not PYVISTA_AVAILABLE:
        logging.debug(f"Skipping visualization '{title}' - PyVista not available")
        return

    if not config.debug.enabled:
        return

    logging.info(f"[DEBUG VIZ] {title}")

    plotter = pv.Plotter()
    plotter.add_text(title, position='upper_edge', font_size=12, color='black')

    colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
    color_idx = 0

    for name, data in meshes.items():
        if data is None:
            continue

        try:
            # Handle different data types
            if isinstance(data, str):
                # STL file path
                mesh = pv.read(data)
                logging.info(f"  {name}: {data} (points: {mesh.n_points})")
            elif isinstance(data, tuple) and len(data) == 3:
                # (voxel_array, dx, shift)
                voxel_array, dx, shift = data
                if isinstance(voxel_array, np.ndarray) and voxel_array.dtype == bool:
                    # Create uniform grid from voxels
                    grid = pv.ImageData(dimensions=voxel_array.shape)
                    grid.spacing = (dx, dx, dx)
                    grid.origin = shift
                    grid.point_data['values'] = voxel_array.flatten(order='F')
                    # Extract surface
                    mesh = grid.contour([0.5])
                    logging.info(f"  {name}: voxel array {voxel_array.shape}, dx={dx}")
                else:
                    continue
            elif isinstance(data, np.ndarray) and data.dtype == bool:
                # Boolean voxel array without spacing
                grid = pv.ImageData(dimensions=data.shape)
                grid.point_data['values'] = data.flatten(order='F')
                mesh = grid.contour([0.5])
                logging.info(f"  {name}: voxel array {data.shape}")
            else:
                # Assume it's already a PyVista mesh
                mesh = data
                logging.info(f"  {name}: mesh object")

            # Add mesh to plotter
            color = colors[color_idx % len(colors)]
            plotter.add_mesh(mesh, color=color, opacity=0.7, label=name)
            color_idx += 1

        except Exception as e:
            logging.warning(f"  Could not visualize {name}: {e}")

    plotter.add_legend()
    plotter.add_axes()
    plotter.show_bounds()
    plotter.show()
