import sys
import time
import numpy as np
import json
import os
import argparse
import logging
from dataclasses import dataclass
from typing import Tuple, List, Optional
import scipy.interpolate as scpinter
import vtk
import slice

from config import PreprocessorConfig, DebugConfig
from readCL import getOpeningsFromCenterline, convertToVoxelspace
from voxelizeStl import voxelize
from createFluidSolid import createWalls
from detectOpenings import detectOpenings, paint_inlets_outlets
from vtk.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtk.util import numpy_support


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


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command-line argument parser.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description='HemoFlow geometry preprocessor - voxelizes vessel geometry and prepares simulation input',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py config.json
  python main.py config.json --log-level DEBUG
  python main.py config.json --debug-outputs fluid_only,geometry
  python main.py config.json --no-debug --output-dir output/
        """
    )

    parser.add_argument(
        'config',
        help='Path to JSON configuration file'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )

    parser.add_argument(
        '--debug-outputs',
        help='Comma-separated list of debug outputs to generate: '
             'fluid_only, wall_fluid, geometry, stent_final, stent_linear, stent_quadratic'
    )

    parser.add_argument(
        '--no-debug',
        action='store_true',
        help='Disable all debug outputs'
    )

    parser.add_argument(
        '--output-dir',
        help='Override output directory from config file'
    )

    return parser

def inRange(value, rangeValue, distance):
    if np.abs(rangeValue-value) < distance:
        return True
    return False

def inRange3D(value3D, rangeValue3D, distance):
    isInRange = True
    for i in range(3):
        isInRange = (isInRange and inRange(value3D[i], rangeValue3D[i], distance) )
    
    return isInRange

def generateCutList(voxelDomainSize: Tuple[int, int, int],
                    radiusTangentVoxelList: List,
                    distance: int) -> np.ndarray:
    """Generate list of domain boundaries to cut for openings.

    Args:
        voxelDomainSize: Size of voxelized domain (x, y, z)
        radiusTangentVoxelList: List of (radius, position, tangent) tuples in voxel space
        distance: Distance threshold for opening detection

    Returns:
        Array of boundary indices to cut [0-5] -> [X-, X+, Y-, Y+, Z-, Z+]
    """
    sidesToCut = np.zeros(6)

    logging.debug(f"Generating cutlist for voxel domain size: {voxelDomainSize}")

    for o in radiusTangentVoxelList:
        pos = o[1]
        logging.debug(f"  Centerline point: {pos}")

        for j in range(3):
            if inRange(pos[j], 0, distance):
                sidesToCut[j*2]=1
            if inRange(pos[j], voxelDomainSize[j], distance):
                sidesToCut[j*2+1]=1

    return np.where(sidesToCut == 1)[0]

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


def voxelize_geometry(config: PreprocessorConfig) -> VoxelizationResult:
    """Voxelize vessel geometry from STL file.

    Args:
        config: Preprocessor configuration

    Returns:
        VoxelizationResult with volume and transformation data
    """
    logging.info("Voxelizing vessel geometry")

    vessel_stl = str(config.resolve_path(config.geometry_stl))

    voxelVol, domainData = voxelize(
        vessel_stl,
        config.target_elements,
        target_dx=config.target_dx
    )

    save_debug_file(config, "fluid_only", voxelVol)

    sx, sy, sz = domainData[0]
    tx, ty, tz = domainData[1]

    logging.info(f"  Voxelized domain size: {voxelVol.shape}")
    logging.info(f"  Domain: {domainData[2]}")
    logging.info(f"  Bounding box: {domainData[3]}")
    logging.debug(f"  Transformation - Translate: {tz}, {tx}, {ty}")
    logging.debug(f"  Transformation - Rotate: 90, 0, 90")
    logging.debug(f"  Transformation - Scale: {sz}, {sx}, {sy}")

    dx = (1.0 / sx) * config.si_factor
    logging.info(f"  dx [m]: {dx}")

    return VoxelizationResult(
        volume=voxelVol,
        scale=domainData[0],
        shift=domainData[1],
        domain_size=domainData[2],
        bbox=domainData[3],
        dx=dx
    )


def extract_openings(config: PreprocessorConfig, voxel_result: VoxelizationResult) -> OpeningData:
    """Extract opening information from centerline data.

    Args:
        config: Preprocessor configuration
        voxel_result: Result from voxelization stage

    Returns:
        OpeningData with radius/tangent list and cut list
    """
    logging.info("Extracting opening information from centerline")

    centerline_file = str(config.resolve_path(config.centerline_vtp))
    radius_tangent_list = getOpeningsFromCenterline(centerline_file)

    logging.debug(f"  Scale: {voxel_result.scale}")
    logging.debug(f"  Translate: {voxel_result.shift}")

    radius_tangent_voxel_list = convertToVoxelspace(
        radius_tangent_list,
        voxel_result.scale,
        voxel_result.shift
    )

    cut_list = generateCutList(
        voxel_result.domain_size,
        radius_tangent_voxel_list,
        config.distance
    )

    logging.info(f"  Sides to cut for openings: {cut_list}")

    return OpeningData(
        radius_tangent_list=radius_tangent_voxel_list,
        cut_list=cut_list
    )


def create_walls(config: PreprocessorConfig,
                 volume: np.ndarray,
                 cut_list: np.ndarray) -> Tuple[np.ndarray, List]:
    """Create wall boundaries and cut opening layers.

    Args:
        config: Preprocessor configuration
        volume: Input voxel volume (fluid only)
        cut_list: List of boundary indices to cut

    Returns:
        Tuple of (volume with walls, slicing indices)
    """
    logging.info("Creating walls and opening inlets/outlets")

    vol_with_walls, sliced = createWalls(volume, cut_list, config.cut_width)

    save_debug_file(config, "wall_fluid", vol_with_walls)

    logging.info(f"  Size after cutting layers: {vol_with_walls.shape}")
    volume_count = np.prod(vol_with_walls.shape)
    fluids = np.count_nonzero(vol_with_walls == 2)
    walls = np.count_nonzero(vol_with_walls == 1)

    logging.info(f"  Volume: {volume_count}")
    logging.info(f"  Fluid nodes: {fluids}")
    logging.info(f"  Fluid ratio: {fluids / volume_count:.3f}")
    logging.info(f"  Walls: {walls}")

    return vol_with_walls, sliced


def detect_openings(config: PreprocessorConfig,
                    volume: np.ndarray,
                    opening_data: OpeningData) -> GeometryResult:
    """Detect and label voxelized openings.

    Args:
        config: Preprocessor configuration
        volume: Volume with walls
        opening_data: Opening data from centerline

    Returns:
        GeometryResult with labeled volume and opening metadata
    """
    logging.info("Detecting and assigning voxel openings")

    inlet_outlets, data = detectOpenings(volume)

    # Save intermediate detection result before matching to centerline
    save_debug_file(config, "openings_detected", data)

    # Calculate opening centers from detected voxels
    opening_centers = []
    for io in inlet_outlets:
        oC = np.zeros(3)
        for (x, y, z) in io:
            oC += np.array((z, x, y))
        opening_centers.append(oC / len(io))

    # Validate opening count matches centerline
    if len(opening_centers) != len(opening_data.radius_tangent_list):
        raise ValueError(
            f"Number of voxelized openings ({len(opening_centers)}) differs "
            f"from centerline openings ({len(opening_data.radius_tangent_list)})"
        )

    # Match voxelized openings to centerline data
    opening_index = []
    opening_radius = []
    opening_normalized_q_ratio = []
    opening_center = []
    inlets_outlets_sorted = []
    opening_normal = []

    # Calculate Murray's law flow ratios (Q ∝ r³)
    r3_tot = np.sum([x[0]**3 for x in opening_data.radius_tangent_list[1:]])

    for ccCL in range(len(opening_data.radius_tangent_list)):
        for ccVox in range(len(opening_centers)):
            cVox = opening_centers[ccVox]
            rCL = opening_data.radius_tangent_list[ccCL]
            cCL = rCL[1]

            if inRange3D(cVox, (cCL[0], cCL[1], cCL[2]), config.distance):
                opening_radius.append(rCL[0] * config.si_factor)
                opening_normalized_q_ratio.append(rCL[0]**3 / r3_tot)
                opening_center.append(cVox)
                opening_normal.append(np.array((rCL[2][0], rCL[2][1], rCL[2][2])))
                inlets_outlets_sorted.append(inlet_outlets[ccVox])

    opening_index, opening_centers_final, painted_openings = paint_inlets_outlets(
        inlets_outlets_sorted,
        data,
        findBoundaryByArea=False
    )

    save_debug_file(config, "geometry", painted_openings)

    if len(opening_index) != len(opening_centers_final):
        logging.warning(
            f"Number of matched openings is incorrect: {len(opening_index)} "
            f"instead of {len(opening_centers_final)}"
        )

    return GeometryResult(
        volume=painted_openings,
        opening_index=opening_index,
        opening_radius=opening_radius,
        opening_normalized_q_ratio=opening_normalized_q_ratio,
        opening_center=opening_center,
        opening_normal=opening_normal
    )


def process_stent(config: PreprocessorConfig,
                  voxel_result: VoxelizationResult,
                  sliced: List,
                  cut_list: np.ndarray) -> StentResult:
    """Voxelize flow diverter geometry using 3-projection method.

    Args:
        config: Preprocessor configuration
        voxel_result: Result from vessel voxelization
        sliced: Slicing indices from wall creation
        cut_list: List of boundary indices to cut

    Returns:
        StentResult with stent volume and resistance coefficients
    """
    logging.info("Voxelizing flow diverter geometry from 3 projections")

    stent_geom_base = str(config.resolve_path(config.stent_mesh_base))
    stent_geom_file = stent_geom_base + "mesh.stl"

    domain_data = (voxel_result.scale, voxel_result.shift, voxel_result.domain_size, voxel_result.bbox)

    # Voxelize from 3 different projections
    logging.info("  Projection #1")
    voxelStent, _ = voxelize(stent_geom_file, config.target_elements, True, domain_data)

    logging.info("  Projection #2")
    voxelStent2, _ = voxelize(stent_geom_file, config.target_elements, True, domain_data, 0)

    logging.info("  Projection #3")
    voxelStent3, _ = voxelize(stent_geom_file, config.target_elements, True, domain_data, 1)

    logging.info("  Processing stent interpolation")
    xg, yg, zg = np.mgrid[0:voxelStent3.shape[0], 0:voxelStent3.shape[1], 0:voxelStent3.shape[2]]

    stent_voxel_linear = 1
    stent_voxel_quadratic = 1

    # Handle inhomogeneous resistance coefficients if available
    if config.inhomogen:
        values_file = stent_geom_base + "values.vtp"
        if os.path.isfile(values_file):
            logging.info("  Loading inhomogeneous resistance values")
            reader = vtkXMLPolyDataReader()
            reader.SetFileName(values_file)
            reader.Update()
            vtpdata = dsa.WrapDataObject(reader.GetOutput())
            stent_points = vtpdata.GetPoints()

            stent_linear = vtpdata.PointData['linearCoeff']
            stent_quadratic = vtpdata.PointData['quadraticCoeff']

            stent_points = scaleAndShiftData(stent_points, voxel_result.scale, voxel_result.shift)

            stent_voxel_linear = scpinter.griddata(stent_points, stent_linear, (xg, yg, zg), method="nearest")
            stent_voxel_quadratic = scpinter.griddata(stent_points, stent_quadratic, (xg, yg, zg), method="nearest")
            logging.info("  Linear and quadratic coefficients interpolated")
        else:
            logging.warning(f"  Inhomogeneous resistance file not found: {values_file}")

    # Merge projections
    logging.info("  Merging projections")
    sdomain_full = np.logical_or(np.logical_or(voxelStent, voxelStent2), voxelStent3)
    linear = stent_voxel_linear * sdomain_full
    quadratic = stent_voxel_quadratic * sdomain_full

    # Apply slicing
    sdomain_full = sdomain_full[sliced[0]:sliced[1], sliced[2]:sliced[3], sliced[4]:sliced[5]]
    linear_full = linear[sliced[0]:sliced[1], sliced[2]:sliced[3], sliced[4]:sliced[5]]
    quadratic_full = quadratic[sliced[0]:sliced[1], sliced[2]:sliced[3], sliced[4]:sliced[5]]

    # Apply cutting for openings
    if 0 in cut_list:
        sdomain_full = sdomain_full[config.cut_width:, :, :]
        linear_full = linear_full[config.cut_width:, :, :]
        quadratic_full = quadratic_full[config.cut_width:, :, :]
    if 1 in cut_list:
        sdomain_full = sdomain_full[:-config.cut_width, :, :]
        linear_full = linear_full[:-config.cut_width, :, :]
        quadratic_full = quadratic_full[:-config.cut_width, :, :]
    if 2 in cut_list:
        sdomain_full = sdomain_full[:, config.cut_width:, :]
        linear_full = linear_full[:, config.cut_width:, :]
        quadratic_full = quadratic_full[:, config.cut_width:, :]
    if 3 in cut_list:
        sdomain_full = sdomain_full[:, :-config.cut_width, :]
        linear_full = linear_full[:, :-config.cut_width, :]
        quadratic_full = quadratic_full[:, :-config.cut_width, :]
    if 4 in cut_list:
        sdomain_full = sdomain_full[:, :, config.cut_width:]
        linear_full = linear_full[:, :, config.cut_width:]
        quadratic_full = quadratic_full[:, :, config.cut_width:]
    if 5 in cut_list:
        sdomain_full = sdomain_full[:, :, :-config.cut_width]
        linear_full = linear_full[:, :, :-config.cut_width]
        quadratic_full = quadratic_full[:, :, :-config.cut_width]

    logging.info(f"  Linear coefficients range: [{np.nanmin(linear_full)}, {np.nanmax(linear_full)}]")
    logging.info(f"  Quadratic coefficients range: [{np.nanmin(quadratic_full)}, {np.nanmax(quadratic_full)}]")
    logging.info(f"  Flow diverter domain size: {sdomain_full.shape}")

    save_debug_file(config, "stent_final", sdomain_full.astype(np.short, copy=False))
    save_debug_file(config, "stent_linear", linear_full.astype(np.single, copy=False))
    save_debug_file(config, "stent_quadratic", quadratic_full.astype(np.single, copy=False))

    return StentResult(
        volume=sdomain_full,
        linear=linear_full,
        quadratic=quadratic_full
    )


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


def main() -> None:
    """Main preprocessing pipeline."""
    # Parse command-line arguments
    parser = setup_argparse()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Load configuration with CLI overrides
    cli_overrides = {
        'output_dir': args.output_dir,
        'debug_outputs': args.debug_outputs,
        'no_debug': args.no_debug,
    }

    try:
        config = PreprocessorConfig.load_from_json(args.config, cli_overrides)
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Configuration error: {e}")
        sys.exit(1)

    logging.info("="*60)
    logging.info("HemoFlow Geometry Preprocessor")
    logging.info("="*60)

    start_time = time.time()

    try:
        # Pipeline execution
        voxel_result = voxelize_geometry(config)

        opening_data = extract_openings(config, voxel_result)

        wall_volume, sliced = create_walls(config, voxel_result.volume, opening_data.cut_list)

        geometry_result = detect_openings(config, wall_volume, opening_data)

        stent_result = None
        if config.has_stent:
            stent_result = process_stent(config, voxel_result, sliced, opening_data.cut_list)

        save_geometry(config, geometry_result, voxel_result, stent_result)

        # Report completion
        elapsed = int(round(time.time() - start_time))
        logging.info("="*60)
        logging.info(f"Preprocessing completed successfully in {elapsed}s")
        logging.info("="*60)

    except Exception as e:
        logging.error(f"Preprocessing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
