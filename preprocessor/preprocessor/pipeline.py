"""Pipeline orchestration for HemoFlow preprocessor."""

import logging
import time
from typing import Tuple, List, Optional
from pathlib import Path
import numpy as np
import os
import scipy.interpolate as scpinter
from vtk.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader

# Import configuration and models
from .config import PreprocessorConfig
from .constants import VoxelLabels
from .models import VoxelizationResult, OpeningData, GeometryResult, StentResult

# Import processing modules
from .voxelization import voxelize
from .centerline import getOpeningsFromCenterline, convertToVoxelspace
from .wall_creation import createWalls
from .opening_detection import detectOpenings, paint_inlets_outlets
from .geometry import generateCutList, inRange3D, scaleAndShiftData, get_opening_face
from .io_utils import save_debug_file, save_geometry
from . import rotation as rotate_geometry
from . import mesh_output


def apply_geometry_rotation(config: PreprocessorConfig) -> None:
    """Apply geometry rotation if enabled in configuration.

    This function rotates both the STL and centerline VTP files to align the inlet
    with a specified bounding box face. The rotated files are saved with '_rotated'
    suffix, and the config is updated to use the rotated files.

    Args:
        config: Preprocessor configuration

    Modifies:
        config.geometry_stl: Updated to rotated STL path if rotation is applied
        config.centerline_vtp: Updated to rotated VTP path if rotation is applied
    """
    if not config.rotation.enabled:
        return

    logging.info("="*60)
    logging.info("Applying geometry rotation preprocessing")
    logging.info("="*60)

    # Resolve input paths
    stl_path = config.resolve_path(config.geometry_stl)
    vtp_path = config.resolve_path(config.centerline_vtp)

    # Generate output paths with '_rotated' suffix
    output_dir = config.resolve_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stl_name = stl_path.stem
    vtp_name = vtp_path.stem
    output_stl = str(output_dir / f"{stl_name}_rotated{stl_path.suffix}")
    output_vtp = str(output_dir / f"{vtp_name}_rotated{vtp_path.suffix}")

    logging.info(f"  Input STL: {stl_path}")
    logging.info(f"  Input VTP: {vtp_path}")
    logging.info(f"  Target axis: {config.rotation.inlet_target_axis}")
    logging.info(f"  Inlet centerline index: {config.rotation.inlet_centerline_index}")
    logging.info(f"  Position at boundary: {config.rotation.position_at_boundary}")

    # Perform rotation
    rotation_matrix, translation = rotate_geometry.rotate_geometry_to_align_inlet(
        str(stl_path),
        str(vtp_path),
        config.rotation.inlet_target_axis,
        output_stl,
        output_vtp,
        inlet_index=config.rotation.inlet_centerline_index,
        position_at_boundary=config.rotation.position_at_boundary
    )

    # Update config to use rotated files
    config.geometry_stl = output_stl
    config.centerline_vtp = output_vtp

    logging.info(f"  Rotated STL saved to: {output_stl}")
    logging.info(f"  Rotated VTP saved to: {output_vtp}")

    # Apply same rotation and translation to coil if present
    if config.has_coil:
        logging.info("  Applying same transformation to coil geometry")
        coil_path = config.resolve_path(config.coil_stl)
        coil_name = coil_path.stem
        output_coil = str(output_dir / f"{coil_name}_rotated{coil_path.suffix}")

        # Create temporary rotated coil
        temp_coil = str(output_dir / f"{coil_name}_temp{coil_path.suffix}")
        rotate_geometry.rotate_stl(str(coil_path), rotation_matrix, temp_coil)

        # Apply translation if used
        if config.rotation.position_at_boundary:
            rotate_geometry.translate_stl(temp_coil, translation, output_coil)
            Path(temp_coil).unlink()  # Clean up temp file
        else:
            Path(temp_coil).rename(output_coil)

        # Update config to use rotated coil
        config.coil_stl = output_coil
        logging.info(f"  Rotated coil saved to: {output_coil}")

    logging.info("="*60)


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
        config.distance,
        use_normals=config.use_normal_for_face_selection
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

    # If rotation was applied, reorder openings to put the inlet first based on target face
    if config.rotation.enabled:
        # Map target axis to face identifier
        axis_to_face_map = {
            '-x': 'X-', '+x': 'X+',
            '-y': 'Y-', '+y': 'Y+',
            '-z': 'Z-', '+z': 'Z+'
        }
        target_face = axis_to_face_map.get(config.rotation.inlet_target_axis.lower())

        if target_face:
            # Find which opening is on the target inlet face
            inlet_idx = None
            for i, center in enumerate(opening_center):
                face = get_opening_face(center, volume.shape, threshold=5.0)
                logging.info(f"  Opening {i}: center={center}, face={face}, radius={opening_radius[i]:.6f}m")
                if face == target_face:
                    inlet_idx = i
                    logging.info(f"  → Identified as INLET (on target face {target_face})")

            # Reorder to put inlet first
            if inlet_idx is not None and inlet_idx != 0:
                logging.info(f"  Reordering: moving opening {inlet_idx} to position 0 (inlet)")
                # Swap inlet to first position
                for lst in [opening_radius, opening_normalized_q_ratio, opening_center,
                           opening_normal, inlets_outlets_sorted]:
                    lst[0], lst[inlet_idx] = lst[inlet_idx], lst[0]
            elif inlet_idx is None:
                logging.warning(f"  Could not find opening on target inlet face {target_face}")

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


def process_coil(config: PreprocessorConfig,
                 geometry_result: GeometryResult,
                 voxel_result: VoxelizationResult,
                 sliced: List,
                 cut_list: np.ndarray) -> np.ndarray:
    """Voxelize coil geometry and mark as wall using 3-projection method.

    Args:
        config: Preprocessor configuration
        geometry_result: Current geometry with openings
        voxel_result: Result from vessel voxelization
        sliced: Slicing indices from wall creation
        cut_list: List of boundary indices to cut

    Returns:
        Updated geometry volume with coil marked as wall (flag=1)
    """
    logging.info("Voxelizing coil geometry from 3 projections")

    coil_stl = str(config.resolve_path(config.coil_stl))
    domain_data = (voxel_result.scale, voxel_result.shift, voxel_result.domain_size, voxel_result.bbox)

    # Voxelize from 3 different projections
    logging.info("  Projection #1")
    voxelCoil1, _ = voxelize(coil_stl, config.target_elements, True, domain_data)
    logging.info(f"    Shape: {voxelCoil1.shape}, Voxels: {np.count_nonzero(voxelCoil1)}")

    logging.info("  Projection #2")
    voxelCoil2, _ = voxelize(coil_stl, config.target_elements, True, domain_data, 0)
    logging.info(f"    Shape: {voxelCoil2.shape}, Voxels: {np.count_nonzero(voxelCoil2)}")

    logging.info("  Projection #3")
    voxelCoil3, _ = voxelize(coil_stl, config.target_elements, True, domain_data, 1)
    logging.info(f"    Shape: {voxelCoil3.shape}, Voxels: {np.count_nonzero(voxelCoil3)}")

    # Merge projections
    logging.info("  Merging projections")
    coil_domain = np.logical_or(np.logical_or(voxelCoil1, voxelCoil2), voxelCoil3)
    logging.info(f"    Merged shape: {coil_domain.shape}, Voxels: {np.count_nonzero(coil_domain)}")

    # Apply slicing
    logging.info(f"  Slicing indices: {sliced}")
    coil_domain = coil_domain[sliced[0]:sliced[1], sliced[2]:sliced[3], sliced[4]:sliced[5]]
    logging.info(f"    After slicing shape: {coil_domain.shape}, Voxels: {np.count_nonzero(coil_domain)}")

    # Apply cutting for openings
    logging.info(f"  Cutting for openings (cut_list: {cut_list})")
    if 0 in cut_list:
        coil_domain = coil_domain[config.cut_width:, :, :]
        logging.info(f"    Cut face 0 (X-), shape: {coil_domain.shape}")
    if 1 in cut_list:
        coil_domain = coil_domain[:-config.cut_width, :, :]
        logging.info(f"    Cut face 1 (X+), shape: {coil_domain.shape}")
    if 2 in cut_list:
        coil_domain = coil_domain[:, config.cut_width:, :]
        logging.info(f"    Cut face 2 (Y-), shape: {coil_domain.shape}")
    if 3 in cut_list:
        coil_domain = coil_domain[:, :-config.cut_width, :]
        logging.info(f"    Cut face 3 (Y+), shape: {coil_domain.shape}")
    if 4 in cut_list:
        coil_domain = coil_domain[:, :, config.cut_width:]
        logging.info(f"    Cut face 4 (Z-), shape: {coil_domain.shape}")
    if 5 in cut_list:
        coil_domain = coil_domain[:, :, :-config.cut_width]
        logging.info(f"    Cut face 5 (Z+), shape: {coil_domain.shape}")

    logging.info(f"    After cutting shape: {coil_domain.shape}, Voxels: {np.count_nonzero(coil_domain)}")

    # Save debug output
    save_debug_file(config, "coil", coil_domain.astype(np.short, copy=False))

    # Check geometry volume shape
    logging.info(f"  Geometry volume shape: {geometry_result.volume.shape}")
    logging.info(f"  Coil domain shape: {coil_domain.shape}")

    if geometry_result.volume.shape != coil_domain.shape:
        logging.error(f"  ERROR: Shape mismatch! geometry={geometry_result.volume.shape} vs coil={coil_domain.shape}")
        return geometry_result.volume

    # Mark coil voxels as WALL (flag=1) in geometry
    geometry_volume = geometry_result.volume.copy()

    # Count how many voxels will be changed
    coil_voxel_count = np.count_nonzero(coil_domain)
    logging.info(f"  Marking {coil_voxel_count} coil voxels as wall (flag=1)")

    # Check what values exist at coil positions before
    coil_positions_before = geometry_volume[coil_domain]
    unique_before, counts_before = np.unique(coil_positions_before, return_counts=True)
    logging.info(f"  Values at coil positions before: {dict(zip(unique_before, counts_before))}")

    # Mark coil as wall
    geometry_volume[coil_domain] = VoxelLabels.WALL

    # Check what values exist at coil positions after
    coil_positions_after = geometry_volume[coil_domain]
    unique_after, counts_after = np.unique(coil_positions_after, return_counts=True)
    logging.info(f"  Values at coil positions after: {dict(zip(unique_after, counts_after))}")

    # Count total walls in final geometry
    wall_count = np.count_nonzero(geometry_volume == VoxelLabels.WALL)
    logging.info(f"  Total wall voxels in final geometry: {wall_count}")

    return geometry_volume


def run_preprocessing_pipeline(config: PreprocessorConfig) -> None:
    """Execute the complete preprocessing pipeline.

    Args:
        config: Preprocessor configuration

    Raises:
        Exception: If any pipeline stage fails
    """
    logging.info("="*60)
    logging.info("HemoFlow Geometry Preprocessor")
    logging.info("="*60)

    start_time = time.time()

    # Step 1: Apply geometry rotation if enabled
    apply_geometry_rotation(config)

    # Step 1.5: Save rescaled meshes in SI units (if enabled)
    mesh_output.save_rescaled_meshes(config)

    # Step 2: Voxelize geometry
    voxel_result = voxelize_geometry(config)

    # Step 3: Extract openings from centerline
    opening_data = extract_openings(config, voxel_result)

    # Step 4: Create walls
    wall_volume, sliced = create_walls(config, voxel_result.volume, opening_data.cut_list)

    # Step 5: Detect and label openings
    geometry_result = detect_openings(config, wall_volume, opening_data)

    # Step 6: Process stent if configured
    stent_result = None
    if config.has_stent:
        stent_result = process_stent(config, voxel_result, sliced, opening_data.cut_list)

    # Step 7: Process coil if configured
    if config.has_coil:
        geometry_result.volume = process_coil(config, geometry_result, voxel_result,
                                               sliced, opening_data.cut_list)
        # Save final geometry with coil for visualization
        save_debug_file(config, "geometry_with_coil", geometry_result.volume.astype(np.short, copy=False))

    # Step 8: Save final geometry
    save_geometry(config, geometry_result, voxel_result, stent_result)

    # Report completion
    elapsed = int(round(time.time() - start_time))
    logging.info("="*60)
    logging.info(f"Preprocessing completed successfully in {elapsed}s")
    logging.info("="*60)

