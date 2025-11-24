"""Configuration management for the HemoFlow preprocessor.

This module provides dataclasses for type-safe configuration management,
replacing the previous JSON-based string configuration with typed fields
and validation.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set


@dataclass
class DebugConfig:
    """Configuration for debug outputs.

    Attributes:
        enabled: Whether debug outputs are enabled
        outputs: Set of debug file names to generate (fluid_only, wall_fluid,
                 geometry, coil, stent_final, stent_linear, stent_quadratic)
        output_dir: Directory for debug files (relative to config file or absolute)
    """
    enabled: bool = False
    outputs: Set[str] = field(default_factory=set)
    output_dir: str = "."

    def should_save(self, output_name: str) -> bool:
        """Check if a specific debug output should be saved."""
        return self.enabled and (not self.outputs or output_name in self.outputs)


@dataclass
class RotationConfig:
    """Configuration for geometry rotation preprocessing.

    Attributes:
        enabled: Whether to apply geometry rotation
        inlet_target_axis: Target axis for inlet alignment (e.g., '-x', '+y', '-z')
        inlet_centerline_index: Index of inlet in centerline (0 = first line start)
        position_at_boundary: If True, translate inlet to bounding box face
    """
    enabled: bool = True
    inlet_target_axis: str = "-x"
    inlet_centerline_index: int = 0
    position_at_boundary: bool = True

    def __post_init__(self):
        """Validate rotation configuration."""
        if self.enabled:
            # Validate axis format
            axis_lower = self.inlet_target_axis.lower().strip()
            if len(axis_lower) != 2 or axis_lower[0] not in ['+', '-'] or axis_lower[1] not in ['x', 'y', 'z']:
                raise ValueError(
                    f"Invalid inlet_target_axis '{self.inlet_target_axis}'. "
                    f"Expected format: '+x', '-y', etc."
                )
            if self.inlet_centerline_index < 0:
                raise ValueError(f"inlet_centerline_index must be non-negative, got {self.inlet_centerline_index}")


@dataclass
class PreprocessorConfig:
    """Configuration for the HemoFlow preprocessor pipeline.

    Attributes:
        geometry_stl: Path to vessel STL file (typically in mm units)
        centerline_vtp: Path to VTP file with centerline data
        stent_mesh_base: Path prefix for stent STL files (empty if no stent)
        coil_stl: Path to coil STL file (empty if no coil)
        target_elements: Target voxel count (mutually exclusive with target_dx)
        target_dx: Voxel size in mm (mutually exclusive with target_elements)
        output_base_name: Prefix for output NPZ and debug files
        output_dir: Directory for output files
        cut_width: Number of voxel layers to cut at openings
        distance: Opening detection threshold in voxels
        si_factor: Conversion factor from mm to meters (default: 0.001)
        inhomogen: Use inhomogeneous stent resistance (default: False)
        use_normal_for_face_selection: Use tangent vectors to select boundary faces (default: True)
        rotation: Rotation configuration for aligning inlet with bounding box
        debug: Debug configuration
        config_dir: Directory containing the config file (for resolving relative paths)
    """
    geometry_stl: str
    centerline_vtp: str
    stent_mesh_base: str = ""
    coil_stl: str = ""
    target_elements: Optional[int] = None
    target_dx: Optional[float] = None
    output_base_name: str = "geometry_"
    output_dir: str = "."
    cut_width: int = 1
    distance: int = 4
    si_factor: float = 0.001
    inhomogen: bool = False
    use_normal_for_face_selection: bool = False
    rotation: RotationConfig = field(default_factory=RotationConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    config_dir: str = "."

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate mutually exclusive resolution parameters
        if self.target_elements is None and self.target_dx is None:
            raise ValueError("Must specify either target_elements or target_dx")
        if self.target_elements is not None and self.target_dx is not None:
            raise ValueError("Cannot specify both target_elements and target_dx")

        # Validate numeric values
        if self.target_elements is not None and self.target_elements <= 0:
            raise ValueError(f"target_elements must be positive, got {self.target_elements}")
        if self.target_dx is not None and self.target_dx <= 0:
            raise ValueError(f"target_dx must be positive, got {self.target_dx}")
        if self.cut_width <= 0:
            raise ValueError(f"cut_width must be positive, got {self.cut_width}")
        if self.distance <= 0:
            raise ValueError(f"distance must be positive, got {self.distance}")
        if self.si_factor <= 0:
            raise ValueError(f"si_factor must be positive, got {self.si_factor}")

        # Validate file existence
        geometry_path = self.resolve_path(self.geometry_stl)
        if not geometry_path.exists():
            raise FileNotFoundError(f"Geometry STL file not found: {geometry_path}")

        centerline_path = self.resolve_path(self.centerline_vtp)
        if not centerline_path.exists():
            raise FileNotFoundError(f"Centerline VTP file not found: {centerline_path}")

        # Stent is optional, but if specified, check it exists
        if self.stent_mesh_base:
            # Check for at least one stent file (with suffix like _0.stl)
            stent_dir = self.resolve_path(self.stent_mesh_base).parent
            stent_prefix = self.resolve_path(self.stent_mesh_base).name
            if stent_dir.exists():
                stent_files = list(stent_dir.glob(f"{stent_prefix}*.stl"))
                if not stent_files:
                    raise FileNotFoundError(f"No stent STL files found matching: {self.stent_mesh_base}*.stl")

        # Coil is optional, but if specified, check it exists
        if self.coil_stl:
            coil_path = self.resolve_path(self.coil_stl)
            if not coil_path.exists():
                raise FileNotFoundError(f"Coil STL file not found: {coil_path}")

    def resolve_path(self, path: str) -> Path:
        """Resolve a path relative to the config file directory.

        Args:
            path: Path string (absolute or relative)

        Returns:
            Resolved Path object
        """
        p = Path(path)
        if p.is_absolute():
            return p
        return (Path(self.config_dir) / p).resolve()

    @property
    def has_stent(self) -> bool:
        """Check if stent processing is enabled."""
        return bool(self.stent_mesh_base)

    @property
    def has_coil(self) -> bool:
        """Check if coil processing is enabled."""
        return bool(self.coil_stl)

    @classmethod
    def load_from_json(cls, config_path: str, cli_overrides: Optional[dict] = None) -> "PreprocessorConfig":
        """Load configuration from JSON file with optional CLI overrides.

        Args:
            config_path: Path to JSON configuration file
            cli_overrides: Optional dictionary of CLI argument overrides

        Returns:
            PreprocessorConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, 'r') as f:
            data = json.load(f)

        # Store config directory for resolving relative paths
        config_dir = str(config_file.parent)

        # Convert JSON data to typed fields
        config_kwargs = {
            'geometry_stl': data.get('geometry_original_stl', ''),
            'centerline_vtp': data.get('centerline_vtp', ''),
            'stent_mesh_base': data.get('stent_mesh_base', ''),
            'coil_stl': data.get('coil_stl', ''),
            'output_base_name': data.get('output_base_name', 'geometry_'),
            'cut_width': int(data.get('cutWidth', '1')),
            'distance': int(data.get('distance', '4')),
            'config_dir': config_dir,
        }

        # Handle resolution parameters (convert empty strings to None)
        target_elements_str = data.get('target_elements', '')
        target_dx_str = data.get('target_dx', '')

        if target_elements_str and target_elements_str.strip():
            config_kwargs['target_elements'] = int(target_elements_str)
        if target_dx_str and target_dx_str.strip():
            config_kwargs['target_dx'] = float(target_dx_str)

        # Handle optional parameters
        if 'si_factor' in data:
            config_kwargs['si_factor'] = float(data['si_factor'])
        if 'inhomogen' in data:
            config_kwargs['inhomogen'] = bool(data['inhomogen'])
        if 'output_dir' in data:
            config_kwargs['output_dir'] = data['output_dir']
        if 'use_normal_for_face_selection' in data:
            config_kwargs['use_normal_for_face_selection'] = bool(data['use_normal_for_face_selection'])

        # Handle rotation configuration
        rotation_config = RotationConfig()
        if 'rotation' in data:
            rotation_data = data['rotation']
            rotation_config.enabled = rotation_data.get('enabled', False)
            if 'inlet_target_axis' in rotation_data:
                rotation_config.inlet_target_axis = rotation_data['inlet_target_axis']
            if 'inlet_centerline_index' in rotation_data:
                rotation_config.inlet_centerline_index = int(rotation_data['inlet_centerline_index'])
            if 'position_at_boundary' in rotation_data:
                rotation_config.position_at_boundary = bool(rotation_data['position_at_boundary'])
        config_kwargs['rotation'] = rotation_config

        # Handle debug configuration
        debug_config = DebugConfig()
        if 'debug' in data:
            debug_data = data['debug']
            debug_config.enabled = debug_data.get('enabled', False)
            if 'outputs' in debug_data:
                debug_config.outputs = set(debug_data['outputs'])
            if 'output_dir' in debug_data:
                debug_config.output_dir = debug_data['output_dir']
        config_kwargs['debug'] = debug_config

        # Apply CLI overrides
        if cli_overrides:
            for key, value in cli_overrides.items():
                if value is not None:  # Only override if explicitly set
                    if key == 'debug_outputs':
                        # Special handling for debug outputs
                        if value:
                            config_kwargs['debug'].outputs = set(value.split(','))
                            config_kwargs['debug'].enabled = True
                    elif key == 'no_debug':
                        if value:
                            config_kwargs['debug'].enabled = False
                    elif key in config_kwargs:
                        config_kwargs[key] = value

        return cls(**config_kwargs)
