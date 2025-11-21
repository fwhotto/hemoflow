# LBM_preprocess tools for HemoFlow

## Usage

See main.py for execution arguments.

Input: JSON config file
Output: voxelized geometry

Notes: 
- The nrrd output is saved since it can be opened in 3DSlicer to investigate the results of the voxelisation.
- The centerline must contain one line per outlet, with each line running from the inlet (source) to one outlet (target)
.
### Specify sides that should contain an opening.
This will practically cut away a layer of voxels from these sides.

Cutlist meaning -> cut one layer from the planes:
    # 0,1 => Xmin, Xmax
    # 2,3 => Ymin, Ymax
    # 4,5 => Zmin, Zmax

### Dependencies

Tested with Anaconda and Python 3.7

- pynrrd (PIP/conda-forge)
- numpy-stl (PIP/conda-forge)

### Known problems

- The triangle size cannot be smaller than the projected voxel size. I.e.: If you see walls appearing inside the fluid domain, or get complaints that the thing is not watertight (but you are sure it is), increase the resolution (also reduce the number of faces on the geometry?).

- The inlet and the pressure outlet are automatically detected as the largest and smallest are opening. Due to the various cut angles, this can be incorrect! Look for a better method based on centerline!

- Also automate the cutlist based on the centerline!

## Configuration Parameters

The preprocessor is configured using a JSON file. Below is a comprehensive description of all available parameters:

### Required Parameters

#### `geometry_original_stl` (string)
- **Description**: Path to the vessel geometry STL file
- **Units**: Typically in millimeters (mm)
- **Example**: `"input/full_closed.stl"`
- **Notes**: Path is relative to the JSON config file location. The geometry is automatically converted from mm to meters using SI_FACTOR (0.001).

#### `centerline_vtp` (string)
- **Description**: Path to the centerline VTP file used for opening detection
- **Format**: VTK PolyData (.vtp)
- **Example**: `"input/centerline_model.vtp"`
- **Requirements**:
  - Must contain one line per outlet
  - Each line runs from inlet (source) to one outlet (target)
  - Used to extract opening positions, radii, tangents, and flow ratios

#### `output_base_name` (string)
- **Description**: Prefix for all output files
- **Example**: `"geometry_"`
- **Output**: Creates files like `geometry_c.npz` (compressed NPZ with all geometry data)
- **Debug outputs** (if DEBUG_MODE enabled):
  - `<prefix>fluid_only.nrrd`: Voxelized vessel before wall creation
  - `<prefix>wall_fluid.nrrd`: After wall boundary creation
  - `<prefix>geometry.nrrd`: Final geometry with opening labels
  - `<prefix>stent_final.nrrd`: Flow diverter geometry (if applicable)

### Resolution Parameters (Use ONE of the following)

#### `target_elements` (string/integer)
- **Description**: Target number of voxel elements in the domain
- **Example**: `"100000"` or `"1000000"`
- **Notes**:
  - Use this OR `target_dx`, NOT both
  - Higher values = finer resolution = more memory/computation time
  - Typical range: 100,000 to 5,000,000 elements
  - Leave `target_dx` as empty string `""` when using this parameter

#### `target_dx` (string/float)
- **Description**: Target voxel size in millimeters
- **Units**: Millimeters (mm)
- **Example**: `"0.1"` for 0.1 mm voxel size
- **Notes**:
  - Use this OR `target_elements`, NOT both
  - Directly specifies spatial resolution
  - Leave `target_elements` as empty string `""` when using this parameter
  - Converted to meters in output (multiplied by SI_FACTOR)

### Flow Diverter/Stent Parameters (Optional)

#### `stent_mesh_base` (string)
- **Description**: Base path/prefix for stent mesh files
- **Example**: `"stent_"` (looks for `stent_mesh.stl`)
- **Format**: Expects `<base>mesh.stl` file
- **Notes**:
  - Leave as empty string `""` if no stent/flow diverter
  - If file exists, preprocessor voxelizes stent from 3 orthogonal projections
  - Can also use `stent_folder` parameter (alternative method)

#### `stent_folder` (string) - Alternative to `stent_mesh_base`
- **Description**: Folder containing stent mesh file with specific naming convention
- **Naming**: Looks for `<stent_folder>_<directory_name>_stent_mesh.stl`
- **Example**: If `"stent_folder": "coiling"` and working in directory `case01/`, looks for `case01/coiling/coiling_case01_stent_mesh.stl`
- **Inhomogeneous coefficients**: Can also load `<base>values.vtp` with spatially varying linearCoeff and quadraticCoeff point data
- **Notes**: Use this OR `stent_mesh_base`, NOT both

### Opening Detection Parameters

#### `cutWidth` (string/integer)
- **Description**: Number of voxel layers to cut away at domain boundaries where openings are detected
- **Default**: `"1"` (currently hardcoded in main.py line 77, but appears in config)
- **Example**: `"1"` or `"2"`
- **Notes**:
  - Cuts away padding layers to create clean openings
  - May need to set to `"2"` if there is more than 1 padding layer
  - **Important**: This parameter is currently NOT read from the config file and uses hardcoded value. Users should modify line 77 in main.py if different value needed.

#### `distance` (string/integer)
- **Description**: Distance threshold (in voxels) for detecting if a centerline point is near a boundary
- **Default**: `"4"` (currently hardcoded in main.py line 78, but appears in config)
- **Example**: `"4"` or `"8"`
- **Notes**:
  - If centerline endpoint is within this distance from domain boundary, it's considered an opening
  - Larger values = more tolerant opening detection
  - Typical range: 4-8 voxels
  - **Important**: This parameter is currently NOT read from the config file and uses hardcoded value. Users should modify line 78 in main.py if different value needed.

### Example Configurations

#### Basic vessel without stent:
```json
{
  "geometry_original_stl": "input/vessel.stl",
  "centerline_vtp": "input/centerline.vtp",
  "stent_mesh_base": "",
  "target_elements": "1000000",
  "target_dx": "",
  "output_base_name": "geometry_",
  "cutWidth": "1",
  "distance": "4"
}
```

#### Vessel with flow diverter using target_dx:
```json
{
  "geometry_original_stl": "input/aneurysm.stl",
  "centerline_vtp": "input/centerline.vtp",
  "stent_mesh_base": "stent_",
  "target_elements": "",
  "target_dx": "0.08",
  "output_base_name": "geometry_",
  "cutWidth": "1",
  "distance": "8"
}
```

#### Using stent_folder parameter:
```json
{
  "geometry_original_stl": "input/vessel.stl",
  "centerline_vtp": "input/centerline.vtp",
  "stent_folder": "coiling",
  "target_elements": "500000",
  "target_dx": "",
  "output_base_name": "geometry_",
  "cutWidth": "1",
  "distance": "4"
}
```

### Hardcoded Parameters in main.py

These parameters are defined in `main.py` and can only be changed by editing the source code:

#### `SI_FACTOR` (line 20)
- **Default**: `0.001`
- **Description**: Conversion factor from STL units to meters
- **Note**: Most STL files are in millimeters, so 0.001 converts mm → m

#### `DEBUG_MODE` (line 22)
- **Default**: `False`
- **Description**: Enables intermediate NRRD output files for debugging
- **Output files when True**:
  - `*_fluid_only.nrrd`: Voxelized vessel geometry
  - `*_wall_fluid.nrrd`: Geometry with walls
  - `*_geometry.nrrd`: Final geometry with labeled openings
  - `*_stent_final.nrrd`: Voxelized stent (if present)
  - `*_stent_linear.nrrd` and `*_stent_quadratic.nrrd`: Coefficient fields
- **Viewing**: Open NRRD files in 3D Slicer for inspection

#### `INHOMOGEN` (line 23)
- **Default**: `False`
- **Description**: Enables spatially inhomogeneous porous media coefficients
- **Requirements**: When `True`, looks for `<stent_base>values.vtp` with point data arrays:
  - `linearCoeff`: Linear Darcy resistance coefficient
  - `quadraticCoeff`: Quadratic Forchheimer coefficient
- **Note**: Coefficients are interpolated to voxel grid using nearest-neighbor method

### Output NPZ File Contents

The preprocessor generates a compressed NPZ file containing:

- `geometryFlag`: 3D array with geometry labels (see Flag table below)
- `dx`: Voxel resolution in meters
- `openingIndex`: Array of opening labels (10, 11, 12, ...)
- `openingRadius`: Opening radii in meters
- `openingNormalizedQRatio`: Flow distribution ratios (Murray's law, Q ∝ R³)
- `openingCenter`: Opening center coordinates in voxel space
- `openingNormal`: Opening normal vectors
- `stent`: 3D boolean array of stent/flow diverter geometry (if present)
- `linear`: Linear resistance coefficient field (if INHOMOGEN enabled)
- `quadratic`: Quadratic resistance coefficient field (if INHOMOGEN enabled)

## Flag table

- 0 unused
- 1 wall
- 2 fluid
- 3 porous
- 10 inlet (velocity)
- 11 outlet 1 (pressure)
- 12... other outlets (velocity)

