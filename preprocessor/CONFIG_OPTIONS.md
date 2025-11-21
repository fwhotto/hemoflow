# Preprocessor Configuration Options

## Required Parameters

### `geometry_original_stl` (string)
Path to vessel STL file. Typically in millimeters.
- **Example**: `"input/vessel.stl"`
- **Relative to**: Config file location (or absolute path)

### `centerline_vtp` (string)
Path to VTP file containing centerline data for opening detection.
- **Example**: `"input/centerline.vtp"`
- **Relative to**: Config file location (or absolute path)

### Resolution (choose ONE)

#### `target_elements` (string)
Target total voxel count. The voxelizer will adjust resolution to approximate this count.
- **Example**: `"1000000"` (1 million voxels)
- **Usage**: Set this OR `target_dx`, not both
- **Leave empty** if using `target_dx`

#### `target_dx` (string)
Direct voxel size specification in millimeters.
- **Example**: `"0.2"` (0.2mm voxel size)
- **Usage**: Set this OR `target_elements`, not both
- **Leave empty** if using `target_elements`

---

## Optional Parameters

### `stent_mesh_base` (string, default: "")
Path prefix for stent/flow diverter STL files. The preprocessor will look for `{prefix}mesh.stl`.
- **Example**: `"input/stent_"` (looks for `input/stent_mesh.stl`)
- **Leave empty**: If no stent/flow diverter
- **Special**: If `inhomogen=true`, also looks for `{prefix}values.vtp` for resistance coefficients

### `output_base_name` (string, default: "geometry_")
Prefix for all output files.
- **Example**: `"my_vessel_"`
- **Generates**: `my_vessel_c.npz`, `my_vessel_fluid_only.nrrd`, etc.

### `output_dir` (string, default: ".")
Directory for output files.
- **Example**: `"output/"` or `"../results/"`
- **Relative to**: Config file location (or absolute path)
- **Note**: Directory must exist or be created

### `cutWidth` (string, default: "1")
Number of voxel layers to cut at each opening boundary.
- **Example**: `"1"` (typical), `"2"` (for thick padding)
- **Range**: Positive integer
- **Purpose**: Ensures openings are exposed at domain boundaries

### `distance` (string, default: "4")
Opening detection threshold in voxels. Centerline endpoints within this distance of a boundary are considered openings.
- **Example**: `"4"` (default), `"8"` (more tolerant), `"12"` (very tolerant)
- **Range**: Positive integer
- **Too small**: May miss real openings
- **Too large**: May create false cuts on multiple boundaries (especially for corner openings)

### `si_factor` (string, default: "0.001")
Conversion factor from STL units to meters.
- **Example**: `"0.001"` (STL in mm → meters), `"0.01"` (STL in cm → meters)
- **Default assumes**: STL files are in millimeters

### `inhomogen` (boolean, default: false)
Use inhomogeneous stent resistance coefficients from VTP file.
- **Example**: `true` or `false`
- **Requires**: Stent VTP file at `{stent_mesh_base}values.vtp` with `linearCoeff` and `quadraticCoeff` point data
- **Purpose**: Spatially varying resistance for realistic stent modeling

---

## Debug Configuration (optional section)

### `debug.enabled` (boolean, default: false)
Master switch for debug outputs.
- **Example**: `true` or `false`
- **Overridden by**: `--no-debug` or `--debug-outputs` CLI flags

### `debug.outputs` (array of strings, default: [])
Specific debug outputs to generate. Empty array = all outputs when enabled.
- **Available outputs**:
  - `"fluid_only"` - Voxelized vessel before wall creation
  - `"wall_fluid"` - After wall boundary creation
  - `"openings_detected"` - Raw detected openings before centerline matching (**useful for debugging mismatches**)
  - `"geometry"` - Final geometry with opening labels
  - `"stent_final"` - Flow diverter geometry
  - `"stent_linear"` - Linear resistance coefficients
  - `"stent_quadratic"` - Quadratic resistance coefficients
- **Example**: `["fluid_only", "geometry", "openings_detected"]`
- **Empty array**: Generates all outputs when `debug.enabled=true`

### `debug.output_dir` (string, default: ".")
Directory for debug NRRD files.
- **Example**: `"debug/"` (separate from main outputs)
- **Relative to**: Config file location (or absolute path)

---

## Command-Line Overrides

These CLI arguments override config file settings:

```bash
python main.py config.json [OPTIONS]

Options:
  --log-level {DEBUG,INFO,WARNING,ERROR}
      Set logging verbosity (default: INFO)

  --debug-outputs fluid_only,geometry,openings_detected
      Enable specific debug outputs (comma-separated, no spaces)
      Automatically enables debug mode

  --no-debug
      Disable all debug outputs (overrides config)

  --output-dir path/
      Override output directory from config
```

---

## Example Configurations

### Minimal Config
```json
{
  "geometry_original_stl": "vessel.stl",
  "centerline_vtp": "centerline.vtp",
  "target_elements": "1000000",
  "target_dx": "",
  "output_base_name": "geometry_",
  "cutWidth": "1",
  "distance": "4"
}
```

### With Stent
```json
{
  "geometry_original_stl": "vessel.stl",
  "centerline_vtp": "centerline.vtp",
  "stent_mesh_base": "stent_",
  "target_elements": "2000000",
  "target_dx": "",
  "output_base_name": "vessel_with_stent_",
  "cutWidth": "1",
  "distance": "8",
  "inhomogen": true
}
```

### With Full Debug
```json
{
  "geometry_original_stl": "vessel.stl",
  "centerline_vtp": "centerline.vtp",
  "target_elements": "1000000",
  "target_dx": "",
  "output_base_name": "geometry_",
  "cutWidth": "1",
  "distance": "4",
  "debug": {
    "enabled": true,
    "outputs": ["fluid_only", "openings_detected", "geometry"],
    "output_dir": "debug/"
  }
}
```

### Using Direct Resolution
```json
{
  "geometry_original_stl": "vessel.stl",
  "centerline_vtp": "centerline.vtp",
  "target_elements": "",
  "target_dx": "0.15",
  "output_base_name": "high_res_",
  "cutWidth": "1",
  "distance": "6"
}
```

---

## Common Usage Patterns

### Quick Run (defaults)
```bash
python main.py config.json
```

### Debug Opening Mismatch
```bash
python main.py config.json --debug-outputs openings_detected --log-level DEBUG
```

### Full Debug Output
```bash
python main.py config.json --debug-outputs fluid_only,wall_fluid,openings_detected,geometry --log-level DEBUG
```

### High Verbosity, No Debug Files
```bash
python main.py config.json --log-level DEBUG --no-debug
```

---

## Troubleshooting

### "Number of voxelized openings differs from centerline openings"
1. Run with `--debug-outputs openings_detected` to visualize detected openings
2. Check if centerline endpoints are at **corners** (near multiple boundaries)
3. Adjust `distance` parameter:
   - Increase (e.g., 12) if openings aren't being matched
   - Decrease (e.g., 4) if too many boundaries are being cut
4. Verify geometry: openings should be on face centers, not corners

### Resolution Too Coarse
- Increase `target_elements` (e.g., 2000000, 3000000)
- Or use `target_dx` with smaller value (e.g., "0.1")

### Stent Not Loading
- Verify `{stent_mesh_base}mesh.stl` exists
- For inhomogeneous resistance, verify `{stent_mesh_base}values.vtp` exists

### Files Not Found
- Check paths are relative to config file location
- Or use absolute paths
- Verify files exist before running
