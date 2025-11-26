# import os.path
# import sys
import logging
import numpy as np
from typing import Tuple

from . import slice
from . import perimeter
# import multiprocessing
from .util import padVoxelArray

from stl import mesh

def voxelize(inputFile, targetElements, isMeshAShell = False, domainData = None, rotation =-1, target_dx=None):
    mesh = list(import_stl_file(inputFile))

    if domainData is None:
        scale, shift, domain, bounding_box = slice.calculateScaleAndShift(mesh, targetElements, target_dx=target_dx)
    else:
        scale, shift, domain, bounding_box = domainData

    mesh = list(slice.scaleAndShiftMesh(mesh, scale, shift))
    #Note: vol should be addressed with vol[z][x][y]
    
    # If it is a shell mesh, rotate here for different projections!
    if rotation == 0:
        mesh = list(slice.swapAxis(mesh, 0, 1))
        domain = [domain[1], domain[0], domain[2]]
    elif rotation == 1:
        mesh = list(slice.swapAxis(mesh, 1, 2))
        domain = [domain[0], domain[2], domain[1]]

    vol = np.zeros((domain[2],domain[0],domain[1]), dtype=bool)

    ## Parallel version
    # procs = []
    # manager = multiprocessing.Manager()
    # return_dict = manager.dict()
    # #pool = multiprocessing.Pool(multiprocessing.cpu_count())
    # pool = multiprocessing.Pool(2)
    # for height in range(int(domain[2])):
    #     p = multiprocessing.Process(
    #         target=render_slice,
    #         args=(mesh, height, domain, return_dict, isMeshAShell)
    #         )
    #     procs.append(p)
    #     p.start()

    # pool.close()

    # for p in procs:
    #     p.join()
    # d = return_dict
    
    ## Serial for debugging
    d={}
    for height in range(int(domain[2])):
        render_slice(mesh, height, domain, d, isMeshAShell)

    for key, value in d.items():
        vol[key] = value

    vol, domain = padVoxelArray(vol)

    # If it is a shell mesh, rotate back here the different projections!
    if rotation == 0:
        vol = np.swapaxes(vol, 1, 2)
    elif rotation == 1:
        #mesh = slice.swapAxisWithZ(mesh, 1)
        vol = np.swapaxes(vol, 0, 2)


    # At the very end here fix the [z][x][y] back to [x][y][z]
    vol = np.swapaxes(vol, 0, 2)
    vol = np.swapaxes(vol, 0, 1)

    return (vol, (scale, shift, domain, bounding_box))

def render_slice(mesh, height, domain, return_dict, isMeshAShell):
    lines = slice.toIntersectingLines(mesh, height)
    prepixel = np.zeros((domain[0], domain[1]), dtype=bool)
    perimeter.linesToVoxels(lines, prepixel, isMeshAShell)
    return_dict[height] = prepixel

def import_stl_file(inputFile):
    imported_mesh = mesh.Mesh.from_file(inputFile)
    return get_stl_faces(imported_mesh)

def get_stl_faces(mesh):
    for i, j, k in zip(mesh.v0, mesh.v1, mesh.v2):
        yield (tuple(i), tuple(j), tuple(k))


def voxelize_with_three_projections(
    stl_file: str,
    target_elements: int,
    domain_data: Tuple,
    geometry_name: str = "geometry"
) -> np.ndarray:
    """Voxelize geometry using 3 orthogonal projections with logical OR merge.

    This method improves coverage for thin structures by voxelizing from
    three different orientations and merging the results.

    Args:
        stl_file: Path to STL file
        target_elements: Target number of voxel elements
        domain_data: Tuple of (scale, shift, domain_size, bbox)
        geometry_name: Name for logging (e.g., "flow diverter", "coil")

    Returns:
        Merged boolean volume from 3 projections
    """
    logging.info(f"  Voxelizing {geometry_name} from 3 projections")

    logging.info("    Projection #1 (default)")
    vol1, _ = voxelize(stl_file, target_elements, True, domain_data)

    logging.info("    Projection #2 (rotation 0)")
    vol2, _ = voxelize(stl_file, target_elements, True, domain_data, 0)

    logging.info("    Projection #3 (rotation 1)")
    vol3, _ = voxelize(stl_file, target_elements, True, domain_data, 1)

    logging.info("    Merging projections")
    merged = np.logical_or(np.logical_or(vol1, vol2), vol3)

    return merged


if __name__ == '__main__':
    #voxelize('Files/Mesh_30000_faces/Mesh_21000.stl', 'Files/meshmesh.nrrd', pow(189, 3))
    #voxelize('test2.stl', 'test2.nrrd', pow(200,3))

    import sys
    if len(sys.argv) < 4:
        print("Usage:", sys.argv[0], "input.nrrd output.nrrd targetElementNum")        
        sys.exit(-1)

    from nrrd import write

    outputVolume, domainData = voxelize(sys.argv[1], int(sys.argv[3])) 

    write(sys.argv[2], outputVolume)