from enum import IntEnum

class VoxelLabels(IntEnum):
    """
    Defines a set of integer labels for categorizing voxel types.
    """
    UNUSED = 0
    WALL = 1
    FLUID = 2
    INLET = 10
    OUTLET = 11
    OUTLET_REST = 12
