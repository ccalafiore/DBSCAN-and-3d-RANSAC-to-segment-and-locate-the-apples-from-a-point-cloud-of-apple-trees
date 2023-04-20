


import typing
import os
import numpy as np
import open3d as o3d
import tools
import calapy as cp
cp.initiate(['txt'])


dir_project = os.getcwd()

with_trees = True
# with_trees = False


max_colors_1 = True
# max_colors_1 = False

coordinates_np, colors_np = tools.load_points(
    dir_project=dir_project, with_trees=with_trees, max_colors_1=max_colors_1, delimiter=' ')

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coordinates_np)
pcd.colors = o3d.utility.Vector3dVector(colors_np)

del coordinates_np, colors_np

# o3d.visualization.draw_geometries([pcd])

step_coordinates = 0.01

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=pcd, voxel_size=step_coordinates)
voxels = voxel_grid.get_voxels()

indices = np.stack(list(vx.grid_index for vx in voxels), axis=0)
max_indices = np.max(indices, axis=0).tolist()  # type: typing.Union[list, np.ndarray, tuple, object]
shape_image_3d = [m + 1 for m in max_indices] + [3]
image_3d = np.full(shape=shape_image_3d, fill_value=0, dtype='f')

# coordinates = np.empty(shape=shape_image_3d, fill_value=0, dtype='f')

indices_vx = np.empty([4], dtype='O')  # type: typing.Union[list, np.ndarray, tuple, object]
indices_vx[3] = slice(0, 3, 1)
for vx in voxels:
    indices_vx[slice(0, 3, 1)] = vx.grid_index.tolist()
    image_3d[tuple(indices_vx.tolist())] = vx.color

max_bound = voxel_grid.get_max_bound()
min_bound = voxel_grid.get_min_bound()

tools.show_2d_slides(image=(image_3d * 255).astype('i'), axis=0)

# o3d.visualization.draw_geometries([voxel_grid])
