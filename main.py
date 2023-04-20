


import os
import typing
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

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(coordinates_np)
# pcd.colors = o3d.utility.Vector3dVector(colors_np)
# o3d.visualization.draw_geometries([pcd])
# pdb = None

coordinates_np, colors_np = tools.preprocess(
    points=coordinates_np, colors=colors_np, nb_points=10, radius=.005, voxel_size=None)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(colors_np)
# pcd.colors = o3d.utility.Vector3dVector(colors_np)

# o3d.visualization.draw_geometries([pcd_some_apples])
print('cluster_dbscan started ...')
timer = cp.clock.Timer()

labels = np.array(pcd.cluster_dbscan(eps=0.004, min_points=40, print_progress=True))

time = timer.get_seconds()
print('cluster_dbscan completed {:.1f} in seconds'.format(time))

max_label = labels.max()
K = max_label + 2

# to show the different clusters uncomment below
# for k in range(0, K, 1):
#
#     logical_inliers = labels == k - 1
#     logical_outliers = np.logical_not(logical_inliers)
#
#     n_points = colors_np.shape[0]
#
#     n_inliers = np.sum(logical_inliers, axis=0)
#     perc_inliers = n_inliers / n_points * 100
#     print('perc_inliers = {:.1f} %'.format(perc_inliers))
#
#     coordinates_np_k = coordinates_np[logical_inliers, :]
#     colors_np_k = colors_np[logical_inliers, :]
#
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(coordinates_np_k)
#     pcd.colors = o3d.utility.Vector3dVector(colors_np_k)
#
#     o3d.visualization.draw_geometries([pcd])

logical_inliers = labels == -1
coordinates_np = coordinates_np[logical_inliers, :]
colors_np = colors_np[logical_inliers, :]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coordinates_np)
pcd.colors = o3d.utility.Vector3dVector(colors_np)

pcd, indexes = pcd.remove_radius_outlier(nb_points=5, radius=0.005)

coordinates_np = np.asarray(pcd.points)
colors_np = np.asarray(pcd.colors)

# o3d.visualization.draw_geometries([pcd])


step_coordinates = 0.005

n_rhos = 3
min_rhos = 0.03
max_rhos = 0.05

distance_threshold = 0.005

threshold_proposes = 0.50

threshold_supp = 0.27

sphere_parameters, inlines, outliers = tools.faster_ransac_multi_sphere_detection_from_point_cloud(
    coordinates_np=coordinates_np, min_rhos=min_rhos, max_rhos=max_rhos,
    threshold_dist=distance_threshold, coord_precision=step_coordinates,
    n_rhos=n_rhos, threshold_prop=threshold_proposes, threshold_supp=threshold_supp)


dir_results = os.path.join(dir_project, 'results')
os.makedirs(name=dir_results, exist_ok=True)

dir_result_points = os.path.join(dir_results, 'apple_points.txt')
dir_result_spheres = os.path.join(dir_results, 'apple_spheres.txt')

apple_points = np.concatenate(
    [coordinates_np[inlines, :].astype('U'), cp.maths.rint(colors_np[inlines, :] * 255).astype('U')],
    axis=1).tolist()  # type: typing.Union[list, np.ndarray, object]

tools.save_points(points=apple_points, directory=dir_result_points, headers=None, delimiter=' ')

headers = ['x_centers', 'y_centers', 'z_centers', 'rhos']
tools.save_points(points=sphere_parameters.tolist(), directory=dir_result_spheres, headers=headers, delimiter=' ')


time = timer.get_seconds()
print('All Completed in {:.1f} seconds'.format(time))


out_pcd = o3d.geometry.PointCloud()
out_pcd.points = o3d.utility.Vector3dVector(coordinates_np[outliers, :])
out_pcd.colors = o3d.utility.Vector3dVector(colors_np[outliers, :])
# o3d.visualization.draw_geometries([out_pcd])

in_pcd = o3d.geometry.PointCloud()
in_pcd.points = o3d.utility.Vector3dVector(coordinates_np[inlines, :])
in_pcd.colors = o3d.utility.Vector3dVector(colors_np[inlines, :])
o3d.visualization.draw_geometries([in_pcd])

# We create a visualizer object that will contain references to the created window, the 3D objects and will listen to callbacks for key presses, animations, etc.
vis = o3d.visualization.Visualizer()
# New window, where we can set the name, the width and height, as well as the position on the screen
vis.create_window(window_name='Apples Replaced with the found Spheres with the found Rhos - 0.01 m', width=1400, height=1000)

# Get options for the renderer for visualizing the coordinate system and changing the background
opt = vis.get_render_option()
opt.show_coordinate_frame = True
opt.background_color = np.asarray([0.2, 0.2, 0.2])

vis.add_geometry(out_pcd)

for parm_a in sphere_parameters:

    # We can easily create primitives like cubes, sphere, cylinders, etc. In our case we create a sphere and specify its radius
    sphere_mesh_a = o3d.geometry.TriangleMesh.create_sphere(radius=parm_a[3] - 0.01)
    sphere_mesh_a.paint_uniform_color([0.2, 0.2, 1.0])
    sphere_mesh_a.translate(translation=(parm_a[0], parm_a[1], parm_a[2]), relative=False)

    # Add the sphere to the visualizer
    vis.add_geometry(sphere_mesh_a)

# We run the visualizer
vis.run()
# Once the visualizer is closed destroy the window and clean up
vis.destroy_window()


