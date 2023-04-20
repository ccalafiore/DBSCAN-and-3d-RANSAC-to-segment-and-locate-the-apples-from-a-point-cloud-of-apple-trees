


import os
import glob
import math
import numpy as np
import open3d as o3d
import calapy as cp
cp.initiate(['txt', 'clock'])


def load_points(dir_project, with_trees=True, max_colors_1=False, delimiter=','):

    timer = cp.clock.Timer()
    print('loading data ...')

    dir_data = os.path.join(dir_project, 'dataset')
    dir_trees = os.path.join(dir_data, 'apple_trees.txt')
    dir_root_apples = os.path.join(dir_data, 'some_apples')
    dirs_apples = glob.glob(os.path.join(dir_root_apples, '2018_01_*.txt'))
    # dirs_apples = dirs_apples[slice(0, 3, 1)]

    if with_trees:
        points_np = cp.txt.csv_file_to_array(
            filename=dir_trees, rows=slice(0, None, 1), columns=slice(0, 6, 1), delimiter=delimiter, dtype='f', encoding=None)
    else:
        n_apples = len(dirs_apples)

        points_np = [cp.txt.csv_file_to_array(
            filename=dirs_apples[a], rows=slice(0, None, 1), columns=slice(0, 6, 1), delimiter=delimiter, dtype='f', encoding=None)
            for a in range(0, n_apples, 1)]

        points_np = np.concatenate(points_np, axis=0)

    coordinates_np = points_np[:, slice(0, 3, 1)]
    colors_np = points_np[:, slice(3, 6, 1)].astype('i')

    del points_np

    if max_colors_1:
        colors_np = colors_np / 255

    time = timer.get_seconds()
    print('loading completed in {sec:.1f} seconds'.format(sec=time))

    return coordinates_np, colors_np


def preprocess(points, colors, nb_points=None, radius=None, voxel_size=None):

    timer = cp.clock.Timer()
    print('preprocessing data ...')

    indexes = points[:, 2] > 0
    points = points[indexes, :]
    colors = colors[indexes, :]

    pcd = None

    if (nb_points is not None) and (radius is not None):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd, indexes = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)

    if voxel_size is not None:

        if pcd is None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    time = timer.get_seconds()
    print('preprocessing completed in {sec:.1f} seconds'.format(sec=time))

    # o3d.visualization.draw_geometries([pcd])

    return points, colors


def distance(p1, p2, axis=1, keepdims=False):

    dist = np.sqrt(np.sum(np.power(p1 - p2, 2), axis=axis, keepdims=keepdims))

    return dist


def faster_ransac_multi_sphere_detection_from_point_cloud(
        coordinates_np, min_rhos, max_rhos, threshold_dist, coord_precision,
        n_rhos=3, threshold_prop=0.4, threshold_supp=0.3):
    """
    step 1: propose all possible spheres with the coordinates_np of points. Any spheres
            with proposes < threshold_prop are dropped. Proposes are normalized in the range between 0 and 1.
            Thus, threshold_prop is also expected between 0 and 1

    step 2: calculate the supporters for all proposed spheres. The supporters of a sphere are the points
            whose distances from the sphere surface is less the threshold_dist. Any spheres
            with supporters < threshold_supp are dropped. Supporters are normalized in the range between 0 and 1.
            Thus, threshold_supp is also expected between 0 and 1

    step 3: search for local maximums of the supporters for the parameters of the proposed spheres

    step 4: splits sphere inliers and outliers
    """

    # step 1

    timer = cp.clock.Timer()
    print('step 1 started ...')

    if threshold_dist < 0:
        raise ValueError('threshold_dist < 0')

    if n_rhos < 1:
        raise ValueError('n_rhos')
    elif n_rhos == 1:
        possible_rhos = np.mean([min_rhos, max_rhos], keepdims=True)
    else:
        if min_rhos > max_rhos:
            min_rhos, max_rhos = max_rhos, min_rhos
        possible_rhos = np.linspace(start=min_rhos, stop=max_rhos, num=n_rhos, endpoint=True, dtype='f')

    tot_points = coordinates_np.shape[0]

    maxes = cp.maths.round_up(x=np.max(coordinates_np, axis=0), delta=coord_precision)
    mins = cp.maths.round_down(x=np.min(coordinates_np, axis=0), delta=coord_precision)

    possible_x = cp.maths.round(np.arange(mins[0] - max_rhos, maxes[0] + max_rhos + coord_precision, coord_precision), delta=coord_precision)
    possible_y = cp.maths.round(np.arange(mins[1] - max_rhos, maxes[1] + max_rhos + coord_precision, coord_precision), delta=coord_precision)
    possible_z = cp.maths.round(np.arange(mins[2] - max_rhos, maxes[2] + max_rhos + coord_precision, coord_precision), delta=coord_precision)

    n_possible_x = len(possible_x)
    n_possible_y = len(possible_y)
    n_possible_z = len(possible_z)

    step_thetas = (math.pi / 180) * 15
    min_thetas = 0.0
    max_thetas = 2 * math.pi

    step_phis = (math.pi / 180) * 15
    min_phis = (math.pi / 180) * 30
    max_phis = math.pi - min_phis

    step_rhos = possible_rhos[1] - possible_rhos[0]
    n_possible_rhos = len(possible_rhos)

    possible_thetas = np.arange(min_thetas, max_thetas - (step_thetas / 10), step_thetas, dtype='f')
    n_possible_thetas = len(possible_thetas)

    possible_phis = np.arange(min_phis, max_phis + (step_phis / 10), step_phis, dtype='f')
    n_possible_phis = len(possible_phis)

    cos_thetas = [math.cos(theta_j) for theta_j in possible_thetas]
    sin_thetas = [math.sin(theta_j) for theta_j in possible_thetas]

    cos_phis = [math.cos(phi_i) for phi_i in possible_phis]
    sin_phis = [math.sin(phi_i) for phi_i in possible_phis]

    shape_accu = [n_possible_x, n_possible_y, n_possible_z, n_possible_rhos]
    accumulator = np.full(shape=shape_accu, fill_value=0, dtype='i')

    for r, rho_r in enumerate(possible_rhos):
        for j in range(0, n_possible_thetas, 1):
            for i in range(0, n_possible_phis, 1):
                x_center = coordinates_np[:, 0] - (rho_r * cos_thetas[j] * sin_phis[i])
                y_center = coordinates_np[:, 1] - (rho_r * sin_thetas[j] * sin_phis[i])
                z_center = coordinates_np[:, 2] - (rho_r * cos_thetas[j])

                index_x_center = cp.maths.rint((x_center - possible_x[0]) / coord_precision)
                index_y_center = cp.maths.rint((y_center - possible_y[0]) / coord_precision)
                index_z_center = cp.maths.rint((z_center - possible_z[0]) / coord_precision)

                accumulator[index_x_center, index_y_center, index_z_center, r] += 1

    max_accu = accumulator.max(initial=0)
    print('max_accu', max_accu)
    accumulator = accumulator / max_accu

    accumulator[accumulator < threshold_prop] = 0
    # accumulator[accumulator >= threshold_prop] = 1

    logical_parms = accumulator > 0
    # logical_parms = accumulator == 1

    indexes_spheres = np.argwhere(logical_parms)

    proposes = np.empty(shape=indexes_spheres.shape[0], dtype='f')

    for a, indexes_a in enumerate(indexes_spheres):
        proposes[a] = accumulator[tuple(indexes_a.tolist())]


    # print('indexes_spheres.shape = ', indexes_spheres.shape)
    # print('indexes_spheres = \n', indexes_spheres)

    parameters_spheres = np.empty(shape=indexes_spheres.shape, dtype='f')

    parameters_spheres[:, 0] = possible_x[indexes_spheres[:, 0]]
    parameters_spheres[:, 1] = possible_y[indexes_spheres[:, 1]]
    parameters_spheres[:, 2] = possible_z[indexes_spheres[:, 2]]

    parameters_spheres[:, 3] = possible_rhos[indexes_spheres[:, 3]]

    print('parameters_spheres.shape = ', parameters_spheres.shape)
    # print('parameters_spheres = \n', parameters_spheres)

    # print('proposes.shape =', proposes.shape)
    # print('proposes =', proposes)

    time_1 = timer.get_seconds()
    print('step_1 completed in {sec:.1f} seconds'.format(sec=time_1))

    # step 2

    timer = cp.clock.Timer()
    print(
        'RANSAC: step 2 started ...\n'
        'Please wait about 5 minutes .\n'
        'If you want to make it quicker, you need to raise threshold_prop,\n'
        'but it will also raise the number of false negatives')

    accumulator = np.full(shape=shape_accu, fill_value=0, dtype='i')

    point_p = np.empty([1, 3], dtype='f')

    for s, parms_s in enumerate(parameters_spheres):

        x_center, y_center, z_center, rho = parms_s

        index_x_center = cp.maths.rint((x_center - possible_x[0]) / coord_precision)
        index_y_center = cp.maths.rint((y_center - possible_y[0]) / coord_precision)
        index_z_center = cp.maths.rint((z_center - possible_z[0]) / coord_precision)

        r = cp.maths.rint((rho - possible_rhos[0]) / step_rhos)

        point_p[0, :] = [x_center, y_center, z_center]

        half_size_cubes = rho + (threshold_dist * 2)

        point_start_cube_p = point_p - half_size_cubes

        point_end_cube_p = point_p + half_size_cubes

        points_cube_p = coordinates_np[
            np.logical_and(
                np.all(coordinates_np > point_start_cube_p, axis=1, keepdims=False),
                np.all(coordinates_np < point_end_cube_p, axis=1, keepdims=False))]

        if points_cube_p.shape[0] > 0:

            distances_from_center = distance(point_p, points_cube_p, axis=1, keepdims=False)

            distances_from_surface = np.abs(distances_from_center - rho)

            n_supporters = np.sum(distances_from_surface < threshold_dist, axis=0)

            accumulator[index_x_center, index_y_center, index_z_center, r] = n_supporters
        else:
            accumulator[index_x_center, index_y_center, index_z_center, r] = 0

    max_accu = accumulator.max(initial=0)
    print('max_accu', max_accu)
    accumulator = accumulator / max_accu

    accumulator[accumulator < threshold_supp] = 0
    # accumulator[accumulator >= threshold_supp] = 1

    logical_parms = accumulator > 0
    # logical_parms = accumulator == 1

    indexes_spheres = np.argwhere(logical_parms)

    supporters = np.empty(shape=indexes_spheres.shape[0], dtype='f')

    for a, indexes_a in enumerate(indexes_spheres):
        supporters[a] = accumulator[tuple(indexes_a.tolist())]

    del accumulator

    # print('indexes_spheres.shape = ', indexes_spheres.shape)
    # print('indexes_spheres = \n', indexes_spheres)

    parameters_spheres = np.empty(shape=indexes_spheres.shape, dtype='f')

    parameters_spheres[:, 0] = possible_x[indexes_spheres[:, 0]]
    parameters_spheres[:, 1] = possible_y[indexes_spheres[:, 1]]
    parameters_spheres[:, 2] = possible_z[indexes_spheres[:, 2]]

    parameters_spheres[:, 3] = possible_rhos[indexes_spheres[:, 3]]

    print('parameters_spheres.shape = ', parameters_spheres.shape)
    # print('parameters_spheres = \n', parameters_spheres)

    # print('supporters.shape =', supporters.shape)
    # print('supporters =', supporters)


    time_2 = timer.get_seconds()
    print('step_2 completed in {sec:.1f} seconds'.format(sec=time_2))

    # supporters = proposes


    # step 3

    timer = cp.clock.Timer()
    print('step 3 started ...')

    point_p = np.empty([1, 3], dtype='f')

    chosen_spheres = []

    while parameters_spheres.shape[0] > 0:

        x_center, y_center, z_center, rho = parameters_spheres[0, :]

    # for s, parms_s in enumerate(parameters_spheres):
    #     x_center, y_center, z_center, rho = parms_s

        point_p[0, :] = [x_center, y_center, z_center]

        # parameters_spheres = np.delete(arr=parameters_spheres, obj=s, axis=0)
        # supporters

        distances_of_centers = distance(point_p, parameters_spheres[:, 0:3:1], axis=1, keepdims=False)

        mask_overlapping_spheres = distances_of_centers < ((rho + parameters_spheres[:, 3]) * 1.5)

        overlapping_spheres = parameters_spheres[mask_overlapping_spheres, :]
        overlapping_supporters = supporters[mask_overlapping_spheres]

        index_local_max = np.argmax(a=overlapping_supporters, axis=0, keepdims=False)
        try:
            n_indexes = len(index_local_max)
            if n_indexes < 1:
                raise ValueError('n_indexes < 1')
            elif n_indexes > 1:
                raise ValueError('n_indexes > 1')
        except TypeError:
            pass

        chosen_spheres.append(overlapping_spheres[[index_local_max], :])

        mask_remaining_spheres = np.logical_not(mask_overlapping_spheres)
        parameters_spheres = parameters_spheres[mask_remaining_spheres, :]
        supporters = supporters[mask_remaining_spheres]

    parameters_spheres = np.concatenate(chosen_spheres, axis=0)

    # print('parameters_spheres = \n', parameters_spheres)
    print('parameters_spheres.shape = ', parameters_spheres.shape)

    time_3 = timer.get_seconds()
    print('step_3 completed in {sec:.1f} seconds\nspheres found'.format(sec=time_3))


    # step 4

    timer = cp.clock.Timer()
    print('step 4 started ...\ngetting inliers and outliers')

    n_spheres = parameters_spheres.shape[0]
    logical_inliners = np.full([tot_points, n_spheres], fill_value=False, dtype='?')

    point_p = np.empty([1, 3], dtype='f')

    for s, parms_s in enumerate(parameters_spheres):

        x_center, y_center, z_center, rho = parms_s

        point_p[0, :] = [x_center, y_center, z_center]

        distances_from_center = distance(point_p, coordinates_np, axis=1, keepdims=False)

        distances_from_surface = np.abs(distances_from_center - rho)

        logical_inliners[:, s] = distances_from_surface < threshold_dist

    inliners = np.any(logical_inliners, axis=1)

    outliers = np.logical_not(inliners)

    time_4 = timer.get_seconds()
    print('step_4 completed in {sec:.1f} seconds'.format(sec=time_4))

    return parameters_spheres, inliners, outliers


def save_points(points, directory=None, headers=None, delimiter=','):

    timer = cp.clock.Timer()
    print('Saving Results...')

    if directory is None:
        directory = 'results.txt'

    cp.txt.lines_to_csv_file(lines=points, directory=directory, headers=headers, delimiter=delimiter)

    time = timer.get_seconds()
    print('Results Saved in {:.1f}'.format(time))

    return None


def show_2d_slides(image, axis, size=None):

    import pygame

    if size is None:
        size = tuple([image.shape[a] for a in range(0, image.ndim, 1) if ((a != axis) and (a < 3))]) # type: typing.Union[tuple, tuple[int, int]]

    pygame.display.init()
    screen = pygame.display.set_mode(size=size)
    running = True

    indexes = [slice(0, image.shape[a], 1) for a in range(0, image.ndim, 1)]

    i = 0
    I = image.shape[axis]

    while running:

        d = 0

        # pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_UP:
                    d = 1
                if event.key == pygame.K_DOWN:
                    d = -1

        i += d

        if i < 0:
            i = 0
        if i >= I:
            i = I - 1

        # print('i =', i)

        indexes[axis] = i

        frame = image[tuple(indexes)]
        # frame = np.swapaxes(image[tuple(indexes)], axis1=0, axis2=1)
        # if factor != 1:
        #     frame = Image.fromarray(obj=frame, mode='RGB')
        #     frame = np.asarray(frame.resize(size=size), dtype='i')

        surf = pygame.surfarray.make_surface(frame[::, ::-1, ::])
        pygame.transform.scale2x(surface=surf)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    pygame.quit()

    return None


def sphere_cloud(center=(0, 0, 0), rho=1, n=1000, sigma=1):

    shape_cloud = [n, 3]

    # maxes = [[360, 180, rho]]
    maxes = [[2 * math.pi, math.pi]]

    spherical_points = np.empty(shape_cloud, dtype='f')

    spherical_points[:, slice(0, 2, 1)] = np.random.random([n, 2]) * maxes
    spherical_points[:, 2] = rho

    thetas = spherical_points[:, 0]
    phis = spherical_points[:, 1]
    rhos = spherical_points[:, 2]

    cartesian_points = np.empty(shape_cloud, dtype='f')

    cartesian_points[:, 0] = center[0] + (rhos * np.sin(phis) * np.cos(thetas))
    cartesian_points[:, 1] = center[1] + (rhos * np.sin(phis) * np.sin(thetas))
    cartesian_points[:, 2] = center[2] + (rhos * np.cos(phis))

    cartesian_points += np.random.normal(loc=0.0, scale=sigma, size=shape_cloud)

    return cartesian_points

