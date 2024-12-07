from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pybdynamics import pbc


# Function to calculate area of a polygon given its vertices
def polygon_area(vertices):
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# create 4 mirror images of original simulation box
# input: positions of (original) particles and length of simulation box
def create_mirror_images(points, length):
    mirror_tot = points
    mirror_new = np.zeros(np.shape(points))

    # go through all neighboring mirror cells
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            mirror_new[:, 0] = points[:, 0] + x * length
            mirror_new[:, 1] = points[:, 1] + y * length

            # do not add the original cell again
            if x == 0 and y == 0:
                continue

            else:
                mirror_tot = np.concatenate((mirror_tot, mirror_new), axis=0)

    return mirror_tot


# Particle resolved local area density using vornooi tessellation
def local_den_particle_res(pos, sigma, length, mask):
    # Create mirror images of the simulation box (of the particle co- ordinates)
    # for better Voronoi tessellation
    mirror_tot = create_mirror_images(pos, length)

    # Create Voronoi diagram
    vor = Voronoi(mirror_tot)

    # fig = voronoi_plot_2d(
    #     vor,
    #     show_vertices=False,
    #     line_colors="orange",
    #     line_width=2,
    #     line_alpha=0.6,
    #     point_size=2,
    # )
    # plt.show()

    dist_threshold = 10.0 * sigma

    # Calculate Voronoi cell areas
    cell_areas = np.full((len(vor.regions)), np.nan)

    for i, region_index in enumerate(vor.point_region):
        # Indices of the voronoi cell region edges
        region = vor.regions[region_index]

        # Check if the distance between the voronoi cell vertex and the particle
        # is less than the threshold distance
        if np.linalg.norm(vor.points[i] - vor.vertices[region]) <= dist_threshold:

            # Discard the particles inside the list mask (particles at edges of the aggregates)
            if i in mask:
                continue

            # Use only the voronoi cells corresponing to the original simulation box
            # and discard the ones corresponding to the mirror images
            if (vor.points[i, 0] > 0) and (vor.points[i, 0] < length):
                if vor.points[i, 1] > 0 and vor.points[i, 1] < length:
                    # Discard regions that extend to infinity
                    if -1 not in region and len(region) > 0:
                        cell_areas[i] = polygon_area(vor.vertices[region])

    # Mask the cell areas to discard the regions that extend to infinity
    # cell_areas = cell_areas[~np.isnan(cell_areas)]

    # Compute particle resolved local area fractions
    local_area_fractions = np.pi * (sigma / 2.0) ** 2 / (cell_areas)

    return local_area_fractions


# Time averaged, particle resolved local area density using vornooi tessellation
def local_den_particle_res_tavg(pos, sigma, length, tstart, tend, mask):
    local_area_fractions = []
    for i in tqdm(range(tstart, tend)):
        local_area_fractions = np.append(
            local_area_fractions,
            local_den_particle_res(pos[:, :, i], sigma, length, mask),
        )

    return local_area_fractions


# Position resolved local area density using vornooi tessellation
def local_den_pos_res(pos, sigma, length, mask):
    # Define the mesh
    mesh_size = length / np.floor(length)
    x = np.arange(0, length, mesh_size)
    y = np.arange(0, length, mesh_size)
    gx, gy = np.meshgrid(x, y)
    mesh_points = np.vstack((gx.flatten(), gy.flatten())).T
    mesh_point_area_fractions = np.full((mesh_points.shape[0]), np.nan)

    # Calculate the particle resolved local area fractions
    local_area_fractions = local_den_particle_res(pos, sigma, length, mask)

    # Determine the closest particle index
    # (and the corresponding Voronoi cell index)
    # from each mesh point
    for i in range(mesh_points.shape[0]):
        dist = np.zeros((pos.shape[0]))
        for j in range(pos.shape[0]):
            # rnorm, r = pbc.dist_mic(
            #     pos.shape[1], mesh_points[i, :], pos[j, :], length, length / 2.0
            # )

            rnorm, r = pbc.dist(pos.shape[1], mesh_points[i, :], pos[j, :])

            dist[j] = rnorm
        closest_particle_index = np.argmin(dist)

        # assign the local area fraction of the closest particle
        # to the mesh point if the particle is not in mask list
        if closest_particle_index not in mask:
            mesh_point_area_fractions[i] = local_area_fractions[closest_particle_index]

    return mesh_point_area_fractions


# Time averaged, position resolved local area density using vornooi tessellation
def local_den_pos_res_tavg(pos, sigma, length, tstart, tend, mask):
    mesh_point_area_fractions = []
    for i in tqdm(range(tstart, tend)):
        mesh_point_area_fractions = np.append(
            mesh_point_area_fractions,
            local_den_pos_res(pos[:, :, i], sigma, length, mask),
        )

    return mesh_point_area_fractions
