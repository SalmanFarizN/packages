import numpy as np
from pybdynamics import pbc
import math


# Function to initialise position of particles so that they do not overlap .. run only once at the beginning
def posini(shape_params, pbc_params, diam):
    pos1 = np.zeros((shape_params[0], shape_params[1]), dtype=np.float64)
    sep = np.zeros((shape_params[0]), dtype=np.float64)
    dimsep = np.zeros((shape_params[1]), dtype=np.float64)

    for i in range(shape_params[0]):
        print(i)
        if i == 0:
            for j in range(shape_params[1]):
                pos1[i, j] = np.random.uniform(0.0, pbc_params[0])
        else:
            for j in range(shape_params[1]):
                pos1[i, j] = np.random.uniform(0.0, pbc_params[0])

            for k in range(i):
                if i != k:
                    # Computing distance with the mirror image condition
                    sep[k], r = pbc.dist_mic(
                        shape_params[1],
                        pos1[i, :],
                        pos1[k, :],
                        pbc_params[0],
                        pbc_params[1],
                    )

            while any(q < diam and q != 0.0 for q in sep):
                for j in range(shape_params[1]):
                    pos1[i, j] = np.random.uniform(0.0, pbc_params[0])
                for k in range(i):
                    if i != k:
                        # Computing distance with the mirror image condition
                        sep[k], r = pbc.dist_mic(
                            shape_params[1],
                            pos1[i, :],
                            pos1[k, :],
                            pbc_params[0],
                            pbc_params[1],
                        )

    return pos1


# Function to initialize particle positions uniformly spaced. May not work for nparticles not a factor of box length
def posini_uniform(shape_params, pbc_params, diam):
    pos1 = np.zeros((shape_params[0], shape_params[1]), dtype=np.float64)
    dist = math.floor((pbc_params[0] / np.sqrt(shape_params[0])) * 10) / 10
    particle_number = 0

    for i in range(1, math.ceil(np.sqrt(shape_params[0])) + 1):
        for j in range(1, math.ceil(np.sqrt(shape_params[0])) + 1):
            if particle_number < shape_params[0]:
                if i * diam * dist < pbc_params[0] and j * diam * dist < pbc_params[0]:
                    pos1[particle_number, 0] = i * diam * dist
                    pos1[particle_number, 1] = j * diam * dist
                    particle_number += 1

    print("particle number:", particle_number)
    if particle_number == shape_params[0]:
        print("successful particle placement")
    print(particle_number)
    print(pos1)

    return pos1


# Function to initialise position of particles so that they do not overlap placed in contact next to each other
def posini2(shape_params, pbc_params, diam):
    pos = np.zeros((shape_params[0], shape_params[1]), dtype=np.float64)

    xax = diam + 0.1
    yax = diam + 0.1
    pos[0, 0] = xax
    pos[0, 1] = yax

    for i in range(1, shape_params[0]):
        print(i)
        if (xax + diam) < (pbc_params[0] - diam):
            xax = xax + diam
        else:
            xax = diam
            yax = yax + diam

        pos[i, 0] = xax
        pos[i, 1] = yax

    return pos


# Funciton to initialise the position of 2 particles such that they are in contact
# and are aligned along the x axis for the non-rp multidirectional system
def active_pair(sigma):
    pos = np.zeros((2, 2), dtype=np.float64)
    xax = 4.0
    yax = 5.0
    pos = np.array([[xax, yax], [xax + sigma, yax]])
    return pos
