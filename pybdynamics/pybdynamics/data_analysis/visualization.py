import os
from pybdynamics.data_analysis import cluster_analysis as cla


# Prepare xyz file to Visualize 2D particle system snapshots in OVITO
def ovito2d(pos, colour2, path, filename, length):
    os.chdir(path)
    file = open(filename, "w")

    # 3D array: trajectory file
    if len(pos.shape) == 3:
        for i in range(0, pos.shape[2] - 1):
            file.write("{}\n".format(pos.shape[0]))
            file.write(
                'Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(
                    length, length, length
                )
            )
            for j in range(pos.shape[0]):
                if j in colour2:
                    file.write("2 {} {}\n".format(pos[j, 0, i], pos[j, 1, i]))

                else:
                    file.write("1 {} {}\n".format(pos[j, 0, i], pos[j, 1, i]))

    # 2D array: Only a single frame
    if len(pos.shape) == 2:
        file.write("{}\n".format(pos.shape[0]))
        file.write(
            'Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(
                length, length, length
            )
        )
        for j in range(pos.shape[0]):
            if j in colour2:
                file.write("2 {} {}\n".format(pos[j, 0], pos[j, 1]))

            else:
                file.write("1 {} {}\n".format(pos[j, 0], pos[j, 1]))

    file.close()


# Prepare xyz file to Visualize 3D particle system snapshots in OVITO
def ovito3d(pos, colour2, path, filename, length):
    os.chdir(path)
    file = open(filename, "w")

    # 3D array: trajectory file
    if len(pos.shape) == 3:
        for i in range(0, pos.shape[2] - 1):
            file.write("{}\n".format(pos.shape[0]))
            file.write(
                'Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(
                    length, length, length
                )
            )
            for j in range(pos.shape[0]):
                if j in colour2:
                    file.write(
                        "2 {} {} {}\n".format(pos[j, 0, i], pos[j, 1, i], pos[j, 2, i])
                    )

                else:
                    file.write(
                        "1 {} {} {}\n".format(pos[j, 0, i], pos[j, 1, i], pos[j, 2, i])
                    )

    # 2D array: Only a single frame
    if len(pos.shape) == 2:
        file.write("{}\n".format(pos.shape[0]))
        file.write(
            'Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(
                length, length, length
            )
        )
        for j in range(pos.shape[0]):
            if j in colour2:
                file.write("2 {} {} {}\n".format(pos[j, 0], pos[j, 1], pos[j, 2]))

            else:
                file.write("1 {} {} {}\n".format(pos[j, 0], pos[j, 1], pos[j, 2]))

    file.close()


# Prepare xyz of snapshots colour coded accoring to states: fluid, disordered, hexagonal, quadratic and chains
def ovito2d_4states(pos, path, filename, nparticles, ndims, sigma, length, hlength):
    os.chdir(path)

    file = open(filename, "w")

    # 2D array: Only a single frame
    if len(pos.shape) == 2:
        z, nc, nclist, r_list = cla.coord_number(
            nparticles, ndims, sigma, length, hlength, pos
        )
        phi4_avg, phi4 = cla.abop(nparticles, ndims, 4, r_list, nc)
        phi6_avg, phi6 = cla.abop(nparticles, ndims, 6, r_list, nc)
        file.write("{}\n".format(pos.shape[0]))
        file.write(
            'Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(
                length, length, length
            )
        )

        for i in range(pos.shape[0]):
            # Chain condition
            if phi4[i] >= 0.8 and phi6[i] >= 0.8 and nc[i] == 2:
                file.write("5 {} {}\n".format(pos[i, 0], pos[i, 1]))

            # HCP Condition
            elif phi6[i] >= 0.8:
                file.write("2 {} {}\n".format(pos[i, 0], pos[i, 1]))

            # Quadratic Condition
            elif phi4[i] >= 0.8:
                file.write("3 {} {}\n".format(pos[i, 0], pos[i, 1]))

            # Fluid Condition
            elif nc[i] <= 2:
                file.write("1 {} {}\n".format(pos[i, 0], pos[i, 1]))

            # None above then disordered
            else:
                file.write("4 {} {}\n".format(pos[i, 0], pos[i, 1]))

    # 3D array: trajectory file
    if len(pos.shape) == 3:
        for i in range(0, pos.shape[2] - 1, 100):
            z, nc, nclist, r_list = cla.coord_number(
                nparticles, ndims, sigma, length, hlength, pos[:, :, i]
            )
            phi4_avg, phi4 = cla.abop(nparticles, ndims, 4, r_list, nc)
            phi6_avg, phi6 = cla.abop(nparticles, ndims, 6, r_list, nc)

            file.write("{}\n".format(pos.shape[0]))
            file.write(
                'Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(
                    length, length, length
                )
            )

            for j in range(pos.shape[0]):
                # Chain condition
                if phi4[j] >= 0.8 and phi6[j] >= 0.8 and nc[j] == 2:
                    file.write("5 {} {}\n".format(pos[j, 0, i], pos[j, 1, i]))

                # HCP Condition
                elif phi6[j] >= 0.8:
                    file.write("2 {} {}\n".format(pos[j, 0, i], pos[j, 1, i]))

                # Quadratic Condition
                elif phi4[j] >= 0.8:
                    file.write("3 {} {}\n".format(pos[j, 0, i], pos[j, 1, i]))

                # Fluid Condition
                elif nc[j] <= 2:
                    file.write("1 {} {}\n".format(pos[j, 0, i], pos[j, 1, i]))

                # None above then disordered
                else:
                    file.write("4 {} {}\n".format(pos[j, 0, i], pos[j, 1, i]))

    file.close()


# For non reciprocal systems
# Prepare xyz of snapshots/videos colour coded accoring to 5
# states: fluid, disordered, hexagonal, quadratic, chains and pairs (one particle of specias A and one of species B)
def ovito2d_nrp_4states(pos, path, filename, nparticles, ndims, sigma, length, hlength):
    os.chdir(path)

    file = open(filename, "w")

    # 2D array: Only a single frame
    if len(pos.shape) == 2:
        z, nc, nclist, r_list = cla.coord_number(
            nparticles, ndims, sigma, length, hlength, pos
        )
        phi4_avg, phi4 = cla.abop(nparticles, ndims, 4, r_list, nc)
        phi6_avg, phi6 = cla.abop(nparticles, ndims, 6, r_list, nc)
        file.write("{}\n".format(pos.shape[0]))
        file.write(
            'Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(
                length, length, length
            )
        )

        for i in range(pos.shape[0]):
            # Chain condition
            if phi4[i] >= 0.8 and phi6[i] >= 0.8 and nc[i] == 2:
                file.write("5 {} {}\n".format(pos[i, 0], pos[i, 1]))

            # HCP Condition
            elif phi6[i] >= 0.8:
                file.write("2 {} {}\n".format(pos[i, 0], pos[i, 1]))

            # Quadratic Condition
            elif phi4[i] >= 0.8:
                file.write("3 {} {}\n".format(pos[i, 0], pos[i, 1]))

            elif nc[i] <= 2:
                # First check if particle has only one neighbour
                if nc[i] == 1:
                    # Pair state
                    # Check if the species of particle i and its neighbour
                    # are different
                    if i < (nparticles / 2) and nclist[i, 1] >= (nparticles / 2):
                        file.write("6 {} {}\n".format(pos[i, 0], pos[i, 1]))
                    elif i >= (nparticles / 2) and nclist[i, 1] < (nparticles / 2):
                        file.write("6 {} {}\n".format(pos[i, 0], pos[i, 1]))
                    else:
                        # Fluid Condition
                        file.write("1 {} {}\n".format(pos[i, 0], pos[i, 1]))
                else:
                    # Fluid Condition
                    file.write("1 {} {}\n".format(pos[i, 0], pos[i, 1]))

            # None above then disordered
            else:
                file.write("4 {} {}\n".format(pos[i, 0], pos[i, 1]))

    # 3D array: trajectory file
    if len(pos.shape) == 3:
        for i in range(0, pos.shape[2] - 1, 1):
            z, nc, nclist, r_list = cla.coord_number(
                nparticles, ndims, sigma, length, hlength, pos[:, :, i]
            )
            phi4_avg, phi4 = cla.abop(nparticles, ndims, 4, r_list, nc)
            phi6_avg, phi6 = cla.abop(nparticles, ndims, 6, r_list, nc)

            file.write("{}\n".format(pos.shape[0]))
            file.write(
                'Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(
                    length, length, length
                )
            )

            for j in range(pos.shape[0]):
                # Chain condition
                if phi4[j] >= 0.8 and phi6[j] >= 0.8 and nc[j] == 2:
                    file.write("5 {} {}\n".format(pos[j, 0, i], pos[j, 1, i]))

                # HCP Condition
                elif phi6[j] >= 0.8:
                    file.write("2 {} {}\n".format(pos[j, 0, i], pos[j, 1, i]))

                # Quadratic Condition
                elif phi4[j] >= 0.8:
                    file.write("3 {} {}\n".format(pos[j, 0, i], pos[j, 1, i]))

                # Fluid Condition
                elif nc[j] <= 2:
                    # First check if particle has only one neighbour
                    if nc[j] == 1:
                        # Pair state
                        # Check if the species of particle j and its neighbour
                        # are different
                        if j < (nparticles / 2) and nclist[j, 1] >= (nparticles / 2):
                            file.write("6 {} {}\n".format(pos[j, 0, i], pos[j, 1, i]))
                        elif j >= (nparticles / 2) and nclist[j, 1] < (nparticles / 2):
                            file.write("6 {} {}\n".format(pos[j, 0, i], pos[j, 1, i]))
                        else:
                            # Fluid Condition
                            file.write("1 {} {}\n".format(pos[j, 0, i], pos[j, 1, i]))
                    else:
                        # Fluid Condition
                        file.write("1 {} {}\n".format(pos[j, 0, i], pos[j, 1, i]))

                # None above then disordered
                else:
                    file.write("4 {} {}\n".format(pos[j, 0, i], pos[j, 1, i]))

    file.close()


# Function to colour code particles according to abop phi_4 and phi_6
def ovito2d_abop(pos, path, filename, nparticles, ndims, sigma, length, hlength):
    os.chdir(path)
    file = open(filename, "w")

    # 2D array: Only a single frame
    if len(pos.shape) == 2:
        z, nc, nclist, r_list = cla.coord_number(
            nparticles, ndims, sigma, length, hlength, pos
        )
        phi4_avg, phi4 = cla.abop(nparticles, ndims, 4, r_list, nc)
        phi6_avg, phi6 = cla.abop(nparticles, ndims, 6, r_list, nc)
        file.write("{}\n".format(pos.shape[0]))
        file.write(
            'Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(
                length, length, length
            )
        )

        for i in range(pos.shape[0]):
            file.write("1 {} {} {} {}\n".format(pos[i, 0], pos[i, 1], phi4[i], phi6[i]))

    if len(pos.shape) == 3:
        for i in range(0, pos.shape[2] - 1, 100):
            z, nc, nclist, r_list = cla.coord_number(
                nparticles, ndims, sigma, length, hlength, pos
            )
            phi4_avg, phi4 = cla.abop(nparticles, ndims, 4, r_list, nc)
            phi6_avg, phi6 = cla.abop(nparticles, ndims, 6, r_list, nc)
            file.write("{}\n".format(pos.shape[0]))
            file.write(
                'Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(
                    length, length, length
                )
            )

            for j in range(pos.shape[0]):
                file.write(
                    "1 {} {} {} {}\n".format(pos[j, 0], pos[j, 1], phi4[j], phi6[j])
                )

    file.close()


# Create .xyz OVITO files for single patch Janus particles
def janus_single(
    pos, path, filename, nparticles, ndims, sigma, length, hlength, direction
):
    os.chdir(path)
    file = open(filename, "w")

    # 2D array: Only a single frame
    if len(pos.shape) == 2:
        file.write("{}\n".format(2 * pos.shape[0]))
        file.write(
            'Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(
                length, length, length
            )
        )

        for i in range(pos.shape[0]):
            file.write("1 {} {}\n".format(pos[i, 0], pos[i, 1]))
            file.write(
                "2 {} {}\n".format(
                    pos[i, 0] + 0.25 * direction[i, 0],
                    pos[i, 1] + 0.25 * direction[i, 1],
                )
            )

    # 3D array: trajectory
    if len(pos.shape) == 3:
        for i in range(0, pos.shape[2] - 100, 100):
            file.write("{}\n".format(2 * pos.shape[0]))
            file.write(
                'Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(
                    length, length, length
                )
            )

            for j in range(pos.shape[0]):
                file.write("1 {} {}\n".format(pos[j, 0, i], pos[j, 1, i]))
                file.write(
                    "2 {} {}\n".format(
                        pos[j, 0, i] + 0.15 * direction[j, 0, i],
                        pos[j, 1, i] + 0.15 * direction[j, 1, i],
                    )
                )

    file.close()


# Create .xyz OVITO files for double patch Janus particles
def janus_double(
    pos, path, filename, nparticles, ndims, sigma, length, hlength, direction
):
    os.chdir(path)
    file = open(filename, "w")

    # 2D array: Only a single frame
    if len(pos.shape) == 2:
        file.write("{}\n".format(3 * pos.shape[0]))
        file.write(
            'Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(
                length, length, length
            )
        )

        for i in range(pos.shape[0]):
            file.write("1 {} {}\n".format(pos[i, 0], pos[i, 1]))
            file.write(
                "2 {} {}\n".format(
                    pos[i, 0] + 0.15 * direction[i, 0],
                    pos[i, 1] + 0.15 * direction[i, 1],
                )
            )
            file.write(
                "2 {} {}\n".format(
                    pos[i, 0] - 0.15 * direction[i, 0],
                    pos[i, 1] - 0.15 * direction[i, 1],
                )
            )

    # 3D array: trajectory
    if len(pos.shape) == 3:
        for i in range(0, pos.shape[2] - 100, 100):
            file.write("{}\n".format(3 * pos.shape[0]))
            file.write(
                'Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(
                    length, length, length
                )
            )

            for j in range(pos.shape[0]):
                file.write("1 {} {}\n".format(pos[j, 0, i], pos[j, 1, i]))
                file.write(
                    "2 {} {}\n".format(
                        pos[j, 0, i] + 0.15 * direction[j, 0, i],
                        pos[j, 1, i] + 0.15 * direction[j, 1, i],
                    )
                )
                file.write(
                    "2 {} {}\n".format(
                        pos[j, 0, i] - 0.15 * direction[j, 0, i],
                        pos[j, 1, i] - 0.15 * direction[j, 1, i],
                    )
                )

    file.close()


# Function to visualize janus particles colour coded according to states: fluid, disordered, hex, kagome
def janus_double_states(
    pos, path, filename, nparticles, ndims, sigma, length, hlength, direction
):
    os.chdir(path)
    file = open(filename, "w")

    # 2D array: Only a single frame
    if len(pos.shape) == 2:
        z, nc, nclist, r_list = cla.coord_number(
            nparticles, ndims, sigma, length, hlength, pos
        )
        phi6_avg, phi6 = cla.abop(nparticles, ndims, 6, r_list, nc)

        file.write("{}\n".format(3 * pos.shape[0]))
        file.write(
            'Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(
                length, length, length
            )
        )

        for i in range(pos.shape[0]):
            # Everything set to disordered by default (whatever remains after the ifs below
            # will be disordered in the end)
            col = 4

            # Fluid condition
            if nc[i] <= 2:
                col = 1

            # Hexagonal condition
            if nc[i] == 6 and phi6[i] > 0.7:
                col = 2

            # Kagome Condition
            if nc[i] == 4 and phi6[i] > 0.7:
                col = 3

            # Particle
            file.write("{} {} {}\n".format(col, pos[i, 0], pos[i, 1]))
            # Two patches
            file.write(
                "5 {} {}\n".format(
                    pos[i, 0] + 0.15 * direction[i, 0],
                    pos[i, 1] + 0.15 * direction[i, 1],
                )
            )
            file.write(
                "5 {} {}\n".format(
                    pos[i, 0] - 0.15 * direction[i, 0],
                    pos[i, 1] - 0.15 * direction[i, 1],
                )
            )

    # 3D array: trajectory
    if len(pos.shape) == 3:
        for i in range(0, pos.shape[2] - 100, 100):
            z, nc, nclist, r_list = cla.coord_number(
                nparticles, ndims, sigma, length, hlength, pos[:, :, i]
            )
            phi6_avg, phi6 = cla.abop(nparticles, ndims, 6, r_list, nc)

            file.write("{}\n".format(3 * pos.shape[0]))
            file.write(
                'Lattice="{} 0.0 0.0 0.0 {} 0.0 0.0 0.0 {}"\n'.format(
                    length, length, length
                )
            )

            for j in range(pos.shape[0]):
                # Everything set to disordered by default (whatever remains after the ifs below
                # will be disordered in the end)
                col = 4

                # Fluid condition
                if nc[j] <= 2:
                    col = 1

                # Hexagonal condition
                if nc[j] == 6 and phi6[j] > 0.7:
                    col = 2

                # Kagome Condition
                if nc[j] == 4 and phi6[j] > 0.7:
                    col = 3

                # Particle
                file.write("{} {} {}\n".format(col, pos[j, 0, i], pos[j, 1, i]))
                # Two patches
                file.write(
                    "5 {} {}\n".format(
                        pos[j, 0, i] + 0.15 * direction[j, 0, i],
                        pos[j, 1, i] + 0.15 * direction[j, 1, i],
                    )
                )
                file.write(
                    "5 {} {}\n".format(
                        pos[j, 0, i] - 0.15 * direction[j, 0, i],
                        pos[j, 1, i] - 0.15 * direction[j, 1, i],
                    )
                )

    file.close()
