import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import distance


def assign_cluster_sizes(nparticles, HexClNr, QuadClNr):
    # defining
    Hex_clsizes = np.zeros(HexClNr, dtype=int)
    Quad_clsizes = np.zeros(QuadClNr, dtype=int)

    nCls = HexClNr + QuadClNr  # total number odf clusters
    avg_particle_number = int(nparticles / nCls)
    stdev = 0.2 * avg_particle_number

    for n in range(HexClNr):
        Hex_clsizes[n] = int(avg_particle_number + stdev * np.random.rand())
    for n in range(QuadClNr):
        Quad_clsizes[n] = int(avg_particle_number + stdev * np.random.randn())

    oversize = nparticles - (np.sum(Hex_clsizes) + np.sum(Quad_clsizes))

    if oversize != 0:
        subtract_or_add = int(oversize / np.abs(oversize))

        while True:
            if HexClNr != 0:
                index = np.random.choice(np.arange(0, HexClNr, 1))
                Hex_clsizes[index] += subtract_or_add
                oversize = nparticles - (np.sum(Hex_clsizes) + np.sum(Quad_clsizes))
                if oversize == 0:
                    break
            if QuadClNr != 0:
                index = np.random.choice(np.arange(0, QuadClNr, 1))
                Quad_clsizes[index] += subtract_or_add
                oversize = nparticles - (np.sum(Hex_clsizes) + np.sum(Quad_clsizes))
                if oversize == 0:
                    break

    if oversize == 0:
        print("Assignment of cluster sizes successful! \n")
        print("Hexagonal cluster sizes:", Hex_clsizes)
        print("Quadratic cluster sizes:", Quad_clsizes)

    return (Hex_clsizes, Quad_clsizes)


def cluster_seeds(boxlength, HexClNr, QuadClNr):
    nCls = HexClNr + QuadClNr
    pos_seeds = np.zeros((nCls, 2))
    for i in range(nCls):
        for dim in range(2):
            pos_seeds[i, dim] = np.random.rand() * boxlength
    print("position seeds: \n", pos_seeds)
    return pos_seeds


def build_cluster(pos, pos_seeds, nparticles, HexClNr, QuadClNr, clType, boxlength, i):
    # setting initial values
    full = "False"
    count_full = 0
    index = 0
    j = 0
    choice_array = np.array([0])
    joint_clNr = HexClNr + QuadClNr

    # creating and appending the start position of the new cluster to pos

    print("pos_seeds", pos_seeds)
    print("pos_seeds -1", pos_seeds[-1, 0])

    if (
        len(pos_seeds[:, 0]) > 1
    ):  # pos[0,0] != pos_seeds[-1,0] and pos[0,1] != pos_seeds[-1,1]:
        choice_array = np.array([len(pos) - 1])
        index = len(pos) - 1

    # selects the type of cluster that must be geneerated
    if clType == "Hex":
        round = 5
        angle_turn = np.pi / 3
        first_angle = np.random.rand(1) * 2 * 3.14

    if clType == "Quad":
        round = 3
        angle_turn = np.pi / 2
        first_angle = 0

    while j < nparticles - 1:
        if full == "True":
            # resetting
            count_full = 0
            full = "False"

            index = np.random.choice(
                choice_array
            )  # chooses a new random particle index; this particle is used to build on

        overlap = "true"
        while overlap == "true":
            if j == 0:
                angle = first_angle
            else:
                angle = angle + angle_turn
            new_pos = pos[index, :] + np.array([np.cos(angle), np.sin(angle)]).reshape(
                -1
            )

            # check pbc
            for dim in range(2):
                if new_pos[dim] < 0.0:
                    new_pos[dim] += boxlength
                if new_pos[dim] >= boxlength:
                    new_pos[dim] -= boxlength

            write_pos = "True"

            # checks if particles are overlapping
            for coord in pos[:, :]:
                dist_overlap = np.linalg.norm(coord - new_pos)
                if dist_overlap < 0.9:
                    overlap = "true"
                    write_pos = "False"
                    count_full += 1

            if write_pos == "True":
                # resetting
                overlap = "False"
                count_full = 0

                pos = np.append(pos, np.array([new_pos]), axis=0)

                choice_array = np.append(choice_array, [len(pos) - 1])

                j += 1
                print(j)

            if count_full > round:
                overlap = "False"
                full = "True"

    if (
        len(pos_seeds[:, 0]) < joint_clNr
    ):  # pos[0,0] != pos_seeds[-1,0] and pos[0,1] != pos_seeds[-1,1]:
        keep_on = True
        while keep_on == True:
            keep_on = False
            pos_seed = np.random.rand(2) * boxlength
            for coord in pos[:, :]:
                dist_overlap = np.linalg.norm(coord - pos_seed)
                if dist_overlap < 6:
                    keep_on = True

            if keep_on == False:
                pos = np.append(pos, np.array([pos_seed]), axis=0)
                pos_seeds = np.append(pos_seeds, np.array([pos_seed]), axis=0)

    return pos, pos_seeds


def move_clusters(pos, pos_seeds, boxlength, Hex_clsizes, Quad_clsizes, closest_dist):
    # create array with joint cluster sizes
    joint_clsizes = np.concatenate((Hex_clsizes, Quad_clsizes), axis=0)

    # print('joint_clsizes',joint_clsizes)
    # cut whose positions array into position arrays for each cluster (stored in list) --------------
    cut_at = 0
    split_pos = []
    k = 0
    for i in joint_clsizes:
        k += 1
        if k < len(joint_clsizes):
            cut_at += i
            print(cut_at)
            split_pos.append(cut_at)

    pos_by_Cl = np.split(pos, split_pos, axis=0)
    # print('pos_by_Cl' , pos_by_Cl)
    # -----------------------------------------------------------------------------------------------

    # check if clusters are to close to each other. If closer than closest_dist, then clusters are moved away from each other ------------------------------
    ckeck_again = "True"
    while ckeck_again == "True":
        check_again = "False"

        for i in range(len(joint_clsizes)):
            for k in range(i):
                dist = distance.cdist(pos_by_Cl[i], pos_by_Cl[k], "euclidean")
                min_dist = np.amin(dist)
                if i == 1 and k == 0:
                    clDist_info = np.array([min_dist])
                else:
                    clDist_info = np.append(clDist_info, [min_dist], axis=0)

                if min_dist < closest_dist:
                    inter_cluster_vector = (
                        pos_seeds[k] - pos_seeds[i]
                    )  # vector pointing from i to k, which means we have to add sth to the k cluster
                    pos_by_Cl[k] += (
                        (closest_dist - int(min_dist))
                        * inter_cluster_vector
                        / np.linalg.norm(inter_cluster_vector)
                    )
                    pos_seeds[k] += (
                        (closest_dist - int(min_dist))
                        * inter_cluster_vector
                        / np.linalg.norm(inter_cluster_vector)
                    )

                    # check pbc
                    for p in range(len(pos_by_Cl[k])):
                        for dim in range(2):
                            if pos_by_Cl[k][p, dim] < 0.0:
                                pos_by_Cl[k][p, dim] += boxlength
                            if pos_by_Cl[k][p, dim] >= boxlength:
                                pos_by_Cl[k][p, dim] -= boxlength
                    for dim in range(2):
                        if pos_seeds[k][dim] < 0.0:
                            pos_seeds[k][dim] += boxlength
                        if pos_seeds[k][dim] >= boxlength:
                            pos_seeds[k][dim] -= boxlength

                    check_again = "True"

        if check_again == "False":
            pos_v = pos_by_Cl[0]
            for x in range(1, len(joint_clsizes)):
                pos_v = np.append(pos_v, pos_by_Cl[x], axis=0)
            break
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------

    return (pos_v, pos_seeds)


# function used to generate clusters
def generate_clusters(nparticles, boxlength, HexClNr, QuadClNr):
    Hex_clsizes, Quad_clsizes = assign_cluster_sizes(nparticles, HexClNr, QuadClNr)

    # determining the positions of the first particle of each cluster
    pos_seeds = np.array([np.random.rand(2) * boxlength])

    # plt.plot(pos_seeds[:,0],pos_seeds[:,1], 'o', color='r', ms=5)

    # initialize position array
    pos = np.array([pos_seeds[0]])

    if HexClNr != 0:
        for i in range(HexClNr):
            pos, pos_seeds = build_cluster(
                pos, pos_seeds, Hex_clsizes[i], HexClNr, QuadClNr, "Hex", boxlength, i
            )

    if QuadClNr != 0:
        for i in range(QuadClNr):
            pos, pos_seeds = build_cluster(
                pos, pos_seeds, Quad_clsizes[i], HexClNr, QuadClNr, "Quad", boxlength, i
            )  # pos_seeds[HexClNr+i,:]

    # checks if created clusters are too close and if so moves them apart
    pos_v, pos_seeds = move_clusters(
        pos, pos_seeds, boxlength, Hex_clsizes, Quad_clsizes, closest_dist=6
    )
    # pos_v = 0

    return pos, pos_v, pos_seeds
