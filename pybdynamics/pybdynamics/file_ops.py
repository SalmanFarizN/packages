import h5py
import numpy as np


# Function for writing to .h5 file
def write(hf, pos, vel, tr, save_count):
    hf["pos"][:, :, save_count] = pos[:, :]
    # hf["vel"][:, :, save_count] = vel[:, :]
    hf["tr"][:, :, save_count] = tr[:, :]
    hf.attrs["saved_point"] = save_count
    hf.flush()


# function for File creation
def hdf5_create(filename, shape_params):
    nparticles = shape_params[0]
    ndims = shape_params[1]
    spoints = shape_params[2] + 1  # +1 for saving the 0th frame
    # Creation of .h5 file
    hf = h5py.File(filename, "a")
    position = hf.create_dataset(
        "pos",
        (nparticles, ndims, spoints),
        maxshape=(nparticles, ndims, None),
        dtype=np.float32,
        chunks=(nparticles, ndims, 1),
    )
    # velocity = hf.create_dataset(
    #     "vel",
    #     (nparticles, ndims, spoints),
    #     maxshape=(nparticles, ndims, None),
    #     dtype=np.float32,
    #     chunks=(nparticles, ndims, 1),
    # )
    passtrack = hf.create_dataset(
        "tr",
        (nparticles, ndims, spoints),
        maxshape=(nparticles, ndims, None),
        dtype=np.float32,
        chunks=(nparticles, ndims, 1),
    )

    # Return the HDF5 file object
    return hf


# Fucntion to open a file and check the last saved frame when continuing
# a simulation
def hdf5_last_check(filename, shape_params, saved):
    nparticles = shape_params[0]
    ndims = shape_params[1]
    spoints = shape_params[2]
    tpoints = spoints * saved

    hf = h5py.File(filename, "a")
    sav_count = hf.attrs["saved_point"]

    print(((sav_count + 1) * 100 / tpoints) * 100, "% Completed!")
    # Calculate the remaining time points to be saved in order to
    # fill the .h5 file
    remaining_spoints = spoints - (sav_count + 1)
    if remaining_spoints <= 0:
        print("Simulation is complete!")
        exit()
    # Calculate the remaining time points to be simulated
    # savedtpoints*saved = remaining_tpoints
    remaining_tpoints = remaining_spoints * saved
    tpoints = remaining_tpoints

    pos = np.array(hf["pos"][:, :, sav_count])
    # vel = np.array(hf["vel"][:, :, sav_count])
    tr = np.array(hf["tr"][:, :, sav_count])

    return hf, pos, tr, sav_count, tpoints


# Function to read the last frame from an HDF5 file
# to start a new simulation
def hdf5_readlast(filename):
    hf = h5py.File(filename, "a")
    pos = np.array(hf["pos"][:, :, hf["pos"].shape[2] - 1])
    tr = np.array(hf["tr"][:, :, hf["tr"].shape[2] - 1])
    hf.close()

    return pos, tr
