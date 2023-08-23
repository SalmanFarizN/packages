import numpy as np
from numba import jit, objmode
from pybdynamics import pbc


# Radial distrubution function calculation
@jit(nopython=True)
def rdf(pos, length, rho, rmax, bins, dr, t_interval):
    hlength = length / 2.0
    r = np.linspace(0, rmax, bins)
    gr = np.zeros((len(r)), dtype=np.float32)
    ndims = pos.shape[1]
    nparticles = pos.shape[0]
    spoints = pos.shape[2]

    for i in range(len(r)):
        width = r[i] + dr
        npp = 0
        d = np.zeros((ndims))
        pointden = (nparticles) / (length**ndims)
        tavg = spoints / t_interval

        # Averaging over multiple timesteps
        for l in range(0, spoints + t_interval, t_interval):
            # Averaging over all particles
            for j in range(nparticles):
                for k in range(nparticles):
                    if j != k:
                        dnorm, d = pbc.dist_mic(
                            ndims, pos[j, :, l], pos[k, :, l], length, hlength
                        )

                        if dnorm > r[i] and dnorm < width:
                            npp = npp + 1

        avglocalden = npp / (nparticles * tavg)
        avglocalden = (avglocalden) / (3.14 * ((r[i] + dr) ** 2 - r[i] ** 2))  # 2D
        gr[i] = avglocalden / (pointden)

    return (gr, r)


# Calculate the unfolded position co ordinates of each particle's trajectory
# for MSD calulcations
@jit(nopython=True)
def unfolded_positions(pos, length, tr):
    nparticles = pos.shape[0]
    ndims = pos.shape[1]
    spoints = pos.shape[2]
    pos_unf = np.zeros((nparticles, ndims, spoints), dtype=np.float64)

    for i in range(spoints):
        pos_unf[:, :, i] = pos[:, :, i] + length * tr[:, :, i]

    return pos_unf


# Calculate MSD (without time windowing)
@jit(nopython=True)
def msd(pos_unf, sigma, saved):
    nparticles = pos_unf.shape[0]
    ndims = pos_unf.shape[1]
    spoints = pos_unf.shape[2]

    msd = np.zeros((spoints), dtype=np.float64)
    taus = np.zeros((spoints), dtype=np.float64)

    for i in range(1, spoints):
        disp = np.zeros((nparticles), dtype=np.float64)
        taus[i] = i * 1e-5 * saved  # 1e-5 tau_B is the timestep
        for j in range(nparticles):
            dnorm, r = pbc.dist(ndims, pos_unf[j, :, i], pos_unf[j, :, 0])
            disp[j] = disp[j] + dnorm**2

        msd[i] = (
            np.sum(disp * sigma**2) / nparticles
        )  # Dimension of disp^2 is sigma^2

    return (msd, taus)


# Compute quantity b_ij=1 if particle i and j are nearest neighbours
# and b_ij=0 otherwise
@jit(nopython=True)
def bij(pos, length, hlength, sigma):
    nparticles = pos.shape[0]
    ndims = pos.shape[1]
    b_ij = np.zeros((nparticles, nparticles), dtype=np.intc)

    for i in range(nparticles):
        for j in range(i + 1, nparticles):  # Avoid double counting
            # Minimum image criteria distance computation
            rnorm, r = pbc.dist_mic(ndims, pos[i, :], pos[j, :], length, hlength)

            if rnorm < 1.15 * sigma:
                b_ij[i, j] = 1
                b_ij[j, i] = 1

    return b_ij


# Function to compute the bond autocorrelation funciton
# 2 particles are defined as being bonded if they are nearest neighbours
@jit(nopython=True)
def bond_ac(pos, length, hlength, sigma):
    nparticles = pos.shape[0]
    ndims = pos.shape[1]
    spoints = pos.shape[2]

    bac = np.zeros((spoints), dtype=np.float64)
    taus = np.zeros((spoints), dtype=np.float64)

    for t in range(1, spoints):
        taus[t] = t * 1e-5
        b_ij0 = bij(pos[:, :, t - 1], length, hlength, sigma)
        b_ij = bij(pos[:, :, t], length, hlength, sigma)
        bac[t] = np.sum(b_ij0 * b_ij) / np.sum(b_ij0)

    return bac, taus
