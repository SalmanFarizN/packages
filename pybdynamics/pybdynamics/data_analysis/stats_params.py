import numpy as np
from numba import jit, objmode
from pybdynamics import pbc


# Radial distrubution function calculation
@jit(nopython=True)
def rdf_old(pos, length, rho, rmax, bins, dr, t_interval):
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


# Function to calculate the distance between all pairs of particles
# in a given frame t
@jit(nopython=True)
def dist_all_pairs(pos, length, hlength, t):
    ndims = pos.shape[1]
    nparticles = pos.shape[0]
    spoints = pos.shape[2]

    distarr = np.full((nparticles * nparticles), np.nan)
    k = 0
    for i in range(nparticles):
        for j in range(nparticles):
            if i != j:
                dnorm, d = pbc.dist_mic(
                    ndims, pos[i, :, t], pos[j, :, t], length, hlength
                )
                distarr[k] = dnorm
                k = k + 1

    return distarr


# Function to calculate angle between all pairs of particles
# in a given frame t (and also the distance between them)
@jit(nopython=True)
def angle_all_pairs(pos, length, hlength, t):
    ndims = pos.shape[1]
    nparticles = pos.shape[0]
    spoints = pos.shape[2]

    anglearr = np.full((nparticles * nparticles), np.nan)
    distarr = np.full((nparticles * nparticles), np.nan)
    k = 0
    for i in range(nparticles):
        for j in range(nparticles):
            if i != j:
                dnorm, d = pbc.dist_mic(
                    ndims, pos[i, :, t], pos[j, :, t], length, hlength
                )

                if d[1] >= 0:
                    anglearr[k] = np.arccos(d[0] / dnorm)
                if d[1] < 0:
                    anglearr[k] = 2 * np.pi - np.arccos(d[0] / dnorm)

                # anglearr[k] = np.arccos(d[0] / dnorm)
                distarr[k] = dnorm
                k = k + 1

    return distarr, anglearr


# Radial distrubution function of a single frame without time averaging
# but averaging over all particles


def rdf(pos, length, rho, rmax, bins, t):
    hlength = length / 2.0
    ndims = pos.shape[1]
    nparticles = pos.shape[0]
    spoints = pos.shape[2]
    # Calculate the distances between all pairs of particles
    distarr = dist_all_pairs(pos, length, hlength, t)

    # Calculate the radial distribution function using histogram
    # of the distances between all pairs of particles
    gr, r = np.histogram(distarr, bins=bins, range=(0, rmax), density=False)
    r_centers = (r[1:] + r[:-1]) / 2
    dr = r[1] - r[0]

    for i in range(len(r_centers)):
        gr[i] = gr[i] / (np.pi * ((r[i] + dr) ** 2 - r[i] ** 2) * rho)

    return (gr, r_centers)


# Angle dependent radial distrubution function of a single frame without time averaging
# but averaging over all particles
def adrdf(pos, length, rho, rmax, rbins, amax, abins, t):
    hlength = length / 2.0
    ndims = pos.shape[1]
    nparticles = pos.shape[0]
    spoints = pos.shape[2]

    # Caalculate the distances and angles between all pairs of particles
    distarr, anglearr = angle_all_pairs(pos, length, hlength, t)

    # Calculate the 2D histogram of the angle dependent RDF
    gr, r, a = np.histogram2d(
        distarr,
        anglearr,
        bins=(rbins, abins),
        range=((0, rmax), (0, amax)),
        density=False,
    )

    r_centers = (r[1:] + r[:-1]) / 2
    a_centers = (a[1:] + a[:-1]) / 2
    dr = r[1] - r[0]
    da = a[1] - a[0]

    for i in range(len(r_centers)):
        gr[i] = gr[i] / (np.pi * ((r[i] + dr) ** 2 - r[i] ** 2) * da * rho)

    return (gr, r_centers, a_centers)


# Time averaged angle dependant radial distrubution function
def adrdf_tavg(pos, length, rho, rmax, rbins, amax, abins, tstart, tend):
    ndims = pos.shape[1]
    nparticles = pos.shape[0]
    spoints = pos.shape[2]

    for i in range(tstart, tend):
        gr, r, a = adrdf(pos, length, rho, rmax, rbins, amax, abins, i)
        if i == tstart:
            gr_tavg = gr
        else:
            gr_tavg = gr_tavg + gr

    gr_tavg = gr_tavg / (tend - tstart)

    return (gr_tavg, r, a)


# Time avergaed radial distrubution function
def rdf_tavg(pos, length, rho, rmax, bins, tstart, tend):
    ndims = pos.shape[1]
    nparticles = pos.shape[0]
    spoints = pos.shape[2]

    for i in range(tstart, tend):
        gr, r = rdf(pos, length, rho, rmax, bins, i)
        if i == tstart:
            gr_tavg = gr
        else:
            gr_tavg = gr_tavg + gr

    gr_tavg = gr_tavg / (tend - tstart)

    return (gr_tavg, r)


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

        msd[i] = np.sum(disp * sigma**2) / nparticles  # Dimension of disp^2 is sigma^2

    return (msd, taus)


# Calculate MSD (with time windowing)
@jit(nopython=True)
def msd_tw(pos_unf, sigma, saved):
    nparticles = pos_unf.shape[0]
    ndims = pos_unf.shape[1]
    spoints = pos_unf.shape[2]

    msd = np.zeros((spoints), dtype=np.float64)
    taus = np.zeros((spoints), dtype=np.float64)

    # loop over different lag-times i
    for i in range(1, spoints):
        taus[i] = i * 1e-5 * saved  # 1e-5 tau_B is the timestep
        tw = 0
        # Loop over all possible time windows
        while i + tw < spoints:
            if tw >= 5:
                break
            disp = np.zeros((nparticles), dtype=np.float64)
            # Compute the displacement of each particle in the time window
            for j in range(nparticles):
                dnorm, r = pbc.dist(ndims, pos_unf[j, :, i + tw], pos_unf[j, :, tw])
                disp[j] = disp[j] + dnorm**2
            # Add the MSD of each time window to the total MSD at that lag-time i
            msd[i] = msd[i] + (
                np.sum(disp * sigma**2) / nparticles
            )  # Dimension of disp^2 is sigma^2
            # Update the time window
            tw += 1

        # Normalize the MSD by the number of time windows if it is non-zero
        if tw != 0:
            msd[i] = msd[i] / tw

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
def bond_ac(pos, length, hlength, sigma, saved):
    nparticles = pos.shape[0]
    ndims = pos.shape[1]
    spoints = pos.shape[2]

    bac = np.zeros((spoints), dtype=np.float64)
    taus = np.zeros((spoints), dtype=np.float64)

    b_ij0 = bij(pos[:, :, 0], length, hlength, sigma)
    # Flatten 2D array to 1D array
    b_ij0 = b_ij0.flatten()
    normalization = np.sum(b_ij0)

    for t in range(1, spoints):
        taus[t] = t * 1e-5 * saved
        b_ij = bij(pos[:, :, t], length, hlength, sigma)
        # Flatten 2D array to 1D arrays
        b_ij = b_ij.flatten()

        bac[t] = np.sum(b_ij0 * b_ij) / normalization

    return bac, taus


# Function to compute the time windowed bond autocorrelation function
# 2 particles are defined as being bonded if they are nearest neighbours
@jit(nopython=True)
def bond_ac_tw(pos, length, hlength, sigma, saved):
    nparticles = pos.shape[0]
    ndims = pos.shape[1]
    spoints = pos.shape[2]

    bac = np.zeros((spoints), dtype=np.float64)
    taus = np.zeros((spoints), dtype=np.float64)

    # Loop over different lag-times i
    for i in range(1, spoints):
        taus[i] = i * 1e-5 * saved
        tw = 0
        # Loop over all possible time windows
        while i + tw < spoints:
            if tw >= 5:
                break
            b_ij0 = bij(pos[:, :, tw], length, hlength, sigma)
            b_ij = bij(pos[:, :, i + tw], length, hlength, sigma)
            # Flatten both 2D arrays to 1D arrays
            b_ij0 = b_ij0.flatten()
            b_ij = b_ij.flatten()

            bac[i] = bac[i] + np.sum(b_ij0 * b_ij) / np.sum(b_ij0)
            tw += 1

        # Normalize the bond autocorrelation by the number of time windows if it is non-zero
        if tw != 0:
            bac[i] = bac[i] / tw

    return bac, taus


# Fucntion to compute the particle level entropy (entropy produced by the particle due to heat transported
# by the particle into the bath) using Stratanovic integration rules
@jit(nopython=True)
def entropy_strat(pos_prev, pos, pos_new, force, kBT, step_size, length, hlength):
    nparticles = pos.shape[0]
    ndims = pos.shape[1]
    force_scalar = np.zeros((nparticles), dtype=np.float64)
    disp_vec = np.zeros((nparticles, ndims), dtype=np.float64)
    disp_scalar = np.zeros((nparticles), dtype=np.float64)
    delta_q = np.zeros((nparticles), dtype=np.float64)
    entropy_prod = np.zeros((nparticles), dtype=np.float64)

    # Caluclate the displacement 2D vector b/w pos_prev and pos_new
    # using mimimum image criteria
    for i in range(nparticles):
        dnorm, disp_vec[i, :] = pbc.dist_mic(
            ndims, pos_new[i, :], pos_prev[i, :], length, hlength
        )
        disp_scalar[i] = dnorm

    disp_vec = (disp_vec) / (2.0 * step_size)

    # Calculate heat produced by each particle into the bath
    # Q_dot = Force * (r(t+dt)-r(t-dt))/2dt
    for i in range(nparticles):
        delta_q[i] = np.dot(force[i, :], disp_vec[i, :])

    # Calculate the entropy produced by each particle into the bath
    # S_dot = delta_q/T
    entropy_prod = delta_q / kBT

    return entropy_prod
