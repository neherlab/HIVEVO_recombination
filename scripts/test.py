# Adds link to the scripts folder
import filenames
from hivevo.HIVreference import HIVreference
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from hivevo.patients import Patient
import trajectory
import tools
import divergence
sys.path.append("../scripts/")



if __name__ == "__main__":
    region = "pol"
    patient_name = "p1"
    patient = Patient.load(patient_name)
    aft = patient.get_allele_frequency_trajectories(region)
    # Masking low depth
    depth = trajectory.get_depth(patient, region)
    depth = np.tile(depth, (6, 1, 1))
    depth = np.swapaxes(depth, 0, 1)
    aft.mask = np.logical_or(aft.mask, ~depth)

    initial_idx = patient.get_initial_indices(region)
    aft_initial = aft[np.arange(aft.shape[0])[:, np.newaxis, np.newaxis], initial_idx, np.arange(aft.shape[-1])]
    aft_initial = aft_initial[:,0,:]

    threshold_low = 0.05
    threshold_high = 0.95

    mask = aft_initial <= threshold_low
    data = aft_initial[:, np.where(np.sum(mask, axis=0))]
    data = data[:,0,:]

    # plt.figure()
    # plt.plot(data)
    # plt.show()

    divergence_matrix = divergence.divergence_matrix(aft)
    div = divergence_matrix[np.arange(aft.shape[0])[:, np.newaxis, np.newaxis], initial_idx, np.arange(aft.shape[-1])]
    div = div[:, 0, :]
    hist, bins = np.histogram(div[-3:,:], bins=1000)
    bins = bins[:-1]
    hist_sum = np.cumsum(hist)
    hist_sum = hist_sum / np.max(hist_sum)

    plt.figure()
    plt.plot(bins, hist_sum, label="hist")
    plt.legend()
    plt.grid()
    plt.show()
