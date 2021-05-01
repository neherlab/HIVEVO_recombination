# Adds link to the scripts folder
import filenames
from hivevo.HIVreference import HIVreference
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from hivevo.patients import Patient
from divergence import get_non_consensus_mask, get_consensus_mask
import trajectory
sys.path.append("../scripts/")

def get_sweep_mask(patient, aft, region, threshold_low=0.05):
    # Masking low depth
    depth = trajectory.get_depth(patient, region)
    depth = np.tile(depth, (6, 1, 1))
    depth = np.swapaxes(depth, 0, 1)
    aft.mask = np.logical_or(aft.mask, ~depth)

    initial_idx = patient.get_initial_indices(region)
    aft_initial = aft[np.arange(aft.shape[0])[:, np.newaxis, np.newaxis],
                      initial_idx, np.arange(aft.shape[-1])]
    aft_initial = aft_initial[:, 0, :]

    mask = aft_initial <= threshold_low
    mask = np.sum(mask, axis=0)
    return mask


if __name__ == "__main__":
    region = "pol"
    patient = Patient.load("p1")
    aft = patient.get_allele_frequency_trajectories(region)
    sweep_mask = get_sweep_mask(patient, aft, region)

    idxs = np.where(sweep_mask)[0]
    fontsize = 24
    ticksize = 16

    plt.figure(figsize= (10, 7))
    plt.plot(patient.dsi / 365, aft[:,:,idxs[2]])
    plt.xlabel("Time [years]", fontsize = fontsize)
    plt.ylabel("Frequency", fontsize = fontsize)
    plt.tick_params(axis="both", labelsize=ticksize)
    plt.grid()
    plt.tight_layout()
    plt.legend(["A", "C", "G" ,"T" ,"-" ,"N"], fontsize = fontsize)
    plt.savefig("Trajectory_example.png", format="png")
    plt.show()
