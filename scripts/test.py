# Adds link to the scripts folder
import filenames
from hivevo.HIVreference import HIVreference
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle
from hivevo.patients import Patient
from trajectory import load_trajectory_dict
from proba_fix import get_nonuniform_bins
sys.path.append("../scripts/")

def get_proba_fix(trajectories, nb_bin=8, freq_range=[0.1, 0.9]):
    """
    Gives the probability of fixation in each frequency bin.
    """

    frequency_bins = get_nonuniform_bins(nb_bin, bin_range=freq_range)

    trajectories = [traj for traj in trajectories if traj.fixation != "active"]  # Remove active trajectories
    traj_per_bin, fixed_per_bin, lost_per_bin, proba_fix = [], [], [], []
    mean_freq_bin = []

    for ii in range(len(frequency_bins) - 1):
        bin_trajectories = [traj for traj in trajectories if np.sum(np.logical_and(
            traj.frequencies >= frequency_bins[ii], traj.frequencies < frequency_bins[ii + 1]), dtype=bool)]

        nb_traj = len(bin_trajectories)
        nb_fix = len([traj for traj in bin_trajectories if traj.fixation == "fixed"])
        nb_lost = len([traj for traj in bin_trajectories if traj.fixation == "lost"])

        traj_per_bin = traj_per_bin + [nb_traj]
        fixed_per_bin = fixed_per_bin + [nb_fix]
        lost_per_bin = lost_per_bin + [nb_lost]
        if nb_traj > 0:
            proba_fix = proba_fix + [nb_fix / nb_traj]
        else:
            proba_fix = proba_fix + [None]

        # Computes the "center" of the bin
        tmp_mean = []
        for traj in bin_trajectories:
            idxs = np.where(np.logical_and(traj.frequencies >=
                                           frequency_bins[ii], traj.frequencies < frequency_bins[ii + 1]))[0]
            tmp_mean = tmp_mean + [traj.frequencies[idxs[0]]]
        mean_freq_bin = mean_freq_bin + [np.ma.mean(tmp_mean)]

    err_proba_fix = np.array(proba_fix) * np.sqrt(1 / (np.array(fixed_per_bin) +
                                                       1e-10) + 1 / np.array(traj_per_bin))

    return mean_freq_bin, proba_fix, err_proba_fix

region = "env"
mut_type = "non_syn"
trajectories = load_trajectory_dict()

freq_bin, proba, _= get_proba_fix(trajectories[region][mut_type], nb_bin=8)

plt.plot(freq_bin, proba)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("Initial frequency")
plt.ylabel("P_fix")
plt.show()
