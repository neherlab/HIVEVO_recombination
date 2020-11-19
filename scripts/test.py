# Adds link to the scripts folder
from proba_fix import get_nonuniform_bins
import tools
from activity import get_average_activity
import copy
import filenames
from hivevo.patients import Patient
from trajectory import Trajectory, create_trajectory_list, create_all_patient_trajectories
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
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
        for traj in bin_trajectories:
            tmp_mean = []
            idxs = np.where(np.logical_and(traj.frequencies >=
                                           frequency_bins[ii], traj.frequencies < frequency_bins[ii + 1]))[0]
            tmp_mean = tmp_mean + traj.frequencies[idxs].data.tolist()
        mean_freq_bin = mean_freq_bin + [np.mean(tmp_mean)]

    err_proba_fix = np.array(proba_fix) * np.sqrt(1 / (np.array(fixed_per_bin) +
                                                       1e-10) + 1 / np.array(traj_per_bin))

    return mean_freq_bin, proba_fix, err_proba_fix


regions = ["env", "pol", "gag"]
trajectories = {}

for region in regions:
    # Create the dictionary with the different regions
    tmp_trajectories = create_all_patient_trajectories(region)
    tmp_trajectories = [traj for traj in tmp_trajectories if traj.t[-1] != 0]
    trajectories[region] = tmp_trajectories

    # Split into sub dictionnaries (rev, non_rev and all)
    rev = [traj for traj in trajectories[region] if traj.reversion == True]
    non_rev = [traj for traj in trajectories[region] if traj.reversion == False]
    syn = [traj for traj in trajectories[region] if traj.synonymous == True]
    non_syn = [traj for traj in trajectories[region] if traj.synonymous == False]
    trajectories[region] = {"rev": rev, "non_rev": non_rev,
                            "syn": syn, "non_syn": non_syn, "all": trajectories[region]}

pfix = {}
for region in regions:
    pfix[region] = {}
    for key in trajectories[region].keys():
        tmp_freq_bin, tmp_proba, tmp_err = get_proba_fix(trajectories[region][key])
        pfix[region][key] = {"freq_bin": tmp_freq_bin, "proba": tmp_proba, "error": tmp_err}

plt.figure()
plt.errorbar(pfix["env"]["rev"]["freq_bin"], pfix["env"]["rev"]["proba"], yerr=pfix["env"]["rev"]["error"])
plt.errorbar(pfix["env"]["non_rev"]["freq_bin"], pfix["env"]["non_rev"]["proba"], yerr=pfix["env"]["non_rev"]["error"])
plt.plot([0, 1], [0, 1], 'k--')
plt.show()

# for region in regions:
#     # Create the dictionary with the different regions
#     tmp_trajectories = create_all_patient_trajectories(region)
#     tmp_trajectories = [traj for traj in tmp_trajectories if traj.t[-1] != 0]
#     trajectories[region] = tmp_trajectories
#
#     # Split into sub dictionnaries (rev, non_rev and all)
#     rev = [traj for traj in trajectories[region] if traj.reversion == True]
#     non_rev = [traj for traj in trajectories[region] if traj.reversion == False]
#     syn = [traj for traj in trajectories[region] if traj.synonymous == True]
#     non_syn = [traj for traj in trajectories[region] if traj.synonymous == False]
#     trajectories[region] = {"rev": rev, "non_rev": non_rev,
#                             "syn": syn, "non_syn": non_syn, "all": trajectories[region]}
