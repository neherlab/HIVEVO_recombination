# Adds link to the scripts folder
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


def get_mean_in_time(trajectories, nb_bins=15, freq_range=[0.4, 0.6]):
    """
    Computes the mean frequency in time of a set of trajectories from the point they are seen in the freq_range window.
    Returns the middle of the time bins and the computed frequency mean.
    """
    # Create bins and select trajectories going through the freq_range
    time_bins = np.linspace(-1000, 2000, nb_bins)
    trajectories = [traj for traj in trajectories if np.sum(np.logical_and(
        traj.frequencies >= freq_range[0], traj.frequencies < freq_range[1]), dtype=bool)]

    # Offset trajectories to set t=0 at the point they are seen in the freq_range and adds all the frequencies / times
    # to arrays for later computation of mean
    t_traj = np.array([])
    f_traj = np.array([])
    for traj in trajectories:
        idx = np.where(np.logical_and(traj.frequencies >=
                                      freq_range[0], traj.frequencies < freq_range[1]))[0][0]
        traj.t = traj.t - traj.t[idx]
        t_traj = np.concatenate((t_traj, traj.t))
        f_traj = np.concatenate((f_traj, traj.frequencies))

    # Binning of all the data in the time bins
    filtered_fixed = [traj for traj in trajectories if traj.fixation == "fixed"]
    filtered_lost = [traj for traj in trajectories if traj.fixation == "lost"]
    freqs, fixed, lost = [], [], []
    for ii in range(len(time_bins) - 1):
        freqs = freqs + [f_traj[np.logical_and(t_traj >= time_bins[ii], t_traj < time_bins[ii + 1])]]
        fixed = fixed + [len([traj for traj in filtered_fixed if traj.t[-1] < time_bins[ii]])]
        lost = lost + [len([traj for traj in filtered_lost if traj.t[-1] < time_bins[ii]])]

    # Computation of the mean in each bin, active trajectories contribute their current frequency,
    # fixed contribute1 and lost contribute 0
    mean = []
    for ii in range(len(freqs)):
        mean = mean + [np.sum(freqs[ii]) + fixed[ii]]
        mean[-1] /= (len(freqs[ii]) + fixed[ii] + lost[ii])

    nb_active = [len(freq) for freq in freqs]
    nb_dead = [fixed[ii] + lost[ii] for ii in range(len(fixed))]

    return 0.5 * (time_bins[1:] + time_bins[:-1]), mean, nb_active, nb_dead


# regions = ["env", "pol", "gag"]
# trajectories = {}
#
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
#
# means = {}
# freq_ranges = [[0.2, 0.4], [0.4, 0.6], [0.6, 0.8]]
# times = []
#
# for freq_range in freq_ranges:
#     means[str(freq_range)] = {}
#     for region in regions:
#         means[str(freq_range)][region] = {}
#         for key in trajectories[region].keys():
#             times, means[str(freq_range)][region][key], _, _ = get_mean_in_time(
#                 trajectories[region][key], freq_range=freq_range)



regions = ["env", "pol", "gag"]
times = [-892.85714286, -678.57142857, -464.28571429, -250.        ,
        -35.71428571,  178.57142857,  392.85714286,  607.14285714,
        821.42857143, 1035.71428571, 1250.        , 1464.28571429,
       1678.57142857, 1892.85714286]

fig, axs = plt.subplots(ncols=3, nrows=2)
for idx_row, split_type in enumerate([["rev", "non_rev"], ["syn", "non_syn"]]):
    for idx_col, region in enumerate(regions):
        freq_range = str([0.4, 0.6])
        axs[idx_row, idx_col].plot(times, means[freq_range][region][split_type[0]], '-.')
        axs[idx_row, idx_col].plot(times, means[freq_range][region][split_type[1]], '--.')
