# Adds link to the scripts folder
import filenames
from hivevo.patients import Patient
from trajectory import Trajectory, load_trajectory_dict
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import copy
import os
import sys
sys.path.append("../../scripts/")


def get_mean_in_time(trajectories, nb_bins=20, freq_range=[0.4, 0.6]):
    """
    Computes the mean frequency in time of a set of trajectories from the point they are seen in the freq_range window.
    Returns the middle of the time bins and the computed frequency mean.
    """
    trajectories = copy.deepcopy(trajectories)

    # Create bins and select trajectories going through the freq_range
    time_bins = np.linspace(-677, 3000, nb_bins)
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
    # fixed contribute 1 and lost contribute 0
    mean = []
    for ii in range(len(freqs)):
        mean = mean + [np.sum(freqs[ii]) + fixed[ii]]
        if len(freqs[ii]) + fixed[ii] + lost[ii] != 0:
            mean[-1] /= (len(freqs[ii]) + fixed[ii] + lost[ii])
        else:
            mean[-1] = np.nan
    nb_active = [len(freq) for freq in freqs]
    nb_dead = [fixed[ii] + lost[ii] for ii in range(len(fixed))]

    return 0.5 * (time_bins[1:] + time_bins[:-1]), mean, nb_active, nb_dead


def make_mean_in_time_dict(trajectories):
    regions = ["env", "pol", "gag", "all"]
    means = {}
    freq_ranges = [[0.2, 0.4], [0.4, 0.6], [0.6, 0.8]]
    times = []

    for freq_range in freq_ranges:
        means[str(freq_range)] = {}
        for region in regions:
            means[str(freq_range)][region] = {}
            for key in trajectories[region].keys():
                times, means[str(freq_range)][region][key], _, _ = get_mean_in_time(
                    trajectories[region][key], freq_range=freq_range)
    return times, means, freq_ranges


def bootstrap_patient_names(patient_names=["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]):
    "Returns a list of patient name bootstrap with replacement (patient can appear more than once)."
    choice = np.random.choice(range(len(patient_names)), len(patient_names))
    names = []
    for ii in choice:
        names += [patient_names[ii]]
    return names


def bootstrap_mean_in_time(trajectories, region, mut_type, freq_range, nb_bootstrap=10):
    """
    Computes the mean_in_time for the given region and mutation type (rev, non_rev etc...) nb_boostrap time
    by bootstrapping patients and returns the average and standard deviation vectors.
    """

    means = []
    for ii in range(nb_bootstrap):
        # Bootstrapping trajectories
        bootstrap_names = bootstrap_patient_names()
        bootstrap_trajectories = []
        for name in bootstrap_names:
            bootstrap_trajectories += [traj for traj in trajectories[region][mut_type] if traj.patient==name]

        # Computing the mean in time for each boostrap
        time, mean, _, _ = get_mean_in_time(bootstrap_trajectories, freq_range=freq_range)
        means += [[mean]]

    means = np.array(means)
    average = np.nanmean(means, axis=0)
    std = np.nanstd(means, axis=0)

    return time, average, std


def make_pfix(nb_bin=8):
    regions = ["env", "pol", "gag", "all"]
    pfix = {}
    for region in regions:
        pfix[region] = {}
        for key in trajectories[region].keys():
            tmp_freq_bin, tmp_proba, tmp_err = get_proba_fix(trajectories[region][key], nb_bin=nb_bin)
            pfix[region][key] = {"freq_bin": tmp_freq_bin, "proba": tmp_proba, "error": tmp_err}
    return pfix


def get_trajectories_offset(trajectories, freq_range):
    trajectories = copy.deepcopy(trajectories)
    trajectories = [traj for traj in trajectories if np.sum(np.logical_and(
        traj.frequencies >= freq_range[0], traj.frequencies < freq_range[1]), dtype=bool)]

    # Offset trajectories to set t=0 at the point they are seen in the freq_range and adds all the frequencies / times
    # to arrays for later computation of mean
    for traj in trajectories:
        idx = np.where(np.logical_and(traj.frequencies >=
                                      freq_range[0], traj.frequencies < freq_range[1]))[0][0]
        traj.t = traj.t - traj.t[idx]
        if traj.fixation == "fixed":
            traj.t = np.append(traj.t, [traj.t[-1] + 300, 3000])
            traj.frequencies = np.append(traj.frequencies, [1, 1])
        elif traj.fixation == "lost":
            traj.t = np.append(traj.t, [traj.t[-1] + 300, 3000])
            traj.frequencies = np.append(traj.frequencies, [0, 0])

    return trajectories


if __name__ == "__main__":
    trajectories = load_trajectory_dict("trajectory_dict")
    # times, means, freq_ranges = make_mean_in_time_dict(trajectories)
    time, mean, std = bootstrap_mean_in_time(trajectories, "pol", "rev", [0.6, 0.8])
    # trajectories_scheme = get_trajectories_offset(trajectories["all"]["rev"], [0.4, 0.6])

    # fontsize=16
    # grid_alpha = 0.5
    # colors = ["C0","C1","C2","C4"]
    # markersize=12
    # freq_ranges = [[0.2, 0.4], [0.4, 0.6], [0.6, 0.8]]
    # regions = ["env","pol","gag"]
    # lines = ["-","--"]
    #
    # fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(14,7), sharey=True)
    #
    # # Plot left
    #
    # for traj in trajectories_scheme:
    #     axs[0].plot(traj.t, traj.frequencies, "k-", alpha=0.1, linewidth=1)
    #
    # axs[0].plot(times, means["[0.4, 0.6]"]["all"]["rev"], '-', color=colors[1])
    #
    # axs[0].set_xlabel("Time [days]", fontsize=fontsize)
    # axs[0].set_ylabel("Frequency", fontsize=fontsize)
    # axs[0].set_ylim([-0.03, 1.03])
    # axs[0].grid(grid_alpha)
    # axs[0].set_xlim([-677, 3000])
    #
    # line1, = axs[0].plot([0], [0], "k-")
    # line2, = axs[0].plot([0], [0], "-", color=colors[1])
    # axs[0].legend([line1, line2], ["Individual trajectories", "Average"], fontsize=fontsize, loc="lower right")
    #
    #
    # # Plot right
    # for ii, freq_range in enumerate(freq_ranges):
    #     axs[1].plot(times, means[str(freq_range)]["all"]["rev"], "-", color=colors[ii])
    #     axs[1].plot(times, means[str(freq_range)]["all"]["non_rev"], "--", color=colors[ii])
    #
    # line1, = axs[1].plot([0], [0], "k-")
    # line2, = axs[1].plot([0], [0], "k--")
    # line3, = axs[1].plot([0], [0], "-", color=colors[0])
    # line4, = axs[1].plot([0], [0], "-", color=colors[1])
    # line5, = axs[1].plot([0], [0], "-", color=colors[2])
    #
    # axs[1].set_xlabel("Time [days]", fontsize=fontsize)
    # # axs[1].set_ylabel("Frequency", fontsize=fontsize)
    # axs[1].set_ylim([-0.03, 1.03])
    # axs[1].grid(grid_alpha)
    # axs[1].legend([line3, line4, line5, line1, line2], ["[0.2, 0.4]", "[0.4, 0.6]", "[0.6, 0.8]", "reversion", "non-reversion"], fontsize=fontsize, ncol=2, loc="lower right")
    #
    # plt.tight_layout()
    # plt.savefig("Reversion_PAC.png", format="png")
    # plt.show()
