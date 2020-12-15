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
from activity import get_average_activity
sys.path.append("../scripts/")


def get_mean_in_time(trajectories, nb_bins=15):
    """
    Computes the mean frequency in time of a set of trajectories from the point they are seen in the freq_range window.
    Returns the middle of the time bins and the computed frequency mean.
    """
    # Create bins and select trajectories going through the freq_range
    time_bins = np.linspace(-110, 3000, nb_bins)

    # Offset trajectories to set t=0 at the point they are seen in the freq_range and adds all the frequencies / times
    # to arrays for later computation of mean
    t_traj = np.array([])
    f_traj = np.array([])
    for traj in trajectories:
        traj.t += 0.5 * traj.t_previous_sample
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
        mean[-1] /= (len(freqs[ii]) + fixed[ii] + lost[ii])

    nb_active = [len(freq) for freq in freqs]
    nb_dead = [fixed[ii] + lost[ii] for ii in range(len(fixed))]

    return 0.5 * (time_bins[1:] + time_bins[:-1]), mean, nb_active, nb_dead


freq_range = [0.2, 0.4]
trajectories = load_trajectory_dict()
trajectories = trajectories["env"]["non_syn"]
trajectories = [traj for traj in trajectories if np.sum(np.logical_and(
    traj.frequencies[0] >= freq_range[0], traj.frequencies[0] < freq_range[1]), dtype=bool)]
time, mean, _, _ = get_mean_in_time(trajectories)

plt.figure()
plt.plot(time, mean, '.-')
plt.grid()
plt.xlabel("Age [days]")
plt.ylabel("Frequency")
plt.ylim([0, 1])
plt.show()
