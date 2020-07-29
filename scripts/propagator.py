from trajectory import Trajectory, create_trajectory_list, filter, create_all_patient_trajectories
from hivevo.patients import Patient
import filenames
import matplotlib.pyplot as plt
import numpy as np
import copy


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
    filtered_fixed = [traj for traj in traj ectories if traj.fixation == "fixed"]
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


if __name__ == "__main__":
    patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]
    # patient_names = ["p1"]
    region = "env"
    nb_bins = 15
    freq_range = [0.4, 0.6]
    fontsize = 16

    trajectories = create_all_patient_trajectories(region, patient_names)
    syn_traj = copy.deepcopy([traj for traj in trajectories if traj.synonymous == True])
    non_syn_traj = copy.deepcopy([traj for traj in trajectories if traj.synonymous == False])

    time_bins, mean_syn, active_syn, dead_syn = get_mean_in_time(syn_traj, nb_bins, freq_range)
    time_bins, mean_non_syn, active_non_syn, dead_non_syn = get_mean_in_time(
        non_syn_traj, nb_bins, freq_range)

    plt.figure()
    plt.plot(time_bins, mean_syn, '.-', label="Synonymous")
    plt.plot(time_bins, mean_non_syn, '.-', label="Non-synonymous")
    plt.xlabel("Time [days]", fontsize=fontsize)
    plt.ylabel("Frequency", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.ylim([0, 1])

    plt.figure()
    plt.plot(time_bins, active_syn, '.-', label="active_syn")
    plt.plot(time_bins, dead_syn, '.-', label="dead_syn")
    plt.plot(time_bins, active_non_syn, '.-', label="active_non_syn")
    plt.plot(time_bins, dead_non_syn, '.-', label="dead_non_syn")
    plt.xlabel("Time [days]", fontsize=fontsize)
    plt.ylabel("# of trajectory", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.show()
