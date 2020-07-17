import numpy as np
import matplotlib.pyplot as plt
import filenames
from hivevo.patients import Patient
from trajectory import Trajectory, create_trajectory_list, filter


def get_duration_distribution(trajectory_list):
    time_bins = np.linspace(0, 2000, 100)
    nb_traj = []

    for time in time_bins:
        filtered_traj = filter(trajectory_list, f"traj.t[-1] >= {time}")
        nb_traj = nb_traj + [len(filtered_traj)]
    return time_bins, nb_traj


def get_freq_distribution(trajectories, nb_bin=10):
    frequency_bins = np.linspace(0.01, 0.99, nb_bin)
    nb_in = []
    for ii in range(len(frequency_bins) - 1):
        nb = len([traj for traj in trajectories if np.sum(np.logical_and(
            traj.frequencies >= frequency_bins[ii], traj.frequencies < frequency_bins[ii + 1]), dtype=bool)])
        nb_in = nb_in + [nb]

    return frequency_bins, nb_in


def plot_duration_distribution(time_bins, nb_traj, freq_min, fontsize=16):
    plt.figure()
    plt.title(f"Trajectory duration freq>{round(freq_min,1)}", fontsize=fontsize)
    plt.plot(time_bins, nb_traj, '.-')
    plt.xlabel("t [days]", fontsize=fontsize)
    plt.ylabel("#Trajectories > t", fontsize=fontsize)
    plt.grid()


def plot_length_distribution(trajectory_list, freq_min, fontsize=16):
    plt.figure()
    plt.title(f"Trajectory length freq>{round(freq_min,1)}", fontsize=fontsize)
    plt.hist([len(x.t) for x in trajectory_list], bins=range(12), align="left", rwidth=0.7)
    plt.xlabel("Length", fontsize=fontsize)
    plt.ylabel("#Trajectories", fontsize=fontsize)


def plot_freq_ditribution(frequency_bins, nb_in, fontsize=16):
    frequency_bins = 0.5 * (frequency_bins[:-1] + frequency_bins[1:])
    plt.figure()
    plt.title("Distribution of trajectories in freq", fontsize=16)
    plt.plot(frequency_bins, nb_in, '.-')
    plt.xlabel("Frequency bin center", fontsize=16)
    plt.ylabel("# Trajectories", fontsize=16)
    plt.yscale("log")
    plt.xlim([0, 1])
    plt.grid()


if __name__ == "__main__":
    patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]
    region = "env"
    trajectories = []

    for patient_name in patient_names:
        patient = Patient.load(patient_name)
        aft = patient.get_allele_frequency_trajectories(region)
        trajectories = trajectories + create_trajectory_list(patient, region, aft)

    ### Trajectory durations ###
    # for freq_min in np.arange(0, 1, 0.2):
    #     filtered_traj = [traj for traj in trajectories if np.sum(traj.frequencies > freq_min, dtype=bool)]
    #     time_bins, nb_traj = get_duration_distribution(filtered_traj)
    #     plot_duration_distribution(time_bins, nb_traj, freq_min)

    ### Trajectory lengths ###
    # for freq_min in np.arange(0, 1, 0.2):
    #     filtered_traj = [x for x in trajectories if np.sum(x.frequencies > freq_min, dtype=bool)]
    #     plot_length_distribution(filtered_traj, freq_min)

    ### Number of trajectory seen in frequency bin ###
    frequency_bins, nb_in = get_freq_distribution(trajectories, nb_bin=11)
    plot_freq_ditribution(frequency_bins, nb_in)

    plt.show()
