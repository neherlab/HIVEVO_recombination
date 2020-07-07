import numpy as np
import matplotlib.pyplot as plt
import filenames
from hivevo.patients import Patient
from trajectory import Trajectory, create_trajectory_list, filter


def get_duration_distribution(trajectory_list):
    time_bins = np.linspace(0, 2000, 100)
    nb_traj = []

    for time in time_bins:
        filtered_traj = filter(trajectories, f"traj.t[-1] >= {time}")
        nb_traj = nb_traj + [len(filtered_traj)]
    return time_bins, nb_traj


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
    plt.hist([len(x.t) for x in trajectory_list], bins=range(aft.shape[0]), align="left", rwidth=0.7)
    plt.xlabel("Length", fontsize=fontsize)
    plt.ylabel("#Trajectories", fontsize=fontsize)


if __name__ == "__main__":

    patient_name = "p1"
    patient = Patient.load(patient_name)
    region = "env"

    aft = patient.get_allele_frequency_trajectories(region)
    trajectories = create_trajectory_list(patient, region, aft)

    ### Trajectory durations ###
    for freq_min in np.arange(0, 1, 0.2):
        filtered_traj = filter(trajectories, f"np.sum(traj.frequencies > {freq_min}, dtype=bool)")
        time_bins, nb_traj = get_duration_distribution(trajectories)
        plot_duration_distribution(time_bins, nb_traj, freq_min)

    ### Trajectory lengths ###
    for freq_min in np.arange(0, 1, 0.2):
        filtered_traj = [x for x in trajectories if np.sum(x.frequencies > freq_min, dtype=bool)]
        plot_length_distribution(filtered_traj, freq_min)

    plt.show()
