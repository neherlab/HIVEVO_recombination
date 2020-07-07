import numpy as np
import matplotlib.pyplot as plt
import filenames
from hivevo.patients import Patient
from trajectory import Trajectory, create_trajectory_list, filter


def get_activity(trajectory_list, normalize=False):
    time_bins = np.linspace(0, 2000, 50)
    filtered_fixed = [traj for traj in trajectory_list if traj.fixation == "fixed"]
    filtered_lost = [traj for traj in trajectory_list if traj.fixation == "lost"]
    filtered_active = [traj for traj in trajectory_list if traj.fixation == "active"]
    fixed, lost, active = [], [], []

    for ii in range(len(time_bins)):
        nb_fixed = len([traj for traj in filtered_fixed if traj.t[-1] < time_bins[ii]])
        nb_lost = len([traj for traj in filtered_lost if traj.t[-1] < time_bins[ii]])
        nb_active = len([traj for traj in trajectory_list if traj.t[-1] >= time_bins[ii]])
        # not adding len([traj for traj in filtered_active if traj.t[-1] < time_bins[ii]]) because we don't know how long they stay active as they are from last timepoint

        fixed = fixed + [nb_fixed]
        lost = lost + [nb_lost]
        active = active + [nb_active]

    sum = np.array(fixed) + np.array(lost) + np.array(active)

    if normalize:
        fixed = np.array(fixed) / sum
        lost = np.array(lost) / sum
        active = np.array(active) / sum
        sum = np.ones_like(fixed)
    return time_bins, fixed, lost, active, sum


def plot_activity(patient, region, time_bins, fixed, lost, active, sum, fontsize=16):
    plt.figure()
    plt.title(f"Activity patient {patient.name} region {region}", fontsize=fontsize)
    plt.plot(time_bins, fixed, label="fixed")
    plt.plot(time_bins, lost, label="lost")
    plt.plot(time_bins, active, label="active")
    if sum[0] != 1:
        plt.plot(time_bins, sum, label="sum")
    plt.legend(fontsize=fontsize)
    plt.xlabel("Time [days]", fontsize=fontsize)
    plt.ylabel("# Trajectories", fontsize=fontsize)
    plt.show()


if __name__ == "__main__":
    patient_name = "p1"
    patient = Patient.load(patient_name)
    region = "env"

    aft = patient.get_allele_frequency_trajectories(region)
    trajectories = create_trajectory_list(patient, region, aft)
    filtered_traj = [traj for traj in trajectories if np.sum(traj.frequencies > 0.2, dtype=bool)]
    filtered_traj = [traj for traj in trajectories if traj.t[-1] > 0]  # Remove 1 point only trajectories

    time_bins, fixed, lost, active, sum = get_activity(filtered_traj)
    plot_activity(patient, region, time_bins, fixed, lost, active, sum)
