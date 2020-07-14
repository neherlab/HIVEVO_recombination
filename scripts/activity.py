from trajectory import Trajectory, create_trajectory_list, filter
from hivevo.patients import Patient
import filenames
import matplotlib.pyplot as plt
import numpy as np


def get_activity(patient, region, normalize=False, remove_one_point_traj=False):
    time_bins = np.linspace(0, 2000, 30)
    aft = patient.get_allele_frequency_trajectories(region)
    trajectories = create_trajectory_list(patient, region, aft)
    filtered_traj = [traj for traj in trajectories if np.sum(traj.frequencies > 0.2, dtype=bool)]
    if remove_one_point_traj:
        filtered_traj = [traj for traj in filtered_traj if traj.t[-1] > 0]  # Remove 1 point only trajectories

    filtered_fixed = [traj for traj in filtered_traj if traj.fixation == "fixed"]
    filtered_lost = [traj for traj in filtered_traj if traj.fixation == "lost"]
    filtered_active = [traj for traj in filtered_traj if traj.fixation == "active"]
    fixed, lost, active = [], [], []

    for ii in range(len(time_bins)):
        nb_fixed = len([traj for traj in filtered_fixed if traj.t[-1] < time_bins[ii]])
        nb_lost = len([traj for traj in filtered_lost if traj.t[-1] < time_bins[ii]])
        nb_active = len([traj for traj in filtered_traj if traj.t[-1] >= time_bins[ii]])
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


def average_activity(patient_names, region, normalize=True, remove_one_point_traj=False):
    for patient_name in patient_names:
        patient = Patient.load(patient_name)

        time_bins, fixed, lost, active, sum = get_activity(patient, region, False, remove_one_point_traj)
        if patient_name == patient_names[0]:
            tot_fixed = np.array(fixed)
            tot_lost = np.array(lost)
            tot_active = np.array(active)
            tot_sum = np.array(sum)
        else:
            tot_fixed += fixed
            tot_lost += lost
            tot_active += active
            tot_sum += sum

    if normalize:
        tot_fixed = np.array(tot_fixed) / tot_sum
        tot_lost = np.array(tot_lost) / tot_sum
        tot_active = np.array(tot_active) / tot_sum
        tot_sum = np.ones_like(tot_fixed)

    return time_bins, tot_fixed, tot_lost, tot_active, tot_sum
<
def plot_average_activity(region, time_bins, fixed, lost, active, sum, fontsize=16):
    plt.figure(figsize=(10,8))
    plt.title(f"Average activity region {region}", fontsize=fontsize)
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
    patient_names = ["p1", "p2", "p4", "p5", "p6", "p8", "p9", "p11"]
    region = "env"
    normalize = True
    remove_one_point_traj = False

    time_bins, fixed, lost, active, sum = average_activity(patient_names, region, normalize, remove_one_point_traj)
    plot_average_activity(region, time_bins, fixed, lost, active, sum)
