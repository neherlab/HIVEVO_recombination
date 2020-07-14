import numpy as np
import matplotlib.pyplot as plt
import filenames
from hivevo.patients import Patient
import trajectory


def get_proba_fix(trajectories, bin_filter="in", nb_bin=8, freq_range=[0.05, 0.95]):
    if bin_filter not in ["in", "through"]:
        raise ValueError("bin_filter parameter must be 'in' or 'through'")

    trajectories = [traj for traj in trajectories if traj.fixation != "active"]  # Remove active trajectories

    frequency_bins = np.linspace(freq_range[0], freq_range[1], nb_bin)
    traj_per_bin, fixed_per_bin, lost_per_bin, proba_fix = [], [], [], []

    for ii in range(len(frequency_bins) - 1):
        # Goes through the bin
        if bin_filter == "through":
            bin_trajectories = [traj for traj in trajectories if np.sum(traj.frequencies >= frequency_bins[ii],
                                                                        dtype=bool) and np.sum(traj.frequencies < frequency_bins[ii + 1], dtype=bool)]

        # Is seen in the bin
        if bin_filter == "in":
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

    frequency_bins = 0.5 * (frequency_bins[:-1] + frequency_bins[1:])

    return frequency_bins, proba_fix, traj_per_bin, fixed_per_bin, lost_per_bin


def plot_proba_fix(patient, region, frequency_bins, proba_fix, traj, fixed, lost, criteria, fontsize=16):
    plt.figure()
    plt.title(f"Proba fix patient {patient.name} region {region} " + criteria + " bin", fontsize=fontsize)
    plt.plot(frequency_bins, proba_fix, '.-')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.ylabel(r"$P_{fix}$", fontsize=fontsize)
    plt.xlabel("Frequency", fontsize=fontsize)

    plt.figure()
    plt.plot(frequency_bins, traj_per_bin, '.-', label="# trajectories")
    plt.plot(frequency_bins, fixed_per_bin, '.-', label="# fixed")
    plt.plot(frequency_bins, lost_per_bin, '.-', label="# lost")
    plt.legend(fontsize=fontsize)
    plt.ylabel("# trajectories", fontsize=fontsize)
    plt.xlabel("Frequency", fontsize=fontsize)
    plt.show()


def average_proba_fix(patient_names, criteria, region, nb_bin=10, remove_one_point_traj=False):
    all_traj_per_bin, all_fixed_per_bin = np.array([]), np.array([])
    for patient_name in patient_names:
        patient = Patient.load(patient_name)

        aft = patient.get_allele_frequency_trajectories(region)
        trajectories = trajectory.create_trajectory_list(patient, region, aft)
        filtered_traj = trajectories
        if remove_one_point_traj:
            filtered_traj = [traj for traj in trajectories if traj.t[-1] > 0]  # Remove 1 point only trajectories

        frequency_bins, _, traj_per_bin, fixed_per_bin, lost_per_bin = get_proba_fix(filtered_traj, criteria, nb_bin=nb_bin)
        if patient_name == patient_names[0]:
            all_traj_per_bin = np.array(traj_per_bin)
            all_fixed_per_bin = np.array(fixed_per_bin)
        else:
            all_traj_per_bin = all_traj_per_bin + traj_per_bin
            all_fixed_per_bin = all_fixed_per_bin + fixed_per_bin

    avg_proba_fix = all_fixed_per_bin / all_traj_per_bin
    err_proba_fix = avg_proba_fix * np.sqrt(1/all_fixed_per_bin + 1/all_traj_per_bin)

    return frequency_bins, avg_proba_fix, all_traj_per_bin, all_fixed_per_bin, err_proba_fix


def plot_average_proba(freq_bin, proba_fix, err_proba_fix, region, criteria):
    plt.figure(figsize=(10,8))
    plt.title(f"Average P_fix region {region} " + criteria + " bin", fontsize=fontsize)
    plt.ylabel(r"$P_{fix}$", fontsize=fontsize)
    plt.xlabel("Frequency", fontsize=fontsize)
    plt.errorbar(freq_bin, proba_fix, yerr=err_proba_fix, fmt='.-', label="Average over patients")
    plt.plot([0,1], [0,1], 'k--', label="neutral expectation")
    plt.legend(fontsize=fontsize)
    plt.show()

if __name__ == "__main__":
    patient_names = ["p1", "p2", "p4", "p5", "p6", "p8", "p9", "p11"]
    criteria = "through"
    region = "env"
    fontsize = 16

    plt.figure()
    plt.title(f"Proba fix region {region} " + criteria + " bin", fontsize=fontsize)
    plt.ylabel(r"$P_{fix}$", fontsize=fontsize)
    plt.xlabel("Frequency", fontsize=fontsize)
    plt.plot([0, 1], [0, 1], 'k--')
    for patient_name in patient_names:
        patient = Patient.load(patient_name)

        aft = patient.get_allele_frequency_trajectories(region)
        trajectories = trajectory.create_trajectory_list(patient, region, aft)
        filtered_traj = trajectories
        # filtered_traj = [traj for traj in trajectories if traj.t[-1] > 0]  # Remove 1 point only trajectories

        frequency_bins, proba_fix, traj_per_bin, fixed_per_bin, lost_per_bin = get_proba_fix(
            filtered_traj, criteria)
        plt.plot(frequency_bins, proba_fix, '.-', label=patient_name)

    plt.legend()
    plt.show()
