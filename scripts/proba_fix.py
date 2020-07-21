import numpy as np
import matplotlib.pyplot as plt
import filenames
from hivevo.patients import Patient
import trajectory


def get_proba_fix(trajectories, nb_bin=8, bin_type="uniform", freq_range=[0.05, 0.95]):
    """
    Gives the probability of fixation in each frequency bin.
    """
    if bin_type not in ["uniform", "nonuniform"]:
        raise ValueError("bin_type must be either uniform or non uniform.")

    if bin_type == "uniform":
        frequency_bins = np.linspace(freq_range[0], freq_range[1], nb_bin)
    elif bin_type == "nonuniform":
        from traj_characterisation import get_nonuniform_bins
        frequency_bins = get_nonuniform_bins(nb_bin, bin_range=freq_range)

    trajectories = [traj for traj in trajectories if traj.fixation != "active"]  # Remove active trajectories
    traj_per_bin, fixed_per_bin, lost_per_bin, proba_fix = [], [], [], []

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


def average_proba_fix(patient_names, region, bin_type="uniform", nb_bin=10, remove_one_point_traj=False):
    # Combining all the trajectories
    trajectories = []
    for patient_name in patient_names:
        patient = Patient.load(patient_name)
        aft = patient.get_allele_frequency_trajectories(region)
        trajectories = trajectories + trajectory.create_trajectory_list(patient, region, aft)

    # Computing proba and error
    frequency_bins, proba_fix, traj_per_bin, fixed_per_bin, lost_per_bin = get_proba_fix(
        trajectories, nb_bin, bin_type)

    err_proba_fix = np.array(proba_fix) * np.sqrt(1 / np.array(fixed_per_bin) + 1 / np.array(traj_per_bin))

    return frequency_bins, proba_fix, traj_per_bin, fixed_per_bin, err_proba_fix


def plot_average_proba(freq_bin, proba_fix, err_proba_fix, region, bin_type, fontsize=16):
    plt.figure(figsize=(10, 8))
    plt.title(f"Average P_fix region {region} " + bin_type + " bin", fontsize=fontsize)
    plt.ylabel(r"$P_{fix}$", fontsize=fontsize)
    plt.xlabel("Frequency", fontsize=fontsize)
    plt.errorbar(freq_bin, proba_fix, yerr=err_proba_fix, fmt='.-', label="Average over patients")
    plt.plot([0, 1], [0, 1], 'k--', label="neutral expectation")
    plt.legend(fontsize=fontsize)
    plt.show()

def get_nonuniform_bins(nb_bins, type="quadra", bin_range=[0.05, 0.95]):
    if type not in ["quadra", "log"]:
        raise ValueError("Type of bins must be either quadra or log.")

    if type == "quadra":
        bins = np.linspace(bin_range[0], bin_range[1], nb_bins + 1)
        non_uniform_bins = bins * bins
        return non_uniform_bins
    elif type == "log":
        return np.logspace(np.log10(bin_range[0]), np.log10(bin_range[1]), nb_bins + 1)


if __name__ == "__main__":
    patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]
    bin_type = "nonuniform"
    region = "env"
    remove_one_point_only = False
    nb_bin = 10

    frequency_bins, proba_fix, traj_per_bin, fixed_per_bin, err_proba_fix = average_proba_fix(
        patient_names, region, bin_type, nb_bin, remove_one_point_only)
    plot_average_proba(frequency_bins, proba_fix, err_proba_fix, region, bin_type)
