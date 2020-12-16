import numpy as np
import matplotlib.pyplot as plt
import filenames
from hivevo.patients import Patient
import trajectory


def get_proba_fix(trajectories, nb_bin=8, freq_range=[0.1, 0.9]):
    """
    Gives the probability of fixation in each frequency bin.
    """

    frequency_bins = get_nonuniform_bins(nb_bin, bin_range=freq_range)

    trajectories = [traj for traj in trajectories if traj.fixation != "active"]  # Remove active trajectories
    traj_per_bin, fixed_per_bin, lost_per_bin, proba_fix = [], [], [], []
    mean_freq_bin = []

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

        # Computes the "center" of the bin
        tmp_mean = []
        for traj in bin_trajectories:
            idxs = np.where(np.logical_and(traj.frequencies >=
                                           frequency_bins[ii], traj.frequencies < frequency_bins[ii + 1]))[0]
            tmp_mean = tmp_mean + [traj.frequencies[idxs[0]]]
        mean_freq_bin = mean_freq_bin + [np.mean(tmp_mean)]

    err_proba_fix = np.array(proba_fix) * np.sqrt(1 / (np.array(fixed_per_bin) +
                                                       1e-10) + 1 / np.array(traj_per_bin))

    return mean_freq_bin, proba_fix, err_proba_fix



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


def plot_average_proba(freq_bin, proba_fix, err_proba_fix, region, bin_type, fontsize=16):
    plt.figure(figsize=(10, 8))
    plt.title(f"Average P_fix region {region} " + bin_type + " bin", fontsize=fontsize)
    plt.ylabel(r"$P_{fix}$", fontsize=fontsize)
    plt.xlabel("Frequency", fontsize=fontsize)
    plt.errorbar(freq_bin, proba_fix, yerr=err_proba_fix, fmt='.-', label="Average over patients")
    plt.plot([0, 1], [0, 1], 'k--', label="neutral expectation")
    plt.legend(fontsize=fontsize)
    plt.show()


def get_nonuniform_bins(nb_bins, type="log", bin_range=[0.05, 0.95]):
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
    fontsize = 16

    trajectories = trajectory.create_all_patient_trajectories(region)
    syn_traj = [traj for traj in trajectories if traj.synonymous == True]
    non_syn_traj = [traj for traj in trajectories if traj.synonymous == False]

    bins, proba_fix_syn, _, _, _, err_proba_fix_syn = get_proba_fix(syn_traj, nb_bin, bin_type)
    bins, proba_fix_non_syn, _, _, _, err_proba_fix_non_syn = get_proba_fix(non_syn_traj, nb_bin, bin_type)

    plt.figure(figsize=(10, 8))
    plt.title(f"Average P_fix region {region} " + bin_type + " bin", fontsize=fontsize)
    plt.ylabel(r"$P_{fix}$", fontsize=fontsize)
    plt.xlabel("Frequency", fontsize=fontsize)
    plt.errorbar(bins, proba_fix_syn, yerr=err_proba_fix_syn, fmt='.-', label="Syn")
    plt.errorbar(bins, proba_fix_non_syn, yerr=err_proba_fix_non_syn, fmt='.-', label="Non_syn")
    plt.plot([0, 1], [0, 1], 'k--', label="neutral expectation")
    plt.legend(fontsize=fontsize)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()
