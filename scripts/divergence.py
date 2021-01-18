# Adds link to the scripts folder
import filenames
from hivevo.HIVreference import HIVreference
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle
import copy
from hivevo.patients import Patient
import trajectory
import tools
sys.path.append("../scripts/")


def get_reversion_mask(patient, region, aft, ref):
    """
    Returns a 3D boolean matrix (timepoint*nucleotide*patient_sequence_length) where True are the positions that
    correspond to the initial nucleotide when it is not the consensus (reversion is possible).
    Removes nucleotide seen too often gapped or not mapped to the reference.
    """
    ref_filter = trajectory.get_reference_filter(patient, region, aft, ref)
    ref_filter = np.tile(ref_filter, (aft.shape[0], aft.shape[1], 1))
    initial_mask = tools.initial_idx_mask(patient, region, aft)
    reversion_mask = trajectory.get_reversion_map(patient, region, aft, ref)
    reversion_mask = np.tile(reversion_mask, (aft.shape[0], 1, 1))

    return np.logical_and(np.logical_and(initial_mask, ref_filter), ~reversion_mask)


def get_non_reversion_mask(patient, region, aft, ref):
    """
    Returns a 3D boolean matrix (timepoint*nucleotide*patient_sequence_length) where True are the positions that
    correspond to the initial nuclotide when it is also the consensus (reversion is not possible).
    Removes nucleotide seen too often gapped or not mapped to the reference.
    """
    ref_filter = trajectory.get_reference_filter(patient, region, aft, ref)
    ref_filter = np.tile(ref_filter, (aft.shape[0], aft.shape[1], 1))
    initial_mask = tools.initial_idx_mask(patient, region, aft)
    reversion_mask = trajectory.get_reversion_map(patient, region, aft, ref)
    reversion_mask = np.tile(reversion_mask, (aft.shape[0], 1, 1))

    return np.logical_and(np.logical_and(initial_mask, ref_filter), reversion_mask)


def get_syn_mask(patient, region, aft):
    """
    Return as 3D boolean matrix where True are the initial positions when there is a synonymous mutation.
    """
    initial_mask = tools.initial_idx_mask(patient, region, aft)
    syn_mask = patient.get_syn_mutations(region, mask_constrained=False)
    syn_mask = np.tile(syn_mask, (aft.shape[0], 1, 1))
    mut_aft = np.copy(aft)
    mut_aft[initial_mask] = 0
    mut_aft[~syn_mask] = 0

    mask = np.array(np.zeros_like(aft, dtype=bool))
    sum_syn_aft = np.sum(np.sum(mut_aft, axis=0), axis=0)
    mask[:, :, sum_syn_aft > 0] = True
    mask[~initial_mask] = False
    return mask


def get_non_syn_mask(patient, region, aft):
    """
    Return as 3D boolean matrix where True are the initial positions when there is a non-synonymous mutation.
    """
    initial_mask = tools.initial_idx_mask(patient, region, aft)
    non_syn_mask = ~patient.get_syn_mutations(region, mask_constrained=False)
    non_syn_mask = np.tile(non_syn_mask, (aft.shape[0], 1, 1))
    mut_aft = np.copy(aft)
    mut_aft[initial_mask] = 0
    mut_aft[~non_syn_mask] = 0

    mask = np.array(np.zeros_like(aft, dtype=bool))
    sum_non_syn_aft = np.sum(np.sum(mut_aft, axis=0), axis=0)
    mask[:, :, sum_non_syn_aft > 0] = True
    mask[~initial_mask] = False
    return mask


def divergence_matrix(aft):
    """
    Returns the divergence matrix in time (same shape as the aft).
    """
    div = np.zeros_like(aft)
    for ii in range(aft.shape[0]):
        div[ii, :, :] = aft[0, :, :] * (1 - aft[ii, :, :])

    return div


def make_divergence_dict(time_average, ref=HIVreference(subtype="any")):
    """
    Creates a dictionary with the divergence in time for all patients and regions.
    """
    regions = ["env", "pol", "gag"]
    patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]

    divergence_dict = {}
    for region in regions:
        divergence_dict[region] = {}
        for patient_name in patient_names:
            divergence_dict[region][patient_name] = {}
            patient = Patient.load(patient_name)
            aft = patient.get_allele_frequency_trajectories(region)
            div = divergence_matrix(aft)

            # Select the 2D (nb_timepoint*nb_nucleotides) subset that correspond to initial indices only
            initial_idx = patient.get_initial_indices(region)
            div_initial = div[np.arange(aft.shape[0])[:, np.newaxis, np.newaxis],
                              initial_idx, np.arange(aft.shape[-1])]
            div_initial = div_initial[:, 0, :]
            mean_div = np.mean(div_initial, axis=1)

            rev_mask = get_reversion_mask(patient, region, aft, ref)
            rev_div = np.reshape(div[rev_mask], (div.shape[0], -1))
            mean_rev_div = np.mean(rev_div, axis=1)

            non_rev_mask = get_non_reversion_mask(patient, region, aft, ref)
            non_rev_div = np.reshape(div[non_rev_mask], (div.shape[0], -1))
            mean_non_rev_div = np.mean(non_rev_div, axis=1)

            syn_mask = get_syn_mask(patient, region, aft)
            syn_div = np.reshape(div[syn_mask], (aft.shape[0], -1))
            mean_syn_div = np.sum(syn_div, axis=1) / aft.shape[-1]

            non_syn_mask = get_non_syn_mask(patient, region, aft)
            non_syn_div = np.reshape(div[non_syn_mask], (aft.shape[0], -1))
            mean_non_syn_div = np.sum(non_syn_div, axis=1) / aft.shape[-1]

            syn_first = div_initial[:, 0::3]
            syn_second = div_initial[:, 1::3]
            syn_third = div_initial[:, 2::3]

            mean_syn_first = np.mean(syn_first, axis=1)
            mean_syn_second = np.mean(syn_second, axis=1)
            mean_syn_third = np.mean(syn_third, axis=1)

            syn_rev_mask = np.logical_and(syn_mask, rev_mask)
            syn_non_rev_mask = np.logical_and(syn_mask, non_rev_mask)
            non_syn_rev_mask = np.logical_and(non_syn_mask, rev_mask)
            non_syn_non_rev_mask = np.logical_and(non_syn_mask, non_rev_mask)

            syn_rev_div = np.reshape(div[syn_rev_mask], (aft.shape[0], -1))
            syn_non_rev_div = np.reshape(div[syn_non_rev_mask], (aft.shape[0], -1))
            non_syn_rev_div = np.reshape(div[non_syn_rev_mask], (aft.shape[0], -1))
            non_syn_non_rev_div = np.reshape(div[non_syn_non_rev_mask], (aft.shape[0], -1))

            mean_syn_rev = np.mean(syn_rev_div, axis=1)
            mean_syn_non_rev = np.mean(syn_non_rev_div, axis=1)
            mean_non_syn_rev = np.mean(non_syn_rev_div, axis=1)
            mean_non_syn_non_rev = np.mean(non_syn_non_rev_div, axis=1)

            # Transforming to regular array as mask is useless after averaging
            divergence_dict[region][patient_name]["rev"] = np.array(mean_rev_div)
            divergence_dict[region][patient_name]["non_rev"] = np.array(mean_non_rev_div)
            divergence_dict[region][patient_name]["syn"] = np.array(mean_syn_div)
            divergence_dict[region][patient_name]["non_syn"] = np.array(mean_non_syn_div)
            divergence_dict[region][patient_name]["all"] = np.array(mean_div)
            divergence_dict[region][patient_name]["dsi"] = np.array(patient.dsi)
            divergence_dict[region][patient_name]["div_all"] = np.array(div_initial)
            divergence_dict[region][patient_name]["div_rev"] = np.array(rev_div)
            divergence_dict[region][patient_name]["div_non_rev"] = np.array(non_rev_div)
            divergence_dict[region][patient_name]["div_syn"] = np.array(syn_div)
            divergence_dict[region][patient_name]["div_non_syn"] = np.array(non_syn_div)
            divergence_dict[region][patient_name]["first"] = np.array(mean_syn_first)
            divergence_dict[region][patient_name]["second"] = np.array(mean_syn_second)
            divergence_dict[region][patient_name]["third"] = np.array(mean_syn_third)
            divergence_dict[region][patient_name]["syn_rev"] = np.array(mean_syn_rev)
            divergence_dict[region][patient_name]["syn_non_rev"] = np.array(mean_syn_non_rev)
            divergence_dict[region][patient_name]["non_syn_rev"] = np.array(mean_non_syn_rev)
            divergence_dict[region][patient_name]["non_syn_non_rev"] = np.array(mean_non_syn_non_rev)

    # Computation of divergence average over all patients using interpolation
    for region in regions:
        time = time_average
        divergence_dict[region]["all"] = {}
        for mut_type in ["rev", "non_rev", "all", "syn", "non_syn", "first", "second", "third", "syn_rev",
                         "syn_non_rev", "non_syn_rev", "non_syn_non_rev"]:
            nb_traj = np.zeros_like(time)
            average_divergence = np.zeros_like(time, dtype=float)
            for patient_name in patient_names:
                tmp_time = time[time < divergence_dict[region][patient_name]["dsi"][-1]]
                tmp_divergence = np.interp(
                    tmp_time, divergence_dict[region][patient_name]["dsi"], divergence_dict[region][patient_name][mut_type])
                average_divergence[:len(tmp_divergence)] += tmp_divergence
                nb_traj[:len(tmp_divergence)] += 1

            divergence_dict[region]["all"][mut_type] = average_divergence / nb_traj

    return divergence_dict


def save_divergence_dict(divergence_dict):
    """
    Saves the divergence dict as a pickle.
    """
    with open("divergence_dict", "wb") as file:
        pickle.dump(divergence_dict, file)


def load_divergence_dict(file_name="divergence_dict"):
    """
    Load the divergence dict from pickle.
    """
    with open(file_name, "rb") as file:
        divergence_dict = pickle.load(file)
    return divergence_dict


def WH_evo_rate(divergence_dict, time, regions=["env", "pol", "gag"]):
    """
    Compute the evolution rate (using gradient) for reversion and non-reversion in the given regions.
    """
    evo_rate_dict = copy.deepcopy(divergence_dict)
    for key in evo_rate_dict.keys():
        for key2 in evo_rate_dict[key]["all"].keys():
            evo_rate_dict[key]["all"][key2] = np.gradient(evo_rate_dict[key]["all"][key2], time)
        evo_rate_dict[key] = evo_rate_dict[key]["all"]
    return evo_rate_dict


if __name__ == "__main__":
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]
    time_average = np.arange(0, 3100, 100)

    divergence_dict = make_divergence_dict(time_average)
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    fontsize = 16

    plt.figure(figsize=(14, 10))
    region = "gag"
    plt.plot(time_average, divergence_dict[region]["all"]["syn_rev"], label="syn_rev")
    plt.plot(time_average, divergence_dict[region]["all"]["syn_non_rev"], label="syn_non_rev")
    plt.plot(time_average, divergence_dict[region]["all"]["non_syn_rev"], label="non_syn_rev")
    plt.plot(time_average, divergence_dict[region]["all"]["non_syn_non_rev"], label="non_syn_non_rev")
    plt.plot(time_average, divergence_dict[region]["all"]["all"], label="all")
    # for ii, region in enumerate(["env", "pol", "gag"]):
    #     plt.plot(time_average, divergence_dict[region]["all"]["syn"], '-', color=colors[ii], label=region)
    #     plt.plot(time_average, divergence_dict[region]["all"]["non_syn"], '--', color=colors[ii])
    #     plt.plot(time_average, divergence_dict[region]["all"]["all"], ':', color=colors[ii])
    # plt.plot([0], [0], 'k-', label="Synonymous")
    # plt.plot([0], [0], 'k--', label="Non-synonymous")
    # plt.plot([0], [0], 'k:', label="All")
    plt.grid()
    plt.xlabel("Time since infection [days]", fontsize=fontsize)
    plt.ylabel("Divergence", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.show()

    # patient = Patient.load("p1")
    # region = "env"
    # aft = patient.get_allele_frequency_trajectories(region)
    # div = divergence_matrix(aft)
    #
    # # Select the 2D (nb_timepoint*nb_nucleotides) subset that correspond to initial indices only
    # initial_idx = patient.get_initial_indices(region)
    # div_tot = div[np.arange(aft.shape[0])[:, np.newaxis, np.newaxis], initial_idx, np.arange(aft.shape[-1])]
    # div_tot = div_tot[:, 0, :]
    # mean_div = np.mean(div_tot, axis=1)
    #
    # syn_mask = get_syn_mask(patient, region, aft)
    # syn_div = np.reshape(div[syn_mask], (aft.shape[0], -1))
    # mean_syn_div = np.sum(syn_div, axis=1) / aft.shape[-1]
    #
    # non_syn_mask = get_non_syn_mask(patient, region, aft)
    # non_syn_div = np.reshape(div[non_syn_mask], (aft.shape[0], -1))
    # mean_non_syn_div = np.sum(non_syn_div, axis=1) / aft.shape[-1]
    #
    # syn_first = div_tot[:, 0::3]
    # syn_second = div_tot[:, 1::3]
    # syn_third = div_tot[:, 2::3]
    #
    # mean_syn_first = np.mean(syn_first, axis=1)
    # mean_syn_second = np.mean(syn_second, axis=1)
    # mean_syn_third = np.mean(syn_third, axis=1)
    #
    # plt.figure()
    # plt.plot(patient.dsi, mean_syn_first, label="first")
    # plt.plot(patient.dsi, mean_syn_second, label="second")
    # plt.plot(patient.dsi, mean_syn_third, label="third")
    # plt.plot(patient.dsi, mean_syn_div, label="syn")
    # plt.plot(patient.dsi, mean_non_syn_div, label="non_syn")
    # plt.legend()
    # plt.xlabel("Days")
    # plt.ylabel("Divergence")
    # plt.grid()
    # plt.show()
