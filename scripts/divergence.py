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


def divergence_matrix(aft):
    """
    Returns the divergence matrix in time (same shape as the aft).
    """
    div = np.zeros_like(aft)
    for ii in range(aft.shape[0]):
        div[ii, :, :] = aft[0, :, :] * (1 - aft[ii, :, :])

    return div


def get_consensus_mask(patient, region, aft, ref=HIVreference(subtype="any")):
    """
    Returns a 1D vector of size aft.shape[-1] where True are the position that correspond to consensus sequences.
    Position that are not mapped to reference or seen too often gapped are always False.
    """
    ref_filter = trajectory.get_reference_filter(patient, region, aft, ref)
    consensus_mask = trajectory.get_reversion_map(patient, region, aft, ref)
    initial_idx = patient.get_initial_indices(region)
    # gives reversion mask at initial majority nucleotide
    consensus_mask = consensus_mask[initial_idx, np.arange(aft.shape[-1])]

    return np.logical_and(ref_filter, consensus_mask)


def get_non_consensus_mask(patient, region, aft, ref=HIVreference(subtype="any")):
    """
    Returns a 1D vector of size aft.shape[-1] where True are the position that do not correspond to consensus sequences.
    Position that are not mapped to reference or seen too often gapped are always False.
    """
    ref_filter = trajectory.get_reference_filter(patient, region, aft, ref)
    consensus_mask = trajectory.get_reversion_map(patient, region, aft, ref)
    initial_idx = patient.get_initial_indices(region)
    consensus_mask = consensus_mask[initial_idx, np.arange(aft.shape[-1])]

    return np.logical_and(ref_filter, ~consensus_mask)


def get_fitness_mask(patient, region, aft, median, range="low"):
    """
    Returns a 1D boolean vector of size aft.shape[-1] where True are the positions corresponding to the 50%
    lowest/highest fitness values. Sites where the fitness values are NANs are always False.
    """

    assert range in ["low", "high"], "range must be eihter low or high"

    fitness_cost = trajectory.get_fitness_cost(patient, region, aft)
    fitness_mask = np.ones_like(fitness_cost, dtype=bool)
    fitness_mask[np.isnan(fitness_cost)] = False  # This is to avoid comparison with NANs

    if range == "low":
        fitness_mask[fitness_mask] = fitness_cost[fitness_mask] < median
    else:
        fitness_mask[fitness_mask] = fitness_cost[fitness_mask] >= median

    return fitness_mask


def get_mean_divergence_patient(patient, region):
    """
    Returns a dictionary with the divergence over time for different categories.
    """

    # Needed data
    aft = patient.get_allele_frequency_trajectories(region)
    div_3D = divergence_matrix(aft)
    fitness = trajectory.get_fitness_cost(patient, region, aft)
    fitness_median = np.median(fitness[~np.isnan(fitness)])
    initial_idx = patient.get_initial_indices(region)
    div = div_3D[np.arange(aft.shape[0])[:, np.newaxis, np.newaxis], initial_idx, np.arange(aft.shape[-1])]
    div = div[:, 0, :]

    # Masks
    consensus_mask = get_consensus_mask(patient, region, aft)
    non_consensus_mask = get_non_consensus_mask(patient, region, aft)
    fitness_low_mask = get_fitness_mask(patient, region, aft, fitness_median, "low")
    fitness_high_mask = get_fitness_mask(patient, region, aft, fitness_median, "high")

    # Mean divergence in time using mask combination
    consensus_div = np.mean(div[:, consensus_mask], axis=-1)
    non_consensus_div = np.mean(div[:, non_consensus_mask], axis=-1)
    consensus_low_div = np.mean(div[:, np.logical_and(consensus_mask, fitness_low_mask)], axis=-1)
    consensus_high_div = np.mean(div[:, np.logical_and(consensus_mask, fitness_high_mask)], axis=-1)
    non_consensus_low_div = np.mean(div[:, np.logical_and(non_consensus_mask, fitness_low_mask)], axis=-1)
    non_consensus_high_div = np.mean(div[:, np.logical_and(non_consensus_mask, fitness_high_mask)], axis=-1)

    div_dict = {"consensus": {}, "non_consensus": {}}
    div_dict["consensus"] = {"low": consensus_low_div, "high": consensus_high_div, "all": consensus_div}
    div_dict["non_consensus"] = {"low": non_consensus_low_div,
                                 "high": non_consensus_high_div, "all": non_consensus_div}

    return div_dict


def make_divergence_dict(time, ref=HIVreference(subtype="any")):
    """
    Creates a dictionary with the divergence in time averaged over patients.
    Format of the dictionary : dict[region][consensus/non_consensus][high/low/all]
    """

    regions = ["env", "pol", "gag"]
    patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]
    fitness_keys = ["low", "high", "all"]

    divergence_dict = {}
    for region in regions:
        divergence_dict[region] = {"consensus": {}, "non_consensus": {}}

        nb_traj = np.zeros_like(time)
        for key in fitness_keys:
            divergence_dict[region]["consensus"][key] = np.zeros_like(time, dtype=float)
            divergence_dict[region]["non_consensus"][key] = np.zeros_like(time, dtype=float)

        for patient_name in patient_names:
            patient = Patient.load(patient_name)
            patient_div_dict = get_mean_divergence_patient(patient, region)

            tmp_time = time[time < patient.dsi[-1]]
            nb_traj[:len(tmp_time)] += 1

            for key in fitness_keys:
                patient_div_dict["consensus"][key] = np.interp(
                    tmp_time, patient.dsi, patient_div_dict["consensus"][key])
                patient_div_dict["non_consensus"][key] = np.interp(
                    tmp_time, patient.dsi, patient_div_dict["non_consensus"][key])

                divergence_dict[region]["consensus"][key][:len(
                    tmp_time)] += patient_div_dict["consensus"][key]
                divergence_dict[region]["non_consensus"][key][:len(
                    tmp_time)] += patient_div_dict["non_consensus"][key]

        for key1 in ["consensus", "non_consensus"]:
            for key2 in fitness_keys:
                divergence_dict[region][key1][key2] = divergence_dict[region][key1][key2] / nb_traj

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
    regions = ["env", "pol", "gag"]

    time = np.arange(0, 3100, 100)
    divergence_dict = make_divergence_dict(time)

    plt.figure()
    for region in regions:
        for key1 in divergence_dict[region].keys():
            plt.plot(time, divergence_dict[region][key1]["all"], label=f"{region} {key1}")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Divergence")
    plt.grid()
    plt.show()
