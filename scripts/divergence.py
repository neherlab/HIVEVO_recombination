# Adds link to the scripts folder
import filenames
from hivevo.HIVreference import HIVreference
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from hivevo.patients import Patient
import trajectory
import tools
sys.path.append("../scripts/")


def get_reversion_mask(patient, region, aft, ref):
    """
    Returns a 3D boolean matrix (timepoint*nucleotide*patient_sequence_length) where True are the positions that
    correspond to the reference nucleotide.
    """
    reversion_mask = trajectory.get_reversion_map(patient, region, aft, ref)
    reversion_mask = np.tile(reversion_mask, (aft.shape[0], 1, 1))
    return reversion_mask


def get_non_reversion_mask(patient, region, aft, ref):
    """
    Returns a 3D boolean matrix (timepoint*nucleotide*patient_sequence_length) where True are the positions that
    do not correspond to the reference nucleotide. Nucleotides that are not mapped to reference and/or too often
    seen gapped are removed.
    """
    ref_filter = trajectory.get_reference_filter(patient, region, aft, ref)
    ref_filter = np.tile(ref_filter, (aft.shape[0], aft.shape[1], 1))
    reversion_mask = trajectory.get_reversion_map(patient, region, aft, ref)
    reversion_mask = np.tile(reversion_mask, (aft.shape[0], 1, 1))
    return np.logical_and(ref_filter, ~reversion_mask)


def divergence_matrix(aft):
    """
    Returns the divergence matrix in time (same shape as the aft).
    """
    div = np.zeros_like(aft)
    for ii in range(aft.shape[0]):
        div[ii, :, :] = aft[0, :, :] * (1 - aft[ii, ::])
    return div


def make_divergence_dict():
    regions = ["env", "pol", "gag"]
    patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]
    ref = HIVreference(subtype="any")

    divergence_dict = {}
    for region in regions:
        divergence_dict[region] = {}
        for patient_name in patient_names:
            divergence_dict[region][patient_name] = {}
            patient = Patient.load(patient_name)
            aft = patient.get_allele_frequency_trajectories(region)
            div = divergence_matrix(aft)
            mean_div = np.mean(np.mean(div, axis=1), axis=1)

            rev_mask = get_reversion_mask(patient, region, aft, ref)
            rev_div = np.reshape(div[rev_mask], (div.shape[0], -1))
            mean_rev_div = np.mean(rev_div, axis=1)

            non_rev_mask = get_non_reversion_mask(patient, region, aft, ref)
            non_rev_div = np.reshape(div[non_rev_mask], (div.shape[0], -1))
            mean_non_rev_div = np.mean(non_rev_div, axis=1)

            # Transforming to regular array as mask is useless after averaging
            divergence_dict[region][patient_name]["rev"] = np.array(mean_rev_div)
            divergence_dict[region][patient_name]["non_rev"] = np.array(mean_non_rev_div)
            divergence_dict[region][patient_name]["all"] = np.array(mean_div)
            divergence_dict[region][patient_name]["dsi"] = np.array(patient.dsi)

    return divergence_dict


if __name__ == "__main__":
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    divergence_dict = make_divergence_dict()

    # plt.plot([0], [0], 'k-', label="Reversion")
    # plt.plot([0], [0], 'k--', label="Non-reversion")
    # plt.grid()
    # plt.xlabel("Time since infection [days]")
    # plt.ylabel("Divergence")
    # plt.legend()
    # plt.show()
