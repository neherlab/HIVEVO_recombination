# Adds link to the scripts folder
import filenames
from hivevo.HIVreference import HIVreference
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from hivevo.patients import Patient
from divergence import get_non_consensus_mask, get_consensus_mask, get_site_mask
import trajectory
sys.path.append("../scripts/")


def get_sweep_mask(patient, aft, threshold_low=0.05):
    # Masking low depth
    depth = trajectory.get_depth(patient, region)
    depth = np.tile(depth, (6, 1, 1))
    depth = np.swapaxes(depth, 0, 1)
    aft.mask = np.logical_or(aft.mask, ~depth)

    initial_idx = patient.get_initial_indices(region)
    aft_initial = aft[np.arange(aft.shape[0])[:, np.newaxis, np.newaxis],
                      initial_idx, np.arange(aft.shape[-1])]
    aft_initial = aft_initial[:, 0, :]

    mask = aft_initial <= threshold_low
    mask = np.sum(mask, axis=0)
    return mask


def fraction_per_region():
    """
    Fraction of consensus and non_consensus site computation. This is for initial sequence for each patient.
    Fraction consensus + non_consensus does not equal one because some regions are excluded due to gaps.
    """
    regions = ["env", "pol", "gag"]
    patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]
    for region in regions:
        consensus = []
        non_consensus = []
        fraction_non_consensus = []
        for patient_name in patient_names:
            patient = Patient.load(patient_name)
            aft = patient.get_allele_frequency_trajectories(region)
            consensus_mask = get_consensus_mask(patient, region, aft)
            non_consensus_mask = get_non_consensus_mask(patient, region, aft)
            consensus += [np.sum(consensus_mask, dtype=int)]
            non_consensus += [np.sum(non_consensus_mask, dtype=int)]
            fraction_non_consensus += [non_consensus[-1] / (consensus[-1] + non_consensus[-1])]

        mean_consensus = np.mean(consensus) / aft.shape[-1]
        std_consensus = np.std(consensus) / aft.shape[-1]
        mean_non_consensus = np.mean(non_consensus) / aft.shape[-1]
        std_non_consensus = np.std(non_consensus) / aft.shape[-1]
        mean_fraction_non_consensus = np.mean(fraction_non_consensus)
        std_fraction_non_consensus = np.std(fraction_non_consensus)
        print(f"Region {region}:")
        print(f"""   Consensus {round(mean_consensus, 2)} += {round(std_consensus, 2)}   Non-consensus {round(mean_non_consensus, 2)} += {round(std_non_consensus, 2)}   Fraction non_consensus {round(mean_fraction_non_consensus,3)} += {round(std_fraction_non_consensus,3)}""")


def fraction_per_site():
    """
    Same as fraction per region but only for pol and with discrimination between 1st 2nd and 3rd position.
    """
    region = "pol"
    patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]
    for ii, site in enumerate(["first", "second", "third"]):
        consensus = []
        non_consensus = []
        fraction_non_consensus = []
        for patient_name in patient_names:
            patient = Patient.load(patient_name)
            aft = patient.get_allele_frequency_trajectories(region)
            site_mask = get_site_mask(aft, ii+1)
            consensus_mask = get_consensus_mask(patient, region, aft)[site_mask]
            non_consensus_mask = get_non_consensus_mask(patient, region, aft)[site_mask]
            consensus += [np.sum(consensus_mask, dtype=int)]
            non_consensus += [np.sum(non_consensus_mask, dtype=int)]
            fraction_non_consensus += [non_consensus[-1] / (consensus[-1] + non_consensus[-1])]

        mean_consensus = np.mean(consensus) / (aft.shape[-1]/3)
        std_consensus = np.std(consensus) / (aft.shape[-1]/3)
        mean_non_consensus = np.mean(non_consensus) / (aft.shape[-1]/3)
        std_non_consensus = np.std(non_consensus) / (aft.shape[-1]/3)
        mean_fraction_non_consensus = np.mean(fraction_non_consensus)
        std_fraction_non_consensus = np.std(fraction_non_consensus)
        print(f"Site {site}:")
        print(f"""   Consensus {round(mean_consensus, 2)} += {round(std_consensus, 2)}   Non-consensus {round(mean_non_consensus, 2)} += {round(std_non_consensus, 2)}   Fraction non_consensus {round(mean_fraction_non_consensus,3)} += {round(std_fraction_non_consensus,3)}""")


if __name__ == "__main__":
    # fraction_per_region()
    fraction_per_site()
