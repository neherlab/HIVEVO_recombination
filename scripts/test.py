# Adds link to the scripts folder
import filenames
from hivevo.HIVreference import HIVreference
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from hivevo.patients import Patient
from divergence import get_non_consensus_mask
import trajectory
sys.path.append("../scripts/")

def get_sweep_mask(patient, aft, threshold_low = 0.05):
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



if __name__ == "__main__":
    region = "pol"
    patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]
    for patient_name in patient_names:
        patient = Patient.load(patient_name)
        aft = patient.get_allele_frequency_trajectories(region)
        non_consensus = get_non_consensus_mask(patient, region, aft)
        sweep = get_sweep_mask(patient, aft)
