from trajectory import Trajectory, create_all_patient_trajectories
from hivevo.patients import Patient
import filenames
import matplotlib.pyplot as plt
import numpy as np
import tools
from hivevo.patients import Patient


def get_missed_sweep_mask(aft):
    """
    Returns a mask where true are the position where a fast sweep is seen.
    """
    mask = np.zeros(aft.shape, dtype=bool)
    mask[1:, :, :] = np.logical_and(aft[:-1, :, :] <= 0.01, aft[1:, :, :] >= 0.99)
    mask[1:, :, :] = np.logical_and(mask[1:, :, :], np.logical_and(
        ~aft.mask[:-1, :, :], ~aft.mask[1:, :, :]))  # to avoid masked data to be included
    mask = np.sum(mask, axis=0, dtype=bool)
    mask = np.logical_and(aft[0, :, :] < 0.01, mask) # Removing things that are not new
    mask = np.tile(mask, (aft.shape[0], 1, 1))
    return mask

def get_missed_sweep(aft):
    missed_sweeps = aft[get_missed_sweep_mask(aft)]
    missed_sweeps = np.reshape(missed_sweeps, (aft.shape[0], -1))
    return missed_sweeps


regions = ["env", "pol", "gag"]
patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]

for region in regions:
    missed_sweep_tot = 0
    for patient_name in patient_names:
        patient = Patient.load(patient_name)
        aft = patient.get_allele_frequency_trajectories(region)
        missed_sweep = get_missed_sweep(aft)
        missed_sweep_tot += missed_sweep.shape[-1]
        print(f"Patient {patient_name} {region}: {missed_sweep.shape[-1]}")

    trajectories = create_all_patient_trajectories(region)
    print(f"Region {region} missed sweeps : {missed_sweep_tot}")
    print(f"Region {region} total trajectories : {len(trajectories)}")

# patient = Patient.load("p1")
# aft = patient.get_allele_frequency_trajectories("env")
# mask = get_missed_sweep_mask(aft)
# missed_sweeps = aft[mask]
# reshaped = np.reshape(missed_sweeps, (aft.shape[0], -1))
