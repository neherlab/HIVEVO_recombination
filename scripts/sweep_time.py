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
    mask[1:, :, :] = np.logical_and(mask[1, :, :], np.logical_or(
        ~aft.mask[:-1, :, :], ~aft.mask[1:, :, :]))  # to avoid masked data to be included
    mask = np.sum(mask, axis=0, dtype=bool)
    mask = np.tile(mask, (aft.shape[0], 1, 1))
    return mask


region = "env"
patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]

# missed_sweep_tot = 0
# for patient_name in patient_names:
#     patient = Patient.load(patient_name)
#     aft = patient.get_allele_frequency_trajectories(region)
#     missed_sweep = get_missed_sweep_number(aft)
#     missed_sweep_tot += missed_sweep
#     print(f"Patient {patient_name}: {missed_sweep}")
# print(missed_sweep_tot)

patient = Patient.load("p1")
aft = patient.get_allele_frequency_trajectories("env")
mask = get_missed_sweep_mask(aft)
missed_sweeps = aft[mask]
reshaped = np.reshape(missed_sweeps, (aft.shape[0], -1))
