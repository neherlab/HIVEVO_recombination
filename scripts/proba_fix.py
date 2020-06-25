import filenames
import numpy as np
import copy
import matplotlib.pyplot as plt
from hivevo.patients import Patient

def get_mutation_positions(patient, region, aft, eps=0.01):
    """
    Return a boolean matrix where True are mutations positions with more than eps frequency.
    Original nucleotides are not considered as mutations.
    """
    mutation_positions = aft > eps
    initial_idx = patient.get_initial_indices(region)
    for ii in range(aft.shape[2]):
        mutation_positions[:, initial_idx[ii], ii] = np.zeros(aft.shape[0]).astype(bool)
    return mutation_positions

def get_fixation_positions(patient, region, aft, eps=0.05, timepoint="any"):
    """
    Return a boolean matrix where True are the mutations with more than 1-eps frequency at some timepoint / last time point.
    timepoint = ["any", "last"]
    """
    fixation_positions = aft > 1-eps
    initial_idx = patient.get_initial_indices(region)
    for ii in range(aft.shape[2]):
        fixation_positions[:, initial_idx[ii], ii] = np.zeros(aft.shape[0]).astype(bool)

    if timepoint == "any":
        return np.sum(fixation_positions, axis=0, dtype=bool)
    elif timepoint == "last":
        return fixation_positions[-1,:,:]
    else:
        raise ValueError("Condition of fixation is not understood.")

patient_name = "p1"
region = "env"

patient = Patient.load(patient_name)
aft = patient.get_allele_frequency_trajectories(region)
mut_pos = get_mutation_positions(patient, region, aft)
fix_pos = get_fixation_positions(patient, region, aft)

# rise_fall_pos = np.logical_and(fix_pos, ~get_fixation_positions(patient, region, aft, timepoint="last"))

# mutations = np.sum(np.sum(mut_pos, axis=0, dtype=bool).astype(int), axis=0)
# fixations = np.sum(fix_pos.astype(int), axis=0)
# plt.plot(mutations)
# plt.plot(fixations)

mut_aft = copy.deepcopy(aft)
mut_aft.mask = ~mut_pos

mut_fixated_aft = copy.deepcopy(aft)
mut_fixated_aft.mask = ~np.tile(fix_pos, (aft.shape[0],1,1))
plt.hist(mut_fixated_aft[~mut_fixated_aft.mask], bins=30)
