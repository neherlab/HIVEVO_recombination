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

def get_fixation_positions(patient, region, aft, eps=0.01, timepoint="any"):
    """
    Return a boolean matrix where True are the mutations with more than 1-eps frequency at some timepoint / last time point.
    timepoint = ["any", "last"]
    """
    fixation_positions = get_mutation_positions(patient, region, aft, 1-eps)

    if timepoint == "any":
        return np.sum(fixation_positions, axis=0, dtype=bool)
    elif timepoint == "last":
        return fixation_positions[-1,:,:]
    else:
        raise ValueError("Condition of fixation is not understood.")



patient_name = "p1"
patient = Patient.load(patient_name)
# region = "env"
plt.figure()
for region in list(patient.annotation.keys())[:10]:

    aft = patient.get_allele_frequency_trajectories(region)
    mut_pos = get_mutation_positions(patient, region, aft)
    fix_pos = get_fixation_positions(patient, region, aft)

    mut_aft = copy.deepcopy(aft)
    mut_aft.mask = ~mut_pos
    mut_fixated_aft = copy.deepcopy(aft)
    mut_fixated_aft.mask = ~np.tile(fix_pos, (aft.shape[0],1,1))

    nonfixing_mut_aft = mut_aft[np.logical_and(mut_pos, mut_fixated_aft.mask)]
    fixing_mut_aft = mut_aft[~mut_fixated_aft.mask]

    h, b = np.histogram(fixing_mut_aft, bins=10, range=(0,1))
    hh, b = np.histogram(mut_aft[~mut_aft.mask], bins=10, range=(0,1))
    bins = 0.5*(b[1:] + b[:-1])

    proba = h/hh

    plt.plot(bins, proba, 'x-', label=region)

plt.plot([0,1], [0,1], 'k-')
plt.xlim([0,1])
plt.ylim([0,1])
plt.legend()
plt.show()
