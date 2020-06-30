import filenames
import numpy as np
import copy
import matplotlib.pyplot as plt
from hivevo.patients import Patient
from tools import get_mutation_positions, get_fixation_positions


patient_name = "p1"
patient = Patient.load(patient_name)
# region = "env"
plt.figure()
plt.title(patient_name, fontsize=16)
for region in ['pol', 'env', 'gag']:

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

    mask = (hh != 0)
    proba = h[mask]/hh[mask]

    plt.plot(bins[mask], proba, 'x-', label=region)

plt.plot([0,1], [0,1], 'k-')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel(r"$\nu$", fontsize=16)
plt.ylabel(r"$P_{fix}$", fontsize=16)
plt.legend()
plt.show()
