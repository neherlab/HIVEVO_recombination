# Adds link to the scripts folder
import filenames
from hivevo.HIVreference import HIVreference
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from hivevo.patients import Patient
import trajectory
from divergence import load_divergence_dict, WH_evo_rate
sys.path.append("../scripts/")


patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]
region = "pol"

plt.figure()
for patient in patient_names:
    patient = Patient.load(patient)
    aft = patient.get_allele_frequency_trajectories(region)
    ref = HIVreference(subtype="any")
    reversion_mask = trajectory.get_reversion_map(patient, region, aft, ref)
    reversion_mask = np.tile(reversion_mask, (aft.shape[0], 1, 1))
    aft2 = np.reshape(aft[reversion_mask], (aft.shape[0], -1))

    nb_non_consensus = []
    for ii in range(aft.shape[0]):
        nb_non_consensus += [np.sum(aft2[ii, :] < 0.5, dtype=int)]
    plt.plot(patient.dsi, nb_non_consensus)
plt.grid()
plt.show()
