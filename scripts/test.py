import numpy as np
import matplotlib.pyplot as plt
import filenames
from hivevo.patients import Patient
from hivevo.HIVreference import HIVreference



patient_names = ["p1"]
# patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]

for patient_name in patient_names:
    region = "pol"
    patient = Patient.load(patient_name)
    coordinates = patient._annotation_to_fragment_indices(region)
    aft = patient.get_allele_frequency_trajectories(region)
    map_to_ref = patient.map_to_external_reference(region)
    ref = HIVreference(subtype="any")
    consensus = ref.get_consensus_in_patient_region(map_to_ref[:,0])
    # consensus_indices = ref.get_consensus_indices_in_patient_region(map_to_ref)
    # print(consensus.shape, map_to_ref.shape)
