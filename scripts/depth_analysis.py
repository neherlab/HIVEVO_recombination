import numpy as np
import matplotlib.pyplot as plt
import filenames
from hivevo.patients import Patient
from trajectory import create_trajectory_list, get_fragment_per_site


patient_name = "p1"
region = "gag"
fontsize = 16

patient = Patient.load(patient_name)
aft = patient.get_allele_frequency_trajectories(region)
trajectories = create_trajectory_list(patient, region, aft)

fragments = get_fragment_per_site(patient, region)
