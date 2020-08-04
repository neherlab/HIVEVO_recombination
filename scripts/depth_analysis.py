import numpy as np
import matplotlib.pyplot as plt
import filenames
from hivevo.patients import Patient
from trajectory import create_trajectory_list, create_all_patient_trajectories


patient_name = "p1"
region = "env"
fontsize = 16

patient = Patient.load(patient_name)
aft = patient.get_allele_frequency_trajectories(region)
trajectories = create_all_patient_trajectories(region)
