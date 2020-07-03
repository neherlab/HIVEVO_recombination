import numpy as np
import copy
import matplotlib.pyplot as plt
import filenames
from hivevo.patients import Patient
from tools import get_mutation_positions, get_fixation_positions, initial_idx_mask
from trajectory import Trajectory, create_trajectory_list
from filter import filter

patient_name = "p1"
patient = Patient.load(patient_name)
region = "env"

aft = patient.get_allele_frequency_trajectories(region)
trajectories = create_trajectory_list(patient, region, aft)

filtered_traj = filter(trajectories, "np.sum(traj.frequencies > 0.5, dtype=bool)")
