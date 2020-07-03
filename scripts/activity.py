import numpy as np
import matplotlib.pyplot as plt
import filenames
from hivevo.patients import Patient
from tools import get_mutation_positions, get_fixation_positions, initial_idx_mask
from trajectory import Trajectory, create_trajectory_list, filter

def get_activity(trajectory_list, time_bins):
    return None


patient_name = "p1"
patient = Patient.load(patient_name)
region = "env"

aft = patient.get_allele_frequency_trajectories(region)
trajectories = create_trajectory_list(patient, region, aft)

filtered_traj = filter(trajectories, "np.sum(traj.frequencies > 0.25, dtype=bool)")
filtered_traj = filter(filtered_traj, "traj.t[-1] > 500")
time_bins = np.linspace(0, 1000, 10)
get_activity(filtered_traj, time_bins)
