import numpy as np
import matplotlib.pyplot as plt
import filenames
from hivevo.patients import Patient
from trajectory import Trajectory, create_trajectory_list, filter


def get_activity(trajectory_list, time_bins):
    return None


patient_name = "p1"
patient = Patient.load(patient_name)
region = "env"

aft = patient.get_allele_frequency_trajectories(region)
trajectories = create_trajectory_list(patient, region, aft)

filtered_traj = [traj for traj in trajectories if np.sum(traj.frequencies > 0.2, dtype=bool)]
# filtered_traj = [traj for traj in trajectories if traj.t[-1] > 0]  # Remove 1 point only trajectories
time_bins = np.linspace(0, 1000, 10)

fixed, lost, active = [], [], []

for ii in range(len(time_bins)-1):
    nb_traj = len([traj for traj in trajectories if traj.t[-1] >= time_bins[ii]])
    # TODO : design the function to get the number of active fixed and lost at a given time
