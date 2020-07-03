import numpy as np
import copy
import matplotlib.pyplot as plt
import filenames
from hivevo.patients import Patient
from tools import get_mutation_positions, get_fixation_positions, initial_idx_mask
from trajectory import Trajectory, create_trajectory_list


def get_rising_positions(patient, region, aft, threshold=0.25):
    """
    Return a 3D (time*letter*genome_position) boolean matrix where True are all the position where
    a mutation is seen for the first time above the threshold value.
    """
    argmax = np.argmax(get_mutation_positions(patient, region, aft, threshold), axis=0)
    result = np.zeros(aft.shape, dtype=bool)
    xx, yy = np.meshgrid(range(aft.shape[2]), range(aft.shape[1]))
    mask = argmax.flatten() != 0
    result[argmax.flatten()[mask], yy.flatten()[mask], xx.flatten()[mask]] = True
    return result

# def get_rising_positions(patient, region, aft, threshold=0.25):
#     get_mutation_positions(patient, region, aft, threshold)
#     rising_positions = np.cumsum(np.cumsum(rising_positions, axis=0, dtype=int), axis=0, dtype=int)
#     rising_positions[rising_positions != 1] = 0
#     return rising_positions.astype(bool)


patient_name = "p1"
patient = Patient.load(patient_name)
region = "env"

aft = patient.get_allele_frequency_trajectories(region)
trajectories = create_trajectory_list(patient, region, aft)

from filter import filter
filtered_traj = filter(trajectories, "np.sum(traj.frequencies > 0.5, dtype=bool)")
