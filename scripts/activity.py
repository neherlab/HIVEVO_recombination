import numpy as np
import copy
import matplotlib.pyplot as plt
import filenames
from hivevo.patients import Patient
from tools import get_mutation_positions, get_fixation_positions, initial_idx_mask
from trajectories import Trajectory, create_trajectory_list


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


def get_trajectory_positions(patient, region, aft, threshold_start=0.1, threshold_end=0.9):
    """
    Return a 3D (time*letter*genome_position) boolean matrix where True are all the position that belong
    to trajectories.
    A trajectory is defined as an aft that starts above the threshold_start value and ends when getting
    above the threshold_end value, below the threshold_start value, or at the last time point if it didn't
    meet any of those conditions.
    """
    mut_pos = get_mutation_positions(patient, region, aft, threshold_start=0.1)
    # TODO


patient_name = "p1"
patient = Patient.load(patient_name)
region = "env"

aft = patient.get_allele_frequency_trajectories(region)
threshold_low = 0.01
threshold_high = 0.99

trajectories = create_trajectory_list(patient, region, aft)
