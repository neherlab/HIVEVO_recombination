import trajectories
import tools
import filenames
import numpy as np
import copy
import matplotlib.pyplot as plt
from hivevo.patients import Patient
from tools import get_mutation_positions, get_fixation_positions, initial_idx_mask


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

trajectories = []
# Exctract the full time series of af for mutations and place them in a 2D matrix as columns
mutation_positions = tools.get_mutation_positions(patient, region, aft, threshold_low)
region_mut_pos = np.sum(mutation_positions, axis=0, dtype=bool)
mask = np.tile(region_mut_pos, (12, 1, 1))
mut_frequencies = aft[mask]
mut_frequencies = np.reshape(mut_frequencies, (12, -1))  # each column is a different mutation
# Removing "mutation" at first time point because we don't know their history, ie rising or falling
mut_frequencies = np.delete(mut_frequencies, np.where(mut_frequencies[0, :] > threshold_low), axis=1)

filter1 = mut_frequencies > threshold_low
filter2 = mut_frequencies < threshold_high

# true for the rest of time points once it hits fixation
filter_fixation = np.cumsum(~filter2, axis=0, dtype=bool)
trajectory_filter = np.logical_and(~filter_fixation, filter1)
new_trajectory_filter = np.logical_and(~trajectory_filter[:-1, :], trajectory_filter[1:, :])
new_trajectory_filter = np.insert(new_trajectory_filter, 0, trajectory_filter[0, :], axis=0)
trajectory_stop_filter = np.logical_and(trajectory_filter[:-1, :], ~trajectory_filter[1:, :])
trajectory_stop_filter = np.insert(trajectory_stop_filter, 0, np.zeros(
    trajectory_stop_filter.shape[1], dtype=bool), axis=0)

# for ii in range(mut_frequencies.shape[1]):
ii = 17
traj = 0
date = patient.dsi[0]
time = patient.dsi - date
# iterate for all trajectories inside this column
for jj, idx_start in enumerate(np.where(new_trajectory_filter[:, ii] == True)[0]):

    if not True in (trajectory_stop_filter[idx_start:, ii] == True):  # still active
        idx_end = None
    else:
        idx_end = np.where(trajectory_stop_filter[:, ii] == True)[0][jj]  # fixed or lost

    if idx_end == None:
        freqs = mut_frequencies[idx_start:, ii]
        t = time[idx_start:]
    else:
        freqs = mut_frequencies[idx_start:idx_end, ii]
        t = time[idx_start:idx_end]

    if idx_end == None:
        fixation = "active"
    elif filter_fixation[idx_end, ii] == True:
        fixation = "fixed"
    else:
        fixation = "lost"

    traj = trajectories.Trajectory(freqs, t, date, fixation, threshold_low, threshold_high, patient, region,
                                   position=None, nucleotide=None)  # TODO: setup position and nucleotide
    trajectories = trajectories + [traj]
    breakpoint()
