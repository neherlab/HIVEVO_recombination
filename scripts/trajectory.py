import numpy as np
import copy
import filenames
from hivevo.patients import Patient
import tools


class Trajectory():
    def __init__(self, frequencies, t, date, t_last_sample, fixation, threshold_low, threshold_high,
                 patient, region, position, nucleotide):

        self.frequencies = frequencies              # Numpy 1D vector
        self.t = t                                  # Numpy 1D vector (in days)
        self.date = date                            # Date at t=0 (int, in days)
        self.t_last_sample = t_last_sample          # Number of days at which last sample was taken for the patient (relative to t[0] = 0)
        self.fixation = fixation                    # "fixed", "active", "lost" (at the next time point)
        self.threshold_low = threshold_low          # Value of threshold_low used for extraction
        self.threshold_high = threshold_high        # Value of threshold_high used for extraction
        self.patient = patient                      # Patient name (string)
        self.region = region                        # Region name, string
        self.position = position                    # Position on the region (int)
        self.nucleotide = nucleotide                # Nucleotide number according to HIVEVO_access/hivevo/sequence alpha

    def __repr__(self):
        return str(self.__dict__)


def create_trajectory_list(patient, region, aft, threshold_low=0.01, threshold_high=0.99):
    """
    Creates a list of trajectories from a patient allele frequency trajectory (aft).
    Select the maximal amount of trajectories:
        - trajectories are extinct before the first time point
        - trajectories are either active, extinct or fixed after the last time point, which is specified
        - a trajectory can be as small as 1 point (extinct->active->fixed, or extinct->active->exctinct)
        - several trajectories can come from a single aft (for ex. extinct->active->extinct->active->fixed)
    """
    trajectories = []
    # Exctract the full time series of af for mutations and place them in a 2D matrix as columns
    mutation_positions = tools.get_mutation_positions(patient, region, aft, threshold_low)
    region_mut_pos = np.sum(mutation_positions, axis=0, dtype=bool)
    mask = np.tile(region_mut_pos, (aft.shape[0], 1, 1))
    mut_frequencies = aft[mask]
    mut_frequencies = np.reshape(mut_frequencies, (aft.shape[0], -1))  # each column is a different mutation

    # Map the original position and nucleotide
    i_idx, j_idx = np.meshgrid(range(mutation_positions.shape[1]), range(
        mutation_positions.shape[2]), indexing="ij")
    coordinates = np.array([i_idx, j_idx])
    coordinates = np.tile(coordinates, (aft.shape[0], 1, 1, 1))
    coordinates = np.swapaxes(coordinates, 1, 3)
    coordinates = np.swapaxes(coordinates, 1, 2)
    coordinates = coordinates[mask]
    coordinates = np.reshape(coordinates, (aft.shape[0], -1, 2))
    # coordinates[t,ii,:] gives the [nucleotide, genome_position] of the mut_frequencies[t,ii] for any t

    # Removing "mutation" at first time point because we don't know their history, ie rising or falling
    mask2 = np.where(mut_frequencies[0, :] > threshold_low)
    mut_freq_mask = mut_frequencies.mask # keep the mask aside as np.delete removes it
    mut_freq_mask = np.delete(mut_freq_mask, mask2, axis=1)
    mut_frequencies = np.array(np.delete(mut_frequencies, mask2, axis=1))
    coordinates = np.delete(coordinates, mask2, axis=1)

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

    # Include the masked points in middle of trajectories (ex [0, 0.2, 0.6, --, 0.8, 1])
    stop_at_masked_filter = np.logical_and(trajectory_stop_filter, mut_freq_mask)
    stop_at_masked_shifted = np.roll(stop_at_masked_filter, 1, axis=0)
    stop_at_masked_shifted[0,:] = False
    stop_at_masked_restart = np.logical_and(stop_at_masked_shifted, new_trajectory_filter)

    new_trajectory_filter[stop_at_masked_restart] = False
    trajectory_stop_filter[np.roll(stop_at_masked_restart, -1, 0)] = False

    breakpoint()

    date = patient.dsi[0]
    time = patient.dsi - date
    # iterate though all columns (<=> mutations trajectories)
    for ii in range(mut_frequencies.shape[1]):
        # iterate for all trajectories inside this column
        for jj, idx_start in enumerate(np.where(new_trajectory_filter[:, ii] == True)[0]):

            if not True in (trajectory_stop_filter[idx_start:, ii] == True):  # still active
                idx_end = None
            else:
                idx_end = np.where(trajectory_stop_filter[:, ii] == True)[0][jj]  # fixed or lost

            if idx_end == None:
                freqs = np.ma.array(mut_frequencies[idx_start:, ii])
                freqs.mask = mut_freq_mask[idx_start:, ii]
                t = time[idx_start:]
            else:
                freqs = np.ma.array(mut_frequencies[idx_start:idx_end, ii])
                freqs.mask = mut_freq_mask[idx_start:idx_end, ii]
                t = time[idx_start:idx_end]

            if idx_end == None:
                fixation = "active"
            elif filter_fixation[idx_end, ii] == True:
                fixation = "fixed"
            else:
                fixation = "lost"

            traj = Trajectory(np.ma.array(freqs), t - t[0], date + t[0], time[-1] - t[0], fixation, threshold_low, threshold_high, patient.name, region,
                              position=coordinates[0, ii, 1], nucleotide=coordinates[0, ii, 0])
            trajectories = trajectories + [traj]
    return trajectories


def filter(trajectory_list, filter_str):
    """
    Evaluate filter_str on all elements (traj) of the trajectory_list and returned the filtered list.
    Return a deepcopy of the elements. Change to a reference instead of copy for performance optimisation.
    It is better to use list comprehension directly ex. : filtered_traj = [x for x in trajectories if np.sum(x.frequencies > freq_min, dtype=bool)]
    """
    return [traj for traj in trajectory_list if eval(filter_str)]


if __name__ == "__main__":
    patient = Patient.load("p3")
    aft = patient.get_allele_frequency_trajectories("env")
    trajectories = create_trajectory_list(patient, "env", aft)
