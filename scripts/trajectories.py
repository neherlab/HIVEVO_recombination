import numpy as np
import filenames
from hivevo.patients import Patient
import tools

class Trajectory():
    def __init__(self, frequencies, t, date, fixation, threshold_low, threshold_high,
                 patient, region, position, nucleotide):

        self.frequencies = frequencies          # Numpy 1D vector
        self.t = t                              # Numpy 1D vector (in days)
        self.date = date                        # Date at t=0 (int, in days)
        self.fixation = fixation                # "fixed", "active", "lost" (at the next time point) 
        self.threshold_low = threshold_low      # Value of threshold_low used for extraction
        self.threshold_high = threshold_high    # Value of threshold_high used for extraction
        self.patient = patient                  # Patient name (string)
        self.region = region                    # Region name, string
        self.position = position                # Position on the region (int)
        self.nucleotide = nucleotide            # Nucleotide number according to HIVEVO_access/hivevo/sequence alpha


def create_trajectory_list(patient, region, aft, threshold_low=0.01, threshold_high=0.99):
    trajectories = []

    # Exctract the full time series of af for mutations and place them in a 2D matrix as columns
    region_mut_pos = np.sum(np.array(tools.get_mutation_positions(patient, region, aft, threshold_low)), axis=0, dtype=bool)
    mask = np.tile(region_mut_pos, (12,1,1))
    mut_frequencies = np.array(aft[mask])
    mut_frequencies = np.reshape(mut_frequencies, (12,-1)) # each column is a different mutation

    filter = np.logical_and(mut_frequencies>threshold_low, mut_frequencies<threshold_high)


    breakpoint()
