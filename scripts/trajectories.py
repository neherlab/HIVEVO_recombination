import numpy as np
import filenames
from hivevo.patients import Patient
import tools

class Trajectory():
    def __init__(self, frequencies, time_points, threshold_low, threshold_high,
                 patient, region, position, nucleotide):

        self.frequencies = frequencies          # Numpy 1D vector
        self.time_points = time_points          # Numpy 1D vector (in days)
        self.threshold_low = threshold_low      # Value of threshold_low used for extraction
        self.threshold_high = threshold_high    # Value of threshold_high used for extraction
        self.patient = patient                  # Patient name (string)
        self.region = region                    # Region name, string
        self.position = position                # Position on the region (int)
        self.nucleotide = nucleotide            # Nucleotide number according to HIVEVO_access/hivevo/sequence alpha


def create_trajectory_list(patient, region, aft, threshold_low, threshold_high):
    trajectories = []
    region_mut_pos = np.sum(np.array(tools.get_mutation_positions(patient, region, aft, threshold_low)), axis=0, dtype=bool)
    breakpoint()
