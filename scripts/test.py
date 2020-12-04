# Adds link to the scripts folder
from proba_fix import get_nonuniform_bins
import tools
from activity import get_average_activity
import copy
import filenames
from hivevo.patients import Patient
from trajectory import Trajectory, create_trajectory_list, create_all_patient_trajectories
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append("../scripts/")
from hivevo.HIVreference import HIVreference



region = "env"
freq_range = [0.2, 0.4]
trajectories = create_all_patient_trajectories(region)
print(len(trajectories))
trajectories = [traj for traj in trajectories if np.sum(np.logical_and(
        traj.frequencies >= freq_range[0], traj.frequencies < freq_range[1]), dtype=bool)]

trajectories2 = [traj for traj in trajectories if np.sum(np.logical_and(
        traj.frequencies[0] >= freq_range[0], traj.frequencies[0] < freq_range[1]), dtype=bool)]

print(len(trajectories))
print(len(trajectories2))
