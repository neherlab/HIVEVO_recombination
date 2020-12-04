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
trajectories = create_all_patient_trajectories(region)
print(len(trajectories))

times = np.linspace(-800, 0, 50)
number = [len([traj for traj in trajectories if traj.t_previous_sample > times[idx]]) for idx in range(len(times))]

plt.figure()
plt.plot(times, number, '.-')
plt.xlabel("Time last sample [days]")
plt.ylabel('Number of trajectories with t_last > t')
plt.grid()
plt.show()
