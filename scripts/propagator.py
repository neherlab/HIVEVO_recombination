from trajectory import Trajectory, create_trajectory_list, filter, create_all_patient_trajectories
from hivevo.patients import Patient
import filenames
import matplotlib.pyplot as plt
import numpy as np
import copy


if __name__ == "__main__":
    # patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]
    patient_names = ["p1"]
    region = "env"
    nb_bin = 10
    fontsize = 16
    freq_range = [0.4, 0.6]

    trajectories = create_all_patient_trajectories(region, patient_names)
    syn_traj = copy.deepcopy([traj for traj in trajectories if traj.synonymous == True])
    non_syn_traj = copy.deepcopy([traj for traj in trajectories if traj.synonymous == False])

    syn_traj = [traj for traj in syn_traj if np.sum(np.logical_and(
        traj.frequencies >= freq_range[0], traj.frequencies < freq_range[1]), dtype=bool)]
    non_syn_traj = [traj for traj in non_syn_traj if np.sum(np.logical_and(
        traj.frequencies >= freq_range[0], traj.frequencies < freq_range[1]), dtype=bool)]

    for traj in syn_traj:
        idx = np.where(np.logical_and(traj.frequencies >= freq_range[0], traj.frequencies < freq_range[1]))[0][0]
        traj.t = traj.t - traj.t[idx]

    for traj in non_syn_traj:
        idx = np.where(np.logical_and(traj.frequencies >= freq_range[0], traj.frequencies < freq_range[1]))[0][0]
        traj.t = traj.t - traj.t[idx]
