from trajectory import Trajectory, create_all_patient_trajectories
from hivevo.patients import Patient
import filenames
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    region = "pol"
    fontsize = 16
    nb_bin = 30

    trajectories = create_all_patient_trajectories(region)
    trajectories = [traj for traj in trajectories if traj.t[-1] > 0] # Remove 1 point trajectories
    syn_traj = [traj for traj in trajectories if traj.synonymous == True]
    non_syn_traj = [traj for traj in trajectories if traj.synonymous == False]

    syn_traj_fixed = [traj for traj in syn_traj if traj.fixation == "fixed"]
    non_syn_traj_fixed = [traj for traj in non_syn_traj if traj.fixation == "fixed"]

    syn_time = [traj.t[-1] for traj in syn_traj_fixed]
    non_syn_time = [traj.t[-1] for traj in non_syn_traj_fixed]

    plt.figure()
    plt.hist(non_syn_time, bins=nb_bin, label="Non_Syn")
    plt.hist(syn_time, bins=nb_bin, label="Syn")
    plt.legend(fontsize=fontsize)
    plt.xlabel("Time for fixation [days]", fontsize=fontsize)
    plt.ylabel("# of trajectories", fontsize=fontsize)
    plt.grid()
    plt.show()
