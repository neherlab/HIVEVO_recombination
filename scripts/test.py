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


def split_traj_fitness(trajectories, first_quantiles=[0.05, 0.5], second_quantiles=[0.5, 0.95]):
    first_quantiles_value = [np.nanquantile([traj.fitness_cost for traj in trajectories], first_quantile) for first_quantile in first_quantiles]
    second_quantiles_value = [np.nanquantile([traj.fitness_cost for traj in trajectories], second_quantile) for second_quantile in second_quantiles]
    traj_low = [traj for traj in trajectories if (traj.fitness_cost > first_quantiles_value[0]) and (
        traj.fitness_cost < first_quantiles_value[1])]
    traj_high = [traj for traj in trajectories if (traj.fitness_cost > second_quantiles_value[0]) and (
        traj.fitness_cost < second_quantiles_value[1])]
    return traj_low, traj_high


regions = ["env", "pol", "gag"]
trajectories = {}

for region in regions:
    # Create the dictionary with the different regions
    tmp_trajectories = create_all_patient_trajectories(region)
    tmp_trajectories = [traj for traj in tmp_trajectories if traj.t[-1] != 0]
    trajectories[region] = tmp_trajectories

    # Split into sub dictionnaries (rev, non_rev and all)
    rev = [traj for traj in trajectories[region] if traj.reversion == True]
    non_rev = [traj for traj in trajectories[region] if traj.reversion == False]
    trajectories[region] = {"rev": rev, "non_rev": non_rev, "all":trajectories[region]}

    for key in trajectories[region].keys():
        traj_low, traj_high = split_traj_fitness(trajectories[region][key])
        trajectories[region][key] = {"low":traj_low, "high":traj_high, "all":trajectories[region][key]}


from propagator import get_mean_in_time
reversions = ["rev", "non_rev"]
fitness = ["low", "high"]
nb_bins = 15
freq_ranges = [[0.2, 0.4], [0.3, 0.7], [0.6, 0.8]]
means = {}

for freq_range in freq_ranges:
    means[str(freq_range)] = copy.deepcopy(trajectories)
    for region in regions:
        for reversion in reversions:
            for f in fitness:
                time_bins, tmp_mean, _, _ = get_mean_in_time(trajectories[region][reversion][f], nb_bins, freq_range)
                means[str(freq_range)][region][reversion][f] = tmp_mean


def plot_means(time_bins, means, freq_ranges, region, reversion, fontsize=16):
    plt.figure(figsize=(14, 10))
    for freq_range in freq_ranges:
        plt.plot(time_bins, means[str(freq_range)][region][reversion]["low"], '.-', label=f"{region}-{reversion}-low")
        plt.plot(time_bins, means[str(freq_range)][region][reversion]["high"], '.-', label=f"{region}-{reversion}-high")
    plt.xlabel("Time [days]", fontsize=fontsize)
    plt.ylabel("Frequency", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.title(f"Region {region}", fontsize=fontsize)
    plt.ylim([0,1])
    plt.grid()
    plt.show()

plot_means(time_bins, means, freq_ranges, "pol", "non_rev")
