import numpy as np

def filter(trajectory_list, filter_str):
    filtered_trajectory_list = []
    for traj in trajectory_list:
        if eval(filter_str):
            filtered_trajectory_list = filtered_trajectory_list + [traj]

    return filtered_trajectory_list
