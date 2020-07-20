import numpy as np
import matplotlib.pyplot as plt
import filenames
from hivevo.patients import Patient
from trajectory import Trajectory, create_trajectory_list, filter


def get_duration_distribution(trajectory_list):
    time_bins = np.linspace(0, 2000, 100)
    nb_traj = []

    for time in time_bins:
        filtered_traj = filter(trajectory_list, f"traj.t[-1] >= {time}")
        nb_traj = nb_traj + [len(filtered_traj)]
    return time_bins, nb_traj


def get_freq_distribution(trajectories, nb_bin=10, bin_range=[0.01, 0.99]):
    frequency_bins = np.linspace(bin_range[0], bin_range[1], nb_bin)
    nb_in = []
    for ii in range(len(frequency_bins) - 1):
        nb = len([traj for traj in trajectories if np.sum(np.logical_and(
            traj.frequencies >= frequency_bins[ii], traj.frequencies < frequency_bins[ii + 1]), dtype=bool)])
        nb_in = nb_in + [nb]

    return frequency_bins, nb_in


def plot_duration_distribution(time_bins, nb_traj, freq_min, fontsize=16):
    plt.figure()
    plt.title(f"Trajectory duration freq>{round(freq_min,1)}", fontsize=fontsize)
    plt.plot(time_bins, nb_traj, '.-')
    plt.xlabel("t [days]", fontsize=fontsize)
    plt.ylabel("#Trajectories > t", fontsize=fontsize)
    plt.grid()


def plot_length_distribution(trajectory_list, freq_min, fontsize=16):
    plt.figure()
    plt.title(f"Trajectory length freq>{round(freq_min,1)}", fontsize=fontsize)
    plt.hist([len(x.t) for x in trajectory_list], bins=range(12), align="left", rwidth=0.7)
    plt.xlabel("Length", fontsize=fontsize)
    plt.ylabel("#Trajectories", fontsize=fontsize)


def plot_freq_ditribution(frequency_bins, nb_in, fontsize=16):
    frequency_bins = 0.5 * (frequency_bins[:-1] + frequency_bins[1:])
    plt.figure()
    plt.title("Distribution of trajectories in freq", fontsize=16)
    plt.plot(frequency_bins, nb_in, '.-')
    plt.xlabel("Frequency bin center", fontsize=16)
    plt.ylabel("# Trajectories", fontsize=16)
    # plt.yscale("log")
    plt.xlim([0, 1])
    plt.grid()


def fit_distribution(frequency_bins, nb_in):
    "Return an exponential fit of the distribution of nb of frequency in each bin"

    def f(x, alpha, beta, gamma):
        return alpha * np.exp(-beta * x) + gamma

    import scipy
    frequency_bins = 0.5 * (frequency_bins[:-1] + frequency_bins[1:])
    fit = scipy.optimize.curve_fit(f, frequency_bins, nb_in, p0=[nb_in[0], 1, nb_in[-1]])
    return fit[0]


def get_nonuniform_bins(nb_bins, fit_params=None, bin_range=[0.05, 0.95]):
    "Return the non_uniform bin edges that try to keep the number of trajectories in each bin constant"
    def f(x, alpha, beta, gamma):
        return alpha * np.exp(-beta * x) + gamma

    if fit_params == None:
        fit_params = [1418.92030293, 13.19671385, 85.01982428] # comes from fit of the distribution over 40 bins

    x = np.linspace(bin_range[0], bin_range[1], nb_bins + 1)
    x = 0.5 * (x[1:] + x[:-1])

    y = f(x, *fit_params)
    y /= np.sum(y)  # at this point y is the "density" of trajectories
    y = 1 / y
    y = np.cumsum(y)
    non_uniform_bins = y / y[-1] * (bin_range[-1] - bin_range[0]) + bin_range[0]
    non_uniform_bins = np.concatenate(([bin_range[0]], non_uniform_bins))
    return non_uniform_bins


if __name__ == "__main__":
    patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]
    region = "env"
    trajectories = []

    for patient_name in patient_names:
        patient = Patient.load(patient_name)
        aft = patient.get_allele_frequency_trajectories(region)
        trajectories = trajectories + create_trajectory_list(patient, region, aft)

    ### Trajectory durations ###
    # for freq_min in np.arange(0, 1, 0.2):
    #     filtered_traj = [traj for traj in trajectories if np.sum(traj.frequencies > freq_min, dtype=bool)]
    #     time_bins, nb_traj = get_duration_distribution(filtered_traj)
    #     plot_duration_distribution(time_bins, nb_traj, freq_min)

    ### Trajectory lengths ###
    # for freq_min in np.arange(0, 1, 0.2):
    #     filtered_traj = [x for x in trajectories if np.sum(x.frequencies > freq_min, dtype=bool)]
    #     plot_length_distribution(filtered_traj, freq_min)

    ### Number of trajectory seen in frequency bin ###
    for nb_bin in [10, 20, 30]:
        non_uniform_bins = get_nonuniform_bins(nb_bin)
        nb_in = []
        for ii in range(len(non_uniform_bins) - 1):
            nb = len([traj for traj in trajectories if np.sum(np.logical_and(
                traj.frequencies >= non_uniform_bins[ii], traj.frequencies < non_uniform_bins[ii + 1]), dtype=bool)])
            nb_in = nb_in + [nb]

        non_uniform_bins = 0.5 * (non_uniform_bins[:-1] + non_uniform_bins[1:])

        plt.plot(non_uniform_bins, nb_in, label=f"{nb_bin} bins")
    plt.legend()
    plt.show()
