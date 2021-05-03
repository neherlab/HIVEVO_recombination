# Adds link to the scripts folder
import filenames
from hivevo.patients import Patient
import trajectory
from divergence import load_divergence_dict, WH_evo_rate, get_mean_divergence_patient, divergence_matrix
from hivevo.HIVreference import HIVreference
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import copy
import os
import sys
import pickle
sys.path.append("../../scripts/")


def create_time_bins(bin_size=400):
    """
    Create time bins for the mean in time analysis. It does homogeneous bins, except for the one at t=0 that
    only takes point where t=0. Bin_size is in days.
    """
    time_bins = [-5, 5]
    interval = [-600, 3000]
    while time_bins[0] > interval[0]:
        time_bins = [time_bins[0] - bin_size] + time_bins
    while time_bins[-1] < interval[1]:
        time_bins = time_bins + [time_bins[-1] + bin_size]

    return np.array(time_bins)


def get_mean_in_time(trajectories, bin_size=400, freq_range=[0.4, 0.6]):
    """
    Computes the mean frequency in time of a set of trajectories from the point they are seen in the freq_range window.
    Returns the middle of the time bins and the computed frequency mean.
    """
    trajectories = copy.deepcopy(trajectories)

    # Create bins and select trajectories going through the freq_range
    time_bins = create_time_bins(bin_size)
    trajectories = [traj for traj in trajectories if np.sum(np.logical_and(
        traj.frequencies >= freq_range[0], traj.frequencies < freq_range[1]), dtype=bool)]

    # Offset trajectories to set t=0 at the point they are seen in the freq_range and adds all the frequencies / times
    # to arrays for later computation of mean
    t_traj = np.array([])
    f_traj = np.array([])
    for traj in trajectories:
        idx = np.where(np.logical_and(traj.frequencies >=
                                      freq_range[0], traj.frequencies < freq_range[1]))[0][0]
        traj.t = traj.t - traj.t[idx]
        t_traj = np.concatenate((t_traj, traj.t))
        f_traj = np.concatenate((f_traj, traj.frequencies))

    # Binning of all the data in the time bins
    filtered_fixed = [traj for traj in trajectories if traj.fixation == "fixed"]
    filtered_lost = [traj for traj in trajectories if traj.fixation == "lost"]
    freqs, fixed, lost = [], [], []
    for ii in range(len(time_bins) - 1):
        freqs = freqs + [f_traj[np.logical_and(t_traj >= time_bins[ii], t_traj < time_bins[ii + 1])]]
        fixed = fixed + [len([traj for traj in filtered_fixed if traj.t[-1] < time_bins[ii]])]
        lost = lost + [len([traj for traj in filtered_lost if traj.t[-1] < time_bins[ii]])]

    # Computation of the mean in each bin, active trajectories contribute their current frequency,
    # fixed contribute 1 and lost contribute 0
    mean = []
    for ii in range(len(freqs)):
        mean = mean + [np.sum(freqs[ii]) + fixed[ii]]
        if len(freqs[ii]) + fixed[ii] + lost[ii] != 0:
            mean[-1] /= (len(freqs[ii]) + fixed[ii] + lost[ii])
        else:
            mean[-1] = np.nan
    nb_active = [len(freq) for freq in freqs]
    nb_dead = [fixed[ii] + lost[ii] for ii in range(len(fixed))]

    return 0.5 * (time_bins[1:] + time_bins[:-1]), mean, nb_active, nb_dead


def make_mean_in_time_dict(trajectories):
    regions = ["env", "pol", "gag", "all"]
    means = {}
    freq_ranges = [[0.2, 0.4], [0.4, 0.6], [0.6, 0.8]]
    times = []

    for freq_range in freq_ranges:
        means[str(freq_range)] = {}
        for region in regions:
            means[str(freq_range)][region] = {}
            for key in trajectories[region].keys():
                times, means[str(freq_range)][region][key], _, _ = get_mean_in_time(
                    trajectories[region][key], freq_range=freq_range)
    return times, means, freq_ranges


def bootstrap_patient_names(patient_names=["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]):
    "Returns a list of patient name bootstrap with replacement (patient can appear more than once)."
    choice = np.random.choice(range(len(patient_names)), len(patient_names))
    names = []
    for ii in choice:
        names += [patient_names[ii]]
    return names


def bootstrap_mean_in_time(trajectories, region, mut_type, freq_range, nb_bootstrap=10):
    """
    Computes the mean_in_time for the given region and mutation type (rev, non_rev etc...) nb_boostrap time
    by bootstrapping patients and returns the average and standard deviation vectors.
    """

    means = []
    for ii in range(nb_bootstrap):
        # Bootstrapping trajectories
        bootstrap_names = bootstrap_patient_names()
        bootstrap_trajectories = []
        for name in bootstrap_names:
            bootstrap_trajectories += [traj for traj in trajectories[region]
                                       [mut_type] if traj.patient == name]

        # Computing the mean in time for each boostrap
        time, mean, _, _ = get_mean_in_time(bootstrap_trajectories, freq_range=freq_range)
        means += [[mean]]

    means = np.array(means)
    average = np.nanmean(means, axis=0)[0, :]
    std = np.nanstd(means, axis=0)[0, :]

    return time, average, std


def make_bootstrap_mean_dict(trajectory_dict, nb_bootstrap=10):
    """
    Generates the dictionary for bootstrapped mean frequency in time. Does it for all regions, the 3 frequency windows
    and rev / non_rav mutations.
    Keys are the following : dict["rev"/"non_rev"]["[0.2,0.4]","[0.4,0.6]","[0.6,0.8]"]
    """
    bootstrap_dict = {"rev": {}, "non_rev": {}}
    for key1 in ["rev", "non_rev"]:
        for key2 in [[0.2, 0.4], [0.4, 0.6], [0.6, 0.8]]:
            times, mean, std = bootstrap_mean_in_time(trajectory_dict, "all", key1, key2, nb_bootstrap)
            bootstrap_dict[key1][str(key2)] = {"mean": mean, "std": std}

    return bootstrap_dict, times


def create_div_bootstrap_dict():
    """
    Create an empty dictionary with the relevant keys. The tip of the dictionnary are lists where there are the
    different divergence values for each of the bootstrapping runs.
    """
    regions = ["env", "pol", "gag"]
    consensus_keys = ["consensus", "non_consensus", "all"]
    fitness_keys = ["low", "high", "all", "first", "second", "third"]

    bootstrap_dict = {}
    for region in regions:
        bootstrap_dict[region] = {}
        for key in consensus_keys:
            bootstrap_dict[region][key] = {}
            if key != "all":
                for key2 in fitness_keys:
                    bootstrap_dict[region][key][key2] = []
            else:
                for key2 in ["all", "first", "second", "third"]:
                    bootstrap_dict[region][key][key2] = []

    return bootstrap_dict


def make_bootstrap_divergence_dict(nb_bootstrap=10, consensus=False):
    """
    Creates a dictionary with the divergence in time for each patient.
    Format of the dictionary : dict[region][patient][consensus/non_consensus/all][high/low/all/first/second/third]
    Turn consensus to True to compute the divergence to consensus sequence instead of founder sequence.
    """

    regions = ["env", "pol", "gag"]
    patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]
    fitness_keys = ["low", "high", "all", "first", "second", "third"]
    time = np.arange(0, 2001, 40)

    # Generating a dictionnary with the divergence for each patient (interpolated to the time vector)
    divergence_dict = {}
    for region in regions:
        divergence_dict[region] = {}
        for patient_name in patient_names:
            patient = Patient.load(patient_name)
            patient_div_dict = get_mean_divergence_patient(patient, region, consensus)
            divergence_dict[region][patient_name] = patient_div_dict

            for key in divergence_dict[region][patient_name].keys():
                for key2 in divergence_dict[region][patient_name][key].keys():
                    divergence_dict[region][patient_name][key][key2] = np.interp(
                        time, patient.dsi, divergence_dict[region][patient_name][key][key2])

    # Bootstrapping the divergence values over patients. Tips of the dict are list with 1 div vector for each of the bootstrapping
    bootstrap_dict = create_div_bootstrap_dict()
    for ii in range(nb_bootstrap):
        bootstrap_names = bootstrap_patient_names()
        for region in regions:
            dict_list = []
            for patient_name in bootstrap_names:
                dict_list += [divergence_dict[region][patient_name]]

            for key in divergence_dict[region][patient_names[0]].keys():
                for key2 in divergence_dict[region][patient_names[0]][key].keys():
                    tmp = np.array([dict[key][key2] for dict in dict_list])
                    bootstrap_dict[region][key][key2] += [np.mean(tmp, axis=0)]

    # Averaging the bootstrapping
    for region in bootstrap_dict.keys():
        for key in bootstrap_dict[region].keys():
            for key2 in bootstrap_dict[region][key].keys():
                tmp = np.array(bootstrap_dict[region][key][key2]).copy()
                bootstrap_dict[region][key][key2] = {"mean": np.mean(tmp, axis=0), "std": np.std(tmp, axis=0)}

    return time, bootstrap_dict


def save(obj, filename):
    """
    Saves the given object using pickle.
    """
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_dict(path="bootstrap_mean_dict"):
    "Load the dict from pickle."
    bootstrap_dict = {}
    with open(path, 'rb') as file:
        bootstrap_dict = pickle.load(file)

    return bootstrap_dict


def make_pfix(nb_bin=8):
    regions = ["env", "pol", "gag", "all"]
    pfix = {}
    for region in regions:
        pfix[region] = {}
        for key in trajectories[region].keys():
            tmp_freq_bin, tmp_proba, tmp_err = get_proba_fix(trajectories[region][key], nb_bin=nb_bin)
            pfix[region][key] = {"freq_bin": tmp_freq_bin, "proba": tmp_proba, "error": tmp_err}
    return pfix


def get_trajectories_offset(trajectories, freq_range):
    trajectories = copy.deepcopy(trajectories)
    trajectories = [traj for traj in trajectories if np.sum(np.logical_and(
        traj.frequencies >= freq_range[0], traj.frequencies < freq_range[1]), dtype=bool)]

    # Offset trajectories to set t=0 at the point they are seen in the freq_range and adds all the frequencies / times
    # to arrays for later computation of mean
    for traj in trajectories:
        idx = np.where(np.logical_and(traj.frequencies >=
                                      freq_range[0], traj.frequencies < freq_range[1]))[0][0]
        traj.t = traj.t - traj.t[idx]
        if traj.fixation == "fixed":
            traj.t = np.append(traj.t, [traj.t[-1] + 500, 3000])
            traj.frequencies = np.append(traj.frequencies, [1, 1])
        elif traj.fixation == "lost":
            traj.t = np.append(traj.t, [traj.t[-1] + 500, 3000])
            traj.frequencies = np.append(traj.frequencies, [0, 0])

    return trajectories


def get_divergence_in_time(region, patient, ref=HIVreference(subtype="any")):
    """
    Returns the 2D matrix with divergence at all genome postion through time. Sites that are too often gapped
    are masked
    """
    aft = patient.get_allele_frequency_trajectories(region)
    ref_filter = trajectory.get_reference_filter(patient, region, aft, ref)
    div_3D = divergence_matrix(patient, region, aft, False)
    initial_idx = patient.get_initial_indices(region)
    div = div_3D[np.arange(aft.shape[0])[:, np.newaxis, np.newaxis], initial_idx, np.arange(aft.shape[-1])]
    div = div[:, 0, :]
    div = np.ma.array(div, mask=np.tile(~ref_filter, (aft.shape[0], 1)))
    return div


# def get_bootstrap_divergence_hist(region, bins, nb_bootstrap=10, nb_last=3):
#     """
#     Computes the histogram using bootstrapping over patients. Use the last nb_last timepoints.
#     """
#
#     hist_dict = {}
#     bootstrap_hist = []
#     for ii in range(nb_bootstrap):
#         bootstrap_names = bootstrap_patient_names()
#
#         divergence_hist = []
#         for name in bootstrap_names:
#             patient = Patient.load(name)
#             div2D = get_divergence_in_time(region, patient)
#             values = div2D[-nb_last:].flatten()
#             hist, bins = np.histogram(values, bins=bins, range=(0, 1))
#             divergence_hist += [hist]
#         divergence_hist = np.sum(divergence_hist, axis=0)
#         bootstrap_hist += [divergence_hist]
#
#     hist_dict = {"mean": np.mean(bootstrap_hist, axis=0), "std": np.std(
#         bootstrap_hist, axis=0), "bins": 0.5*(bins[1:] +bins[:-1])}
#     return hist_dict

def get_divergence_cumulative_sum(patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]):
    """
    Returns the divergence values for the last 3 datapoints of each patient. Returns both the raw values and
    the cumulative sum (normalized to 1).
    Taking only one every sampling time points as there is a lot of data.
    """
    region = "pol"
    nb_last = 3
    sampling = 20

    all_values = []
    for name in patient_names:
        patient = Patient.load(name)
        div2D = get_divergence_in_time(region, patient)
        values = div2D[-nb_last:].flatten()
        all_values += list(values[~values.mask])
    all_values = np.sort(all_values)
    cum_sum = np.cumsum(all_values)
    cum_sum /= cum_sum[-1]
    values = np.concatenate((all_values[::sampling], np.array([all_values[-1]])))
    cumulative = np.concatenate((cum_sum[::sampling], np.array([cum_sum[-1]])))
    return values, cumulative


def mean_in_time_plot(fontsize=16, fill_alpha=0.15, grid_alpha=0.5):
    trajectories = load_trajectory_dict("trajectory_dict")
    # times, means, freq_ranges = make_mean_in_time_dict(trajectories)
    # bootstrap_dict, times = make_bootstrap_mean_dict(trajectories, 100)
    # save(bootstrap_dict, "bootstrap_dict")
    times = create_time_bins()
    times = 0.5 * (times[:-1] + times[1:]) / 365
    bootstrap_dict = load_dict()
    trajectories_scheme = get_trajectories_offset(trajectories["all"]["rev"], [0.4, 0.6])

    colors = ["C0", "C1", "C2", "C4"]
    freq_ranges = [[0.2, 0.4], [0.4, 0.6], [0.6, 0.8]]

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(14, 7), sharey=True)

    # Plot left

    for traj in trajectories_scheme:
        axs[0].plot(traj.t / 365, traj.frequencies, "k-", alpha=0.1, linewidth=1)

    mean = bootstrap_dict["rev"]["[0.4, 0.6]"]["mean"]
    std = bootstrap_dict["rev"]["[0.4, 0.6]"]["std"]
    axs[0].plot(times, mean, '-', color=colors[1])
    axs[0].fill_between(times, mean - std, mean + std, color=colors[1], alpha=fill_alpha)

    axs[0].set_xlabel("Time [years]", fontsize=fontsize)
    axs[0].set_ylabel("Frequency", fontsize=fontsize)
    axs[0].set_ylim([-0.03, 1.03])
    axs[0].grid(grid_alpha)
    axs[0].set_xlim([-677 / 365, 3000 / 365])

    line1, = axs[0].plot([0], [0], "k-")
    line2, = axs[0].plot([0], [0], "-", color=colors[1])
    axs[0].legend([line1, line2], ["Individual trajectories", "Average"],
                  fontsize=fontsize, loc="lower right")

    # Plot right
    for ii, freq_range in enumerate(freq_ranges):
        for key, line in zip(["rev", "non_rev"], ["-", "--"]):
            mean = bootstrap_dict[key][str(freq_range)]["mean"]
            std = bootstrap_dict[key][str(freq_range)]["std"]
            axs[1].plot(times, mean, line, color=colors[ii])
            axs[1].fill_between(times, mean - std, mean + std, color=colors[ii], alpha=fill_alpha)

    line1, = axs[1].plot([0], [0], "k-")
    line2, = axs[1].plot([0], [0], "k--")
    line3, = axs[1].plot([0], [0], "-", color=colors[0])
    line4, = axs[1].plot([0], [0], "-", color=colors[1])
    line5, = axs[1].plot([0], [0], "-", color=colors[2])

    axs[1].set_xlabel("Time [years]", fontsize=fontsize)
    # axs[1].set_ylabel("Frequency", fontsize=fontsize)
    axs[1].set_ylim([-0.03, 1.03])
    axs[1].grid(grid_alpha)
    axs[1].legend([line3, line4, line5, line1, line2], ["[0.2, 0.4]", "[0.4, 0.6]", "[0.6, 0.8]",
                                                        "reversion", "non-reversion"], fontsize=fontsize, ncol=2, loc="lower right")

    plt.tight_layout()
    plt.savefig("Reversion_DEHV.png", format="png")
    plt.show()


def divergence_region_plot(figsize=(14, 10), fontsize=20, tick_fontsize=18,
                           colors=["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
                           fill_alpha=0.15):
    time_average = np.arange(0, 2001, 40) / 365
    divergence_dict = load_dict("bootstrap_div_dict")

    plt.figure(figsize=(14, 10))
    for ii, region in enumerate(["env", "pol", "gag"]):
        mean = divergence_dict[region]["all"]["all"]["mean"]
        std = divergence_dict[region]["all"]["all"]["std"]
        plt.plot(time_average, mean, '-', color=colors[ii], label=region)
        plt.fill_between(time_average, mean + std, mean - std, color=colors[ii], alpha=fill_alpha)
    plt.grid()
    plt.xlabel("Time since infection [years]", fontsize=fontsize)
    plt.ylabel("Divergence", fontsize=fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("Divergence_region.png", format="png")
    plt.show()


def divergence_consensus_plot(figsize=(14, 10), fontsize=20, tick_fontsize=18,
                              colors=["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
                              fill_alpha=0.15):
    text_index = 40
    time_average = np.arange(0, 2001, 40) / 365
    divergence_dict = load_dict("bootstrap_div_dict")

    plt.figure(figsize=(14, 10))
    for ii, region in enumerate(["env", "pol", "gag"]):
        mean = divergence_dict[region]["consensus"]["all"]["mean"]
        std = divergence_dict[region]["consensus"]["all"]["std"]
        plt.plot(time_average, mean, '-', color=colors[ii], label=region)
        plt.fill_between(time_average, mean - std, mean + std, alpha=fill_alpha, color=colors[ii])

        mean = divergence_dict[region]["non_consensus"]["all"]["mean"]
        std = divergence_dict[region]["non_consensus"]["all"]["std"]
        plt.plot(time_average, mean, '--', color=colors[ii])
        plt.fill_between(time_average, mean - std, mean + std, alpha=fill_alpha, color=colors[ii])

        plt.plot(time_average, divergence_dict[region]["all"]["all"]["mean"], color="0.3")

    plt.plot([0], [0], "k-", label="consensus")
    plt.plot([0], [0], "k--", label="non_consensus")
    plt.grid()
    plt.xlabel("Time since infection [years]", fontsize=fontsize)
    plt.ylabel("Divergence", fontsize=fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("Divergence_consensus.png", format="png")
    plt.show()


def divergence_site_plot(figsize=(14, 10), fontsize=20, tick_fontsize=18,
                         colors=["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
                         fill_alpha=0.15):
    time_average = np.arange(0, 2001, 40) / 365
    divergence_dict = load_dict("bootstrap_div_dict")

    plt.figure(figsize=figsize)
    region = "pol"
    for ii, site in enumerate(["first", "second", "third"]):
        mean = divergence_dict[region]["consensus"][site]["mean"]
        std = divergence_dict[region]["consensus"][site]["std"]
        plt.plot(time_average, mean, '-', color=colors[ii])
        plt.fill_between(time_average, mean - std, mean + std, alpha=fill_alpha, color=colors[ii])

        mean = divergence_dict[region]["non_consensus"][site]["mean"]
        std = divergence_dict[region]["non_consensus"][site]["std"]
        plt.plot(time_average, mean, '--', color=colors[ii])
        plt.fill_between(time_average, mean - std, mean + std, alpha=fill_alpha, color=colors[ii])

    plt.plot([0], [0], "k-", label="consensus")
    plt.plot([0], [0], "k--", label="non_consensus")
    plt.plot([0], [0], "-", label="first", color=colors[0])
    plt.plot([0], [0], "-", label="second", color=colors[1])
    plt.plot([0], [0], "-", label="third", color=colors[2])
    plt.grid()
    plt.xlabel("Time since infection [years]", fontsize=fontsize)
    plt.ylabel("Divergence", fontsize=fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("Divergence_sites.png", format="png")
    plt.show()


def divergence_histogram(figsize=(14, 10), fontsize=20, tick_fontsize=18,
                         colors=["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
                         fill_alpha=0.15):

    values, cumulative = get_divergence_cumulative_sum()
    bins = np.linspace(0,values[-1],9)
    idxs = [np.where(values>=bin)[0][0] for bin in bins]

    hist = []
    for ii in range(len(bins)-1):
        hist += [cumulative[idxs[ii+1]]-cumulative[idxs[ii]]]

    middle_bins = 0.5*(bins[1:] + bins[:-1])

    plt.figure(figsize=figsize)
    # Cumulative distribution
    plt.plot(values, cumulative, color="C1", label="Cumulative distribution")
    # Histogram
    plt.bar(middle_bins, hist, color="C0", width=0.95*middle_bins[0]*2, label="Distribution")
    plt.xlabel("Divergence values", fontsize=fontsize)
    plt.ylabel("Contribution to divergence", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.xlim([-0.01, 1.01])
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.grid()
    plt.tight_layout()
    plt.savefig("Divergence_hist.png", format="png")
    plt.show()




if __name__ == "__main__":
    # mean_in_time_plot()
    # divergence_region_plot()
    # divergence_consensus_plot()
    # divergence_site_plot(colors=["C3", "C8", "C9"])
    divergence_histogram()
