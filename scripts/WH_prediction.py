# Adds link to the scripts folder
import filenames
from hivevo.HIVreference import HIVreference
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import math
import sys
import os
import random
from numba import jit
from hivevo.patients import Patient
from divergence import load_divergence_dict, WH_evo_rate
sys.path.append("../scripts/")


@jit(nopython=True)  # makes it ~10 times faster
def simulation_step(x, dt, rate_rev, rate_non_rev):
    "Returns the boolean vector x(t+dt) from x(t). Time unit is day, rates are per day and per nucleotide."
    nb_consensus = len(x[x])
    nb_non_consensus = len(x) - nb_consensus
    nb_rev = np.random.poisson(nb_non_consensus * rate_rev * dt)
    nb_non_rev = np.random.poisson(nb_consensus * rate_non_rev * dt)

    idxs_consensus = np.where(x)[0]
    idxs_non_consensus = np.where(~x)[0]

    mut_rev = np.random.choice(idxs_non_consensus, nb_rev)
    mut_non_rev = np.random.choice(idxs_consensus, nb_non_rev)

    idxs_mutation = np.concatenate((mut_rev, mut_non_rev))
    x[idxs_mutation] = ~x[idxs_mutation]
    return x


def initialize_seq(sequence_length, freq_non_consensus):
    "Returns a boolean vector specified length where values are False with the specified frequency (chosen at random)"
    x_0 = np.ones(sequence_length, dtype=bool)
    nb_non_consensus = round(sequence_length * freq_non_consensus)
    idxs = np.random.choice(sequence_length, nb_non_consensus)
    x_0[idxs] = False
    return x_0


def initialize_fixed_point(sequence_length, rate_rev, rate_non_rev):
    "Return initialize_seq with frequence accroding to the fixed point."
    freq_non_consensus = rate_non_rev / (rate_rev + rate_non_rev)
    return initialize_seq(sequence_length, freq_non_consensus)


def run_simulation(x, simulation_time, dt, rate_rev, rate_non_rev, sampling_time):
    """
    Runs a simulation and stores the sampled sequences the matrix sequences (nb_nucleotide * nb_sequences).
    x is modified during the simulation. The original sequence is included in the sequences matrix, in the first row.
    """
    ii = 0
    time = np.arange(0, simulation_time + 1, dt)
    nb_samples = simulation_time // sampling_time
    sequences = np.zeros(shape=(len(x), nb_samples + 1), dtype=bool)

    for t in time:
        if (t % sampling_time == 0):
            sequences[:, ii] = x
            ii += 1

        x = simulation_step(x, dt, rate_rev, rate_non_rev)
    return sequences


def run_simulation_group(x_0, simulation_time, dt, rate_rev, rate_non_rev, sampling_time, nb_sim):
    """
    Runs several simulation starting from the same x_0, and returns a 3D matrix containing the sequences
    (nb_nucleotide * nb_sequences * nb_simulation)
    """
    nb_samples = simulation_time // sampling_time
    sequences = np.zeros(shape=(len(x_0), nb_samples + 1, nb_sim), dtype=bool)

    for ii in range(nb_sim):
        x = np.copy(x_0)
        sim_matrix = run_simulation(x, simulation_time, dt, rate_rev, rate_non_rev, sampling_time)
        sequences[:, :, ii] = sim_matrix

    return sequences


@jit(nopython=True)
def hamming_distance(a, b):
    """
    Returns the hamming distance between sequence a and b. Sequences must be 1D and have the same length.
    """
    return np.count_nonzero(a != b)


def distance_to_initial(sequences):
    """
    Returns a 2D matrix (timepoint*nb_sim) of hamming distance to the initial sequence.
    """
    result = np.zeros((sequences.shape[1], sequences.shape[2]))
    for ii in range(sequences.shape[1]):
        for jj in range(sequences.shape[2]):
            result[ii, jj] = hamming_distance(sequences[:, 0, jj], sequences[:, ii, jj])
    return result


def distance_to_pairs(sequences):
    """
    Returns a 2D matrix (timepoint*nb_pair_combination) of distance between sequences at each time point.
    """
    result = np.zeros((sequences.shape[1], math.comb(sequences.shape[2], 2)))
    for ii in range(sequences.shape[1]):
        counter = 0
        for jj in range(sequences.shape[2]):
            for kk in range(jj + 1, sequences.shape[2]):
                result[ii, counter] = hamming_distance(sequences[:, ii, jj], sequences[:, ii, kk])
                counter += 1
    return result


if __name__ == '__main__':
    evo_rates = {
        "pol": {"consensus": {"first": 1.98e-6, "second": 1.18e-6, "third": 5.96e-6},
                "non_consensus": {"first": 2.88e-5, "second": 4.549e-5, "third": 2.06e-5}
                }
    }

    # These are per nucleotide per year, need to change it for per day to match the simulation
    BH_rates = {"all": 0.0009372268087945193, "first": 0.0006754649449205438,
                "second": 0.000407792658976286, "third": 0.0017656284793794623}

    nb_simulation = 10
    simulation_time = 36500  # in days
    dt = 10
    time = np.arange(0, simulation_time + 1, dt)
    sampling_time = 10 * dt
    sequence_length = 2500

    # True is consensus, False is non consensus
    x_0 = np.ones(sequence_length, dtype=bool)
    sequences = run_simulation_group(x_0, simulation_time, dt, rate_rev,
                                     rate_non_rev, sampling_time, nb_simulation)
    distance_initial = distance_to_initial(sequences)
    mean_distance_initial = np.mean(distance_initial, axis=-1)
    distance_pairs = distance_to_pairs(sequences)
    mean_distance_pairs = np.mean(distance_pairs, axis=-1)

    x = time[::10]
    saturation = 2 * rate_rev * rate_non_rev * len(x_0) / (rate_rev + rate_non_rev)**2
    tau = 1 / (rate_rev + rate_non_rev)
    theory = saturation * (1 - np.exp(-time / tau))

    plt.figure()
    plt.plot(x, mean_distance_initial, label="Mean distance to initial")
    plt.plot(time, theory, "k--", label="x")
    plt.xlabel("Time [years]")
    plt.ylabel("Hamming distance")
    plt.legend()
    # plt.xscale("log")
    # plt.yscale("log")
    plt.grid()
    plt.show()
