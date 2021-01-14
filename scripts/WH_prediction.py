# Adds link to the scripts folder
import filenames
from hivevo.HIVreference import HIVreference
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random
from hivevo.patients import Patient
from divergence import load_divergence_dict, WH_evo_rate
sys.path.append("../scripts/")


def simulation_step(x, dt, rate_rev, rate_non_rev):
    "Returns the boolean vector x(t+dt) from x(t). Time unit is day, rates are per day and per nucleotide."
    nb_consensus = len(x[x == True])
    nb_non_consensus = len(x[x == False])
    nb_rev = random_round(nb_non_consensus * rate_rev * dt)
    nb_non_rev = random_round(nb_consensus * rate_non_rev * dt)

    idxs_consensus = np.arange(len(x))[x == True]
    idxs_non_consensus = np.arange(len(x))[x == False]

    mut_rev = np.random.choice(idxs_non_consensus, nb_rev)
    mut_non_rev = np.random.choice(idxs_consensus, nb_non_rev)

    idxs_mutation = np.concatenate((mut_rev, mut_non_rev))
    x[idxs_mutation] = ~x[idxs_mutation]
    return x


def random_round(number):
    "Stochastic rounding of number. For example, random_round(3.14) will give 3 with prba 0.86, and 4 with proba 0.14."
    floor = int(number)
    remainder = number % 1
    number = floor
    if random.random() < remainder:
        number = floor + 1
    return number


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


evo_rates = {
    "env": {"rev": 4.359e-5, "non_rev": 6.734e-6},
    "pol": {"rev": 2.500e-5, "non_rev": 2.946e-6},
    "gag": {"rev": 3.562e-5, "non_rev": 3.739e-6}
}

patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]
regions = ["env", "pol", "gag"]
simulation_time = 3650  # in days
dt = 1

sequence_length = 2500
rate_rev = evo_rates["env"]["rev"]
rate_non_rev = evo_rates["env"]["non_rev"]

# True is consensus, False is non consensus
x_0 = initialize_fixed_point(sequence_length, rate_rev, rate_non_rev)
x = simulation_step(x_0, dt, rate_rev, rate_non_rev)
