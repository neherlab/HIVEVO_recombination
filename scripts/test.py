# Adds link to the scripts folder
import filenames
from hivevo.HIVreference import HIVreference
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from hivevo.patients import Patient
import trajectory
from divergence import load_divergence_dict, WH_evo_rate
sys.path.append("../scripts/")


def delta(t, mu_plus, mu_minus):
    prefactor = (2 * mu_plus * mu_minus) / (mu_plus + mu_minus)
    result = prefactor * (t - (1 - np.exp(-(mu_plus + mu_minus) * t)) / (mu_plus + mu_minus))
    return result


def distance_initial(t, mu_plus, mu_minus):
    prefactor = (2 * mu_plus * mu_minus) / (mu_plus + mu_minus)**2
    distance = prefactor * (1 - np.exp(-(mu_plus + mu_minus) * t))
    return distance


def distance_approx(t, mu_plus, mu_minus):
    return (2 * mu_plus * mu_minus) / (mu_plus + mu_minus) * t


def delta_relative(t, mu_plus, mu_minus):
    return (mu_plus + mu_minus) * t / (1 - np.exp(-(mu_plus + mu_minus) * t)) - 1


if __name__ == "__main__":
    t = np.linspace(0.001, 100, 1000)

    # mu_plus = 0.0011096
    # mu_minus = 0.01154495
    # delta = delta(t, mu_plus, mu_minus)
    # distance = distance_initial(t, mu_plus, mu_minus)
    # distance_approx = distance_approx(t, mu_plus, mu_minus)
    # delta_r = delta / distance
    # delta_r2 = delta_relative(t,mu_plus, mu_minus)
    #
    # plt.figure()
    # plt.plot(t, distance, label="Real distance")
    # plt.plot(t, distance_approx, label="Without reversions")
    # plt.plot(t, delta, label="Error")
    # plt.plot(t, delta_r, label="Relative error")
    # plt.plot(t, delta_r2, "k--", label="Relative error analytical")
    # plt.legend()
    # plt.grid()

    evo_rates = {
        "pol": {"consensus": {"first": 1.98e-6, "second": 1.18e-6, "third": 5.96e-6},
                "non_consensus": {"first": 2.88e-5, "second": 4.549e-5, "third": 2.06e-5}}
    }

    plt.figure()
    for nuc in ["first", "second", "third"]:
        plt.plot(t, delta_relative(t, evo_rates["pol"]["consensus"][nuc]
                                   * 365, evo_rates["pol"]["non_consensus"][nuc] * 365), label=nuc+" nucleotides")
        plt.legend()
    plt.xlabel("Time [years]")
    plt.ylabel("Relative error")
    plt.grid()
    plt.show()
