# Adds link to the scripts folder
import filenames
from hivevo.HIVreference import HIVreference
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from hivevo.patients import Patient
from divergence import load_divergence_dict, WH_evo_rate
sys.path.append("../scripts/")

def run_simulation(simulation_time, x_0, dt, rate_non_rev, rate_rev):
    x = np.zeros(simulation_time)
    x[0] = x_0
    times = np.arange(0, simulation_time, dt)
    for ii in range(len(times)-1):
        x[ii+1] =


evo_rates = {
    "env": {"rev": 4.359e-5, "non_rev": 6.734e-6},
    "pol": {"rev": 2.500e-5, "non_rev": 2.946e-6},
    "gag": {"rev": 3.562e-5, "non_rev": 3.739e-6}
}

simulation_time = 3650 # in days
