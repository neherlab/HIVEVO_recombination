# Adds link to the scripts folder
import filenames
from hivevo.HIVreference import HIVreference
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle
from hivevo.patients import Patient
from divergence import load_divergence_dict
sys.path.append("../scripts/")

patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]
time_average = np.arange(0, 3100, 100)

divergence_dict = load_divergence_dict()
div_vector = {}
for ii, region in enumerate(["env", "pol", "gag"]):
    div_vector[region] = {}
    div_vector[region]["div_all"] = np.array([])
    div_vector[region]["div_rev"] = np.array([])
    div_vector[region]["div_non_rev"] = np.array([])
    for patient_name in patient_names:

        div_vector[region]["div_all"] = np.concatenate(
            (div_vector[region]["div_all"], divergence_dict[region][patient_name]["div_all"][:, :].flatten()))
        div_vector[region]["div_rev"] = np.concatenate(
            (div_vector[region]["div_rev"], divergence_dict[region][patient_name]["div_rev"][:, :].flatten()))
        div_vector[region]["div_non_rev"] = np.concatenate(
            (div_vector[region]["div_non_rev"], divergence_dict[region][patient_name]["div_non_rev"][:, :].flatten()))

div_vector["all"] = {}
for div_type in ["div_all", "div_rev", "div_non_rev"]:
    div_vector["all"][div_type] = np.array([])
    for region in ["env", "pol", "gag"]:
        div_vector["all"][div_type] = np.concatenate((div_vector["all"][div_type], div_vector[region][div_type]))

hist_dict = {}
for ii, region in enumerate(["env", "pol", "gag"]):
    hist_dict[region] = {}
    for div_type in ["div_all", "div_rev", "div_non_rev"]:
        hist_dict[region][div_type], bins = np.histogram(div_vector[region][div_type], bins=100, range=(0,1))

bins = bins[:-1]

colors = ["C0", "C1", "C2"]
linestyle = ["-", "--", ":"]

plt.figure()
for ii, region in enumerate(["env", "pol", "gag"]):
    for jj, div_type in enumerate(["div_all", "div_rev", "div_non_rev"]):
        tmp = hist_dict[region][div_type]*bins / np.sum(hist_dict[region][div_type]*bins)
        tmp = np.cumsum(tmp)
        plt.plot(bins, tmp, linestyle=linestyle[jj], color=colors[ii])

for linestyle, label in zip(linestyle, ["all", "rev", "non_rev"]):
    plt.plot([0], [0], linestyle=linestyle, color="k", label=label)

for color, label in zip(colors, ["env", "pol", "gag"]):
    plt.plot([0], [0], color=color, label=label)

plt.grid()
plt.xlabel("Divergence value")
plt.ylabel("Relative frequency (cumulative)")
# plt.yscale("log")
plt.legend()
plt.show()
