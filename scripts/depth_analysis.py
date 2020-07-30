import numpy as np
import matplotlib.pyplot as plt
import filenames
from hivevo.patients import Patient


# patient_names = ["p1"]
patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]
fontsize = 16

for patient_name in patient_names:
    patient = Patient.load(patient_name)
    depth = patient.get_fragment_depth()
    plt.figure()
    plt.title(patient_name, fontsize=fontsize)
    plt.imshow(depth.transpose(), aspect="auto", vmin=0, vmax=100)
    plt.xlabel("Sample", fontsize=fontsize)
    plt.ylabel("Fragment", fontsize=fontsize)
    plt.colorbar()

plt.show()
