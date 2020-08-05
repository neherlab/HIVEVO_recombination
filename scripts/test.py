import numpy as np
import matplotlib.pyplot as plt
import filenames
from hivevo.patients import Patient


patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]
sum = 0
sum2 = 0

for patient_name in patient_names:
    patient = Patient.load(patient_name)
    depth = patient.get_fragment_depth()
    depth.mask = np.logical_or(depth.mask, depth < 10)
    print(patient_name)
    sum += np.sum(depth.mask)
    sum2 += np.sum(~depth.mask)
print(sum)
print(sum2)

test
