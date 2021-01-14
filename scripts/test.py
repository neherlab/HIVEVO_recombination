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

def test():
    d = {}
    for idx1 in ["a", "b", "c"]:
        d[idx1] = {}
        for idx2 in ["A", "B", "C"]:
            d[idx1][idx2] = {}
            for idx3 in ["d", "e", "f"]:
                d[idx1][idx2][idx3] = {}
                for idx4 in ["D", "E", "F"]:
                    d[idx1][idx2][idx3][idx4] = np.random.rand(10)
                    d[idx1][idx2][idx3][idx4] += 2
    return d



if name == "__main__":
    test()

# x = np.array([[[0.,  1.,  2.,  3.],
#                [4.,  5.,  6.,  7.],
#                [8.,  9., 10., 11.],
#                [12., 13., 14., 15.],
#                [16., 17., 18., 19.]],
#               [[0.5,  1.5,  2.5,  3.5],
#                [4.5,  5.5,  6.5,  7.5],
#                [8.5,  9.5, 10.5, 11.5],
#                [12.5, 13.5, 14.5, 15.5],
#                [16.5, 17.5, 18.5, 19.5]]])
#
# idx1 = np.array([0, 1])
# idx2 = np.array([0, 1, 2, 3, 4])
# idx3 = np.array([0, 2, 2, 0, 1])
#
# y = x[idx1[:, np.newaxis, np.newaxis], idx2, idx3]
# y = x[idx1[:, np.newaxis, np.newaxis], idx2, idx3[np.newaxis, np.newaxis, :]]
