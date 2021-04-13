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



if __name__ == "__main__":
