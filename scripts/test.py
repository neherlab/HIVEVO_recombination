# Adds link to the scripts folder
import filenames
from hivevo.HIVreference import HIVreference
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from hivevo.patients import Patient
from divergence import get_non_consensus_mask, get_consensus_mask
import trajectory
sys.path.append("../scripts/")


if __name__ == "__main__":
