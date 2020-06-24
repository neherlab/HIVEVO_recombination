import os
import sys

# HIVEVO_access repository location
HIVEVO_PATH = "/home/valentin/Desktop/Scicore/druell0000/PhD/HIVEVO_access"
sys.path.append(HIVEVO_PATH)

# HIVEVO data folder
HIVEVO_ROOT_DATA_PATH = "/home/valentin/Desktop/Scicore/GROUP/data/MiSeq_HIV_Karolinska/"
os.environ["HIVEVO_ROOT_DATA_FOLDER"] = HIVEVO_ROOT_DATA_PATH
