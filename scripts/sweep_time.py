from trajectory import Trajectory, create_all_patient_trajectories
from hivevo.patients import Patient
import filenames
import matplotlib.pyplot as plt
import numpy as np
import tools
from hivevo.patients import Patient


def get_missed_sweep_mask(aft):
    """
    Returns a mask where true are the position where a fast sweep is seen.
    """
    mask = np.zeros(aft.shape, dtype=bool)
    mask[1:, :, :] = np.logical_and(aft[:-1, :, :] <= 0.01, aft[1:, :, :] >= 0.99)
    mask[1:, :, :] = np.logical_and(mask[1:, :, :], np.logical_and(
        ~aft.mask[:-1, :, :], ~aft.mask[1:, :, :]))  # to avoid masked data to be included
    mask = np.sum(mask, axis=0, dtype=bool)
    mask = np.logical_and(aft[0, :, :] < 0.01, mask)  # Removing things that are not new
    mask = np.tile(mask, (aft.shape[0], 1, 1))
    return mask


def get_missed_sweep(aft):
    missed_sweeps = aft[get_missed_sweep_mask(aft)]
    missed_sweeps = np.reshape(missed_sweeps, (aft.shape[0], -1))
    return missed_sweeps


def initial_traj_under_threshold(region, patient_name):
    patient = Patient.load(patient_name)
    aft = patient.get_allele_frequency_trajectories(region)
    # Masking low depth
    depth = trajectory.get_depth(patient, region)
    depth = np.tile(depth, (6, 1, 1))
    depth = np.swapaxes(depth, 0, 1)
    aft.mask = np.logical_or(aft.mask, ~depth)

    initial_idx = patient.get_initial_indices(region)
    aft_initial = aft[np.arange(aft.shape[0])[:, np.newaxis, np.newaxis],
                      initial_idx, np.arange(aft.shape[-1])]
    aft_initial = aft_initial[:, 0, :]

    threshold_low = 0.05
    threshold_high = 0.95

    mask = aft_initial <= threshold_low
    data = aft_initial[:, np.where(np.sum(mask, axis=0))]
    data = data[:, 0, :]

    plt.figure()
    plt.plot(data)
    plt.show()


def divergence_contribution(region, patient_name):
    patient = Patient.load(patient_name)
    aft = patient.get_allele_frequency_trajectories(region)
    initial_idx = patient.get_initial_indices(region)
    divergence_matrix = divergence.divergence_matrix(patient, region, aft)
    div = divergence_matrix[np.arange(aft.shape[0])[:, np.newaxis, np.newaxis],
                            initial_idx, np.arange(aft.shape[-1])]
    div = div[:, 0, :]
    hist, bins = np.histogram(div[-3:, :], bins=1000)
    bins = bins[:-1]
    hist_sum = np.cumsum(hist)
    hist_sum = hist_sum / np.max(hist_sum)

    plt.figure()
    plt.plot(bins, hist_sum, label="hist")
    plt.legend()
    plt.grid()
    plt.show()


def get_mean_sweep_per_year(region, patient_names=["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"], threshold=0.05):
    nb = 0
    for patient_name in patient_names:
        patient = Patient.load(patient_name)
        aft = patient.get_allele_frequency_trajectories(region)

        initial_idx = patient.get_initial_indices(region)
        aft_initial = aft[np.arange(aft.shape[0])[:, np.newaxis, np.newaxis],
                          initial_idx, np.arange(aft.shape[-1])]
        aft_initial = aft_initial[:, 0, :]

        mask = aft_initial <= threshold
        tmp = np.where(np.sum(mask, axis=0))[0]
        nb += tmp.shape[0] / patient.ysi[-1]
    return nb / len(patient_names)


if __name__ == "__main__":
    #############
    regions = ["env", "pol", "gag"]
    patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]

    for region in regions:
        missed_sweep_tot = 0
        for patient_name in patient_names:
            patient = Patient.load(patient_name)
            aft = patient.get_allele_frequency_trajectories(region)
            missed_sweep = get_missed_sweep(aft)
            missed_sweep_tot += missed_sweep.shape[-1]
            print(f"Patient {patient_name} {region}: {missed_sweep.shape[-1]}")

        trajectories = create_all_patient_trajectories(region)
        print(f"Region {region} missed sweeps : {missed_sweep_tot}")
        print(f"Region {region} total trajectories : {len(trajectories)}")

    # patient = Patient.load("p1")
    # aft = patient.get_allele_frequency_trajectories("env")
    # mask = get_missed_sweep_mask(aft)
    # missed_sweeps = aft[mask]
    # reshaped = np.reshape(missed_sweeps, (aft.shape[0], -1))

    ##############
    region = "pol"
    patient_name = "p1"

    # initial_traj_under_threshold(region, patient_name)
    # divergence_contribution(region, patient_name)
    nb = get_mean_sweep_per_year(region)
