import numpy as np
import copy
import filenames
from hivevo.patients import Patient
import tools
from hivevo.HIVreference import HIVreference


class Trajectory():
    def __init__(self, frequencies, t, date, t_last_sample, fixation, threshold_low, threshold_high,
                 patient, region, position, nucleotide, synonymous):

        self.frequencies = frequencies              # Numpy 1D vector
        self.t = t                                  # Numpy 1D vector (in days)
        self.date = date                            # Date at t=0 (int, in days)
        # Number of days at which last sample was taken for the patient (relative to t[0] = 0)
        self.t_last_sample = t_last_sample
        self.fixation = fixation                    # "fixed", "active", "lost" (at the next time point)
        self.threshold_low = threshold_low          # Value of threshold_low used for extraction
        self.threshold_high = threshold_high        # Value of threshold_high used for extraction
        self.patient = patient                      # Patient name (string)
        self.region = region                        # Region name, string
        self.position = position                    # Position on the region (int)
        self.nucleotide = nucleotide                # Nucleotide number according to HIVEVO_access/hivevo/sequence alpha
        self.synonymous = synonymous                # True if this trajectory is part of synonymous mutation

    def __repr__(self):
        return str(self.__dict__)


def create_trajectory_list(patient, region, aft, threshold_low=0.01, threshold_high=0.99, syn_constrained=False):
    """
    Creates a list of trajectories from a patient allele frequency trajectory (aft).
    Select the maximal amount of trajectories:
        - trajectories are extinct before the first time point
        - trajectories are either active, extinct or fixed after the last time point, which is specified
        - a trajectory can be as small as 1 point (extinct->active->fixed, or extinct->active->exctinct)
        - several trajectories can come from a single aft (for ex. extinct->active->extinct->active->fixed)
        - masked datapoints (low depth / coverage) are included only if in the middle of a trajectory (ie. [0.2, --, 0.6] is kept, but [--, 0.2, 0] gives [0.2] and [0.5, --, 1] gives [0.5])
    """
    trajectories = []
    # Adding masking for low depth fragments
    depth = get_depth(patient, region)
    depth = np.tile(depth, (6, 1, 1))
    depth = np.swapaxes(depth, 0, 1)
    aft.mask = np.logical_or(aft.mask, ~depth)

    # Filter the aft to remove positions where there is no reference or seen to often gapped TODO
    aft = aft[:,:, get_reference_filter(patient, region, aft)]

    # Exctract the full time series of af for mutations and place them in a 2D matrix as columns
    mutation_positions = tools.get_mutation_positions(patient, region, aft, threshold_low)
    region_mut_pos = np.sum(mutation_positions, axis=0, dtype=bool)
    mask = np.tile(region_mut_pos, (aft.shape[0], 1, 1))
    mut_frequencies = aft[mask]
    mut_frequencies = np.reshape(mut_frequencies, (aft.shape[0], -1))  # each column is a different mutation

    # Map the original position and nucleotide
    i_idx, j_idx = np.meshgrid(range(mutation_positions.shape[1]), range(
        mutation_positions.shape[2]), indexing="ij")
    coordinates = np.array([i_idx, j_idx])
    coordinates = np.tile(coordinates, (aft.shape[0], 1, 1, 1))
    coordinates = np.swapaxes(coordinates, 1, 3)
    coordinates = np.swapaxes(coordinates, 1, 2)
    coordinates = coordinates[mask]
    coordinates = np.reshape(coordinates, (aft.shape[0], -1, 2))
    # coordinates[t,ii,:] gives the [nucleotide, genome_position] of the mut_frequencies[t,ii] for any t

    # Removing "mutation" at first time point because we don't know their history, ie rising or falling
    mask2 = np.where(mut_frequencies[0, :] > threshold_low)
    mut_freq_mask = mut_frequencies.mask  # keep the mask aside as np.delete removes it
    mut_freq_mask = np.delete(mut_freq_mask, mask2, axis=1)
    mut_frequencies = np.array(np.delete(mut_frequencies, mask2, axis=1))
    coordinates = np.delete(coordinates, mask2, axis=1)

    filter1 = mut_frequencies > threshold_low
    filter2 = mut_frequencies < threshold_high

    # true for the rest of time points once it hits fixation
    filter_fixation = np.cumsum(~filter2, axis=0, dtype=bool)
    trajectory_filter = np.logical_and(~filter_fixation, filter1)
    new_trajectory_filter = np.logical_and(~trajectory_filter[:-1, :], trajectory_filter[1:, :])
    new_trajectory_filter = np.insert(new_trajectory_filter, 0, trajectory_filter[0, :], axis=0)
    trajectory_stop_filter = np.logical_and(trajectory_filter[:-1, :], ~trajectory_filter[1:, :])
    trajectory_stop_filter = np.insert(trajectory_stop_filter, 0, np.zeros(
        trajectory_stop_filter.shape[1], dtype=bool), axis=0)

    # Include the masked points in middle of trajectories (ex [0, 0.2, 0.6, --, 0.8, 1])
    stop_at_masked_filter = np.logical_and(trajectory_stop_filter, mut_freq_mask)
    stop_at_masked_shifted = np.roll(stop_at_masked_filter, 1, axis=0)
    stop_at_masked_shifted[0, :] = False
    stop_at_masked_restart = np.logical_and(stop_at_masked_shifted, new_trajectory_filter)

    new_trajectory_filter[stop_at_masked_restart] = False
    trajectory_stop_filter[np.roll(stop_at_masked_restart, -1, 0)] = False

    syn_mutations = patient.get_syn_mutations(region, mask_constrained=syn_constrained)

    date = patient.dsi[0]
    time = patient.dsi - date
    # iterate though all columns (<=> mutations trajectories)
    for ii in range(mut_frequencies.shape[1]):
        # iterate for all trajectories inside this column
        for jj, idx_start in enumerate(np.where(new_trajectory_filter[:, ii] == True)[0]):

            if not True in (trajectory_stop_filter[idx_start:, ii] == True):  # still active
                idx_end = None
            else:
                idx_end = np.where(trajectory_stop_filter[:, ii] == True)[0][jj]  # fixed or lost

            if idx_end == None:
                freqs = np.ma.array(mut_frequencies[idx_start:, ii])
                freqs.mask = mut_freq_mask[idx_start:, ii]
                t = time[idx_start:]
            else:
                freqs = np.ma.array(mut_frequencies[idx_start:idx_end, ii])
                freqs.mask = mut_freq_mask[idx_start:idx_end, ii]
                t = time[idx_start:idx_end]

            if idx_end == None:
                fixation = "active"
            elif filter_fixation[idx_end, ii] == True:
                fixation = "fixed"
            else:
                fixation = "lost"

            position = coordinates[0, ii, 1]
            nucleotide = coordinates[0, ii, 0]
            traj = Trajectory(np.ma.array(freqs), t - t[0], date + t[0], time[-1] - t[0], fixation, threshold_low, threshold_high, patient.name, region,
                              position=position, nucleotide=nucleotide, synonymous=syn_mutations[nucleotide, position])
            trajectories = trajectories + [traj]
    return trajectories


def filter(trajectory_list, filter_str):
    """
    Evaluate filter_str on all elements (traj) of the trajectory_list and returned the filtered list.
    Return a deepcopy of the elements. Change to a reference instead of copy for performance optimisation.
    It is better to use list comprehension directly ex. : filtered_traj = [x for x in trajectories if np.sum(x.frequencies > freq_min, dtype=bool)]
    """
    return [traj for traj in trajectory_list if eval(filter_str)]


def create_all_patient_trajectories(region, patient_names=[]):
    if patient_names == []:
        patient_names = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p11"]

    trajectories = []
    for patient_name in patient_names:
        patient = Patient.load(patient_name)
        aft = patient.get_allele_frequency_trajectories(region)
        trajectories = trajectories + create_trajectory_list(patient, region, aft)

    return trajectories


def get_fragment_per_site(patient, region):
    """
    Returns a list of fragment associated to each position in the region.
    """
    fragment_list = [[]] * len(patient._region_to_indices(region))
    frag = patient._annotation_to_fragment_indices(region)
    fragment_names = [*frag][2:]

    for ii in range(len(fragment_list)):
        for frag_name in fragment_names:
            if ii in frag[frag_name][0]:
                fragment_list[ii] = fragment_list[ii] + [frag_name]

    return fragment_list, fragment_names


def get_fragment_depth(patient, fragment):
    "Returns the depth of the fragment for each time point."
    return [s[fragment] for s in patient.samples]


def associate_depth(fragments, fragment_depths, fragment_names):
    "Associate a bolean array (true where coverage is ok) to each positions of the region."
    bool_frag_depths = np.array(fragment_depths) == "ok"
    depths = []
    for ii in range(len(fragments)):
        if len(fragments[ii]) == 1:  # Site only belongs to 1 fragment
            depths += [bool_frag_depths[np.where(np.array(fragment_names) == fragments[ii][0])[0][0]]]
        elif len(fragments[ii]) == 2:  # Site belongs to 2 fragments => take the best coverage
            depth1 = bool_frag_depths[np.where(np.array(fragment_names) == fragments[ii][0])[0][0]]
            depth2 = bool_frag_depths[np.where(np.array(fragment_names) == fragments[ii][1])[0][0]]
            depths += [np.logical_or(depth1, depth2)]
        else:
            raise(ValueError("Number of fragments for each site must be either 1 or 2."))

    return np.swapaxes(np.array(depths), 0, 1)


def get_depth(patient, region):
    """
    Returns nb_timepoint*nb_site boolean matrix where True are samples where the depth was labeled "ok" in the tsv files.
    """
    fragments, fragment_names = get_fragment_per_site(patient, region)
    fragment_depths = [get_fragment_depth(patient, frag) for frag in fragment_names]
    return associate_depth(fragments, fragment_depths, fragment_names)


def get_reference_filter(patient, region, aft, gap_treshold=0.1):
    """
    Returns a 1D boolean vector where False are the positions (in aft.shape[-1]) that are unmapped to reference or too often gapped.
    """
    ref = HIVreference(subtype="any")
    map_to_ref = patient.map_to_external_reference(region)
    ungapped_genomewide = ref.get_ungapped(gap_treshold)
    ungapped_region = ungapped_genomewide[map_to_ref[:, 0]]

    # excludes the positions that are not mapped to the reference (i.e. insertions as alignement is unreliable)
    mask1 = np.in1d(np.arange(aft.shape[-1]), map_to_ref[:, 2])

    # excludes positions that are often gapped in the reference (i.e. where the alignement is unreliable)
    mask2 = np.in1d(np.arange(aft.shape[-1]), map_to_ref[ungapped_region,2])

    return np.logical_and(mask1, mask2)


if __name__ == "__main__":
    region = "pol"
    patient = Patient.load("p1")
    aft = patient.get_allele_frequency_trajectories(region)
    aft_mask = get_reference_filter(patient, region, aft)
    trajectories = create_trajectory_list(patient, region, aft)
