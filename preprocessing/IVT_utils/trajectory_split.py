import numpy as np


def ivt(velocities, vel_threshold, min_fix_duration, sampleRate):
    """ Extract saccades and fixations from a list of velocities

    :param velocities:  ndarray
        1D array of velocities
    :param vel_threshold: float
        minimum velocity for a saccade
    :param min_fix_duration:
        fixations shorter than this threshold will be added to the sorrounding saccade
    :param sampleRate: float
        Sensor sample rate in Hz
    :return: list of tuples, list of tuples
        List of saccades and list of fixations. Each List contains tuples that contain start and
        end frame.
    """
    assert len(velocities) > 0, "IVT needs at least one element"

    # internally we work with frames, so convert to frame number
    mdf_frames = min_fix_duration * sampleRate

    # mark every possible fixation frame with true
    fixx = velocities < vel_threshold
    # find all changes from saccade to fixation and the other way around
    # these changes mark the end of each sub sequence
    diffs = fixx[:-1] != fixx[1:]

    # The first element is always a start and the last an end.
    # This expands the length of the marker arrays to the input size again
    start_marks = np.insert(diffs, 0, True)
    end_marks = np.append(diffs, True)

    # get the indices for each sub-region
    starts = np.where(start_marks)[0]
    ends = np.where(end_marks)[0]

    # for each sub-region, is it a fixation?
    is_fix = fixx[starts]
    # filter fixations that are too short and combine them with the surrounding saccades
    frame_durations = (ends - starts) + 1
    # all groups we should delete
    rem_ids = np.where((frame_durations < mdf_frames) & is_fix)[0]

    # mark removed fixations as saccades
    fixx[starts[rem_ids]] = False

    # remove short fixations start and end and further:
    # 1. for every short fixation that is not first, also remove the previous saccade's end
    rem_ids_end = np.concatenate([rem_ids, rem_ids - 1])
    ends = np.delete(ends, rem_ids_end[(rem_ids_end < len(ends) - 1) & (rem_ids_end >= 0)])
    # 2. for every short fixation that is not last, also remove the successor saccade's start
    rem_ids_start = np.concatenate([rem_ids, rem_ids + 1])
    starts = np.delete(starts, rem_ids_start[(rem_ids_start <= len(starts) - 1) & (rem_ids_start > 0)])

    # take every second item as it alternates between fixation and saccade
    saccades = zip(starts[::2], ends[::2])
    fixations = zip(starts[1::2], ends[1::2])
    if fixx[starts[0]]:
        # if the xy sequence starts with a fixation, switch lists
        saccades, fixations = fixations, saccades

    return list(saccades), list(fixations)


def sliding_window(list_len, window_len, step_size, sample_rate):
    # TODO: add sample rate?
    return zip(range(0, list_len, step_size), range(window_len - 1, list_len, step_size))
