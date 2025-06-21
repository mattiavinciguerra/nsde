import numpy as np # type: ignore
from IVT_utils import dataset_helpers as helpers
from IVT_utils.trajectory_split import ivt
from IVT_utils.features import calculate_velocity

def classify_raw_IVT(xy, sample_rate, screen_params, vel_threshold=50, min_fix_duration=.1):
    """ Generate feature vectors for all saccades and fixations in a trajectory

    :param xy: ndarray
        2D array of gaze points (x,y)
    :param label: any
         single class this trajectory belongs to
    :return: list, list
        list of feature vectors (each is a 1D ndarray) and list of classes(these will all be the same)
    """

    angle = helpers.convert_pixel_coordinates_to_angles(xy, *screen_params.values())
    smoothed_angle = np.asarray(helpers.savgol_filter_trajectory(angle)).T
    smoothed_vel_xy = calculate_velocity(smoothed_angle, sampleRate=sample_rate)
    smoothed_vel = np.linalg.norm(smoothed_vel_xy, axis=1)
    smoothed_pixels = helpers.convert_angles_to_pixel_coordinates(smoothed_angle, *screen_params.values())

    sacs, fixs = ivt(smoothed_vel, vel_threshold=vel_threshold, min_fix_duration=min_fix_duration, sampleRate=sample_rate)
    # sacs, fixs: list of couples of indices of start and end of saccades and fixations in the input data structure

    fixations = []
    for fix in fixs:
        fixxy = xy[fix[0]:fix[1]] # List of coordinates of the fixation
        if len(fixxy) < 1:
            continue
        fixxy = np.array(fixxy)
        mean = np.mean(fixxy, axis=0)
        fixxy -= mean
        fixations.append(fixxy)
        durs = (len(fixxy)/sample_rate)*1000

    saccades = []
    for sac in sacs:
        sacxy = xy[sac[0]:sac[1]] # List of coordinates of the saccade
        if len(sacxy) < 1:
            continue
        sacxy = np.array(sacxy)
        mean = np.mean(sacxy, axis=0)
        sacxy -= mean
        saccades.append(sacxy)
        durs = (len(sacxy)/sample_rate)*1000

    return fixations, saccades