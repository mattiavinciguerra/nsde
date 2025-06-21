import numpy as np
from scipy.signal import savgol_filter


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.
       From: https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    assert len(y.shape) == 1
    return np.isnan(y), lambda z: z.nonzero()[0]


def convert_angles_to_pixel_coordinates(ab, pix_per_mm, screen_dist_mm, screen_res):
    """Converts trajectory measured in viewing angles to a trajectory in screen pixels.
    Angles are measured from the center viewpoint in horizontal and vertical direction.

    :param ab: ndarray
        2D array of gaze points defined through angles (a,b) horizontal and vertical
    :param pix_per_mm: ndarray
        screen density in x and y
    :param screen_dist_mm: float
        distance between subject and screen
    :param screen_res: ndarray
        screen resolution in pixels x,y
    :return: ndarray
        trajectory in screen pixel coordinates xy with same shape as ab
    """
    return pix_per_mm * np.tan(np.radians(ab)) * screen_dist_mm + screen_res / 2


def convert_pixel_coordinates_to_angles(xy, pix_per_mm, screen_dist_mm, screen_res):
    """ convert pixel coordinates into angles coordinates

    @See convert_angles_to_pixel_coordinates

    :return: ndarray
        trajectory in angles ab with same shape as xy
    """

    return np.degrees(np.arctan2((xy - screen_res/2), screen_dist_mm * pix_per_mm))


def savgol_filter_trajectory(xy, frame_size = 15, pol_order=6):
    """ filter the data using savitzky_Golay
    input:
    xy: 2D array with equal length filtered: angle_x(Xorg), angle_y(Yorg)
    frame_size used for savitzky_Golay filter
    pol_order polynomial order used for savitzky_Golay filter

    Return:
    filtered x and y vectors

    """

    xf = savgol_filter(xy[:, 0], frame_size, pol_order)
    yf = savgol_filter(xy[:, 1], frame_size, pol_order)

    return xf,yf


def smooth_velocities(xf, yf, sampleRate):
    """ generate smooth velocities
    input:
    filtered x and y vectors
    sampleRate: Sensor sample rate in Hz

    Return:
    1D np.arrays of smoothed velocities
    """

    # calculate x velocity
    diff_x = xf[:-1] - xf[1:]
    vel_x = diff_x * sampleRate

    # calculate y velocity
    diff_y = yf[:-1] - yf[1:]
    vel_y = diff_y * sampleRate

    smoothed_velocities = np.asarray([vel_x, vel_y]).T
    norm_our_velocities = np.linalg.norm(smoothed_velocities, axis=1)

    return norm_our_velocities

def convert_gaze_pixel_into_eye_coordinates(xy_pix, pix_per_mm, screen_dist_mm, screen_res):
    """Converts pixel gaze point to mm 3d point.

    :param xy: ndarray
        2D array of gaze points (x,y)
    :param pix_per_mm: ndarray
        screen density in x and y
    :param screen_dist_mm: float
        distance between subject and screen
    :param screen_res: ndarray
        screen resolution in pixels x,y
    :return: ndarray
        trajectory 3d mm coordinates with shape as xyz
        x = (w/w_pix) x_screen - (w/2)
        y = (h/h_pix) y_screen - (h/2)
        z = screen_dist_mm
    """
    assert xy_pix.shape[1] == 2
    assert xy_pix.shape[0] >= 1  # check if the input has at least 1 row

    screen_size_mm = screen_res / pix_per_mm    # we use get_screen_params function without screen_size_mm in our previous calculations, so we calculate it explicitly.
    xy_mm = (1/pix_per_mm) * (xy_pix) - screen_size_mm / 2
    z_mm = np.full((1, len(xy_mm)), screen_dist_mm)  # create a 1 x n array filled with z coordinate
    xyz_mm = np.concatenate((xy_mm, z_mm.T), axis=1)

    return xyz_mm