import numpy as np


def event_hotpixel_removal(ev, size, threshold=100):
    """
    Exclude all possible hotpixel through event distribution histogram
    :param ev: the event data
    :param size: the DVS sensor size
    :param threshold: find pixels whose distribution gradient is higher than the threshold
    :return: the event data
    """
    _bins, _range = size, [[0, size[0]], [0, size[1]]]
    hist = np.histogram2d(ev[:, 1], ev[:, 2], bins=_bins, range=_range)
    
    idx = 0
    
    return ev[idx]


def event_polarity_removal(ev, size):
    """

    :return:
    """
    ev[:, 3] = 0
    return ev
