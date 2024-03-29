import numpy as np


def crop_and_pad(img, r_offset, c_offset):
    img = img[:img.shape[0] - r_offset, :img.shape[1] - c_offset]
    return np.pad(img, ((0, r_offset), (c_offset, 0)), 'constant', constant_values=0)


def get_cross_correlation(img1, img2, r_offset, c_offset):
    if (r_offset != 0) or (c_offset != 0):
        img2 = crop_and_pad(img2, r_offset, c_offset)
    cc = np.sum(img1 * img2)
    return cc
