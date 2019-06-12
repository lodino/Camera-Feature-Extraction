import numpy as np


def get_element(img, coordinate):
    x = coordinate[0]
    y = coordinate[1]
    size = img.shape[:2]
    if (x >= size[0]) or (y >= size[1]) or (x < 0) or (y < 0):
        return 0
    else:
        return img[x, y]


def crop_and_pad(img, r_offset, c_offset):
    img = img[:img.shape[0] - r_offset, :img.shape[1] - c_offset]
    return np.pad(img, ((0, r_offset), (c_offset, 0)), 'constant', constant_values=0)


def get_cross_correlation(img1, img2, r_offset, c_offset):
    # for i in range(size[0]):
    #     for j in range(size[1]):
    #         cross_correlation += get_element(img1, (i, j)) * get_element(img2, (i - r_offset, j - c_offset))
    img2 = crop_and_pad(img2, r_offset, c_offset)
    return np.correlate(np.ravel(img1), np.ravel(img2))[0]
