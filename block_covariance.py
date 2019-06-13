import numpy as np
import random


def get_block_covariance(img, k):
    vec = []
    size = img.shape[:2]
    num_vecs = [size[0] // k, size[1] // k]
    r_list = random.sample(list(range(num_vecs[0])), 3 * k ** 2)
    c_list = random.sample(list(range(num_vecs[1])), 3 * k ** 2)
    for row in r_list:
        for col in c_list:
            vec.append(np.ravel(img[row*k:(row+1)*k, col*k:(col+1)*k, :]))
    cov_mat = np.ravel(np.cov(vec))
    return np.ravel(cov_mat)
