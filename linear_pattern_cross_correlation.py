import numpy as np


# FROM: https://stackoverflow.com/questions/14297012/estimate-autocorrelation-using-python
def estimated_autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x - x.mean()
    r = np.correlate(x, x, mode='full')[-n:]
    assert np.allclose(r, np.array([(x[:n - k] * x[-(n - k):]).sum() for k in range(n)]))
    result = r / (variance * (np.arange(n, 0, -1)))
    return result


# Given one (random) channel of the fingerprint of an image
def get_autocorrelation_feature(img):
    vecs = []
    series = np.sum(img, axis=0)  # expect to have the shape of the height of img
    autocorrelation = estimated_autocorrelation(series)
    vec_len = img.shape[1] // 8
    for i in range(8):
        vec = []
        for j in range(vec_len):
            vec.append(autocorrelation[j*8])
        vecs.append(vec)
    return vecs
