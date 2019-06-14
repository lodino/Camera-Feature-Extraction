import cv2
import numpy as np
import matplotlib.pyplot as plt


LG_SIZE = (4032, 3024)
MOTOX_SIZE = (3120, 4160)
MOTOD_SIZE = (2432, 4320)
MOTON_SIZE = (4130, 3088)
SONY_SIZE = (4000, 6000)
HTC_SIZE = (2688, 1520)
SGN_SIZE = (2322, 4128)
SGS_SIZE = (2322, 4128)
IP4_SIZE = (2448, 3264)
IP6_SIZE = (2448, 3264)


size_dict = {'iP6': IP6_SIZE, 'iP4s': IP4_SIZE, 'GalaxyS4': SGS_SIZE, 'GalaxyN3': SGN_SIZE, 'MotoNex6': MOTOX_SIZE,
             'MotoMax': MOTOD_SIZE, 'MotoX': MOTOX_SIZE, 'HTC-1-M7': HTC_SIZE, 'Nex7': SONY_SIZE, 'LG5x': LG_SIZE}


def get_fingerprint(imgs, cam):
    numerator = 0
    dominator = 0
    for img_path in imgs:
        '''
            Given that there are problems with the shape when using cv2.imread, use plt.imread instead, 
            which return a ndarray as well. Also notice that cv2 uses BGR, but plt here uses RGB.
            Considering we treat each channel equally, so I just did not modify other parts of code accordingly.
        '''
        img = plt.imread(img_path)
        if img.shape[:2] != size_dict[cam]:
            img = np.transpose(img, (1, 0, 2))
        denoised = cv2.GaussianBlur(img, (3, 3), 0)  # denoise the input image
        w = img - denoised
        # Consider some entries are 0 (divide-by-zero err)
        numerator += w * img
        numerator[np.isnan(numerator)] = 0
        numerator[np.isinf(numerator)] = 0
        dominator += img ** 2
        dominator[np.isnan(dominator)] = 0
        numerator[np.isinf(dominator)] = 0

    return numerator / dominator
