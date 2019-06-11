import cv2
import matplotlib.pyplot as plt


def get_fingerprint(imgs):
    numerator = 0
    dominator = 0
    for img_path in imgs:
        '''
            Given that there are problems with the shape when using cv2.imread, use plt.imread instead, 
            which return a ndarray as well. Also notice that cv2 uses BGR, but plt here uses RGB.
            Considering we treat each channel equally, so I just did not modify other parts of code accordingly.
        '''
        img = plt.imread(img_path)
        denoised = cv2.GaussianBlur(img, (3, 3), 0)  # denoise the input image
        w = img - denoised
        numerator += w * img
        dominator += img ** 2
    return numerator / dominator
