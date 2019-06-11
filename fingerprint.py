import cv2


def get_fingerprint(imgs):
    numerator = 0
    dominator = 0
    for img in imgs:
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)  # denoise the input image
        w = img - denoised
        numerator += w * img
        dominator += img ** 2
    return numerator / dominator
