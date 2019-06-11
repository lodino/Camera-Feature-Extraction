import cv2


def get_moments(img):
    moments_list = []
    for i in range(3):
        m = cv2.moments(img[:, :, i])
        moments_list.append(m['nu20'])
        moments_list.append(m['nu02'])
        moments_list.append(m['nu11'])
    return moments_list
