def get_element(img, coordinate):
    x = coordinate[0]
    y = coordinate[1]
    size = img.shape[:2]
    if (x >= size[0]) or (y >= size[1]) or (x < 0) or (y < 0):
        return 0
    else:
        return img[x, y]


def get_cross_correlation(img1, img2, offsetx, offsety):
    cross_correlation = 0
    size = img1.shape[:2]
    for i in range(size[0]):
        for j in range(size[1]):
            cross_correlation += get_element(img1, (i, j)) * get_element(img2, (i - offsetx, j - offsety))
    return cross_correlation
