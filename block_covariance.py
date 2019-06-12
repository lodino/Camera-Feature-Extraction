def get_block_covariance(img, k):
    vecs = []
    size = img.shape[:2]
    block_size = [size[0] // k, size[1] // k]
    for x in range(block_size[0]):
        for y in range(block_size[1]):
            # Each of (i, j)th element in the block represents a vec. In regular cases, k=2 or k=3
            vec = []
            for i in range(k):
                for j in range(k):
                    for channel in range(3):
                        vec.append(img[x+i*block_size[0], y+j*block_size[1], channel])
            vecs += vec
    return vecs  # Length should be 3k^2*3k^2
