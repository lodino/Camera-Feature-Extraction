import block_covariance
import cross_correlation
import fingerprint
import linear_pattern_cross_correlation
import load_img
import statistical_moments
import collector
from sklearn.decomposition import PCA
import argparse
from itertools import permutations
import re


def extract_camera_name(path: str) -> str:
    name = re.match(r'/?(\S*/)*(\S*.\S*)', path).group(2)
    camera_name = re.match(r'\((\S*)\)\S*$', name).group(1)
    return camera_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-f', '--format', default='jpg')
    parser.add_argument('-d', '--dir', default='data/')
    args = parser.parse_args()
    properties = {'input_dir': args.dir, 'format': args.format}

    # Load imgaes
    imgs = load_img.load_img_from_dir(properties['input_dir'], properties['format'])
    img_collector = collector.ImageCollector()
    for filename in imgs:
        camera = extract_camera_name(filename)
        img_collector[camera].append(filename)
    cameras = img_collector.imgs.keys()

    feature_collector = collector.FeatureCollector()

    # Get fingerprint
    for camera in cameras:
        imgs = img_collector.imgs[camera]
        fp = [fingerprint.get_fingerprint(imgs)]
        feature_collector.fingerprints[camera] = fp

    for camera in cameras:
        fp = feature_collector.fingerprints[camera]
        # Get statistic normalized central moments of each channel of the fingerprint
        feature_collector.moments[camera] = statistical_moments.get_moments(fp)

        # Get cross-correlation
        cross_correlations = []
        for pair in list(permutations([0, 1, 2], 2)):
            img1 = fp[camera][:, :, pair[0]]
            img2 = fp[camera][:, :, pair[1]]
            for i in range(4):
                for j in range(4):
                    cross_correlations.append(cross_correlation.get_cross_correlation(img1, img2, i, j))

        # Get linear-pattern correlation (just take one channel is ok, here choose red)
        feature_collector.linear_correlations[camera] = linear_pattern_cross_correlation.get_autocorrelation_feature(
            fp[:, :, 2])

        # Get block covariance
        feature_collector.block_covariances_1[camera] = block_covariance.get_block_covariance(fp, 2)
        feature_collector.block_covariances_2[camera] = block_covariance.get_block_covariance(fp, 3)
# TODO: PCA to reduce the dim of features
