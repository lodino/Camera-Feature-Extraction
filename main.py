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
import numpy as np
import json


def eliminate_nan_inf(arr):
    arr[np.isnan(arr)] = 0
    arr[np.isinf(arr)] = 0
    return arr


def extract_camera_name(path: str) -> str:
    name = re.match(r'/?(\S*/)*(\S*.\S*)', path).group(2)
    camera_name = re.match(r'\((\S*)\)\S*$', name).group(1)
    return camera_name


def extract_features(fp, cc_pca, bc_pca_1, bc_pca_2, lcc_pca, final_pca):
    # Get statistic normalized central moments of each channel of the fingerprint
    moments = statistical_moments.get_moments(fp)

    # Get cross-correlation
    cross_correlations = []
    for pair in list(permutations([0, 1, 2], 2)):
        img1 = fp[:, :, pair[0]]
        img2 = fp[:, :, pair[1]]
        for i in range(4):
            for j in range(4):
                cross_correlations.append(cross_correlation.get_cross_correlation(img1, img2, i, j))
    cc = np.array([cross_correlations])
    eliminate_nan_inf(cc)

    # Get linear-pattern correlation (just take one channel is ok, here choose red)
    linear_correlations = linear_pattern_cross_correlation.get_autocorrelation_feature(fp[:, :, 2])
    lcc = np.array([linear_correlations])
    eliminate_nan_inf(lcc)

    # Get block covariance
    bc1 = block_covariance.get_block_covariance(fp, 2)
    bc1 = eliminate_nan_inf(bc1)
    bc2 = block_covariance.get_block_covariance(fp, 3)
    bc2 = eliminate_nan_inf(bc2)

    all_features = np.concatenate((cc_pca.transform(cc), bc_pca_1.transform([bc1]),
                                   bc_pca_2.transform([bc2]), lcc_pca.transform(lcc), [moments]))
    eliminate_nan_inf(all_features)
    final_features = final_pca.transform(all_features)
    return final_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-f', '--format', default='jpg')
    parser.add_argument('-d', '--dir', default='data/')
    args = parser.parse_args()
    properties = {'input_dir': args.dir, 'format': args.format}

    # Load imgaes
    print('Loading images...')
    imgs = load_img.load_img_from_dir(properties['input_dir'], properties['format'])
    img_collector = collector.ImageCollector()
    for filename in imgs:
        camera = extract_camera_name(filename)
        img_collector[camera].append(filename)
    cameras = list(img_collector.imgs.keys())
    print('Images loaded!')

    feature_collector = collector.FeatureCollector()

    # Get fingerprint
    print('Generating fingerprints...')
    for camera in cameras:
        imgs = img_collector.imgs[camera]
        fp = fingerprint.get_fingerprint(imgs, camera)
        fp = eliminate_nan_inf(fp)
        feature_collector.fingerprints[camera] = fp
    print('Finished!')

    print('Extracting features...')
    for camera in cameras:
        fp = feature_collector.fingerprints[camera]
        # Get statistic normalized central moments of each channel of the fingerprint
        print(f"Start extracting moments of {camera}...")
        feature_collector.moments[camera] = statistical_moments.get_moments(fp)

        # Get cross-correlation
        print(f"Start extracting cross-correlation of {camera}...")
        cross_correlations = []
        for pair in list(permutations([0, 1, 2], 2)):
            img1 = fp[:, :, pair[0]]
            img2 = fp[:, :, pair[1]]
            for i in range(4):
                for j in range(4):
                    cross_correlations.append(cross_correlation.get_cross_correlation(img1, img2, i, j))
        feature_collector.cross_correlations[camera] = np.array(cross_correlations)

        # Get linear-pattern correlation (just take one channel is ok, here choose red)
        print(f"Start extracting linear-pattern correlation of {camera}...")
        feature_collector.linear_correlations[camera] = linear_pattern_cross_correlation.get_autocorrelation_feature(
            fp[:, :, 2])

        # Get block covariance
        print(f"Start extracting block covariance of {camera}...")
        bc1 = block_covariance.get_block_covariance(fp, 2)
        bc1 = eliminate_nan_inf(bc1)
        feature_collector.block_covariances_1[camera] = bc1
        bc2 = block_covariance.get_block_covariance(fp, 3)
        bc2 = eliminate_nan_inf(bc2)
        feature_collector.block_covariances_2[camera] = bc2
        print(f"Finished extraction of {camera}")
    print('Finished!')

    # Feature reduction
    print('Reducing features...')
    cc_pca = PCA(n_components=4, random_state=42)
    bc_pca_1 = PCA(n_components=4, random_state=42)
    bc_pca_2 = PCA(n_components=4, random_state=42)
    lcc_pca = PCA(n_components=4, random_state=42)

    original_cc = np.array([feature_collector.cross_correlations[camera] for camera in cameras])
    original_bc_1 = np.array([feature_collector.block_covariances_1[camera] for camera in cameras])
    original_bc_2 = np.array([feature_collector.block_covariances_2[camera] for camera in cameras])
    original_lcc = np.array([feature_collector.linear_correlations[camera] for camera in cameras])

    original_cc = eliminate_nan_inf(original_cc)
    original_bc_1 = eliminate_nan_inf(original_bc_1)
    original_bc_2 = eliminate_nan_inf(original_bc_2)
    original_lcc = eliminate_nan_inf(original_lcc)

    transformed_cc = cc_pca.fit_transform(original_cc)
    transformed_bc_1 = bc_pca_1.fit_transform(original_bc_1)
    transformed_bc_2 = bc_pca_2.fit_transform(original_bc_2)
    transformed_lcc = lcc_pca.fit_transform(original_lcc)

    all_features = np.concatenate((transformed_cc, transformed_bc_1, transformed_bc_2, transformed_lcc,
                                   np.array([feature_collector.moments[camera] for camera in cameras])), axis=1)
    final_pca = PCA(n_components=5, random_state=42)
    eliminate_nan_inf(all_features)
    final_features = final_pca.fit_transform(all_features)
    features_dict = dict()
    for ind in range(len(cameras)):
        features_dict[cameras[ind]] = final_features[ind]
    print('Finished!')

    print('FINAL FEATURES:')
    print(features_dict)
    # print("Converting single points...")
    # scatters = dict()
    # for camera in cameras:
    #     scatters[camera] = []
    #     imgs = img_collector.imgs[camera]
    #     for img in imgs:
    #         fp = fingerprint.get_fingerprint([img], camera)
    #         scatters[camera].append(extract_features(fp, cc_pca, bc_pca_1, bc_pca_2, lcc_pca, final_pca).tolist())
    # print("Finished!")
    # print("Saving info of scatters...")
    # scatters_txt = json.dumps(scatters)
    # with open('scatters.txt', 'w+') as f:
    #     f.write(scatters_txt)
    # print("Saved to scatters.txt!")
