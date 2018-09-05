import cv2
import numpy as np
from imutils import paths
import collections
import argparse
import sys
import time

do_print_results=False

ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query_dir", required=True, help="query dir")
ap.add_argument("-t", "--train_dir", required=True, help="train dir")
ap.add_argument("-o", "--output_file", required=True, help="output file")

args = vars(ap.parse_args())

query_dir = args["query_dir"]
train_dir = args["train_dir"]
output_file = args["output_file"]

print('query_dir: {}'.format(query_dir))
print('train_dir: {}'.format(train_dir))
print('output_file: {}'.format(output_file))


def write_results(output, query_img, score):
    f = open(output, 'a')
    f.write('{},{}\n'.format(score,query_img))
    f.close()

def percentage(part, whole):
    return 100 * float(part) / float(whole)

def print_score(query_img, img_cnt, all_matches, good_matches, homograph_matches, homography_inlier_matches,
                homography_outlier_matches):
    print '\nprint_score all_matches: {} good_matches: {}'.format(len(all_matches[0]), len(good_matches[0]))
    print('homograph_matches: {} inliers: {}'.format(len(homograph_matches), homography_inlier_matches))
    if len(homograph_matches) != 0:
        print 'percent: {}'.format(percentage(homography_inlier_matches, len(all_matches[0])))

def feature_matcher(query_img, train_folder):
    min_match_count = 10

    img1 = cv2.imread(query_img, 0)
    surf = cv2.xfeatures2d.SURF_create(800)
    kp1, des1 = surf.detectAndCompute(img1, None)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    img_cnt = 0
    total_matches = 0
    total_homography = 0
    total_inliers = 0
    total_outliers = 0
    for train_img in paths.list_images(train_folder):
        try:
            img2 = cv2.imread(train_img, 0)
            surf = cv2.xfeatures2d.SURF_create(800)
            kp2, des2 = surf.detectAndCompute(img2, None)
            img_cnt = img_cnt + 1

            matches = bf.knnMatch(des1, des2, k=2)
            total_matches = total_matches + len(matches)

            good = []
            for m, n in matches:

                if m.distance < 0.7 * n.distance:
                    good.append(m)

            if len(good) > min_match_count:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()

                if not M is None and not M.all() == None:
                    total_homography = total_homography + len(matchesMask)
                    res = collections.Counter(matchesMask)

                    total_outliers = total_outliers + res[0]
                    total_inliers = total_inliers + res[1]
                    mask_ratio = percentage(res[1], len(matchesMask))

        except Exception as e:

            pass

    if do_print_results:
        print_score(query_img, img_cnt, total_matches, good, total_homography, total_inliers, total_outliers)
    return img_cnt, total_matches, total_homography, total_inliers, total_outliers

results = []
for query_img in paths.list_images(query_dir):
    start_time = time.time()
    # top_train_img, mask = feature_matcher(query_img, train_dir)
    img_cnt, total_matches, total_homography, total_inliers, total_outliers = feature_matcher(query_img, train_dir)
    score = percentage(total_homography, total_matches)
    score = total_homography
    write_results(output_file, query_img, score)