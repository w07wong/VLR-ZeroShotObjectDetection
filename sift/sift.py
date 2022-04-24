import numpy as np
import cv2
from matplotlib import pyplot as plt

# https://stackoverflow.com/questions/51606215/how-to-draw-bounding-box-on-best-matches

def crop(image):
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

def get_sift_bb(scene_img, target_img, gt_bb, save_viz=False, viz_fname=''):
    # Crop background from target_img
    target_img = crop(target_img)

    # Initiate SIFT detector
    orb = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with ORB
    target_keypoints, target_descriptors = orb.detectAndCompute(target_img,None)
    scene_keypoints, scene_descriptors = orb.detectAndCompute(scene_img,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    # Match descriptors.
    matches = bf.match(target_descriptors, scene_descriptors)

    # Sort them in the order of their distance.
    # matches = [m for m in macthes if m.distance < ]
    matches = sorted(matches, key = lambda x:x.distance)

    # Set threshold based on distance
    good_matches = matches[:50]
    # print([m.distance for m in sorted(good_matches, key=lambda x:x.distance)])

    if len(good_matches) >= 10:
        src_pts = np.float32([target_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([scene_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1 , 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC ,5.0)
        matchesMask = mask.ravel().tolist()
        h, w = target_img.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(pts, M)

        dst = dst.squeeze()
        # x_min = int(min(dst[0][0], dst[1][0]))
        # x_max = int(max(dst[2][0], dst[3][0]))
        # y_min = int(min(dst[0][1], dst[1][1]))
        # y_max = int(max(dst[2][1], dst[2][1]))
        # pred_bb = [x_min, y_min, x_max, y_max]
        # print(pred_bb)
        x_min = max(0, int(min(dst[0][0], dst[1][0])))
        x_max = min(scene_img.shape[1], int(max(dst[2][0], dst[3][0])))
        y_min = max(0, int(min(dst[0][1], dst[1][1])))
        y_max = min(scene_img.shape[0], int(max(dst[2][1], dst[2][1])))
        pred_bb = [x_min, y_min, x_max, y_max]
        if save_viz:
            dst += (w, 0)  # adding offset
            draw_params = dict(matchColor = (0, 255, 0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags=2)

            img3 = cv2.drawMatches(target_img, target_keypoints, scene_img, scene_keypoints, good_matches, None, **draw_params)

            # Draw bounding box in red
            # img3 = cv2.polylines(img3, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)
            dst = dst.squeeze()
            x_min = max(w, int(min(dst[0][0], dst[1][0])))
            x_max = min(img3.shape[1], int(max(dst[2][0], dst[3][0])))
            y_min = max(0, int(min(dst[0][1], dst[1][1])))
            y_max = min(img3.shape[0], int(max(dst[2][1], dst[2][1])))
            img3 = cv2.rectangle(img3, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

            # Draw ground truth bounding box in blue
            img3 = cv2.rectangle(img3, (int(gt_bb[0] + w), int(gt_bb[1])), (int(gt_bb[2] + w), int(gt_bb[3])), (255, 0, 0), 2)

            # plt.figure(figsize=(10,10))
            # plt.imsave(viz_fname, img3)
            # plt.close()
            cv2.imwrite(viz_fname, img3)

        return pred_bb
    else:
        return [0, 0, scene_img.shape[0], scene_img.shape[1]]