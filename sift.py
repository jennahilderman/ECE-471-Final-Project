import cv2
import numpy as np
import os

duck_directory = "./duck_images/"
test_frame_file = "./scene_images/frame_55.038.png"

def CallSift(current_frame, test_mode=False):
    duck_images = []
    folder = duck_directory
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            duck_images.append(img)

    scene_image = current_frame
    # convert images to grayscale
    duck_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                   for img in duck_images]
    scene_image = cv2.cvtColor(scene_image, cv2.COLOR_BGR2GRAY)
    # create SIFT object
    sift = cv2.SIFT_create()
    # detect SIFT features from duck images
    keypoints_duck = []
    descriptors_duck = []
    result = [sift.detectAndCompute(img, None) for img in duck_images]
    keypoints_duck = [res[0] for res in result]
    descriptors_duck = [res[1] for res in result]

    # detect SIFT features from scene image
    keypoints_scene, descriptors_scene = sift.detectAndCompute(
        scene_image, None)

    # feature match each duck descriptor/keypoint with scene descriptor/keypoint
    for i in range(len(descriptors_duck)):
        img1 = duck_images[i]
        keypoints_1 = keypoints_duck[i]
        img2 = scene_image
        keypoints_2 = keypoints_scene
        desc1 = descriptors_duck[i]
        desc2 = descriptors_scene

        # BFMatcher
        # # create feature matcher
        # bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        # # match descriptors of both images
        # matches = bf.match(desc1, desc2)

        # # sort matches by distance
        # matches = sorted(matches, key=lambda x: x.distance)
        # # for match in matches:
        # #     print(match.distance)

        # # take only matches within 700 distance
        # good_matches = [m for m in matches if m.distance < 700]

        ##########

        # KNN
        FLANN_INDEX_KDTREE = 0
        des1 = np.float32(desc1)
        des2 = np.float32(desc2)
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good_matches = []
        for m,n in matches:
            if m.distance < 0.3*n.distance:
                good_matches.append(m)
        ##########

        # if good_matches does not have at least 4 matches, means no match
        if len(good_matches) >= 4:

            src_pts = np.float32(
                [keypoints_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [keypoints_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            # print(dst_pts)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                matchesMask = mask.ravel().tolist()
                h, w = img1.shape
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                                  [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                dst += (w, 0)  # adding offset

                # img3 = cv2.drawMatches(img1,keypoints_1,scene_image,keypoints_scene,matches[:50], None,**draw_params)
                if test_mode:
                    draw_params = dict(matchColor=(0, 0, 255),  # draw matches in green color
                                       singlePointColor=None,
                                       matchesMask=matchesMask,  # draw only inliers
                                       flags=2)

                    img3 = cv2.drawMatches(img1,keypoints_1,img2,keypoints_scene,good_matches,None,**draw_params) # for knn
                    # img3 = cv2.drawMatches(img1, keypoints_1, scene_image, keypoints_scene,
                    #                        matches[:50], None, flags=2)  # for euclidian distance

                    # Draw bounding box in Red
                    img3 = cv2.polylines(
                        img3, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

                if not test_mode:
                    # get the middle point of the bounding box or 4 points of dst
                    # print(dst)

                    # print(centroid)
                    dst_moffset = dst - (w, 0)
                    x = [p[0][0] for p in dst_moffset]
                    y = [p[0][1] for p in dst_moffset]
                    centroid = ((max(x) + min(x))/2, (max(y) + min(y))/2)
                    # print("Returning " + str(centroid))
                    return centroid
            else:
                if test_mode:
                    img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[0][:50], img2, flags=2) # for knn
                    # img3 = cv2.drawMatches(img1, keypoints_1, scene_image, keypoints_scene,
                    #                       matches[:50], None, flags=2)  # for euclidian distance

        else:
            if test_mode:
                img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[0][:50], img2, flags=2) # for knn
                # img3 = cv2.drawMatches(img1, keypoints_1, scene_image, keypoints_scene,
                #                        matches[:50], None, flags=2)  # for euclidian distance

        if test_mode:
            cv2.imshow("result", img3)
            cv2.waitKey()


if __name__ == "__main__":
    scene_image = cv2.imread(
        test_frame_file)
    CallSift(scene_image, test_mode=True)
