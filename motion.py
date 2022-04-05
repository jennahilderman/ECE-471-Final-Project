# https://towardsdatascience.com/image-analysis-for-beginners-creating-a-motion-detector-with-opencv-4ca6faba4b42

import cv2
import numpy as np

image_1 = "./scene_images_motion/frame_0_20220404-163348.png"
image_2 = "./scene_images_motion/frame_0_20220404-163349.png"
test_frame_file = "./scene_images/frame_55.038.png"

prev_frame = None


def CallMotion(current_frame, test_mode=False):
    global prev_frame

    # 1. Load image
    img_bgr = current_frame
    # img_rgb = cv2.cvtColor(src=img_bgr, code=cv2.COLOR_BGR2RGB)

    # 2. Prepare image; grayscale and blur
    prepared_frame = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    prepared_frame = cv2.GaussianBlur(
        src=prepared_frame, ksize=(5, 5), sigmaX=0)

    if prev_frame is None:
        prev_frame = prepared_frame
        return None

    # calculate difference and update previous frame
    diff_frame = cv2.absdiff(src1=prev_frame, src2=prepared_frame)

    prev_frame = prepared_frame

    # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
    kernel = np.ones((5, 5))
    diff_frame = cv2.dilate(diff_frame, kernel, 1)

    # 5. Only take different areas that are different enough (>20 / 255)
    thresh_frame = cv2.threshold(
        src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]

    contours, _ = cv2.findContours(
        image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    if True:
        cv2.drawContours(image=img_bgr, contours=contours, contourIdx=-1,
                         color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    for contour in contours:
        # if cv2.contourArea(contour) < 50:
        #     # too small: skip!
        #     continue
        (x, y, w, h) = cv2.boundingRect(contour)

        # case where crosshair gets stuck at the bottom
        if y > current_frame.shape[0] - 42:
            continue

        # if contour is around a box of size 40-42 (crosshair), skip
        # usually w and h around 40 - 42
        if 38 <= w <= 42 and 38 <= h <= 42:
            continue

        # hacky method, probably needs improvement
        # if the contour is vertical means change somewhat the y direction, then it is a duck dying, skip
        # however if the alive duck is flying somewhat vertically they unfortunately get filtered out too.
        if w < h:
            continue

        if not test_mode:
            return ((x + (x + w)) / 2, (y + (y + w)) / 2)

        cv2.rectangle(img=img_bgr, pt1=(x, y), pt2=(
            x + w, y + h), color=(0, 255, 0), thickness=2)

    if test_mode:
        cv2.imshow('Motion detector', img_bgr)
        cv2.waitKey()


if __name__ == "__main__":
    image_1 = cv2.imread(
        image_1)
    prev_frame = cv2.imread(image_2)
    CallMotion(image_1, test_mode=True)
