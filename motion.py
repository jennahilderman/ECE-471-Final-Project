# Motion Detection or Background Subtraction steps referred from : 
# https://towardsdatascience.com/image-analysis-for-beginners-creating-a-motion-detector-with-opencv-4ca6faba4b42

import cv2
import numpy as np

image_1 = "./scene_images_motion/frame_0_20220404-163348.png"
image_2 = "./scene_images_motion/frame_0_20220404-163349.png"
test_frame_file = "./scene_images/frame_55.038.png"

prev_frame = None


def CallMotion(current_frame, test_mode=False):
    global prev_frame

    # For Testing    
    # prev_frame = cv2.imread(image_2)
    # prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # prev_frame = cv2.GaussianBlur(
    #     src=prev_frame, ksize=(5, 5), sigmaX=0)
    
    # Load image
    img_bgr = current_frame

    # Prepare image; grayscale and blur
    prepared_frame = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    prepared_frame = cv2.GaussianBlur(
        src=prepared_frame, ksize=(5, 5), sigmaX=0)

    if prev_frame is None:
        prev_frame = prepared_frame
        return None

    # Calculate difference
    diff_frame = cv2.absdiff(src1=prev_frame, src2=prepared_frame)
    
    if test_mode:
        cv2.imshow('Motion detector', diff_frame)
        cv2.waitKey()
    
    # Update previous frame
    prev_frame = prepared_frame

    # Dilute the image a bit to make differences more seeable; more suitable for contour detection
    kernel = np.ones((5, 5))
    diff_frame = cv2.dilate(diff_frame, kernel, 1)
    
    if test_mode:
        cv2.imshow('Frame Difference', diff_frame)
        cv2.waitKey()

    contours, _ = cv2.findContours(
        image=diff_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    if test_mode:
        cv2.drawContours(image=img_bgr, contours=contours, contourIdx=-1,
                         color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow('Contours Drawn', img_bgr)
        cv2.waitKey()

    for contour in contours:
        # During the demo, there were levels with rain,
        # Adding this countour area check to make sure they are above a certain size basically removed all rain false positives.
        if cv2.contourArea(contour) < 800:
            # too small: skip!
            continue
        
        (x, y, w, h) = cv2.boundingRect(contour)

        # This is a case where crosshair gets stuck at the bottom
        # Happens because if crosshair is at the bottom, when it moves, the bounding box will be flat
        if y > current_frame.shape[0] - 42:
            continue

        # If contour is around a box of size 38 - 44 (crosshair), skip
        # Usually w and h around 38 - 44
        if 38 <= w <= 44 and 38 <= h <= 44:
            continue

        # Falling ducks are being detected and being shot at. 
        # Thus, bounding boxes that are shaped vertically with their heights being more than their width, are filtered out
        if w < h:
            continue
        print("Motion detected at: ", x, y, w, h)
        if not test_mode:
            return ((x + (x + w)) / 2, (y + (y + w)) / 2)

        cv2.rectangle(img=img_bgr, pt1=(x, y), pt2=(
            x + w, y + h), color=(0, 255, 0), thickness=2)

    if test_mode:
        cv2.imshow('', img_bgr)
        cv2.waitKey()


if __name__ == "__main__":
    image_1 = cv2.imread(
        image_1)
    prev_frame = cv2.imread(image_2)
    CallMotion(image_1, test_mode=True)
