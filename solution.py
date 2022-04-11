import time

from sift import CallSift
from motion import CallMotion
from tensor_code import tensor_main

import cv2
"""
Replace following with your own algorithm logic

Two random coordinate generator has been provided for testing purposes.
Manual mode where you can use your mouse as also been added for testing purposes.
"""
prev_frame = None

def GetLocation(move_type, env, current_frame):
    # time.sleep(1) #artificial one second processing time
    
    
    #Use relative coordinates to the current position of the "gun", defined as an integer below
    if move_type == "relative":
        """
        North = 0
        North-East = 1
        East = 2
        South-East = 3
        South = 4
        South-West = 5
        West = 6
        North-West = 7
        NOOP = 8
        """
        coordinate = env.action_space.sample() 
    #Use absolute coordinates for the position of the "gun", coordinate space are defined below
    else:
        """
        (x,y) coordinates
        Upper left = (0,0)
        Bottom right = (W, H) 
        """
        # coordinate = env.action_space_abs.sample()
        result_rotate = cv2.rotate(current_frame, cv2.cv2.ROTATE_90_CLOCKWISE)
        result_flip = cv2.flip(result_rotate, 1)
        result_BGR = cv2.cvtColor(result_flip, cv2.COLOR_RGB2BGR)
        
        # 1. SIFT Feature Matching Solution
        # coordinate = CallSift(result_BGR, test_mode = False)
    
        # 2. Motion detection solution
        coordinate = CallMotion(result_BGR, test_mode = False)

        # print(coordinate)
        
    return [{'coordinate' : coordinate, 'move_type' : move_type}]
