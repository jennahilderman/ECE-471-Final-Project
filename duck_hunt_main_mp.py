import argparse
import gym
import cv2
import numpy as np
import pygame
import multiprocessing as mp

import time

import ece471_duckhunt as dh 
from ece471_duckhunt import envs
from ece471_duckhunt.envs import duckhunt_env

from solution import GetLocation 

# Required version for the following packages
print(f"Duck Hunt version: {dh.__version__} (=1.2.0)")
print(f"OpenCV version: {cv2.__version__} (=4.X)")
print(f"NumPy version: {np.__version__} (=1.19+)")
print(f"OpenGym version: {gym.__version__} (=0.18.0)")

""" If your algorithm isn't ready, it'll perform a NOOP """
def noop():
    return [{'coordinate' : 8, 'move_type' : 'relative'}]

download_directory = "./scene_images/"
download_mode = False


# Helper to download the current frame
def download_image(current_frame, filename):
    """
    Download the current frame from the game.
    """
    # convert filename to bgr
    result_rotate = cv2.rotate(current_frame, cv2.cv2.ROTATE_90_CLOCKWISE)
    result_flip = cv2.flip(result_rotate, 1)
    result_BGR = cv2.cvtColor(result_flip, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, result_BGR)

""" Here is the main loop for your algorithm """
def main(args):
    time_to_dl = 1
    dl_counter = 0
    future = None
    executor = mp.Pool(processes=4)

    while True:
         
        train_start = time.time()

        """ 
        Use the `current_frame` from either env.step of env.render
        to determine where to move the scope.
        
        current_frame : np.ndarray (width, height, 3), np.uint8, RGB
        """
        current_frame = env.render() 
        
        """
            The game needs to continue while you process the previous image so 
            we will be using multithreading (as pygame cannot be multithreaded directly).
            This should be realitively save, as it's a single thread (and not a process).
           
            A no-operation is sent to the game while waiting for your algorithm to finish.

            You can remove this restriction while developing your algorithm, but
            during the demo this is the setup that will be run.  Example with dummy arguments
            have been provided to you.
        """
        
        if args.move_type == 'manual':
            #manual mode
            result = [{"coordinate" : pygame.mouse.get_pos(), 'move_type' : "absolute"}]
        else:
            if future is None:
                result = noop()
                future = executor.apply_async(GetLocation, args=('absolute',env.action_space if args.move_type == "relative" else env.action_space_abs,current_frame))
            elif future.ready():
                result = future.get()
                future = None
       
        """
        Pass the current location (and location type) you want the "gun" place.
        The action "shoot" is automatically done for you.
        Returns: 
                current_frame: np.dnarray (W,H,3), np.uint8, RGB: current frame after `coordinate` is applied.
                level_done: True if the current level is finished, False otw 
                game_done: True if all the levels are finished, False otw 
                info: dict containing current game information (see API guide)
        
        """
        for res in result[:10]:
            coordinate  = res['coordinate']
            move_type   = res['move_type']
            current_frame, level_done, game_done, info = env.step(coordinate, move_type)
            if level_done or game_done:
                break

        # check if time reached time_to_dl
        # if info['time'] exists
        level_time = info.get("time")
        if level_time and download_mode:
            if level_time > time_to_dl:
                time_to_dl = int(time_to_dl) + 6
                download_image(current_frame, download_directory + "frame_" + str(dl_counter) + "_" + time.strftime("%Y%m%d-%H%M%S") + ".png")

        if level_done:
            """ Indicates the level has finished. Any post-level cleanup your algorithm may need """
            executor.close()
            executor.join()
            executor.terminate()
            pass

        if game_done:
            """ All levels have finished."""
            print(info)
            break

if __name__ == "__main__":
    desc="ECE 471 - Duck Hunt Challenge"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-m", "--move-type", 
            default="relative", choices=["relative", "absolute", "manual"], 
            help="Use relative or absolute coordinates, for testing purposes.  You may switch between modes in your algorithm. Manual mode use mouse to move (experimental, untested on Windows)")
    parser.add_argument("-a", "--move-amount", type=int, default=1,
            help="When using relative coordinates, set the delta move amount (default=1)")
    parser.add_argument("-l", "--level", type=int, default=0,
            help="Level to run: 0-TODO  (default=0). Level 0 randomly plays all levels.")
    parser.add_argument("-q", "--quiet", action="store_true",
            help="No visual output or debugging messages")
    parser.add_argument("-r", "--randomize", action="store_true", 
            help="Randomize the levels when level=0.")
    
    parser.add_argument("-w", "--win-size", type=int, nargs=2, default=(1024,768),
            help="Override game window size.  During evaluation and demo, this will be set to the default (1024, 768)")
    parser.add_argument("-d", "--duration", type=int, default=0,
            help="Override level duration. During evaluation and demo, this will be set to the level dependent")
    parser.add_argument("-s", "--seed", type=int, default=None,
            help="Override random seed for reproducability. During evaluation and demo, this will be set to the same for all groups.")
    
    args = parser.parse_args()
    
    #Construct the Duck Hunt Environment
    env = gym.make("DuckHunt-v0", 
                    move_amount=args.move_amount,
                    quiet=args.quiet,
                    level=args.level,
                    shape=args.win_size,
                    duration=args.duration,
                    seed=args.seed,
                    randomize=args.randomize,
                    )


    main(args)
