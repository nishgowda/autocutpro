#!/usr/bin/env python3
"""
    @file: main.py
    @author: Nish Gowda
    
    The purpose of this file is to purely be the
    main pipeline between the video_splice, object_tracker, and motion_detection files.
    Given the inputted command line arguments, the program extends
    the values to the functions of the aformentioned files and
    executes the functions.
"""

from video_splice import *
from object_tracker import *
from motion_detection import *
import sys

def run_motion_detection(video_file, outpath, motion_percent):
    motion_detector = MotionDetection()
    video_splice = VideoSplice()
    motion_detector.compare_frames(video_file)
    frames = motion_detector.frames
    video_splice.cut_motion_video(video_file, outpath, motion_percent, frames)

def run_obj_tracker(video_file, outpath):
        obj_tracker = ObjectTracker()
        video_splice = VideoSplice()
        obj_tracker.track_video(video_file, outpath)
        object_frames = obj_tracker.obj_frames
        print("Detected the following objects in the supplied video")
        print(', '.join(str(obj) for obj in set(obj_tracker.objects)))
        
        desired_objects = []
        while True:
            resp = input("Enter the objects you want to cut along (press quit to end) ")
            if resp == "quit":
                break
            if resp not in set(obj_tracker.objects):
                desired_objects.append(resp)
            else:
                print(resp, " is not one of the options")
        video_splice.cut_tracker_video(video_file, outpath, desired_objects, object_frames)

if __name__ == "__main__":
    option = sys.argv[1]
    if option == "motion":
        video_file = sys.argv[2]
        outpath = sys.argv[3]
        motion_percent = sys.argv[4:]
        run_motion_detection(video_file, outpath, motion_percent)
    elif option == "object":
        video_file = sys.argv[2]
        outpath = sys.argv[3]
        run_obj_tracker(video_file, outpath)
    else:
        sys.exit("You supplied an invalid option")
