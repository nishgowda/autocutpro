"""
    @file: main.py
    @author: Nish Gowda
    
    The purpose of this file is to purely be the
    main pipeline between the video_splice and object_tracker files/motion_detection.
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

def run_obj_tracker(video_file, outpath, object_list):
        obj_tracker = ObjectTracker()
        video_splice = VideoSplice()
        obj_tracker.track_video(video_file, outpath)
        object_frames = obj_tracker.objects
        video_splice.cut_tracker_video(video_file, outpath, object_list, object_frames)

if __name__ == "__main__":
    option = sys.argv[1]
    if option == 'motion':
        video_file = sys.argv[2]
        outpath = sys.argv[3]
        motion_percent = sys.argv[4:]
        run_motion_detection(video_file, outpath, motion_percent)
    elif option == 'object':
        video_file = sys.argv[2]
        outpath = sys.argv[3]
        object_list = sys.argv[4:]
        run_obj_tracker(video_file, outpath, object_list)
    else:
        print("You supplied an invalid option")
