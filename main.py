"""
    @file: main.py
    @author: Nish Gowda 2020
    About: The purpose of this file is to purely be the
    junction between the video_splice and object_tracker files/motion_detection.
    Given the inputted command line arguments, the program extends
    the values to the functions of the aformentioned files and
    executes the functions.
"""


from video_splice import *
from object_tracker import *
from motion_detection import *
import sys


class Main():
    def __init__(self):
        self.option = ''
        self.video_file = ""
        sys.outpath = ""
        self.filename = ""
        self.motion_percent = []
        self.object_list = []
    def run(self):
        if self.option == 'object':
            obj_tracker = ObjectTracker()
            video_splice = VideoSplice()
            obj_tracker.track_video(self.video_file, self.outpath)
            object_frames = obj_tracker.objects
            video_splice.cut_tracker_video(self.video_file, self.outpath, self.object_list, object_frames)
        elif self.option == 'motion':
            motion_detector = MotionDetection()
            video_splice = VideoSplice()
            motion_detector.compare_frames(self.video_file)
            frames = motion_detector.frames
            video_splice.cut_motion_video(self.video_file, self.outpath, self.motion_percent,frames)
        else:
            print('Not an option')

if __name__ == "__main__":
    main = Main()
    main.option = sys.argv[1]
    if main.option == 'motion':
        main.video_file = sys.argv[2]
        main.outpath = sys.argv[3]
        main.filename = sys.argv[4]
        main.motion_percent = sys.argv[5:]
        main.object_list = []
    elif main.option == 'object':
        main.video_file = sys.argv[2]
        main.outpath = sys.argv[3]
        main.filename = sys.argv[4]
        main.object_list = sys.argv[5:]
    main.run()
