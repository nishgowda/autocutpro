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
        self.filename = ""
        self.motion_percent = 0
        self.object_list = []
    def run(self):
        if self.option == 'object':
            obj_tracker = ObjectTracker()
            video_splice = VideoSplice()
            obj_tracker.track_video(self.video_file)
            object_frames = obj_tracker.objects
            video_splice.cut_video(self.video_file, self.object_list, self.filename, object_frames)
        elif self.option == 'motion':
            motion_detector = MotionDetection()
            video_splice = VideoSplice()
            motion_detector.compare_frames(self.video_file)
            frames = motion_detector.frames
            video_splice.cut_motion_video(self.video_file, self.filename, float(self.motion_percent),frames)
        else:
            print('Not an option')

if __name__ == "__main__":
    main = Main()
    main.option = sys.argv[1]
    if main.option == 'motion':
        main.video_file = sys.argv[2]
        main.filename = sys.argv[3]
        main.motion_percent = sys.argv[4]
        main.object_list = []
    elif main.option == 'object':
        main.video_file = sys.argv[2]
        main.filename = sys.argv[3]
        main.object_list = sys.argv[4:]
    main.run()
