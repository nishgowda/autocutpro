"""
    @file: motion_detection.py
    @author: Nish Gowda 2020
    About: this file compares each frame by
    computing the difference between their
    max and min values
"""



import cv2
import numpy as np
from PIL import Image
import collections
from progress.bar import Bar
import time
class MotionDetection():

    def __init__(self):
        self.total_diff = 0.0
        self.frames = {}

    #computes the difference between each frame in video
    def compare_frames(self, videopath):
        vid = cv2.VideoCapture(videopath)
        video_length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        old_frame = None
        bar = Bar('Comparing frames in video', max=video_length, suffix='%(percent)d%%')
        while (True):
            ret, frame = vid.read()

            if not ret:
                break
            pilimg = Image.fromarray(frame)
            prevFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not ret:
                break
            if old_frame is not None:
                diff_frame = prevFrame - old_frame
                diff_frame -= diff_frame.min()
                disp_frame = np.uint8(255.0*diff_frame/float(diff_frame.max()))
                avg_frame = (np.mean(disp_frame) / 255.0)
                avg_frame = round(avg_frame * 100)
                #print(avg_frame)
                self.frames.update({avg_frame : frame})
                bar.next()
            old_frame = prevFrame
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
        #print(self.frames)
        cv2.destroyAllWindows()
