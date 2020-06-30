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
            prevFrame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
            pilimg2 = Image.fromarray(prevFrame)
            img = np.array(pilimg)
            prevImg = np.array(pilimg2)

            imgWidth = img.shape[0]
            imgHeight = img.shape[1]
            numPixels = img.size
            diffR, diffG, diffB = 0.0, 0.0, 0.0
            
            if old_frame is not None:

                colorsB1, colorsG1, colorsR1 = cv2.split(img)
                colorsB2, colorsG2, colorsR2 = cv2.split(prevImg)
                diffR += np.sum(colorsR1 - colorsR2) / 255.0
                diffG += np.sum(colorsG1 - colorsG2) / 255.0
                diffB += np.sum(colorsB1 - colorsB2) / 255.0
                diffR /= numPixels
                diffG /= numPixels
                diffB /= numPixels
                self.total_diff = (diffR + diffG + diffB) / 3.0
                self.total_diff = round(self.total_diff * 100)
                #print(self.total_diff)


                self.frames.update({self.total_diff : frame})
            bar.next()
            old_frame = prevFrame
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
        #print(self.frames)
        cv2.destroyAllWindows()
