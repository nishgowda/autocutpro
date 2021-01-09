"""
    @file: motion_detection.py
    @author: Nish Gowda 2020
    @about: this file compares each frame by
    computing the difference between their
    rgb values for every pixel.
"""
import cv2
import numpy as np
from PIL import Image
import collections
from progress.bar import Bar
import time

class MotionDetection:
    def __init__(self):
        self.total_diff = 0.0
        self.frames = {}
    #computes the rgb value for each pixel in each frame
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
            # Compare the rgb value of each pixel between the current and previous frames
            diffR, diffG, diffB = 0.0, 0.0, 0.0
            if old_frame is not None:
                # .split grabs the rgb (in order of grb) of the img
                colorsB1, colorsG1, colorsR1 = cv2.split(img)
                colorsB2, colorsG2, colorsR2 = cv2.split(prevImg)
                # Grab the sums of each channels difference -- matrix or np array and divide by 255
                diffR += np.sum(colorsR1 - colorsR2) / 255.0
                diffG += np.sum(colorsG1 - colorsG2) / 255.0
                diffB += np.sum(colorsB1 - colorsB2) / 255.0
                # divide each difference by num pixels to get the avg difference per chanel
                diffR /= numPixels
                diffG /= numPixels
                diffB /= numPixels
                # get the avg of all three channels
                self.total_diff = (diffR + diffG + diffB) / 3.0
                self.total_diff = round(self.total_diff * 100)
                # update our dictionary to grab the difference of the current and previous frame as well as the current frame
                self.frames.update({self.total_diff : frame})
            bar.next()
            old_frame = prevFrame
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
        cv2.destroyAllWindows()
