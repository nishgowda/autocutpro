"""
    @file: motion_detection.py
    @author: Nish Gowda 2020
    
    The purpose of this program is to compare each frame by
    computing the difference between their rgb values for every pixel.
    This is later applied to video_splice.py to connect each frame to 
    a full video.
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
            prev_frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
            pilimg2 = Image.fromarray(prev_frame)
            img = np.array(pilimg)
            prev_img = np.array(pilimg2)

            img_width = img.shape[0]
            img_height = img.shape[1]
            num_pixels = img.size
            # Compare the rgb value of each pixel between the current and previous frames
            diff_r, diff_g, diff_b = 0.0, 0.0, 0.0
            if old_frame is not None:
                # .split grabs the rgb (in order of grb) of the img
                colors_b1, colors_g1, colors_r1 = cv2.split(img)
                colors_b2, colors_g2, colors_r2 = cv2.split(prev_img)
                # Grab the sums of each channels difference -- matrix or np array and divide by 255
                diff_r += np.sum(colors_r1 - colors_r2) / 255.0
                diff_g += np.sum(colors_g1 - colors_g2) / 255.0
                diff_b += np.sum(colors_b1 - colors_b2) / 255.0
                # divide each difference by num pixels to get the avg difference per chanel
                diff_r /= num_pixels
                diff_g /= num_pixels
                diff_b /= num_pixels
                # get the avg of all three channels
                self.total_diff = (diff_r + diff_g + diff_b) / 3.0
                self.total_diff = round(self.total_diff * 100)
                # update our dictionary to grab the difference of the current and previous frame as well as the current frame
                self.frames.update({self.total_diff : frame})
            bar.next()
            old_frame = prev_frame
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
        cv2.destroyAllWindows()
