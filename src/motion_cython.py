import cv2
import numpy as np
from PIL import Image
import itertools
import processing_module
class MotionDetection():

    def __init__(self):
        self.total_diff = 0.0
        self.frames = []

    def compare_rgb(self, videopath):
        vid = cv2.VideoCapture(videopath)
        old_frame = None
        while (True):
            ret, frame = vid.read()

            if not ret:
                break

            pilimg = Image.fromarray(frame)


            prevFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            pilimg2 = Image.fromarray(prevFrame)
            img = np.array(pilimg)
            prevImg = np.array(pilimg2)

            w = img.shape[0]
            h = img.shape[1]
            imgWidth, imgHeight = [], []
            imgWidth.append(w)
            imgHeight.append(h)
            numPixels = img.size
            if not ret:
                break
            diffR, diffG, diffB = 0.0, 0.0, 0.0
            if old_frame is not None:
              avg_frame = processing_module.compare_rgb(img, prevImg)
              print(avg_frame)
              self.total_diff = avg_frame
              self.frames.append(self.total_diff)
            old_frame = prevFrame

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

        cv2.destroyAllWindows()
        print(self.frames)

if __name__ == "__main__":
    motion_detection = MotionDetection()
    motion_detection.compare_rgb("videos/short-test.mp4")
