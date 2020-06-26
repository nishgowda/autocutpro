from models import *
from utils import *
from object_tracker import ObjectTracker
import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from datetime import datetime
from collections import defaultdict
from PIL import Image
import cv2
from sort import *
from progress.bar import Bar
# load weights and set defaults

# load model and put into eval mode



class VideoSplice():

    def cut_video(self, videopath, object_list, filename, obj_frames):
        #global objects
        #print(object_list)

        vid = cv2.VideoCapture(videopath)
        vid.set(cv2.CAP_PROP_BUFFERSIZE, 2)

                #cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
                #cv2.resizeWindow('Stream', (800,600))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        ret,frame=vid.read()
        vw = frame.shape[1]
        vh = frame.shape[0]
        filepath = f"edits/{filename}.mp4"
        print(filepath)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        outvideo = cv2.VideoWriter(filepath,fourcc,20.0,(vw,vh))
        bar = Bar('WRITING SEQUENCE TO VIDEO ', max=len(obj_frames), suffix='%(percent)d%%')

        for obj in object_list:
            if obj in obj_frames:
                    for frames in obj_frames.get(obj):
                        outvideo.write(frames)
                        bar.next()
            else:
                print(f"{obj} is not detected")

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                    break
        print("\nSsaved edited video to output file")
        cv2.destroyAllWindows()
        outvideo.release()
