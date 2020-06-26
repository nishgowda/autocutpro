"""
    File: video_slice.py
    Author: Nish Gowda 2020
    About: selects the frames of the selected objects
    and splices them together to create a new scene of
    the video.
    This is meant to be built on top of the object_tracker.py
    file.
"""
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




class VideoSplice():
# Stitches together the frames of the targeted objects to create a sequence
    def cut_video(self, videopath, object_list, filename, obj_frames):
        vid = cv2.VideoCapture(videopath)
        vid.set(cv2.CAP_PROP_BUFFERSIZE, 2)
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
        print("\nSaved edited video to output file as ", filepath)
        cv2.destroyAllWindows()
        outvideo.release()
