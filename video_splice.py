"""
    @file: video_slice.py
    @author: Nish Gowda
    
    This selects the frames of the selected objects or motion
    and splices them together to create a new scene of
    the video. This is meant to be built on top of the object_tracker.py
    and motion_detection.py files.
"""
from models import *
import utils
from object_tracker import ObjectTracker
from motion_detection import MotionDetection
import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from datetime import datetime
from collections import defaultdict
from PIL import Image
import cv2
import sort

class VideoSplice:
    def __init__(self):
        self.split = "\n------------------------------------"
    # stitches together the frames of the targeted objects to create a sequence
    def cut_tracker_video(self, videopath, outpath,  object_list, obj_frames):
        vid = cv2.VideoCapture(videopath)
        old_vid_length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        vid.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        ret,frame=vid.read()
        vw = frame.shape[1]
        vh = frame.shape[0]
        filepath = outpath
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        outvideo = cv2.VideoWriter(filepath, fourcc, 20.0, (vw, vh))
    
        if "random" in object_list:
            obj = random.choice(list(obj_frames))
            for frames in obj_frames.get(obj):
                outvideo.write(frames)
        else:
            for obj in object_list:
                if obj in obj_frames:
                    # grab the frame from the dictionary and write it to the out video
                        for frames in obj_frames.get(obj):
                            outvideo.write(frames)
                else:
                    print(f"{obj} is not detected")
                ch = 0xFF & cv2.waitKey(1)
                if ch == 27:
                        break
        cv2.destroyAllWindows()
        outvideo.release()
        new_video = cv2.VideoCapture(filepath)
        new_video_length = int(new_video.get(cv2.CAP_PROP_FRAME_COUNT))
        percent_edited = (((old_vid_length - new_video_length) / old_vid_length) * 100)
        print(self.split)
        print("Edited out {:.0f}% of source video".format(percent_edited))
        print(self.split)
        print("Saved edited video to output file as ", filepath)


    # Stitches together the the frames that have a motion threshold of greater than or equal to the input to create a sequence.
    def cut_motion_video(self, videopath, outpath,  motion_percent, frames):
        vid = cv2.VideoCapture(videopath)
        vid.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        old_vid_length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        ret,frame=vid.read()
        vw = frame.shape[1]
        vh = frame.shape[0]
        filepath = outpath
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        outvideo = cv2.VideoWriter(filepath,fourcc,20.0,(vw,vh))
        edited_frames = []
        print(motion_percent)
        if "random" in motion_percent:
            for avg_frame, frame in frames.items():
                top_percent = (avg_frame / len(frames)) * 100
                compared_percent1 = random.randint(1,99)
                compared_percent2 = random.randint(compared_percent1,100)
                if top_percent >= float(compared_percent1) or top_percent <= float(compared_percent2):
                    edited_frames.append(avg_frame)
                    edited_frames = list(set(edited_frames)) # ensures that there are no duplicate frames in list
        else:
            for avg_frame, frame in frames.items():
                # calculates what percentage of the values each frame belongs to
                top_percent = (avg_frame / len(frames)) * 100
                if top_percent >= float(motion_percent[0]) or top_percent <= float(top_percent[1]):
                    edited_frames.append(avg_frame)
                    edited_frames = list(set(edited_frames)) # ensures that there are no duplicate frames in list
        for percent_frame in edited_frames:
            # grab the frames from the dictionary in motion_detection algorithm and write it to the video
            outvideo.write(frames.get(percent_frame))
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                    break
        cv2.destroyAllWindows()
        outvideo.release()
        new_video = cv2.VideoCapture(filepath)
        new_video_length = int(new_video.get(cv2.CAP_PROP_FRAME_COUNT))
        percent_edited = (((old_vid_length - new_video_length) / old_vid_length) * 100)
        print(self.split)
        print("Edited out: {:.0f}% of source video".format(percent_edited))
        print(self.split)
        print("Saved edited video to: ", filepath)

