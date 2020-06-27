"""
    @file: video_slice.py
    @author: Nish Gowda 2020
    About: selects the frames of the selected objects
    and splices them together to create a new scene of
    the video.
    This is meant to be built on top of the object_tracker.py
    and motion_detection.py
    file.
"""
from models import *
from utils import *
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
from sort import *
from progress.bar import Bar




class VideoSplice():
# Stitches together the frames of the targeted objects to create a sequence
    def cut_tracker_video(self, videopath, object_list, filename, obj_frames):
        vid = cv2.VideoCapture(videopath)
        old_vid_length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
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
        new_video_length = int(outvideo.get(cv2.CAP_PROP_FRAME_COUNT))
        percent_edited = round((new_video_length / old_vid_length) * 100, 1)
        print("\n------------------------------------")
        print("Edited out ", percent_edited, "% of video")
        print("\n------------------------------------")
        print("\nSaved edited video to output file as ", filepath)
        cv2.destroyAllWindows()
        outvideo.release()
        
# Stitches together the the frames that have a motion threshold of greater than or equal to the input to create a sequence.
    def cut_motion_video(self, videopath, filename,  motion_percent, frames):
        vid = cv2.VideoCapture(videopath)
        vid.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        old_vid_length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        ret,frame=vid.read()
        vw = frame.shape[1]
        vh = frame.shape[0]
        filepath = f"edits/{filename}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        outvideo = cv2.VideoWriter(filepath,fourcc,20.0,(vw,vh))
        edited_frames = []
        for avg_frame, frame in frames.items():
            top_percent = (avg_frame / len(frames)) * 100
            #print(top_percent)
            if top_percent >= motion_percent:
                edited_frames.append(avg_frame)
                edited_frames = list(set(edited_frames))
                #print(edited_frames)
        bar = Bar('WRITING SEQUENCE TO VIDEO ', max=len(edited_frames), suffix='%(percent)d%%')
        for percent_frame in edited_frames:
            outvideo.write(frames.get(percent_frame))
            bar.next()
            time.sleep(0.1)
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                    break
        cv2.destroyAllWindows()
        outvideo.release()
        new_video_length = int(outvideo.get(cv2.CAP_PROP_FRAME_COUNT))
        percent_edited = round((new_video_length / old_vid_length) * 100, 1)
        print("\n------------------------------------")
        print("Edited out: ", percent_edited, "% of video")
        print("------------------------------------")
        print("\nSaved Edited video to: ", filepath)
