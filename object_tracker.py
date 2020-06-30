"""
    File: object_tracker.py
    Author: Nish Gowda 2020
    About: The purpose of this file is to detect the objects
    in each frame of a given video. The detect image function
    uses the sort and model file to detect images that are
    within the model class in a given frame. With that, it stores the
    contents of the detected object and the frames that object
    is located in a dictionary that is then used in the video_slice
    file.
"""
from models import *
from utils import *

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
config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4

# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.eval()

classes = utils.load_classes(class_path)
Tensor = torch.FloatTensor


class ObjectTracker():
    def __init__(self):
        self.objects = {}

    def detect_image(self, img):
        # scale and pad image
        ratio = min(img_size/img.size[0], img_size/img.size[1])
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)
        img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
             transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                            (128,128,128)),
             transforms.ToTensor(),
             ])
        # convert image to Tensor
        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input_img = Variable(image_tensor.type(Tensor))
        # run inference on the model and get detections
        with torch.no_grad():
            detections = model(input_img)
            detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
        return detections[0]
    def track_video(self, video_file):
        videopath = str(video_file)
        colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]
        vid = cv2.VideoCapture(videopath)
        vid.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        mot_tracker = Sort()

        #cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('Stream', (800,600))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        ret,frame=vid.read()
        vw = frame.shape[1]
        vh = frame.shape[0]
        video_length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        print ("Video size", vw,vh)
        filepath = videopath.replace(".mp4", f"-{random.randint(1, 100)}.mp4")
        filepath = filepath.replace("videos/", "out/")
        print(filepath)
        outvideo = cv2.VideoWriter(filepath,fourcc,20.0,(vw,vh))

        frames = 0
        starttime = time.time()
        bar = Bar('WRITING SEQUENCE TO VIDEO ', max=video_length, suffix='%(percent)d%%')
        while(True):
            ret, frame = vid.read()

            if not ret:
                break
            frames += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pilimg = Image.fromarray(frame)
            detections = self.detect_image(pilimg)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = np.array(pilimg)
            pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
            pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
            unpad_h = img_size - pad_y
            unpad_w = img_size - pad_x
            if detections is not None:
                tracked_objects = mot_tracker.update(detections.cpu())

                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                    cls = classes[int(cls_pred)]
                    obj = f"{cls}-{obj_id}" # The identity of the objects found
                    self.objects.setdefault(obj, []).append(frame)
                    box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                    box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                    y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                    x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                    color = colors[int(obj_id) % len(colors)]
                    cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                    cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
                    cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)


            outvideo.write(frame)
            bar.next()
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
        totaltime = time.time()-starttime
        print("\n", frames, "frames", totaltime/frames, "s/frame")
        print("Saved file as " + str(filepath))
        cv2.destroyAllWindows()
        outvideo.release()
