from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from datetime import datetime

from PIL import Image
import cv2
from sort import *
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
objects = {}



def detect_image(img):
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
def track_video(video_file):
    global objects
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
    print ("Video size", vw,vh)
    filepath = videopath.replace(".mp4", f"-{random.randint(1, 100)}.mp4")
    filepath = filepath.replace("videos/", "out/")
    print(filepath)
    outvideo = cv2.VideoWriter(filepath,fourcc,20.0,(vw,vh))

    frames = 0
    starttime = time.time()
    objects = {}

    while(True):
        ret, frame = vid.read()

        if not ret:
            break
        frames += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(frame)
        detections = detect_image(pilimg)

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
                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                color = colors[int(obj_id) % len(colors)]
                cls = classes[int(cls_pred)]
                cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
                cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
                print(vid.get(cv2.CAP_PROP_FPS))
                start_time = time.time()
                print("--- %s seconds ---" % (time.time() - start_time))
                    #print(cls + " " + str(obj_id))
                obj = f"{cls}-{obj_id}"
                objects.__setitem__(obj,frame)

        #cv2.imshow('Stream', frame)
        outvideo.write(frame)

        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
    totaltime = time.time()-starttime

    print(frames, "frames", totaltime/frames, "s/frame")
    print("Saved file as " + str(filepath))
    print(objects)
    cv2.destroyAllWindows()
    outvideo.release()

def cut_video(videopath, object_list, filename):
    global objects
    print(object_list)
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
    for obj in object_list:
        if obj in objects:
            frames = objects.get(obj)
            print(frames)
            outvideo.write(frames)
            print(f"...writing {obj} with {frames}")
        else:
            print(f"{obj} is not detected")
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
    print("Saved edited video to output file")
    cv2.destroyAllWindows()
    outvideo.release()
if __name__ == "__main__":
    video_file = sys.argv[1]
    filename = sys.argv[2]
    object_list = sys.argv[3:]
    print(object_list)
    track_video(video_file)
    cut_video(video_file, object_list, filename)
