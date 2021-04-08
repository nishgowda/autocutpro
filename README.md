# autocutpro
 
Autonomous video editing powered by Object Tracking and Motion Detection

## What is this?
The first method of video editing is through the use of **Object Tracking**. Using PyTorch, YOLOv3, and OpenCV a [deep learning model](https://github.com/abewley/sort) is made to track objects in a given video. Using this model, the user specifies which objects in a given video they would like to scan through and will then make cuts along the frames of these objects in the video and splice them together to create a new scene.

The other method of video editing is by using **Motion Detection**. Each frame of a given video is compared by computing the difference between the RGB channels of each pixel and the video is cut along the given motion threshold

## How to Run:
Firt clone this repositiory and then install the required dependencies (preferably in your virtual environment) with ``pip``.
```
pip install requirements.txt
```
### Object Tracking
  1. Run the shell file download_weights.sh or run 
  ```wget https://pjreddie.com/media/files/yolov3.weights``` (only need to do this once)
  2. Specify object tracking method
  3. Specify a directory of the video you would like to read and specify the output directory for the edited copy. (You can also choose random to select a random object in the video for fun)
  4. Once it detects objects, choose the objects from the displayed list to edit the video around
  ### Usage:

  ```
  $ python3 main.py object videos/short-clip.mp4 out/test-output2.mp4
  ```
### Motion Detection:
  1. Specify motion algorithm
  2. Provide the directory of the video you would like to read, specify what the filename and output should be and the motion threshold (Random is also an option if you want to select a random motion percentage)
  ### Usage:
  ```
  $ python3 main.py motion videos/short-clip.mp4 edits/motion-test-variety-new.mp4 10 15
  ```
