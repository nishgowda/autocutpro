# Smart Clips
Automating the Video Editing process through Object Tracking and Motion Detection
## HOW?
The first method of video editing is through the use of **Object Tracking**. Using PyTorch, YOLOv3, and OpenCV a deep learning model is made to track objects in a given video. Using this model, my algorithm allows the user to specify which objects in a given video they would like to scan through and will then make cuts along the frames of these objects in the video and splice them together to create a new scene. If the object is one of the available classes that the model can detect, then this allows the user to save themselves of having to go through all the amount of footage just to edit a given scene with a specific object.

The other method of video editing is by using **Motion Detection**. Each frame of a given video is compared by computing the difference between the RGB channels of each pixel. Given an inputted motion threshold (max and min), the algorithm uses the computed differences and splices together the frames that contain motion between the max and min threshold percentages.
## Installation:
```
git clone https://github.com/nishgowda/Smart-Clips
```

## How to Run:
### For Object Tracking
  1. cd into directory
  2. Run the shell file download_weights.sh or run ```wget https://pjreddie.com/media/files/yolov3.weights``` (only need to do this once)
  3. Specify object tracking method
  4. Specify a directory of the video you would like to read, specify the output directory for the edited copy and the objects you would like to detect. (You can also choose random to select a random object in the video for fun)
  ### Example:

  ```
  $ python3 main.py object videos/short-clip.mp4 out/test-output2.mp4 car-3.0 person-15.0
  ```
### For Motion Detection:
  1. cd into directory
  2. Specify motion algorithm
  3. Provide the directory of the video you would like to read, specify what the filename and output should be and the motion threshold (Random is also an option if you want to select a random motion percentage)
  ### Example:
  ```
  $ python3 main.py motion videos/short-clip.mp4 edits/motion-test-variety-new.mp4 10 15
  ```
*A project by Nish Gowda*
