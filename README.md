# Smart Clips
Automating the Video Editing process through Object Tracking and Motion Detection
## HOW?
Using PyTorch, YOLOv3, and OpenCV a deep learning model can be made to track objects in a given video. Using this model, my algorithm allows the user to specify which objects in a given video they would like to scan through and will then make cuts along the frames of these objects in the video and splice them together to create a new scene. If the object is one of the available classes that the model can detect, then this allows the user to save themselves of having to go through all the amount of footage just to edit a given scene with a specific object. After running the program you will notice two videos have been created. One that shows all the objects tracked in the video with boxes identifying them, and the other will be an edited version of your original inputed video that edited out everything besides the objects you specified.

The other method of video editing is by using Motion Detection. Each frame of a given video is compared by computing the difference between the RGB channels of each pixel. Given an inputted motion threshold (max and min), the algorithm uses the computed differences and splices together the frames that contain said amount of motion.
## Installation:
```
git clone: https://github.com/nishgowda/Object-Tracking-Video-Editor
```

## How to Run:
### For Object Tracking
  1. cd into directory
  2. Run the shell file download_weights.sh or run ```wget https://pjreddie.com/media/files/yolov3.weights```
  3. Specify object tracking method
  4. Specify a directory of the video you would like to read, specify the output directory for the edited copy and the objects you would like to detect.
  ### Example:

  ```
  $ python3 main.py object videos/short-clip.mp4 test-output2.mp4 car-3.0 person-15.0
  ```
### For Motion Detection:
  1. cd into directory
  2. Specify motion algorithm
  3. Provide the directory of the video you would like to read, specifiy what the filename and output should be and the motion threshold
  ### Example:
  ```
  $ python3 main.py motion videos/short-test.mp4 motion-test-new.mp4 10 15
  ```
*A project by Nish Gowda*
