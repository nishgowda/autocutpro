# Object-Tracking-Video-Editor
Automating the Video Editing process through Object Tracking
## HOW?
Using PyTorch and YOLOv3, it's simple enough to create a deep learning model that can track objects in a given video. Using this model, my algorithm allows the user to specify which objects in a given video they would like to scan through and will then make cuts along the frames of these objects in the given video and splice them together to create a new scene. If the object is one of the available classes that the model can detect, then this allows the user to save themselves of having to go through all the amount of footage just to edit a given scene with a specific object. After running the program you will notice two videos have been created. 1 that shows all the objects tracked in the video with boxes identifying them. And the other one will be the edited version of your original inputed video that edited out everything besides the objects you specified.
## Installation:
```
git clone :https://github.com/nishgowda/Object-Tracking-Video-Editor
```

## How to Run:
  1. cd into directory
  2. Specify a directory of the video you would like to read, specify the output directory for the edited copy and the objects you would like to detect.
### Example: 
  ```
  $ python3 object_tracker.py videos/short-clip.mp4 test-output2.mp4 car-3.0 person-15.0
  ```

*A project by Nish Gowda*
