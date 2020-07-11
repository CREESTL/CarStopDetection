# Overview
I already have a repo for `car counting` and tracking but here I decided to detect `car stop`.

So this repo is based on [CarCounterYOLOv3](https://github.com/CREESTL/CarCounterYOLOv3) with some changes in it's structure and code.
____

 At the very beginning a thought that it was enough just to check if centroid coords on N and N+1 frames are the same. That would indicate that a car is standing still.
 
 __BUT__
 
 YOLOv3 works not perfectly and even if I can see that the car is not moving it's centroid is "shaking" on the video.
 That means that it's coords won't be exactly the same of two different frames.
 
 So I decided to consider a car not moving if it's centroid's coords are SLIGHTLY changing but are still not very different on separate frames.
 
 So `how it works`:
 - If distance between car centroid's coords on 1 and 2 frames is shorter than a minimum (we can change it) then we put this centroid in a dictionary.
 - Dictionary looks like this
       ID (key) -> Number of frames on which distance that I was talking about above is "kept"(value)
 - If on the 3rd frame situation is the same (distance is small) than the number of frames increases. And so on...
 - If this continues for some time (or some amount of frames) then we can tell that the car is stopped.
 - If after decreasing the distance starts increasing than we can tell that the car is moving again.
 
 (<i>for more detailed explanation see comments in code</i>)

How to `run it`:
- Clone/Download this project.
- Download YOLOv3 `.weights` [here](https://yadi.sk/d/-WQu7wm-T3dhzg) and put it to `/yolo` folder
- Get all necessary modules via `pip install -r 'requirements.txt'.
- Go to the directory with this project.
- Download some videos of cars driving aroud o just use a test `video01.mp4` that I've uploaded.
- Type `python car_stop_detector.py -y yolo --input videos/'PATH_TO_YOUR_VIDEO'.mp4 --output output --skip-frames 5` and hit `Enter`.
- Enjoy!


