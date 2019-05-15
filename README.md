# ROBOT ASSIGNMENT 2

### Note:


### First Time Setup:
```
pip install -r requirement.txt
```

### Run:
- Generate Top-view picture:
```
python generate_topview.py + (scene_number)
```
- Main:
```
(sudo) python ai2thor.py + (scene_number)
```

## Challenge-01: Object detection and Classification on First View Cameras.
Link Download pre-traned :https://bitly.vn/3h60 (due to the BigFile)
Once you have the Robot file, run
'''
sudo python AI2THOR.py
'''
When the First view of Robot show Up, Use W, S, A, D for Move up, down, left, right. Use Key arrow to rotate the camera of robot

If you want to change Sceen, press 'q'
You can take picture and detect Object in room whenever you want after press key 'f'

## Challenge-02: Drawing Trajectories on Top-view Map of the room.
Overview: 
- Capture Top-View Image
- Display Location of Robot in map on Top-View Image
- Update Real-time new Location of Robot each step and distinguish by RGB color<br>
Controller:
- Move: Arrow Keys
- Change Views Rotation: W-A-S-D
- Detect Object By YOLO: 'F' key
- Detect Object By ai2thor metadata: 'P' key<br>
Note:
Take a rest between key press to ensure the stability of application


