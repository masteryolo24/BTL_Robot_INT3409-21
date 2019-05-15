import ai2thor.controller
import time
import cv2
import numpy as np
from pynput import keyboard
import time
import random
controller = ai2thor.controller.Controller(quality = 'Low', fullscreen = False)
controller.start(player_screen_width = 700, player_screen_height = 300)
scene = random.randint(1, 30)
controller.reset('FloorPlan' + str(scene))
visibilityDistance = 10
renderObjectImage = True
event = controller.step(dict(action='Initialize', gridSize=0.25, cameraY = 0.6, renderObjectImage = renderObjectImage, visibilityDistance = visibilityDistance))

"""def count_Type(event):
    array = []
    for o in event.metadata['objects']:
        if o['visible'] == True and o['objectType'] != 'CounterTop' and o['objectType'] != 'Window':
            array.append(o['objectType'])
    array = list(set(array))
    return len(array)
print(count_Type(event)[1])"""
def draw_box(event):
    r = 255
    g = 0
    b = 0
    cv2.imwrite('test1.png', event.cv2img)
    img = cv2.imread("test1.png", 3)
    for o in event.metadata['objects']:
        if o['visible'] == True and o['objectType'] != 'CounterTop' and o['objectType'] != 'Window':
            a = event.instance_detections2D[o['objectId']]
            cv2.rectangle(img, (a[0], a[1]), (a[2], a[3]), (r, g, b), 1)
            cv2.putText(img, o['objectType'], (a[0], a[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
    return img
def display_image(img):
    cv2.imshow("image", img)
    cv2.namedWindow("image",cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

rotation = 0
horizon = 0
def on_press(key):
    global rotation, horizon, event
    try:
        if key.char == 'w':
            event = controller.step(dict(action='MoveAhead'))
        elif key.char == 's':
            event = controller.step(dict(action ='MoveBack'))
        elif key.char == 'd':
            rotation +=10
            event = controller.step(dict(action = 'Rotate', rotation = rotation))
        elif key.char == 'a':
            rotation -=10
            event = controller.step(dict(action = 'Rotate', rotation = rotation))
        elif key.char == 'c':
            display_image(draw_box(event))
            #cv2.imwrite('test1.png', event.cv2img)
    except:
        if key == keyboard.Key.up:
            horizon -=10
            event = controller.step(dict(action = 'Look', horizon = horizon))
        elif key == keyboard.Key.down:
            horizon +=10
            event = controller.step(dict(action = 'Look', horizon = horizon))

        elif key == keyboard.Key.right:
            event = controller.step(dict(action = 'MoveRight'))
        elif key == keyboard.Key.left:
            event = controller.step(dict(action = 'MoveLeft'))

def on_release(key):

    try:

        if key.char == 'w':
            event = controller.step(dict(action='MoveAhead'))
        elif key.char == 's':
            event = controller.step(dict(action ='MoveBack'))
    except:
        if key == keyboard.Key.esc:
            return False
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()