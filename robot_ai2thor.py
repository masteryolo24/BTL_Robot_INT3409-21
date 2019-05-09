import ai2thor.controller
from ai2thor.controller import Controller
import time
import cv2
import numpy as np
from pynput import keyboard
from time import sleep
import random
import math
import matplotlib.pyplot as plt 
from PIL import Image, ImageDraw
import copy 
import numpy as np
rotation = 0
horizon =0
class ThorPositionTo2DFrameTranslator(object):
    def __init__(self, frame_shape, cam_position, orth_size):
        self.frame_shape = frame_shape
        self.lower_left = np.array((cam_position[0], cam_position[2])) - orth_size
        self.span = 2 * orth_size

    def __call__(self, position):
        if len(position) == 3:
            x, _, z = position
        else:
            x, z = position

        camera_position = (np.array((x, z)) - self.lower_left) / self.span
        return np.array(
            (
                round(self.frame_shape[0] * (1.0 - camera_position[1])),
                round(self.frame_shape[1] * camera_position[0]),
            ),
            dtype=int,
        )

def display_topview_image():
    image = cv2.imread("topview.png")
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def position_to_tuple(position):
    return (position["x"], position["y"], position["z"])

def get_agent_map_data(c):
    c.step({"action": "ToggleMapView"})
    cam_position = c.last_event.metadata["cameraPosition"]
    cam_orth_size = c.last_event.metadata["cameraOrthSize"]
    pos_translator = ThorPositionTo2DFrameTranslator(c.last_event.frame.shape, position_to_tuple(cam_position), cam_orth_size)
    to_return = {
        "frame" : c.last_event.frame,
        "cam_position": cam_position,
        "cam_orth_size": cam_orth_size,
        "pos_translator": pos_translator,
    }
    c.step({"action": "ToggleMapView"})
    return to_return

def add_agent_view_triangle(
    position, rotation, frame, pos_translator, scale=1.0, opacity=0.7
):
    p0 = np.array((position[0], position[2]))
    p1 = copy.copy(p0)
    p2 = copy.copy(p0)

    theta = -2 * math.pi * (rotation / 360.0)
    rotation_mat = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    offset1 = scale * np.array([-1, 1]) * math.sqrt(2) / 2
    offset2 = scale * np.array([1, 1]) * math.sqrt(2) / 2

    p1 += np.matmul(rotation_mat, offset1)
    p2 += np.matmul(rotation_mat, offset2)

    img1 = Image.fromarray(frame.astype("uint8"), "RGB").convert("RGBA")
    img2 = Image.new("RGBA", frame.shape[:-1])  # Use RGBA

    opacity = int(round(255 * opacity))  # Define transparency for the triangle.
    points = [tuple(reversed(pos_translator(p))) for p in [p0, p1, p2]]
    draw = ImageDraw.Draw(img2)
    img = Image.alpha_composite(img1, img2)
    return np.array(img.convert("RGB"))

def save_topview_image():
    controller = ai2thor.controller.Controller(quality = 'High', fullscreen = False)
    controller.start(player_screen_width = 300, player_screen_height = 300)
    scene = random.randint(1, 30)
    controller.reset('FloorPlan' + str(scene))
    visibilityDistance = 10
    renderObjectImage = True
    event = controller.step(dict(action='Initialize', gridSize=0.25, cameraY = 0.6, renderObjectImage = renderObjectImage, visibilityDistance = visibilityDistance))

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
        #cv2.namedWindow("image",cv2.WND_PROP_FULLSCREEN)
        #cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)    
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_press(key):
        global rotation, horizon, event
        try:
            """image = cv2.imread("topview.png")
                                                cv2.imshow("image", image)
                                                cv2.waitKey(0)
                                                cv2.destroyAllWindows()"""
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


    t = get_agent_map_data(controller)
    new_frame = add_agent_view_triangle(
        position_to_tuple(controller.last_event.metadata["agent"]["position"]),
        controller.last_event.metadata["agent"]["rotation"]["y"],
        t["frame"],
        t["pos_translator"],
    )
    plt.imshow(new_frame)
    plt.axis('off')
    plt.savefig('topview.png')

    
    
"""image = cv2.imread("topview.png")
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()"""
save_topview_image()


