import ai2thor.controller
import time
import cv2
import numpy as np
from pynput import keyboard
import time
import random
import math
import matplotlib.pyplot as plt 
from PIL import Image, ImageDraw, ImageChops
import copy 
import numpy as np
import robot_ai2thor
"""def collisionXZ(scene):
    controller = ai2thor.controller.BFSController()
    controller.start()
    controller.search_all_closed('FloorPlan' + str(scene))
    arrayX = []
    arrayZ = []
    for o in controller.grid_points:
        arrayX.append(o['x'])
        arrayZ.append(o['z'])

    arrayX = list(set(arrayX))
    arrayZ = list(set(arrayZ))
    return arrayX, arrayZ

def collisionZ(scene):
    controller = ai2thor.controller.BFSController()
    controller.start()
    controller.search_all_closed('FloorPlan' + str(scene))
    arrayZ = []
    for o in controller.grid_points:
        arrayZ.append(o['z'])

    arrayZ = list(set(arrayZ))
    return arrayZ

arrayX, arrayZ = collisionXZ(16)"""
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

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


def position_to_tuple(position):
    return (position["x"], position["y"], position["z"])


def get_agent_map_data(c):
    c.step({"action": "ToggleMapView"})
    cam_position = c.last_event.metadata["cameraPosition"]
    cam_orth_size = c.last_event.metadata["cameraOrthSize"]
    pos_translator = ThorPositionTo2DFrameTranslator(
        c.last_event.frame.shape, position_to_tuple(cam_position), cam_orth_size
    )
    to_return = {
        "frame": c.last_event.frame,
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
image = cv2.imread("topview.png")
x = int(image.shape[0] / 2)
y = int(image.shape[1] / 2)
def on_press(key):
    global rotation, horizon, event, x, y
    image = cv2.imread("topview.png")
    #print(event.metadata['agent']['position'])
    #cv2.circle(image,(int(event.metadata['agent']['position']['x'])+ 320 , int(event.metadata['agent']['position']['z']) + 240), 5, (0,255,0),-1)
    #print(int(event.metadata['agent']['position']['x'])+ 320 , int(event.metadata['agent']['position']['z']) + 240)
    #display_image(image)
    print(event.metadata['agent']['rotation'])
    cv2.circle(image, (x, y+10), 3, (255,0,0), -1)
    display_image(image)
    try:
        print(event.metadata['agent']['position']['x'], event.metadata['agent']['position']['y'], event.metadata['agent']['position']['z'])
        if key.char == 'w':
            event = controller.step(dict(action='MoveAhead'))
            print(event.metadata['lastAction'])
            print(event.metadata['lastActionSuccess'])
            if event.metadata['lastActionSuccess'] == True:
                y +=20

        elif key.char == 's':
            event = controller.step(dict(action ='MoveBack'))
            print(event.metadata['lastAction'])
            print(event.metadata['lastActionSuccess'])
            if event.metadata['lastActionSuccess'] == True:
                y -=20
        elif key.char == 'd':
            #rotation +=10
            event = controller.step(dict(action = 'RotateRight'))
        elif key.char == 'a':
            #rotation -=10
            event = controller.step(dict(action = 'RotateLeft'))
        """
        elif key.char == 'c':
            display_image(draw_box(event))
            #cv2.imwrite('test1.png', event.cv2img)
        """

    except:
        if key == keyboard.Key.up:
            horizon -=10
            event = controller.step(dict(action = 'Look', horizon = horizon))
        elif key == keyboard.Key.down:
            horizon +=10
            event = controller.step(dict(action = 'Look', horizon = horizon))

        elif key == keyboard.Key.right:
            event = controller.step(dict(action = 'MoveRight'))
            x +=15
        elif key == keyboard.Key.left:
            event = controller.step(dict(action = 'MoveLeft'))
            x -=15
    
def on_release(key):
    if key == keyboard.Key.esc:
        return False


if __name__ == "__main__":  
    scene = 16  
    """c = ai2thor.controller.Controller(quality = 'High', fullscreen = False)
    c.start(player_screen_width = 300, player_screen_height = 300)
    scene = 15
    #scene = random.randint(1, 30)
    c.reset("FloorPlan" + str(scene))
 
    t = get_agent_map_data(c)
    new_frame = add_agent_view_triangle(
        position_to_tuple(c.last_event.metadata["agent"]["position"]),
        c.last_event.metadata["agent"]["rotation"]["y"],
        t["frame"],
        t["pos_translator"],
    )
    plt.imshow(new_frame)
    plt.axis('off')
    plt.savefig('topview.png')
    im = Image.open('topview.png')
    im = trim(im)
    im.save("topview.png")
   
    plt.show()"""
	
    controller = ai2thor.controller.Controller(quality = 'Low', fullscreen = False)
    controller.start(player_screen_width = 300, player_screen_height = 300)
    controller.reset('FloorPlan' + str(scene))

    visibilityDistance = 10
    renderObjectImage = True
    event = controller.step(dict(action='Initialize', gridSize=0.25, cameraY = 0.6, renderObjectImage = renderObjectImage, visibilityDistance = visibilityDistance))
    with keyboard.Listener(
               on_press=on_press,
               on_release=on_release) as listener:
        listener.join()

