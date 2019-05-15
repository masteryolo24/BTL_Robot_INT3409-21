import ai2thor.controller
import cv2
import numpy as np
from pynput import keyboard
import random
import math
import matplotlib.pyplot as plt 
from PIL import Image, ImageDraw, ImageChops
import copy 
import argparse
import sys
rotation = 0
horizon =0
r = 255
g = 70
b = 0
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

def takePicture(event):
    set_confidence = 0.1
    set_threshold = 0.3
    cv2.imwrite("pic.png", event.cv2img)
    # load the COCO class labels
    labelsPath = 'yolo-object-detection/yolo-coco/coco.names'
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


    # derive the paths to the YOLO weights and model configuration
    weightsPath = 'yolo-object-detection/yolo-coco/yolov3.weights'
    configPath = 'yolo-object-detection/yolo-coco/yolov3.cfg'

    # load Yolo on coco dataset
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # load input Image
    image = cv2.imread('pic.png')
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
    # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > set_confidence:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, set_confidence, set_threshold)
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow("Image", image)
    cv2.waitKey(1)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("sence", type = int, help = 'number of sence', default = 16)

    return parser.parse_args(argv)

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

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
image = cv2.imread("topview.png")
x = int(image.shape[0] / 2 + 20)
y = int(image.shape[1] / 2 + 60)
def save_topview_image(args):
    controller = ai2thor.controller.Controller(quality = 'High', fullscreen = False)
    controller.start(player_screen_width = 300, player_screen_height = 300)
    scene = args.sence
    #scene = random.randint(1, 30)
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

    print(x, y)
    
    def on_press(key):
        global rotation, horizon, event, x, y, image, r, g ,b 
        print(x, y)
        try:
            
            if key.char == 'w': 
                event = controller.step(dict(action='MoveAhead'))
                draw_topview2(controller)
                print(event.metadata['lastActionSuccess'])
                if event.metadata['lastActionSuccess'] == True and (event.metadata['agent']['rotation']['y']) == 90.0:
                    x+=12
                    if g < 255:
                        g +=30
                    else:
                        b +=30
                    if b > 255:
                        b = 0
                elif event.metadata['lastActionSuccess'] == True and (event.metadata['agent']['rotation']['y']) == 180.0:
                    y+=17
                elif event.metadata['lastActionSuccess'] == True and (event.metadata['agent']['rotation']['y']) == 270.0:
                    x-=12
                elif event.metadata['lastActionSuccess'] == True and (event.metadata['agent']['rotation']['y']) == 0.0:
                    y-=17        
                cv2.circle(image, (x, y), 3, (r,g,b), -1)
                display_image(image)
            elif key.char == 's':
                event = controller.step(dict(action ='MoveBack'))
                if event.metadata['lastActionSuccess'] == True and (event.metadata['agent']['rotation']['y']) == 90.0:
                    x-=12
                elif event.metadata['lastActionSuccess'] == True and (event.metadata['agent']['rotation']['y']) == 180.0:
                    y-=17
                elif event.metadata['lastActionSuccess'] == True and (event.metadata['agent']['rotation']['y']) == 270.0:
                    x+=12
                elif event.metadata['lastActionSuccess'] == True and (event.metadata['agent']['rotation']['y']) == 0.0:
                    y+=17        
                cv2.circle(image, (x, y), 3, (0,255,0), -1)
                display_image(image)
            elif key.char == 'd':
                event = controller.step(dict(action = 'RotateRight'))
                display_image(image)
            elif key.char == 'a':
                event = controller.step(dict(action = 'RotateLeft'))
                display_image(image)
            elif key.char == 'p':
                display_image(draw_box(event))
            elif key.char == 'c':
                draw_topview2(controller)
                image = cv2.imread("topview2.png")
                display_image(image)
            elif key.char == 'q':
                takePicture(event)
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
        if key == keyboard.Key.esc:
            return False
    
    def draw_topview1(controller):
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
        im = Image.open('topview.png')
        im = trim(im)
        im.save("topview.png")     

    def draw_topview2(controller):
        t = get_agent_map_data(controller)
        new_frame = add_agent_view_triangle(
            position_to_tuple(controller.last_event.metadata["agent"]["position"]),
            controller.last_event.metadata["agent"]["rotation"]["y"],
            t["frame"],
            t["pos_translator"],
        )
        plt.imshow(new_frame)
        plt.axis('off')
        plt.savefig('topview2.png')
        im = Image.open('topview2.png')
        im = trim(im)
        im.save("topview2.png")

    with keyboard.Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()
if __name__ == "__main__":
    save_topview_image(parse_arguments(sys.argv[1:]))
