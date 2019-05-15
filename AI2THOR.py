import ai2thor.controller
import numpy as np
import keyboard
import time
import random
import cv2
import os
player_size = 500
controller = ai2thor.controller.Controller()
controller.start(player_screen_width=player_size * 1.5, player_screen_height=player_size)
event = controller.step(dict(action='Initialize', gridSize=0.25, renderObjectImage = False))

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
def pickUp(event):
        for o in event.metadata['objects']:
            if o['visible'] and o['pickupable']:
                event = controller.step(dict(action='PickupObject', objectId=o['objectId']), raise_for_failure=True)
                object_id = o['objectId']
                break
rotate = 0	
while 1:
	#cv2.imshow('image', event.cv2img)
	#cv2.waitKey(1)
	if keyboard.is_pressed('right'):
		rotate +=15
		event = controller.step(dict(action='Rotate',rotation=rotate))
		#time.sleep(0.1)
	elif keyboard.is_pressed('left'):
		rotate -=15
		event = controller.step(dict(action='Rotate',rotation=rotate))
		time.sleep(0.1)
	elif keyboard.is_pressed('up'):
		event = controller.step(dict(action='LookUp'))
		time.sleep(0.1)
	elif keyboard.is_pressed('down'):
		event = controller.step(dict(action='LookDown'))
		time.sleep(0.1)
	elif keyboard.is_pressed('w'):		
		event = controller.step(dict(action='MoveAhead'))
		time.sleep(0.05)
	elif keyboard.is_pressed('s'):
		event = controller.step(dict(action='MoveBack'))		
		time.sleep(0.05)
	elif keyboard.is_pressed('d'):
		time.sleep(0.05)
		event = controller.step(dict(action='MoveRight'))
	elif keyboard.is_pressed('a'):
		time.sleep(0.05)
		event = controller.step(dict(action='MoveLeft'))
	elif keyboard.is_pressed('f'):
		takePicture(event)
	elif keyboard.is_pressed('c'):
		time.sleep(0.1)
		pickUp(event)
	elif keyboard.is_pressed('v'):
		event = controller.step(dict(action='DropHandObject'))
	elif keyboard.is_pressed('q'):
		time.sleep(0.05)
		controller.reset('FloorPlan'+ str(random.randint(1, 30)))
		controller.step(dict(action='Initialize', gridSize=0.25))
		controller.step(dict(action = 'InitialRandomSpawn', randomSeed = 0, forceVisible = False, maxNumRepeats = 5))
	elif keyboard.is_pressed('esc'):
		#event.stop()
		break