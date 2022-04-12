import cv2
import numpy as np

### Idea: Use pretrained YOLO v3 model

### Steps:
# 1. Get frames from the webcam
# 2. Load YOLO v3 model:
# - Collect the names of our classes by using coco.names file
# - Import the model file (the network(configuration) and the trained weights)
# 3. Need to input the actual image to the network. Cannot input the plain image that we're getting from our webcam
# into the network. Because the network only accepts a particular type of format (blob)
# A blob: is an N-dimensional array stored in a C-contiguous fashion
# 4. Find object presented and the corresponding value of bounding box, confidence.
# 5. Apply Non max Suppression to remove overlapping boxes

cap = cv2.VideoCapture(0)
whT = 320  # because we're using yolov3 - 320
confThreshold = 0.5
nmsThreshold = 0.3

classesFile = '../Pretrained Models/coco.names'
classNames = []

with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')  # extract classes based on new lines
print(classNames)
print(len(classNames))

# Load model
modelConfiguration = '../Pretrained Models/yolov3.cfg'
modelWeights = '../Pretrained Models/yolov3.weights'

# Create the network
# dnn => deep neural network
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Declare that we're going to use OpenCV as the backend and the CPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs, img):
    hT, wT, cT = img.shape
    boundingBoxes = []  # contains bx, by, bw, bh
    classIds = []  # contains 80 predicted probabilies of 80 classes
    confs = []  # contains confidence (probabilities of class that satisfies the threshold
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            # Check if that confidence satisfy threshold or not
            if confidence >= confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)  # coor of center point
                boundingBoxes.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    # print(len(boundingBoxes))
    indices = cv2.dnn.NMSBoxes(boundingBoxes, confs, confThreshold, nmsThreshold)  # use NNM to output the indices of
    # bounding boxes # you want to keep
    # print(indices)

    # Loop in the indices to draw the bounding boxes
    for i in indices:
        box = boundingBoxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


while True:
    success, img = cap.read()

    # Set up inputs
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    # print(layerNames)

    # Get the names of output layers
    outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)
    # print(net.getUnconnectedOutLayers())

    # Set forward pass to the network using the above blob and we have the outputs
    # We got 3 different outputs bc of 3 different output layers
    outputs = net.forward(outputNames)
    # print(outputs[0].shape)  # 300x85 (85 represents for predicted probability of each of 80 classes and 5 other params: pc, bx, by, bw, bh)
    # print(outputs[1].shape)  # 1200x85 (300, 1200, 4800 represent for the number of bounding boxes)
    # print(outputs[2].shape)  # 4800x85
    # print(len(outputs[0][0])) # first row's 85 values

    findObjects(outputs, img)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
