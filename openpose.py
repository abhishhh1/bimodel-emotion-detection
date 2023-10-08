import cv2
import time
import numpy as np

protoFile = "coco/pose_deploy_linevec.prototxt"
weightsFile = "coco/pose_iter_440000.caffemodel"
nPoints = 18
POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

def makeSkelton(frame):

    #frame = cv2.imread(frame)
    frame = cv2.resize(frame,(640, 960))
    frameCopy = cv2.imread('black.jpg',cv2.IMREAD_COLOR)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    frameCopy = cv2.resize(frameCopy,(frameWidth, frameHeight))
    threshold = 0.1

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    #print("Using CPU device")

    t = time.time()
    # input image dimensions for the network
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),(0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold :
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frameCopy, points[partA], points[partB], (0, 255, 255), 8)
            cv2.circle(frameCopy, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    #cv2.imshow('Output-Skeleton.jpg', frame)
    cv2.imwrite('intermediate/Output-Skeleton.jpg', frameCopy)
    return frameCopy

    #print("Total time taken : {:.3f}".format(time.time() - t))
    cv2.waitKey(0)
    
#makeSkelton('input/image.jpeg')
    
