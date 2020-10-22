import cv2
import numpy as np
from math import sqrt

min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([235,173,127],np.uint8)

def get_skin_mask(image):
    image = cv2.GaussianBlur(image, (7,7), 0)
    # image = correctIllumination(image) 
    # Get pointer to video frames from primary device

    imageYCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
    # cv2.imshow('detect', skinRegionYCrCb)

    # skinYC
    # rCb = cv2.bitwise_and(image, image, mask = skinRegionYCrCb)
    return skinRegionYCrCb

def getMask(img):
    ## convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ## mask of green (36,0,0) ~ (70, 255,255)
    mask = cv2.inRange(hsv, (36, 0, 0), (70, 255,255))
    mask = cv2.bitwise_not(mask, mask)
    return mask

def distanceTwoPoint(x1, y1, x2, y2):
    return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

#norfair
def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

def get_id_location_objects(objects):
    list_objects = [] 
    for obj in objects:
        id = obj.id
        for point, live in zip(obj.estimate, obj.live_points):
            if live:
                list_objects.append((id, point))
    return list_objects

def get_centroid(box): #x,y,w,h
    x,y,w,h = box
    return np.array([x+ w/2, y + h/2])

