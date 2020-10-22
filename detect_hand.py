import cv2
import numpy as np
from utils import  distanceTwoPoint
import random

def findContours(mask):
    if cv2.getVersionMajor() in [2, 4]:
        # OpenCV 2, OpenCV 4 case
        cnts, _ = cv2.findContours(mask, cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_NONE)
    else:
        # OpenCV 3 case
        _, cnts, _ = cv2.findContours(mask, cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_NONE)
    return cnts

#mask is bibary image
#return: - location of brightest
#        - radius
#        - image of distance transform
def getDistanceTransform(mask):
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    r = np.amax(dist)  
    indices = np.where(dist == r)
    y,x = indices[0][0], indices[1][0]
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    return (x,y), r, dist

def getForearmCener(mask, y_location_of_hand):
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    r = np.amax(dist[y_location_of_hand:dist.shape[0], :])  
    indices = np.where(dist[y_location_of_hand:dist.shape[0], :] == r)
    y,x = indices[0][0], indices[1][0]
    y = y + y_location_of_hand
    return (x,y), r

#return location of palm and radius
def getPalmCenter(mask, thresh, k_dis_to_center = 0.2, k_radius = 1):
    # print('Accept min size hand = ', thresh)
    maskcp = np.copy(mask)
    x_res = 0
    y_res = 0
    r_res = 0
    x_center = mask.shape[1]/2
    y_center = mask.shape[0]/2
    # i = 0
    while True:
        dist = cv2.distanceTransform(maskcp, cv2.DIST_L2, 3)
        r = np.amax(dist)  
        if r >= thresh:
            indices = np.where(dist == r)
            y,x = indices[0][0], indices[1][0]
            #neu gan tam anh hon
            # print('find',(x,y), r)
            dis1 = distanceTwoPoint(x, y, x_center, y_center)
            dis2 = distanceTwoPoint(x_res, y_res, x_center, y_center)
            
            # print('dis', k_dis_to_center*dis1)
            # print('rrr', k_radius*r)
            # print('add', k_radius*r - k_dis_to_center*dis1)
            
            if k_radius*r - k_dis_to_center*dis1 > k_radius*r_res - k_dis_to_center*dis2:
                x_res = x
                y_res = y
                r_res = r
                # print('updated!')
            # print('res',(x_res,y_res), r_res)
            # print('-------------------------')
            cv2.circle(maskcp, (x,y), int(1.3*r), (0,0,0),-1)
            # cv2.imshow('maskCP', maskcp)
            # cv2.waitKey(0)
        else:
            break
    # print('(x,y), R: ',(x_res,y_res), r_res)
    return (x_res, y_res), r_res

#input image 
#return: mask of hand and mask of forearm
def hand_mask_segmentation(gray, min_size_hand):
    mask = np.zeros(gray.shape, dtype=np.uint8)
    x_centroid = int(mask.shape[1]/2)
    y_centroid = int(mask.shape[0]/2)

    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cnts = findContours(thresh)
  
    if (len(cnts) != 0):
        maxCnt = max(cnts, key = cv2.contourArea)
        cv2.drawContours(mask, [maxCnt], 0, (255,255,255), -1)
        cv2.drawContours(gray, [maxCnt], 0, (0,0,0), -1)
    else:
        return None, None
   
    cv2.imshow('mask for detect', mask)
    #mask with only one hand (biggest contour)
    org_mask = np.copy(mask)

    (x_palm,y_palm), r_palm = getPalmCenter(mask, min_size_hand)
    cv2.circle(mask, (x_palm,y_palm), int(1.5*r_palm), (0,0,0),-1)
    
    maskcp = np.copy(mask)
    cv2.circle(maskcp, (x_centroid, y_centroid), 
                        int(distanceTwoPoint(x_centroid, y_centroid, x_palm, y_palm)), (0,0,0), -1)
    (x,y), r = getForearmCener(maskcp, 0)

    if (r > r_palm/3.0): #forearm
        cnts = findContours(mask)
        for cnt in cnts:
            d = cv2.pointPolygonTest(cnt, (x,y), True) #check if (X,Y) inside the contour
            if (d < 0): #is not the forearm
                cv2.drawContours(mask, [cnt], 0, (0,0,0), -1)
    else:
        # return only hand, forarm is None
        return org_mask, None

    org_mask[mask > 0] = 0 #org_mask is mask of hand, mask is mask of forearm

    return org_mask, mask


# input gray image (mask of arm)
def detectHandByDistanceTrans(mask, min_size_hand, detect_two_hand=True):
    hand_palm_locations = []
    
    mask_detect = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    #after this function, one hand was removed from gray
    hand_mask_1, forearm_mask_1 = hand_mask_segmentation(mask, min_size_hand)

    if hand_mask_1 is None: #if can not find a hand
        return np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8), hand_palm_locations

    #for showing
    mask_detect[hand_mask_1 > 0] = (0,255,0)
    if forearm_mask_1 is not None:
        mask_detect[forearm_mask_1 > 0] = (179,250,255)

    cnts = findContours(hand_mask_1)
    if (len(cnts)) > 0:
        # x,y,w,h = cv2.boundingRect(cnts[0])
        hand_palm_locations.append(cv2.boundingRect(cnts[0]))
    
    if detect_two_hand:
        #find hand2
        hand_mask_2, forearm_mask_2 = hand_mask_segmentation(mask, min_size_hand)

        if hand_mask_2 is not None: #if can not find a hand
            mask_detect[hand_mask_2 > 0] = (0,255,0)
            
            if forearm_mask_2 is not None:
                mask_detect[forearm_mask_2 > 0] = (179,250,255)

            cnts = findContours(hand_mask_2)
            if (len(cnts)) > 0:
                hand_palm_locations.append(cv2.boundingRect(cnts[0]))
    
    # print(hand_palm_locations)
    return mask_detect, hand_palm_locations
