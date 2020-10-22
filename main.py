import argparse
import cv2
import numpy as np
from detect_hand import detectHandByDistanceTrans
from utils import get_skin_mask, euclidean_distance, get_centroid, get_id_location_objects, getMask
#tracking library
import norfair
from sort import Sort
from norfair import Detection, Tracker, Video

#sort tracker
sort_tracker = Sort(max_age=50, min_hits=2, iou_threshold=0.1)
ids = []

norfair_tracker = Tracker(
        distance_function=euclidean_distance,
        distance_threshold=100
    )


def testVideo(video_path, tracker, write=False):
    cap = cv2.VideoCapture(video_path)
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    if write:
        videoWriter = cv2.VideoWriter('detect.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width*2,frame_height))
   
    to_show = np.zeros((frame_height, frame_width*2, 3), dtype=np.uint8)
    
    import time
    count = 0
    t = time.time()
    fps = 0
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if ret == False:
            break

        mask = get_skin_mask(frame)
        mask_d, hand_locations = detectHandByDistanceTrans(mask, min_size_hand=40, detect_two_hand=True)
       
        if tracker == 'nofair': 
            detections = [
                Detection(get_centroid(box), data=box)
                for box in hand_locations
            ]

            tracked_objects = norfair_tracker.update(detections=detections) 
            # norfair.draw_points(mask_detect, detections)
            norfair.draw_tracked_objects(mask_d, tracked_objects, id_size=1, id_thickness=2, draw_points=True, radius=3)
            # list_obj = get_id_location_objects(tracked_objects)
        else:
             #sort tracking method
            sort_bb = []
            for bbox in hand_locations:
                x,y,w,h = bbox
                sort_bb.append([x,y,x+w,y+h])
            predict=sort_tracker.update(np.array(sort_bb))
            for pre in predict:
                x1, y1, x2, y2,id=int(pre[0]),int(pre[1]),int(pre[2]),int(pre[3]),int(pre[4])
                # print('predict: ', (x1, y1, x2, y2))
                if id not in ids:
                    ids.append(id)

                cv2.putText(mask_d, str(id), (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX ,  
                        1, (0,100,255) , 2, cv2.LINE_AA) 
                cv2.rectangle(mask_d,(x1,y1) ,(x2,y2), (255,255,0), 2)


        count = (count + 1) % 10
        if count == 0:
            fps = 10 / (time.time() - t)
            t = time.time()
        
        cv2.putText(mask_d, 'FPS: ' + str(int(fps)), (7, 40), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 255, 0), 2, cv2.LINE_AA) 


        to_show[:,0:frame_width] = frame
        to_show[:,frame_width:2*frame_width] = mask_d

        # sth = cv2.addWeighted(frame, 0.7, mask_d, 0.3, 1)
        
        # cv2.imshow('detect', sth)
        cv2.imshow('Detect', to_show)

        key = cv2.waitKey(1)
        
        if write:
            videoWriter.write(to_show)
        
        if key == 27:
            break
        elif key == ord('p'):
            cv2.waitKey(0)
    cap.release()   
    # videoWriter.release()

def testMultiImage(folder_path):
    import glob
    import imutils
    files = glob.glob(folder_path + '*.jpg')
    
    
    i = 0
    while i < len(files):
        src = cv2.imread(files[i])
        src = imutils.resize(src, 640)
        # print(src.shape)
        #if mask image
        arm_mask = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # arm_mask = get_skin_mask(src)
        # cv2.imshow('src', arm_mask)

        mask_d, _ = detectHandByDistanceTrans(arm_mask, min_size_hand=25)
        # cv2.imshow('sec', dis_trans)
        # cv2.imshow('detect', mask_d)
        
        w = src.shape[1]
        h = src.shape[0]
        to_show = np.zeros((h, w*2, 3), dtype=np.uint8)

        mask = get_skin_mask(src)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        to_show[:,0:w] = src
        to_show[:,w:2*w] = mask_d
        # cv2.imwrite('stick_hand.jpg', to_save)
        cv2.imshow('maskk', to_show)
        # videoWriter.write(to_save)
    
        key = cv2.waitKey(30)
        print(key)
        if key == 27:
            break
        
        if key == ord('p') or key == ord('P'):
            cv2.waitKey(0)
        elif key == 81:
            i = i-1
        else:
            i = i+1

        print('image number =  ', i)
        



parser = argparse.ArgumentParser()
parser.add_argument('--video_path', type=str, default="", help='Path the video to test')
parser.add_argument("--tracker", type=str, default="nofair", help="Video files to process")
parser.add_argument("--write", type=bool, default=False, help="Save video result")
parser.add_argument("--detect_two_hand", type=bool, default=True, help="Detect two hands")

args = parser.parse_args()


# testVideo(args.video_path, args.tracker, args.write)
# testVideo('data/v13.mp4')
# testVideo(0, 'sort')

testMultiImage('data/hands/')
