import cv2 

counter = 0
cap = cv2.VideoCapture(0)
print("[INFO] starting video stream...")

tracker = cv2.TrackerMOSSE_create()


def drawbox(img,bbox):
    x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(img,(x,y),((x+w),(y+h)),(255,0,255),3,1)
    cv2.putText(frame, "moving", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

while(True):
    #refresh 1 msec
    key = cv2.waitKey(1) & 0xFF
    success, frame = cap.read()

    if key == ord("s"):
        bbox = cv2.selectROI("Tracking", frame, False)
        tracker.init(frame,bbox)
    
    
    success,bbox = tracker.update(frame)
    if(success):
        drawbox(frame, bbox)

    else:
        cv2.putText(frame, "Not tracking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow('Tracking', frame)

    #press to close the window
    if key == ord("q"):
        print('[INFO] Stopping the video')
        break