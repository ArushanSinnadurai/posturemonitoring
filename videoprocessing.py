import cv2
import dlib
import numpy as np
import math
import os
from urllib.request import urlretrieve
import time
from datetime import datetime

# Website server
url = "http://192.168.0.147/picture"

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

font = cv2.FONT_HERSHEY_SIMPLEX 

# read the image
cap = cv2.VideoCapture(0)

ret, img = cap.read()
size = img.shape

# Camera internals
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )


def getTimeStamp():
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S.%f)")
    print("timestamp =", timestampStr)
    return timestampStr

if __name__ == '__main__':

    parent_path = os.getcwd()
    print(parent_path)

    path = os.path.join(parent_path, "Pictures")
    if not os.path.exists(path):
        print("Session Creation created at the directory %s" %path)
        os.makedirs(path)
    else:
        print("%s is already created" %path)
    os.chdir(path)

    while True:
        
        image_Str = "image_" + str(getTimeStamp()) + ".jpg"
        urlretrieve(url,image_Str) 
        img = cv2.imread(image_Str)
        
        # Convert image into grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Use detector to find landmarks
        faces = detector(gray)

        for face in faces:
            x1 = face.left()  # left point
            y1 = face.top()  # top point
            x2 = face.right()  # right point
            y2 = face.bottom()  # bottom point

            # Create landmark object
            landmarks = predictor(image=gray, box=face)

            image_points = np.array([
                            (landmarks.part(30).x,landmarks.part(30).y),     # Nose tip
                            (landmarks.part(8).x,landmarks.part(8).y),      # Chin
                            (landmarks.part(36).x,landmarks.part(36).y),     # Left eye left corner
                            (landmarks.part(45).x,landmarks.part(45).y),     # Right eye right corne
                            (landmarks.part(48).x,landmarks.part(48).y),     # Left Mouth corner
                            (landmarks.part(54).x,landmarks.part(54).y)      # Right mouth corner
                        ], dtype="double")

            dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
        
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                
            for p in image_points:
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            
            p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            cv2.line(img, p1, p2, (0, 255, 255), 2)

            try:
                m = (p2[1] - p1[1])/(p2[0] - p1[0])
                ang1 = int(math.degrees(math.atan(m)))
            except:
                ang1 = 90

            cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
        # show the image
        cv2.imshow(winname="Face", mat=img)

        # Exit when escape is pressed
        key = cv2.waitKey(1000) & 0xff
        if key == ord("q"):
            break

# When everything done, release the video capture and video write objects
cap.release()

# Close all windows
cv2.destroyAllWindows()
