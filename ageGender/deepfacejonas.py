import cv2
from deepface import DeepFace
import time
start = time.time()
cap = cv2.VideoCapture('vids/0.mp4')
count = 0

while(cap.isOpened()): 
    ret, frame = cap.read() 
    if ret == True: 
        count += 1
        start = time.time()
        objs = DeepFace.extract_faces(img_path = frame, 
                detector_backend='yolov8',
            )
        print(time.time() - start)
    else:
        break

cap.release() 
cv2.destroyAllWindows() 
print(time.time() - start)
