import cv2
from deepface import DeepFace
from collections import Counter

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
padding=20

import time



""" Video display """

def DisplayVid():
    cap = cv2.VideoCapture('vids/6.mp4')
    ages, genders = [], []
    while True:
        ret, frame = cap.read()
        if ret:
            objs = DeepFace.extract_faces(img_path = frame, 
                detector_backend='yolov8',
            )
            bboxes = [(bbox['x'], bbox['y'], bbox['x'] + bbox['w'], bbox['y'] + bbox['h']) for bbox in [face['facial_area'] for face in objs]]
            for bbox in bboxes:
                face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                ages.append(age)     
                genders.append(gender)           
        else:
            break
    cap.release()
    print(Counter(ages))
    print(Counter(genders))

start = time.time()
DisplayVid()
print(time.time() - start)