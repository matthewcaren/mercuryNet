import cv2
from deepface import DeepFace
import time
import os
import pandas as pd
import json
import numpy as np
import shutil
from tqdm import tqdm

def get_center(region):
    return int(region['x'] + 0.5*region['w']), int(region['y'] + 0.5*region['h'])

data = pd.read_csv('./data/avspeech_test_langs.csv')
dirs = os.listdir('./vids_2')
to_remove = []
counter = 0
for i in (range(len(dirs))):
    dir = dirs[i]
    # if i%100 == 0:
    #     print(dir)
    if dir[0] != '.':
        try:
            start = time.time()
            json_data = json.load(open(f'vids_2/{dir}/{dir}_feat.json'))
            print(time.time() - start)
            if json_data['race'] == 0:
                counter += 1
        except:
             print('wtf')

print(counter)
#             cap = cv2.VideoCapture(f'vids/{dir}/{dir}.mp4')
#             if not cap.isOpened():
#                 to_remove.append(dir)
#             else:
#                 correct_box = {'race': None, 'age': None, 'gender': None, 'emotion': None}
#                 vid = '_'.join(dir.split('_')[:-1])
#                 start_time = float(dir.split('_')[-1])
#                 row = data[(data['ID'] == vid) & (data['s'] == start_time)]
#                 ex_x, ex_y = row.iloc[0]['x'], row.iloc[0]['y']
#                 json_data = json.load(open(f'vids/{dir}/{dir}_feat.json'))
#                 lang = json_data['lang']
#                 while(cap.isOpened()): 
#                     ret, frame = cap.read() 
#                     if ret == True: 
#                         shape_x, shape_y = frame.shape[1], frame.shape[0]
#                         head_center = np.array([shape_x*ex_x, shape_y*ex_y])
#                         objs = DeepFace.analyze(img_path = frame, 
#                                 detector_backend='yolov8',
#                                 actions = ['age', 'gender', 'race', 'emotion'],
#                                 enforce_detection=False,
#                                 silent=True
#                             )
#                         centers = [np.array(get_center(obj['region'])) for obj in objs]
#                         dists = [np.linalg.norm(center - head_center) for center in centers]
#                         if len(dists) == 0:
#                             pass
#                         elif (min(dists) < 20): 
#                             if (len(dists) == 1) or (dists[np.argsort(dists)[1]] > 3*min(dists)):
#                                 correct_box = objs[np.argmin(dists)]
#                                 break
#                     else:
#                         break
#                 facial_attributes = {'race': correct_box['race'], 'gender': correct_box['gender'], 'emotion': correct_box['emotion'], 'age': correct_box['age'], 'lang': lang}
#                 json.dump(facial_attributes, open(f'./vids/{dir}/{dir}_feat.json', 'w'))
        
#             cap.release() 
#             cv2.destroyAllWindows() 
#         except Exception as e:
#             print(e)
#             print("Something failed for", dir)

# print(to_remove)
# json.dump(to_remove, open('to_remove.json', 'w'))
