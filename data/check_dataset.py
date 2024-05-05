import json
import os
import numpy as np
import shutil
from tqdm import tqdm

root_dir = 'vids'
still_to_go = []
import cv2

for vid_dir in tqdm(os.listdir(root_dir)):
    failed = False
    json_fail = False
    if vid_dir[0] != '.':
        if len(os.listdir(f'{root_dir}/{vid_dir}')) != 5:
            print("Wrong length")
            failed =True
               
        path = f'{root_dir}/{vid_dir}/{vid_dir}'
        json_data = json.load(open(path + '_feat.json'))
        if json_data['race'] == 0 or len(json_data['race']) != 6 or len(json_data['gender']) != 2 or type(json_data['lang']) != str:
            failed = True
            json_fail = True
            print('Oh no json')
        frames = np.load(path + '_frames.npy')
        if len(frames) == 0:
            failed = True
            print('Oh no frames')
        if len(frames.shape) != 4:
            print(frames.shape, vid_dir)
            failed = True
            print('Oh no frames shape')
        pros = np.load(path + '_pros.npy')
        if pros.shape[0] != 3:
            failed = True
            print("oh no pros")
        if pros.shape[1] < 2:
            failed = True
            print('oh no pros len')
        if abs(pros.shape[1] - frames.shape[0]) > 5:
            failed = True
            print('Uneven lengths')
            print(pros.shape[1] - frames.shape[0])
            

        # fpath = f'./{root_dir}/{vid_dir}/{vid_dir}.mp4'
        # cap = cv2.VideoCapture(fpath)
        # frame_counter = 0
        # while(cap.isOpened()): 
        #     ret, frame = cap.read() 
        #     if ret == True:
        #         frame_counter += 1
        #     else:
        #         break
        # cap.release() 
        # cv2.destroyAllWindows() 
        # if frame_counter != frames.shape[0]:
        #     print("Frame lenght mismatch")
        #     print(frame_counter, frames.shape[0], vid_dir)
        if failed:
            still_to_go.append(vid_dir)

print(len(still_to_go))

