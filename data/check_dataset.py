import json
import os
import numpy as np

root_dir = './vids_2'
for dir in os.listdir(root_dir):
    failed = False
    if dir[0] != '.':
        path = f'{root_dir}/{dir}/{dir}'
        json_data = json.load(open(path + '_feat.json'))
        if json_data['race'] == 0:
            failed = True
            #print('Oh no json')
        frames = np.load(path + '_frames.npy')
        if len(frames) == 0:
            failed = True
            #print('Oh no frames')
        if len(frames.shape) != 3:
            failed = True
           # print('Oh no frames shape')
        pros = np.load(path + '_pros.npy')
        if pros.shape[0] != 3:
            failed = True
          #  print("oh no pros")
        if pros.shape[1] < 2:
            failed = True
         #  print('oh no pros len')
        if failed == False:
            print('Wow')
    
    