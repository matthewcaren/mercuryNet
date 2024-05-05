import json
import os
from tqdm import tqdm

root_dir = 'vids'
for vid_dir in tqdm(os.listdir(root_dir)):
    try:
        json.load(open(f'{root_dir}/{vid_dir}/{vid_dir}_feat.json'))
    except:
        facial_attributes = {'race': 0, 'gender': 0, 'emotion': 0, 'age': 0, 'lang': 'nolang'}
        json.dump(facial_attributes, open(f'{root_dir}/{vid_dir}/{vid_dir}_feat.json', 'w'))