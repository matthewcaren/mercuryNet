import json
import os
from tqdm import tqdm

for dir in tqdm(os.listdir('./vids')):
    try:
        json.load(open(f'./vids/{dir}/{dir}_feat.json'))
    except:
        facial_attributes = {'race': 0, 'gender': 0, 'emotion': 0, 'age': 0, 'lang': 'nolang'}
        json.dump(facial_attributes, open(f'./vids/{dir}/{dir}_feat.json', 'w'))