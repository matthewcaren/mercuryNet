import os
import torch
import json
import numpy as np
import time

class AVSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, windows, overlap=30, window_size=90):        
        self.windows = windows
        self.overlap = overlap
        self.window_size = window_size
        self.lang_embeddings = json.load(open('data/lang_embeddings.json'))

    def get_windows(self, num_images):
        num_windows = (num_images - self.overlap) // (self.window_size - self.overlap)
        num_frames_in_window = num_windows*(self.window_size - self.overlap) + self.overlap
        amount_to_chop_front = (num_images - num_frames_in_window) // 2
        windows = []
        for i in range(num_windows):
            start = i*(self.window_size - self.overlap) + amount_to_chop_front
            windows.append([start, start + self.window_size])
        return windows

    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        path, window = self.windows[idx]
        pros_path = path + '_pros.npy'
        metadata_path = path + '_feat.json'
        frames = np.load(path + '_frames.npy')
        windowed_frames = frames[window[0]:window[1], :, :]
        json_data = json.load(open(metadata_path))
        metadata_embd = self.lang_embeddings[json_data['lang']].copy()
        age = json_data['age']/80
        gender = [json_data['gender']['Woman']/100, json_data['gender']['Man']/100]
        race = [json_data['race'][r]/100 for r in ("asian", "indian", "black", "white", "middle eastern", "latino hispanic")]
        metadata_embd.append(age)
        metadata_embd.extend(gender)
        metadata_embd.extend(race)
        metadata_embd = torch.tensor(metadata_embd)

        target = np.load(pros_path)[:, window[0]:window[1]]
        if target.shape[1] != 90:
            target = np.concatenate([target, target[:, -1:]], axis=1)
        target = torch.tensor(target).T.type(torch.FloatTensor)
        
        imgs = windowed_frames / 255.
        imgs = torch.tensor(imgs).permute(3,0,1,2)
        return imgs, target, metadata_embd