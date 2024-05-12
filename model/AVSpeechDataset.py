import os
import torch
import json
import numpy as np
import time

F0_STD = 41.20620875707121
L_STD = 0.03757316168641439

class AVSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, windows, lang_embeddings='data/lang_embeddings.json'):        
        self.root_dir = root_dir
        self.windows = windows
        self.lang_embeddings = json.load(open(lang_embeddings))
        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        # Load the data
        path, start, end = self.windows[idx]
        path = os.path.join(self.root_dir, path)
        pros_path = path + '_pros.npy'
        metadata_path = path + '_feat.json'
        frames = np.load(path + '_frames.npy')
        windowed_frames = frames[start:end, :, :]
        json_data = json.load(open(metadata_path))
        metadata_embd = self.lang_embeddings[json_data['lang']].copy()
        
        # Make metadata
        age = json_data['age']/80
        gender = [json_data['gender']['Woman']/100, json_data['gender']['Man']/100]
        race = [json_data['race'][r]/100 for r in ("asian", "indian", "black", "white", "middle eastern", "latino hispanic")]
        metadata_embd.append(age)
        metadata_embd.extend(gender)
        metadata_embd.extend(race)
        metadata_embd = torch.tensor(metadata_embd)
        
        #Prepare target
        target = np.load(pros_path)[:, start:end]
        target = torch.tensor(target).T
        target[torch.isnan(target)] = 0
        imgs = windowed_frames / 255.
        imgs = torch.tensor(imgs).permute(3,0,1,2)
        target_f0, target_voiced, target_amp = target[:,0], target[:,1], target[:,2]
        target_f0 -= (torch.mean(target_f0) * 90 / (target_f0 != 0).sum()).unsqueeze(0)
        target_amp -= (torch.mean(target_amp) * 90 / (target_amp != 0).sum()).unsqueeze(0)
        target_f0 /= F0_STD
        target_amp /= L_STD
        target = torch.stack([target_f0, target_voiced, target_amp], dim=1)
        target[torch.isnan(target)] = 0
        return imgs.type(torch.FloatTensor), target.type(torch.FloatTensor), metadata_embd.type(torch.FloatTensor)