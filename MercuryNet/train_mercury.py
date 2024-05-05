import torch
from model.model import MercuryNet, MercuryNetLoss
from tqdm import tqdm
from datetime import datetime
import os
import numpy as np
from torch.utils.data import DataLoader
import json

class AVSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, directories, overlap=30, window_size=90):        
        self.all_paths = []
        self.all_pros = []
        self.windows = []
        self.overlap = overlap
        self.window_size = window_size

        for vid_dir in directories:
            images = [os.path.join(root_dir, vid_dir,d) for d
                      in os.listdir(os.path.join(root_dir, vid_dir)) 
                      if d.endswith('.jpg')]
            self.all_paths.append(images)

        for paths in self.all_paths:
            vid_windows = self.get_windows(len(paths))
            for window in vid_windows:
                self.windows.append((paths, window))
        
        for vid_dir in directories:
            frames = np.load(f'{root_dir}/{vid_dir}/{vid_dir}_frames.npy')
            num_frames = frames.shape[0]
            vid_windows = self.get_windows(num_frames)
            for window in vid_windows:
                self.windows.append((f'{root_dir}/{vid_dir}/{vid_dir}', window))

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
        target = torch.tensor(target).T.type(torch.FloatTensor)
        
        imgs = windowed_frames / 255.
        imgs = torch.tensor(imgs).permute(3,0,1,2)
        return imgs, target, metadata_embd


def train(model, train_dataloader, val_dataloader, optimizer, epochs):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    loss_func = MercuryNetLoss()

    for epoch in range(epochs):
        print("\nstarting epoch", epoch)
        train_batches = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        val_batches = enumerate(val_dataloader)
        model.train()
        for batch_idx, (data, target, metadata_embd) in train_batches:
            
            data_mps = data.to(device)
            target_mps = target.to(device)
            metadata_embd_mps = metadata_embd.to(device)
            optimizer.zero_grad()

            output = model(data_mps, metadata_embd_mps)

            train_loss = loss_func(output, target_mps)

            train_loss.backward()
            optimizer.step()
        print("train loss:", train_loss)
        model.eval()

        for batch_idx, (data, target, metadata_embd) in val_batches:
            data_mps = data.to(device)
            target_mps = target.to(device)
            metadata_embd_mps = metadata_embd.to(device)

            output = model(data_mps, metadata_embd_mps)
            val_loss = loss_func(output, target_mps)
        print("val loss:", val_loss)

    checkpoint_path = f"/MercuryNet/checkpoints/ckpt_{datetime.today().strftime('%d_%H-%M')}.pt"
    
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": epochs,
        },
        checkpoint_path,
    )


def test(model, test_dataloader):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    loss_func = MercuryNetLoss()

    test_batches = enumerate(test_dataloader)
    for batch_idx, (data, target, metadata_embd) in test_batches:
        data_mps = data.to(device)
        target_mps = target.to(device)
        metadata_embd_mps = metadata_embd.to(device)

        output = model(data_mps, metadata_embd_mps)
        test_loss = loss_func(output, target_mps)

    print("test loss:", test_loss)

def segment_data(root_dir, desired_datause):
    all_directories = [vid_dir for vid_dir in os.listdir(root_dir) if vid_dir[0] != '.']
    actual_datacount = min(len(all_directories), desired_datause)
    train_count = int(actual_datacount*0.7)
    val_count = int(actual_datacount*0.15)
    test_count = int(actual_datacount*0.15)
    total_count = train_count + val_count + test_count
    all_data = np.random.choice(all_directories, total_count, replace=False)
    train_data = all_data[:train_count]
    val_data = all_data[train_count:train_count+val_count]
    test_data = all_data[train_count+val_count:]
    return train_data, val_data, test_data

def run_training_pass(root_dir, data_count=100, epochs=8, batch_size=16):
    model = MercuryNet()

    train_data, val_data, test_data = segment_data(root_dir, data_count)
    train_dataset = AVSpeechDataset(root_dir, train_data)
    val_dataset = AVSpeechDataset(root_dir, val_data)
    test_dataset = AVSpeechDataset(root_dir, test_data)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    optim = torch.optim.Adam(model.parameters())
    train(model, train_dataloader, val_dataloader, optim, epochs=epochs)
    test(model, test_dataloader)



run_training_pass('vids_2', batch_size=8, epochs=16)
