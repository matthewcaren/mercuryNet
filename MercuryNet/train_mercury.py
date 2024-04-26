import torch
from model.model import MercuryNet, MercuryNetLoss
from tqdm import tqdm
import torch.nn.functional as F
from datetime import datetime
import os
from torchvision import transforms
import cv2
import numpy as np
from torch.utils.data import DataLoader

class AVSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, overlap=30, window_size=90):        
        self.all_paths = []
        self.all_pros = []
        self.windows = []
        self.overlap = overlap
        self.window_size = window_size

        directories = [dir for dir in os.listdir(root_dir) if dir[0] != '.']
        for dir in directories:
            images = [os.path.join(root_dir, dir,d) for d
                      in os.listdir(os.path.join(root_dir, dir)) 
                      if d.endswith('.jpg')]
            self.all_paths.append(images)

        for paths in self.all_paths:
            vid_windows = self.get_windows(len(paths))
            for window in vid_windows:
                self.windows.append((paths, window))

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
        paths, window = self.windows[idx]
        paths = sorted(paths, key = lambda x: int(x.split('_')[-1].split('.')[0]))
        windowed_paths = paths[window[0]:window[1]]
        pros_path = '_'.join(paths[0].split('_')[:-1])+'_pros.npy'
        target = np.load(pros_path)[:, window[0]:window[1]]
        target = torch.tensor(target).T.type(torch.FloatTensor)
        sz = (96, 96)
        imgs = [cv2.resize(cv2.imread(filename), sz) for filename in windowed_paths]
        imgs = np.asarray(imgs) / 255.
        imgs = torch.tensor(imgs).permute(3,0,1,2)
        return imgs, target


def train(model, dataloader, optimizer, epochs):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    model.train()
    train_loss = []

    loss_func = MercuryNetLoss()
    batches = tqdm(enumerate(dataloader), total=len(dataloader))
    
    for epoch in range(epochs):
        print("starting epoch", epoch)
        for batch_idx, (data, target) in batches:
            data_mps = data.to(device)
            target_mps = target.to(device)
            optimizer.zero_grad()

            output = model(data_mps)

            loss = loss_func(output, target_mps)
            
            loss.backward()
            optimizer.step()

            print("loss:", loss)

    checkpoint_path = f"./checkpoints/ckpt_{datetime.today().strftime('%d_%H-%M')}.pt"
    
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": epochs,
        },
        checkpoint_path,
    )

    return train_loss


# # do some training!
# model = MercuryNet()
# train_dataset = AVSpeechDataset('../vids_10')
# dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# optim = torch.optim.Adam(model.parameters())

# train(model, dataloader, optim, 4)