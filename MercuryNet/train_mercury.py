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
    def __init__(self, root_dir, transform=None):        
        directories = [dir for dir in os.listdir(root_dir) if dir[0] != '.']
        self.transform = transform
        self.all_paths = []

        for dir in directories:
            images = [os.path.join(root_dir, dir,d) for d
                      in os.listdir(os.path.join(root_dir, dir)) 
                      if (os.path.isfile(os.path.join(root_dir, dir, d)) and d[-4:]=='.jpg')]
            self.all_paths.append(images)

    def __len__(self):
        return len(self.all_paths)
    
    def __getitem__(self, idx):
        paths = self.all_paths[idx]
        imgs = []
        sz = (96, 96)
        for filename in paths:
            img = cv2.imread(filename)
            img = cv2.resize(img, sz)
            imgs.append(img)
        imgs = np.asarray(imgs) / 255.
        imgs = torch.tensor(imgs)
        if self.transform is not None:
            imgs = self.transform(imgs)
        return imgs


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

    checkpoint_path = f"../checkpoints/ckpt_{datetime.today().strftime('%d_%H-%M')}.pt"
    
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": epochs,
        },
        checkpoint_path,
    )

    return train_loss


# do some training!
model = MercuryNet()
train_dataset = AVSpeechDataset('../vids_10', None)
dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
optim = torch.optim.Adam(model.parameters())

train(model, dataloader, optim, 4)