import torch
from model.model import MercuryNet, MercuryNetLoss
import tqdm
import torch.nn.functional as F
from datetime import datetime
import os


class AVSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):        
        directories = [os.listdir(root_dir)]

        self.all_paths = []

        for dir in directories:
            images = [d for d
                      in os.listdir(os.path.join(root_dir, dir))
                      if (os.path.isfile(os.path.join(root_dir, dir, d)) and d[-4:]=='.jpg')]
            self.all_paths.append([images])
        
        


def train(dataloader, optimizer, epochs):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    model = MercuryNet()
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

    checkpoint_path = f"../checkpoints/ckpt_{datetime.today().strftime("%Y-%m-%d_%H-%M")}.pt"
    
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": epochs,
        },
        checkpoint_path,
    )

    return train_loss
