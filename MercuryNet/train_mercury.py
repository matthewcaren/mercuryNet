import torch
from model.model import MercuryNet, MercuryNetLoss
import tqdm
import torch.nn.functional as F
from datetime import datetime


def train(dataloader, optimizer, epochs):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    model = MercuryNet()
    model.train()
    train_loss = []

    batches = tqdm(enumerate(dataloader), total=len(dataloader))
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in batches:
            data_mps = data.to(device)
            target_mps = target.to(device)
            optimizer.zero_grad()
            output = model(data_mps)

            loss = MercuryNetLoss()
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
