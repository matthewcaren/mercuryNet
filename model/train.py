import torch
from model import load_model
from tqdm import tqdm
from datetime import datetime
import numpy as np
from AVSpeechDataset import AVSpeechDataset
from torch.utils.data import DataLoader
from loss import MercuryNetLoss, HumanReadableLoss
import random
from hparams import hparams as hps
import argparse
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_grad_flow(named_parameters, epoch):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    ave_grads = ave_grads
    layers = layers
    layers = [l.split('.')[1] for l in layers]
    max_grads = max_grads
    plt.figure(figsize=(24, 6))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.yscale('log')
    plt.ylim(bottom = -0.001, top=0.5) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(f'model/results/grad_viz_{epoch}.jpg')
    
def train(model, device, train_dataloader, val_dataloader, optimizer, epochs):
    loss_func = MercuryNetLoss()
    for epoch in range(epochs):
        
        train_batches = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Training for epoch {epoch}')
        model.train()
        torch.cuda.empty_cache()
        for batch_idx, (data, target, metadata_embd) in train_batches:
            data, target, metadata_embd = data.to(device), target.to(device), metadata_embd.to(device)
            optimizer.zero_grad()
            output = model(data, metadata_embd)
            train_loss = loss_func(output, target)
            train_loss.backward()
            optimizer.step()
        print("train loss:", train_loss)
        
        model.eval()
        torch.cuda.empty_cache()
        val_batches = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f'Validation for epoch {epoch}')
        this_epoch_loss = []
        for batch_idx, (data, target, metadata_embd) in val_batches:
            data, target, metadata_embd = data.to(device), target.to(device), metadata_embd.to(device)
            output = model(data, metadata_embd)
            val_loss = loss_func(output, target)
        print("val loss:", val_loss)
    

    checkpoint_path = f"model/checkpoints/ckpt_{datetime.today().strftime('%d_%H-%M')}.pt"
    
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": epochs,
        },
        checkpoint_path,
    )
   # np.save(f"model/checkpoints/valid_loss_{datetime.today().strftime('%d_%H-%M')}.npy", np.array(human_readable_loss_list))


def test(model, device, test_dataloader):
    loss_func = MercuryNetLoss()
    test_batches = tqdm(enumerate(test_dataloader),total=len(test_dataloader), desc='Testing model')
    for batch_idx, (data, target, metadata_embd) in test_batches: 
        data, target, metadata_embd = data.to(device), target.to(device), metadata_embd.to(device)
        output = model(data, metadata_embd)
        test_loss = loss_func(output, target)
    print("test loss:", test_loss)

def segment_data(windows, desired_datause):
    actual_datacount = min(len(windows), desired_datause)
    train_count = int(actual_datacount*0.70)
    val_count = int(actual_datacount*0.15)
    test_count = int(actual_datacount*0.15)
    total_count = train_count + val_count + test_count
    samples = [windows[0]] * actual_datacount
#     samples = random.sample([i for i in range(len(windows))], total_count)
#     all_data = [windows[ix] for ix in samples]
    all_data = samples
    train_data = all_data[:train_count]
    val_data = all_data[train_count:train_count+val_count]
    test_data = all_data[train_count+val_count:]
    return train_data, val_data, test_data

def run_training_pass(root_dir, window_loc, data_count=100, epochs=8, batch_size=16, ckpt_pth='model/checkpoints/lip2wav.pt'):
    print("Starting")    
    model, device = load_model()
    param_count = filter(lambda p: p.requires_grad, model.parameters())
    param_count = sum([np.prod(p.size()) for p in param_count])
    print("total trainable decoder weights:", param_count)
    
    windows = np.load(window_loc, allow_pickle=True)
    windows = [(path, int(start), int(end)) for path, start, end in list(windows) if int(end) - int(start) == 90]
    train_windows, val_windows, test_windows = segment_data(windows, data_count)
    train_dataset = AVSpeechDataset(root_dir, train_windows)
    val_dataset = AVSpeechDataset(root_dir, val_windows)
    test_dataset = AVSpeechDataset(root_dir, test_windows)
    train_dataloader = DataLoader(train_dataset, num_workers = 16, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers = 16,  batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, num_workers = 16,  batch_size=batch_size, shuffle=True)

    optim = torch.optim.Adam(model.parameters(), lr=0.0004)
    train(model, device, train_dataloader, val_dataloader, optim, epochs=epochs)
    test(model, device, test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Training Model')
    parser.add_argument('-r', '--root_dir', help = 'Video directory', required=True)
    parser.add_argument('-w', '--window_loc', help = 'Window directory', required=True)
    parser.add_argument('-c', '--count', help = 'How many windows to include in dataset', default=100000)
    parser.add_argument('-b', '--batch_size', help = 'Batch Size', default=24)
    parser.add_argument('-e', '--epochs', help = 'Epochs', default=1)
    parser.add_argument('-cp', '--checkpoint', help = 'Checkpoint Location', default=None)
    args = parser.parse_args()
    count = int(args.count)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    torch.autograd.set_detect_anomaly(True)
    start_time = time.time()
    run_training_pass(args.root_dir, args.window_loc, data_count=count, batch_size=batch_size, epochs=epochs, ckpt_pth=args.checkpoint)
    print(f'Total training time for {epochs} epochs and {count} windows is {time.time() - start_time} seconds')

   
