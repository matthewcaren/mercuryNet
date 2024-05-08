import torch
import numpy as np
from tqdm import tqdm
import os
import argparse
from model import MercuryNet, load_model
from AVSpeechDataset import AVSpeechDataset
from torch.utils.data import DataLoader
from loss import MercuryNetLoss, HumanReadableLoss
from make_windows import make_all_windows
from datetime import datetime


def test_model(root_dir, ckpt_pth):
    windows = make_all_windows(root_dir)
    test_dataset = AVSpeechDataset(root_dir, windows)
    test_dataloader = DataLoader(test_dataset, num_workers = 2, batch_size=8, shuffle=True)
    test_batches = tqdm(enumerate(test_dataloader),  total=len(test_dataloader), desc='Testing model')
    model, device = load_model(ckpt_pth)
    model.eval()
    loss_func = MercuryNetLoss()
    human_readable_loss = HumanReadableLoss()
    human_readable_loss_list = []
    for batch_idx, (data, target, metadata_embd) in test_batches:
        data_mps = data.to(device)
        target_mps = target.to(device)
        metadata_embd_mps = metadata_embd.to(device)
        output = model(data_mps, metadata_embd_mps)
        test_loss = loss_func(output, target_mps)
        human_readable_loss_list.append(list(human_readable_loss(output, target_mps)))
    np.save(f"model/results/loss_{datetime.today().strftime('%d_%H-%M')}.npy", np.array(human_readable_loss_list))
    print("Final loss:", test_loss)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_dir", help="Video Directory", required=True)
    parser.add_argument("-cp", "--checkpoint", help="Path to trained checkpoint", required=True)
    args = parser.parse_args()
    test_model(args.root_dir, args.checkpoint)
