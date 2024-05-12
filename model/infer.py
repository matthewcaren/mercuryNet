import numpy as np
from tqdm import tqdm
import argparse
from model import load_model
from AVSpeechDataset import AVSpeechDataset
from torch.utils.data import DataLoader
from loss import MercuryNetLoss, HumanReadableLoss
from make_windows import make_all_windows
from datetime import datetime
import random

def test_model(root_dir, windows, ckpt_pth):
    model, device = load_model(ckpt_pth)
    model.eval()
    test_dataset = AVSpeechDataset(root_dir, windows)
    test_dataloader = DataLoader(test_dataset, num_workers = 16, batch_size=1, shuffle=True)
    test_batches = tqdm(enumerate(test_dataloader),  total=len(test_dataloader), desc='Testing model')
    loss_func = MercuryNetLoss()
    pred_list, target_list = [], []
    for batch_idx, (data, target, metadata_embd) in test_batches:
        data, target, metadata_embd = data.to(device), target.to(device), metadata_embd.to(device)
        output = model(data, metadata_embd)
        pred_list.append(output.detach().cpu().numpy())
        target_list.append(target.detach().cpu().numpy())
        test_loss = loss_func(output, target)
    np.save(f"nb/model_output_{datetime.today().strftime('%d_%H-%M')}.npy", np.array([pred_list, target_list]))
    print("Final loss:", test_loss)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_dir", help="Video Directory", required=True)
    parser.add_argument('-w', '--window_loc', help = 'Window directory', required=True)
    parser.add_argument('-c', '--count', help = 'How many windows to include in dataset', default=100)
    parser.add_argument("-cp", "--checkpoint", help="Path to trained checkpoint", required=True)
    
    args = parser.parse_args()
    windows = np.load(args.window_loc)
    windows = [(path, int(start), int(end)) for path, start, end in list(windows)]
#     selected_windows = random.sample(windows, int(args.count))
    selected_windows = windows[0:1]*int(args.count)
    test_model(args.root_dir, selected_windows, args.checkpoint)
