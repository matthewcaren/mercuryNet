from train_mercury import AVSpeechDataset
import time
import torch

start = time.time()
dataset = AVSpeechDataset('./vids_10', overlap=30, window_size=90)
dataset[0]
print(time.time() - start)