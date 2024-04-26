from train_mercury import AVSpeechDataset
import time
import torch

start = time.time()
dataset = AVSpeechDataset('./vids', overlap=30, window_size=90)
print(time.time() - start)