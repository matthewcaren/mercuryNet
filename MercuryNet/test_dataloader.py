from train_mercury import AVSpeechDataset
from torchvision import transforms


dataset = AVSpeechDataset('./vids_10', 30)
print(len(dataset))
data, target = dataset[0]
print(data.shape)
print(target.shape)