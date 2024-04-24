from train_mercury import AVSpeechDataset
from torchvision import transforms


transform_func = transforms.Normalize(0.5, 0.5, 0.5)

dataset = AVSpeechDataset('./vids_10', transform_func)
print(len(dataset))
print(dataset.all_paths[0])
