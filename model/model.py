import torch
from torch import nn
from torch.nn import functional as F

def load_model(ckpt_pth=None):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    print("using device:", device)
    model = MercuryNet(device)
    if ckpt_pth is not None:
        checkpoint_dict = torch.load(ckpt_pth, map_location=device)["model"]
        print("Loaded checkpoints from", ckpt_pth)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    model = model.eval()
    return model.to(device), device


class MercuryNet(nn.Module):
    def __init__(self, device):
        super(MercuryNet, self).__init__()
        
        self.device = device
        self.layer1 = nn.Sequential(torch.nn.Conv3d(3, 32, kernel_size=(7, 7, 7), stride=(1, 1, 1), padding=(3, 3, 3), bias=True),
                                    torch.nn.ReLU())
        self.max_pool = torch.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer2 = nn.Sequential(torch.nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=True),
                                    torch.nn.ReLU())
        self.layer3 = nn.Sequential(torch.nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=True),
                                    torch.nn.ReLU())
        self.layer4 = nn.Sequential(torch.nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=True),
                                    torch.nn.ReLU())
        self.layer5 = nn.Sequential(torch.nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),  bias=True),
                                    torch.nn.ReLU())

        self.avg_pool = nn.AvgPool3d(kernel_size=(1, 6, 6), stride=(1, 1, 1))
        self.fc1 = nn.Sequential(
                                    torch.nn.Linear(512, 128),
                                    torch.nn.ReLU())
        self.fc2 = nn.Sequential(
                                    torch.nn.Linear(128, 32),
                                    torch.nn.ReLU())
        self.fc3 = nn.Sequential(   torch.nn.Linear(32, 3))

    def forward(self, vid_inputs, metadata_embd):
        x = vid_inputs.type(torch.cuda.FloatTensor)
        x = self.layer1(x)
        x = self.max_pool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avg_pool(x).squeeze(4).squeeze(3).permute(0,2,1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x