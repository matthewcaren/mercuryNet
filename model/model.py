import torch
from torch import nn
from math import sqrt
import sys
sys.path.append('../')
from hparams import hparams as hps
from torch.autograd import Variable
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm, ConvNorm2D, ConvNorm3D
from utils.util import to_var, get_mask_from_lengths, mode
import numpy as np

def load_model(ckpt_pth):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    print("using device:", device)

    checkpoint_dict = torch.load(ckpt_pth, map_location=device)["model"]
    model = MercuryNet(device)
    
#     model_dict = model.state_dict()
#     pretrained_dict = {k: v for k, v in checkpoint_dict.items() if k in model_dict}
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)
    
    model = mode(model, True).eval()
    return model, device

class Encoder3D(nn.Module):
    """Encoder module:
    - Three 3-d convolution banks
    - Bidirectional LSTM
    """

    def __init__(self, hparams):
        super(Encoder3D, self).__init__()

        self.hparams = hparams
        self.out_channel = hps.num_init_filters
        self.in_channel = 3
        convolutions = []

        for i in range(hps.encoder_n_convolutions):
            if i == 0:
                conv_layer = nn.Sequential(
                    ConvNorm3D(
                        self.in_channel,
                        self.out_channel,
                        kernel_size=5,
                        stride=(1, 2, 2),
                        # padding=int((hparams.encoder_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="relu",
                    ),
                    ConvNorm3D(
                        self.out_channel,
                        self.out_channel,
                        kernel_size=3,
                        stride=1,
                        # padding=int((hparams.encoder_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="relu",
                        residual=True,
                    ),
                    ConvNorm3D(
                        self.out_channel,
                        self.out_channel,
                        kernel_size=3,
                        stride=1,
                        # padding=int((hparams.encoder_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="relu",
                        residual=True,
                    ),
                )
                convolutions.append(conv_layer)
            else:
                conv_layer = nn.Sequential(
                    ConvNorm3D(
                        self.in_channel,
                        self.out_channel,
                        kernel_size=3,
                        stride=(1, 2, 2),
                        # padding=int((hparams.encoder_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="relu",
                    ),
                    ConvNorm3D(
                        self.out_channel,
                        self.out_channel,
                        kernel_size=3,
                        stride=1,
                        # padding=int((hparams.encoder_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="relu",
                        residual=True,
                    ),
                    ConvNorm3D(
                        self.out_channel,
                        self.out_channel,
                        kernel_size=3,
                        stride=1,
                        # padding=int((hparams.encoder_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="relu",
                        residual=True,
                    ),
                )
                convolutions.append(conv_layer)

            if i == hps.encoder_n_convolutions - 1:
                conv_layer = nn.Sequential(
                    ConvNorm3D(
                        self.out_channel,
                        self.out_channel,
                        kernel_size=3,
                        stride=(1, 3, 3),
                        # padding=int((hparams.encoder_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="relu",
                    )
                )
                convolutions.append(conv_layer)

            self.in_channel = self.out_channel
            self.out_channel *= 2
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(
            hparams.encoder_embedding_dim,
            int(hparams.encoder_embedding_dim / 2),
            1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(conv(x), 0.5, self.training)

        # [bs x 90 x encoder_embedding_dim]
        x = (x.permute(0, 2, 1, 3, 4).squeeze(4).squeeze(3).contiguous())
        
        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        outputs, _ = self.lstm(x)              
        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(conv(x), 0.5, self.training)

        x = x.permute(0, 2, 1, 3, 4).squeeze(4).squeeze(3).contiguous()
        outputs, _ = self.lstm(x)  # x:B,T,C

        return outputs


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

#         self.fc1 = torch.nn.Sequential(
#             torch.nn.Linear(447, 1024),
#             torch.nn.ReLU(),
#             torch.nn.Linear(1024, 1024),
#             torch.nn.ReLU()
#         )

#         self.conv1 = torch.nn.Sequential(
#             # depth-wise conv to expand channel space
#             ConvNorm(in_channels=1024, out_channels=1024, kernel_size=5),
#             torch.nn.ReLU(),
#             ConvNorm(in_channels=1024, out_channels=512, kernel_size=3),
#             torch.nn.ReLU(),
#             ConvNorm(in_channels=512, out_channels=512, kernel_size=3),
#             torch.nn.ReLU(),
#         )

#         self.fc2 = torch.nn.Sequential(
#             torch.nn.Linear(512, 512),
#             torch.nn.ReLU(),
#             torch.nn.Linear(512, 256),
#             torch.nn.ReLU(),
#         )

#         self.conv2 = torch.nn.Sequential(
#             ConvNorm(in_channels=256, out_channels=256, kernel_size=3),
#             torch.nn.ReLU(),
#             ConvNorm(in_channels=256, out_channels=256, kernel_size=3),
#             torch.nn.ReLU(),
#             ConvNorm(in_channels=256, out_channels=128, kernel_size=3),
#             torch.nn.ReLU(),
#             ConvNorm(in_channels=128, out_channels=64, kernel_size=3),
#             torch.nn.ReLU(),
#         )

#         self.fc3 = torch.nn.Sequential(
#             torch.nn.Linear(64, 64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, 32),
#             torch.nn.ReLU(),
#         )

#         self.fc_out = torch.nn.Sequential(
#             torch.nn.Linear(32, 3)
#         )

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(447, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
        )
    

        self.conv1 = torch.nn.Sequential(
            # depth-wise conv to expand channel space
            torch.nn.BatchNorm1d(1024),
            ConvNorm(in_channels=1024, out_channels=1024, kernel_size=5),
            torch.nn.ReLU(),
            ConvNorm(in_channels=1024, out_channels=512, kernel_size=3),
            torch.nn.ReLU(),
#             torch.nn.LayerNorm([512, 90])
        )
        
        self.conv2 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(512),
            # depth-wise conv to expand channel space
            ConvNorm(in_channels=512, out_channels=512, kernel_size=3),
            torch.nn.ReLU(),
            ConvNorm(in_channels=512, out_channels=512, kernel_size=3),
            torch.nn.ReLU(),
#             torch.nn.LayerNorm([512, 90])
        )
        
        self.conv3 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(512), 
            # depth-wise conv to expand channel space
            ConvNorm(in_channels=512, out_channels=512, kernel_size=3),
            torch.nn.ReLU(),
            ConvNorm(in_channels=512, out_channels=512, kernel_size=3),
            torch.nn.ReLU(),
#             torch.nn.LayerNorm([512, 90])
        )
            
        self.conv4 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(512),
            # depth-wise conv to expand channel space
            ConvNorm(in_channels=512, out_channels=512, kernel_size=3),
            torch.nn.ReLU(),
            ConvNorm(in_channels=512, out_channels=512, kernel_size=3),
            torch.nn.ReLU(),
#             torch.nn.LayerNorm([512, 90])
        )
            
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(256),
            ConvNorm(in_channels=256, out_channels=256, kernel_size=3),
            torch.nn.ReLU(),
            ConvNorm(in_channels=256, out_channels=256, kernel_size=3),
            torch.nn.ReLU(),
#             torch.nn.LayerNorm([256, 90])
        )
        
        self.conv6 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(256),
            ConvNorm(in_channels=256, out_channels=256, kernel_size=3),
            torch.nn.ReLU(),
            ConvNorm(in_channels=256, out_channels=256, kernel_size=3),
            torch.nn.ReLU(),
#             torch.nn.LayerNorm([256, 90])
        )
        
        self.conv7 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(256),
            ConvNorm(in_channels=256, out_channels=256, kernel_size=3),
            torch.nn.ReLU(),
            ConvNorm(in_channels=256, out_channels=256, kernel_size=3),
            torch.nn.ReLU(),
#             torch.nn.LayerNorm([256, 90])
        )
        
                
        self.conv8 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(256),
            ConvNorm(in_channels=256, out_channels=256, kernel_size=3),
            torch.nn.ReLU(),
            ConvNorm(in_channels=256, out_channels=256, kernel_size=3),
            torch.nn.ReLU(),
#             torch.nn.LayerNorm([256, 90])
        )
        
        
        self.conv9 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(256),
            ConvNorm(in_channels=256, out_channels=256, kernel_size=3),
            torch.nn.ReLU(),
            ConvNorm(in_channels=256, out_channels=256, kernel_size=3),
            torch.nn.ReLU(),
#             torch.nn.LayerNorm([256, 90])
        )
        
        self.conv10 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(256),
            ConvNorm(in_channels=256, out_channels=256, kernel_size=3),
            torch.nn.ReLU(),
            ConvNorm(in_channels=256, out_channels=256, kernel_size=3),
            torch.nn.ReLU(),
#             torch.nn.LayerNorm([256, 90])
        )
        
        self.conv11 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(256),
            ConvNorm(in_channels=256, out_channels=256, kernel_size=3),
            torch.nn.ReLU(),
            ConvNorm(in_channels=256, out_channels=256, kernel_size=3),
            torch.nn.ReLU(),
#             torch.nn.LayerNorm([256, 90])
        )
        self.conv12 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(256),
            ConvNorm(in_channels=256, out_channels=256, kernel_size=3),
            torch.nn.ReLU(),
            ConvNorm(in_channels=256, out_channels=128, kernel_size=3),
            torch.nn.ReLU(),
#             torch.nn.LayerNorm([128, 90])
        )
        
        self.conv13 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(128),
            ConvNorm(in_channels=128, out_channels=128, kernel_size=3),
            torch.nn.ReLU(),
            ConvNorm(in_channels=128, out_channels=128, kernel_size=3),
            torch.nn.ReLU(),
#             torch.nn.LayerNorm([128, 90])
        )
            
        self.conv14 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(128),
            ConvNorm(in_channels=128, out_channels=128, kernel_size=3),
            torch.nn.ReLU(),
            ConvNorm(in_channels=128, out_channels=128, kernel_size=3),
            torch.nn.ReLU(),
#             torch.nn.LayerNorm([128, 90])
        )
        self.conv15 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(128),
            ConvNorm(in_channels=128, out_channels=128, kernel_size=3),
            torch.nn.ReLU(),
            ConvNorm(in_channels=128, out_channels=128, kernel_size=3),
            torch.nn.ReLU(),
#             torch.nn.LayerNorm([128, 90])
        )
        

        self.fc3 = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
        )

        self.fc_out = torch.nn.Sequential(
            torch.nn.Linear(32, 3)
        )
        

    def forward(self, x):
        x = self.fc1(x)         # expand feature space      (time, 1024)
        x = x.transpose(-2, -1) # flip time, channel dims   (1024, time)

        x = self.conv1(x)       # 1st conv block           
        x = x + self.conv2(x)
        x = x + self.conv3(x)
        x = x + self.conv4(x)
        x = x.transpose(-2, -1) # flip feat, time dims      (time, 512)
        x = self.fc2(x)         # linear                    (time, 256)
        x = x.transpose(-2, -1) # flip feat, time dims      (256, time)
        x_old = x.clone()
        x = x + self.conv5(x)       # 2nd conv block            (256, time)
        x = x + self.conv6(x)
        x = x + self.conv7(x)
        x = x + self.conv8(x)
        x = x + self.conv9(x)
        x = x + self.conv10(x)
        x = x + self.conv11(x) + x_old
        x = self.conv12(x)
        x = x + self.conv13(x)
        x = x + self.conv14(x)
        x = x + self.conv15(x)
        x = x.transpose(-2, -1) # flip feat, time dims      (time, 64)

        x = self.fc3(x)         # linear                    (time, 32)
        x = self.fc_out(x)      # linear                    (time, 3)
        
        softplus_f0 = F.softplus(x[:,:,0])
        softplus_amp = F.softplus(x[:,:,2])
        voiced_flag = torch.sigmoid(x[:,:,1])
        
        return torch.stack((softplus_f0, voiced_flag, softplus_amp), dim=2)
    
class MercuryNet(nn.Module):
    def __init__(self, device):
        super(MercuryNet, self).__init__()
        self.mask_padding = hps.mask_padding
        self.n_frames_per_step = hps.n_frames_per_step
        self.device = device
        self.encoder = Encoder3D(hps).to(self.device)
        self.decoder = Decoder().to(self.device)

    def forward(self, vid_inputs, metadata_embd):
        vid_lengths = torch.tensor(vid_inputs.shape[0])

        embedded_inputs = vid_inputs.type(torch.FloatTensor)

        encoder_outputs = self.encoder(
            embedded_inputs.to(self.device), vid_lengths.to(self.device)
        )
                
        metadata_embd = metadata_embd.type(torch.FloatTensor).to(self.device)
        metadata_embd = metadata_embd.unsqueeze(1).repeat((1, encoder_outputs.shape[1], 1))
                
        decoder_input = torch.cat((encoder_outputs, metadata_embd), dim=2)    
        decoder_output = self.decoder(decoder_input)
        return decoder_output
    