import torch
from torch import nn
from math import sqrt
import sys
sys.path.append('../')
from hparams import hparams as hps
from torch.autograd import Variable
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm, ConvNorm2D, ConvNorm3D
from utils.util import to_var, get_mask_from_lengths
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(447, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU()
        )

        self.conv1 = torch.nn.Sequential(
            # depth-wise conv to expand channel space
            ConvNorm(in_channels=1024, out_channels=1024, kernel_size=5),
            torch.nn.ReLU(),
            ConvNorm(in_channels=1024, out_channels=512, kernel_size=3),
            torch.nn.ReLU(),
            ConvNorm(in_channels=512, out_channels=512, kernel_size=3),
            torch.nn.ReLU(),
        )

        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
        )

        self.conv2 = torch.nn.Sequential(
            ConvNorm(in_channels=256, out_channels=256, kernel_size=3),
            torch.nn.ReLU(),
            ConvNorm(in_channels=256, out_channels=256, kernel_size=3),
            torch.nn.ReLU(),
            ConvNorm(in_channels=256, out_channels=128, kernel_size=3),
            torch.nn.ReLU(),
            ConvNorm(in_channels=128, out_channels=64, kernel_size=3),
            torch.nn.ReLU(),
        )

        self.fc3 = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
        )

        self.fc_out = torch.nn.Sequential(
            torch.nn.Linear(32, 3)
        )
        

    def forward(self, x):
        x = self.fc1(x)         # expand feature space      (time, 512)
        x = x.transpose(-2, -1) # flip time, channel dims   (512, time)

        x = self.conv1(x)       # 1st conv block            (512, time)
        x = x.transpose(-2, -1) # flip feat, time dims      (time, 512)

        x = self.fc2(x)         # linear                    (time, 256)
        x = x.transpose(-2, -1) # flip feat, time dims      (256, time)

        x = self.conv2(x)       # 2nd conv block            (256, time)
        x = x.transpose(-2, -1) # flip feat, time dims      (time, 64)

        x = self.fc3(x)         # linear                    (time, 32)
        x = self.fc_out(x)      # linear                    (time, 3)
        
        softplus_f0 = F.softplus(x[:,:,0])
        softplus_amp = F.softplus(x[:,:,2])
        voiced_flag = x[:,:,1]
        
        return torch.stack((softplus_f0, voiced_flag, softplus_amp), dim=2)


def is_end_of_frames(output, eps=0.2):
    return (output.data <= eps).all()


class MercuryNet(nn.Module):
    def __init__(self):
        super(MercuryNet, self).__init__()
        self.mask_padding = hps.mask_padding
        self.n_frames_per_step = hps.n_frames_per_step
        self.encoder = Encoder3D(hps).to(device)
        self.decoder = Decoder().to(device)

    def forward(self, vid_inputs, metadata_embd):
        vid_lengths = torch.tensor(vid_inputs.shape[0])

        embedded_inputs = vid_inputs.type(torch.FloatTensor)

        encoder_outputs = self.encoder(
            embedded_inputs.to(device), vid_lengths.to(device)
        )
                
        metadata_embd = metadata_embd.type(torch.FloatTensor).to(device)
        metadata_embd = metadata_embd.unsqueeze(1).repeat((1, encoder_outputs.shape[1], 1))
                
        decoder_input = torch.cat((encoder_outputs, metadata_embd), dim=2)    
        decoder_output = self.decoder(decoder_input)
        return decoder_output
    

    # def inference(self, inputs, mode="train"):
    #     if mode == "train":
    #         vid_inputs, vid_lengths = inputs
    #     else:
    #         vid_inputs = inputs
    #         vid_inputs = to_var(torch.from_numpy(vid_inputs)).float()
    #         vid_inputs = vid_inputs.permute(3, 0, 1, 2).unsqueeze(0).contiguous()

    #     embedded_inputs = vid_inputs.type(torch.FloatTensor)
    #     encoder_outputs = self.encoder.inference(embedded_inputs.to(device))
    #     decoder_output = self.decoder(encoder_outputs)
    #     return decoder_output