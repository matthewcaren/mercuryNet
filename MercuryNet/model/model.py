import torch
from torch import nn
from math import sqrt
import sys
sys.path.append('../')
from hparams import hparams as hps
from torch.autograd import Variable
from torch.nn import functional as F
from model.layers import ConvNorm, LinearNorm, ConvNorm2D, ConvNorm3D
from utils.util import to_var, get_mask_from_lengths
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MercuryNetLoss(nn.Module):
    def __init__(self):
        super(MercuryNetLoss, self).__init__()

    def forward(self, model_output, targets):
        target_f0, target_voiced, target_amp = targets[0,:], targets[1,:], targets[2,:]

        # only care about f0 when it's voiced (so there's a valid ground truth)
        masked_f0_output = model_output[0,:]
        masked_f0_output[target_voiced == 0] = 0
        target_f0[target_voiced == 0] = 0

        target_f0 = torch.nan_to_num(torch.log(target_f0))
        output_f0 = torch.nan_to_num(torch.log(masked_f0_output))

        target_amp = torch.nan_to_num(torch.log(target_amp))
        output_amp = torch.nan_to_num(torch.log(model_output[2,:]))

        loss = 0
        loss += hps.f0_penalty * torch.nn.MSELoss(output_f0, target_f0)
        loss += hps.voiced_penalty * torch.nn.MSELoss(target_voiced, model_output[1,:])
        loss += hps.amp_penalty * torch.nn.MSELoss(target_amp, output_amp)
        return loss


class Encoder(nn.Module):
    """Encoder module:
    - Three 1-d convolution banks
    - Bidirectional LSTM
    """

    def __init__(self):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hps.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(
                    hps.encoder_embedding_dim,
                    hps.encoder_embedding_dim,
                    kernel_size=hps.encoder_kernel_size,
                    stride=1,
                    padding=int((hps.encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(hps.encoder_embedding_dim),
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(
            hps.encoder_embedding_dim,
            int(hps.encoder_embedding_dim / 2),
            1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


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
        # for i in range(len(self.convolutions)):
        # 	if i==0 or i==1 or i ==2:
        # 		with torch.no_grad():
        # 			x = F.dropout(self.convolutions[i](x), 0.5, self.training)
        # 	else:
        # 		x = F.dropout(self.convolutions[i](x), 0.5, self.training)

        x = (
            x.permute(0, 2, 1, 3, 4).squeeze(4).squeeze(3).contiguous()
        )  # [bs x 90 x encoder_embedding_dim]
        # print(x.size())
        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        # x = nn.utils.rnn.pack_padded_sequence(
        # 	x, input_lengths, batch_first=True)

        # self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        print("output size", outputs.size())
        # outputs, _ = nn.utils.rnn.pad_packed_sequence(
        # 	outputs, batch_first=True)
        # print('outputs', outputs.size())

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(conv(x), 0.5, self.training)

        x = x.permute(0, 2, 1, 3, 4).squeeze(4).squeeze(3).contiguous()
        # self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)  # x:B,T,C

        return outputs


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(384, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU()
        )

        self.conv1 = torch.nn.Sequential(
            # depth-wise conv to expand channel space
            ConvNorm(in_channels=512, out_channels=512, kernel_size=5),
            torch.nn.ReLU(),
            ConvNorm(in_channels=512, out_channels=512, kernel_size=3),
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

        x[0] = torch.exp(x[0])  # make f0 log scale
        x[2] = torch.exp(x[2])  # make loudness log scale (dB)

        return x


def is_end_of_frames(output, eps=0.2):
    return (output.data <= eps).all()


class MercuryNet(nn.Module):
    def __init__(self):
        super(MercuryNet, self).__init__()
        self.num_mels = hps.num_mels
        self.mask_padding = hps.mask_padding
        self.n_frames_per_step = hps.n_frames_per_step
        self.embedding = nn.Embedding(hps.n_symbols, hps.symbols_embedding_dim)
        std = sqrt(2.0 / (hps.n_symbols + hps.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder3D(hps).to(device)
        self.decoder = Decoder().to(device)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
        text_padded = to_var(text_padded).long()
        input_lengths = to_var(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_var(mel_padded).float()
        gate_padded = to_var(gate_padded).float()
        output_lengths = to_var(output_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded),
        )

    def parse_batch_vid(self, batch):
        (
            vid_padded,
            input_lengths,
            mel_padded,
            gate_padded,
            target_lengths,
            split_infos,
            embed_targets,
        ) = batch
        vid_padded = to_var(vid_padded).float()
        input_lengths = to_var(input_lengths).float()
        mel_padded = to_var(mel_padded).float()
        gate_padded = to_var(gate_padded).float()
        target_lengths = to_var(target_lengths).float()

        max_len_vid = split_infos[0].data.item()
        max_len_target = split_infos[1].data.item()

        mel_padded = to_var(mel_padded).float()

        return (
            (vid_padded, input_lengths, mel_padded, max_len_vid, target_lengths),
            (mel_padded, gate_padded),
        )

    def forward(self, inputs):
        vid_inputs, vid_lengths, mels, max_len, output_lengths = inputs
        vid_lengths, output_lengths = vid_lengths.data, output_lengths.data

        embedded_inputs = vid_inputs.type(torch.FloatTensor)

        encoder_outputs = self.encoder(
            embedded_inputs.to(device), vid_lengths.to(device)
        )

        decoder_output = self.decoder(encoder_outputs)
        return decoder_output

    def inference(self, inputs, mode="train"):
        if mode == "train":
            vid_inputs, vid_lengths, mels, max_len, output_lengths = inputs
        else:
            vid_inputs = inputs
            vid_inputs = to_var(torch.from_numpy(vid_inputs)).float()
            vid_inputs = vid_inputs.permute(3, 0, 1, 2).unsqueeze(0).contiguous()

        embedded_inputs = vid_inputs.type(torch.FloatTensor)
        encoder_outputs = self.encoder.inference(embedded_inputs.to(device))
        decoder_output = self.decoder(encoder_outputs)
        return decoder_output
    

    def teacher_infer(self, inputs, mels):
        il, _ = torch.sort(
            torch.LongTensor([len(x) for x in inputs]), dim=0, descending=True
        )
        vid_lengths = to_var(il)

        embedded_inputs = self.embedding(inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, vid_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=vid_lengths
        )

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments]
        )
