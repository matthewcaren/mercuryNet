import torch
from torch import nn
from math import sqrt
from hparams import hparams as hps
from torch.autograd import Variable
from torch.nn import functional as F
from model.layers import ConvNorm, LinearNorm, ConvNorm3D
from utils.util import to_var, get_mask_from_lengths

device = torch.device("cpu")


class MercuryNetLoss(nn.Module):
    def __init__(self):
        super(MercuryNetLoss, self).__init__()

    def forward(self, model_output, targets, iteration):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        slice = torch.arange(0, gate_target.size(1), hps.n_frames_per_step)
        gate_target = gate_target[:, slice].view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        p = hps.p
        mel_loss = nn.MSELoss()(p * mel_out, p * mel_target)
        mel_loss_post = nn.MSELoss()(p * mel_out_postnet, p * mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        # added
        l1_loss = nn.L1Loss()(mel_target, mel_out)
        return (
            mel_loss,
            mel_loss_post,
            l1_loss,
            gate_loss,
        )  # , ((mel_loss+mel_loss_post)/(p**2)+gate_loss+l1_loss).item()


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(
            2,
            attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1,
        )
        self.location_dense = LinearNorm(
            attention_n_filters, attention_dim, bias=False, w_init_gain="tanh"
        )

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(
        self,
        attention_rnn_dim,
        embedding_dim,
        attention_dim,
        attention_location_n_filters,
        attention_location_kernel_size,
    ):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(
            attention_rnn_dim, attention_dim, bias=False, w_init_gain="tanh"
        )
        self.memory_layer = LinearNorm(
            embedding_dim, attention_dim, bias=False, w_init_gain="tanh"
        )
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(
            attention_location_n_filters, attention_location_kernel_size, attention_dim
        )
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory, attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, num_mels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att we;[ights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(
            torch.tanh(processed_query + processed_attention_weights + processed_memory)
        )

        energies = energies.squeeze(-1)
        return energies

    def forward(
        self,
        attention_hidden_state,
        memory,
        processed_memory,
        attention_weights_cat,
        mask,
    ):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat
        )

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [
                LinearNorm(in_size, out_size, bias=False)
                for (in_size, out_size) in zip(in_sizes, sizes)
            ]
        )

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


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
        # self.encoder = Encoder()
        self.encoder = Encoder3D(hps).to(device)
        self.decoder = Decoder().to(device)
        self.postnet = Postnet().to(device)

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

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths, True)  # (B, T)
            mask = mask.expand(self.num_mels, mask.size(0), mask.size(1))  # (80, B, T)
            mask = mask.permute(1, 0, 2)  # (B, 80, T)

            outputs[0].data.masked_fill_(mask, 0.0)  # (B, 80, T)
            outputs[1].data.masked_fill_(mask, 0.0)  # (B, 80, T)
            slice = torch.arange(0, mask.size(2), self.n_frames_per_step)
            outputs[2].data.masked_fill_(
                mask[:, 0, slice], 1e3
            )  # gate energies (B, T//n_frames_per_step)

        return outputs

    def forward(self, inputs):
        vid_inputs, vid_lengths, mels, max_len, output_lengths = inputs
        vid_lengths, output_lengths = vid_lengths.data, output_lengths.data

        embedded_inputs = vid_inputs.type(torch.FloatTensor)
        # print('vid_inputs',vid_inputs)

        encoder_outputs = self.encoder(
            embedded_inputs.to(device), vid_lengths.to(device)
        )
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=vid_lengths
        )

        s_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments], output_lengths
        )

    def inference(self, inputs, mode="train"):
        if mode == "train":
            vid_inputs, vid_lengths, mels, max_len, output_lengths = inputs
        else:
            vid_inputs = inputs
            vid_inputs = to_var(torch.from_numpy(vid_inputs)).float()
            vid_inputs = vid_inputs.permute(3, 0, 1, 2).unsqueeze(0).contiguous()

        # vid_lengths, output_lengths = vid_lengths.data, output_lengths.data
        # embedded_inputs = self.embedding(inputs).transpose(1, 2)

        embedded_inputs = vid_inputs.type(torch.FloatTensor)

        encoder_outputs = self.encoder.inference(embedded_inputs.to(device))
        print("ENC", encoder_outputs.shape)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs)
        mel_outputs_postnet = self.postnet(mel_outputs)

        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments]
        )

        return outputs

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
