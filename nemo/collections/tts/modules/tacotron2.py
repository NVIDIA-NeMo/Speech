# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.autograd import Variable
from torch.nn import functional as F

from nemo.collections.tts.modules.submodules import Attention, ConvNorm, LinearNorm, Prenet
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import (
    EmbeddedTextType,
    LengthsType,
    LogitsType,
    MelSpectrogramType,
    SequenceToSequenceAlignmentType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging


class Encoder(NeuralModule):
    def __init__(
        self, encoder_n_convolutions: int, encoder_embedding_dim: int, encoder_kernel_size: int,
    ):
        """
        Tacotron 2 Encoder. A number of convolution layers that feed into a LSTM

        Args:
            encoder_n_convolutions (int): Number of convolution layers.
            encoder_embedding_dim (int): Final output embedding size. Also used to create the convolution and LSTM layers.
            encoder_kernel_size (int): Kernel of the convolution front-end.
        """
        super().__init__()

        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = torch.nn.Sequential(
                ConvNorm(
                    encoder_embedding_dim,
                    encoder_embedding_dim,
                    kernel_size=encoder_kernel_size,
                    stride=1,
                    padding=int((encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain='relu',
                ),
                torch.nn.BatchNorm1d(encoder_embedding_dim),
            )
            convolutions.append(conv_layer)
        self.convolutions = torch.nn.ModuleList(convolutions)

        self.lstm = torch.nn.LSTM(
            encoder_embedding_dim, int(encoder_embedding_dim / 2), 1, batch_first=True, bidirectional=True,
        )

    @property
    def input_types(self):
        return {
            "token_embedding": NeuralType(('B', 'D', 'T'), EmbeddedTextType()),
            "token_len": NeuralType(('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "encoder_embedding": NeuralType(('B', 'T', 'D'), EmbeddedTextType()),
        }

    @typecheck()
    def forward(self, *, token_embedding, token_len):
        for conv in self.convolutions:
            token_embedding = F.dropout(F.relu(conv(token_embedding)), 0.5, self.training)

        token_embedding = token_embedding.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = token_len.cpu().numpy()

        token_embedding = torch.nn.utils.rnn.pack_padded_sequence(
            token_embedding, input_lengths, batch_first=True, enforce_sorted=False
        )

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(token_embedding)

        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs


class Decoder(NeuralModule):
    def __init__(
        self,
        n_mel_channels: int,
        n_frames_per_step: int,
        encoder_embedding_dim: int,
        attention_dim: int,
        attention_location_n_filters: int,
        attention_location_kernel_size: int,
        attention_rnn_dim: int,
        decoder_rnn_dim: int,
        prenet_dim: int,
        max_decoder_steps: int,
        gate_threshold: float,
        p_attention_dropout: float,
        p_decoder_dropout: float,
        early_stopping: bool,
        prenet_p_dropout: float = 0.5,
    ):
        """
        Tacotron 2 Decoder. Consists of a 2 layer LSTM, one of which interfaces with the attention mechanism while the
        other is used as a regular LSTM. Includes the prenet and attention modules as well.

        Args:
            n_mel_channels (int): Number of mel channels to output
            n_frames_per_step (int): Number of spectrogram frames to predict per decoder step.
            encoder_embedding_dim (int): The size of the output from the encoder.
            attention_dim (int): The output dimension of the attention layer.
            attention_location_n_filters (int): Channel size for the convolution used the attention mechanism.
            attention_location_kernel_size (int): Kernel size for the convolution used the attention mechanism.
            attention_rnn_dim (int): The output dimension of the attention LSTM layer.
            decoder_rnn_dim (int): The output dimension of the second LSTM layer.
            prenet_dim (int): The output dimension of the prenet.
            max_decoder_steps (int): For evaluation, the max number of steps to predict.
            gate_threshold (float): At each step, tacotron 2 predicts a probability of stopping. Rather than sampling,
                this module checks if predicted probability is above the gate_threshold. Only in evaluation.
            p_attention_dropout (float): Dropout probability on the attention LSTM.
            p_decoder_dropout (float): Dropout probability on the second LSTM.
            early_stopping (bool): In evaluation mode, whether to stop when all batches hit the gate_threshold or to
                continue until max_decoder_steps.
            prenet_p_dropout (float): Dropout probability for prenet. Note, dropout is on even in eval() mode.
                Defaults to 0.5.
        """
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.encoder_embedding_dim = encoder_embedding_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout
        self.early_stopping = early_stopping

        self.prenet = Prenet(n_mel_channels * n_frames_per_step, [prenet_dim, prenet_dim], prenet_p_dropout)

        self.attention_rnn = torch.nn.LSTMCell(prenet_dim + encoder_embedding_dim, attention_rnn_dim)

        self.attention_layer = Attention(
            attention_rnn_dim,
            encoder_embedding_dim,
            attention_dim,
            attention_location_n_filters,
            attention_location_kernel_size,
        )

        self.decoder_rnn = torch.nn.LSTMCell(attention_rnn_dim + encoder_embedding_dim, decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            decoder_rnn_dim + encoder_embedding_dim, n_mel_channels * n_frames_per_step
        )

        self.gate_layer = LinearNorm(decoder_rnn_dim + encoder_embedding_dim, 1, bias=True, w_init_gain='sigmoid')

    @property
    def input_types(self):
        input_dict = {
            "memory": NeuralType(('B', 'T', 'D'), EmbeddedTextType()),
            "memory_lengths": NeuralType(('B'), LengthsType()),
        }
        if self.training:
            input_dict["decoder_inputs"] = NeuralType(('B', 'D', 'T'), MelSpectrogramType())
        return input_dict

    @property
    def output_types(self):
        output_dict = {
            "mel_outputs": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "gate_outputs": NeuralType(('B', 'T'), LogitsType()),
            "alignments": NeuralType(('B', 'T', 'T'), SequenceToSequenceAlignmentType()),
        }
        if not self.training:
            output_dict["mel_lengths"] = NeuralType(('B'), LengthsType())
        return output_dict

    @typecheck()
    def forward(self, *args, **kwargs):
        if self.training:
            return self.train_forward(**kwargs)
        return self.infer(**kwargs)

    def get_go_frame(self, memory):
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0), int(decoder_inputs.size(1) / self.n_frames_per_step), -1,
        )
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        # Add a -1 to prevent squeezing the batch dimension in case
        # batch is 1
        gate_outputs = torch.stack(gate_outputs).squeeze(-1).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        cell_input = torch.cat((decoder_input, self.attention_context), -1)

        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell)
        )
        self.attention_hidden = F.dropout(self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1), self.attention_weights_cum.unsqueeze(1)), dim=1,
        )
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory, attention_weights_cat, self.mask,
        )

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat((self.attention_hidden, self.attention_context), -1)

        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell)
        )
        self.decoder_hidden = F.dropout(self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat((self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def train_forward(self, *, memory, decoder_inputs, memory_lengths):
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments

    def infer(self, *, memory, memory_lengths):
        decoder_input = self.get_go_frame(memory)

        if memory.size(0) > 1:
            mask = ~get_mask_from_lengths(memory_lengths)
        else:
            mask = None

        self.initialize_decoder_states(memory, mask=mask)

        mel_lengths = torch.zeros([memory.size(0)], dtype=torch.int32).to(memory.device)
        not_finished = torch.ones([memory.size(0)], dtype=torch.int32).to(memory.device)

        mel_outputs, gate_outputs, alignments = [], [], []
        stepped = False
        while True:
            decoder_input = self.prenet(decoder_input, inference=True)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            dec = torch.le(torch.sigmoid(gate_output.data), self.gate_threshold).to(torch.int32).squeeze(1)

            not_finished = not_finished * dec
            mel_lengths += not_finished

            if self.early_stopping and torch.sum(not_finished) == 0 and stepped:
                break
            stepped = True

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if len(mel_outputs) == self.max_decoder_steps:
                logging.warning("Reached max decoder steps %d.", self.max_decoder_steps)
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments, mel_lengths


class Postnet(NeuralModule):
    def __init__(
        self,
        n_mel_channels: int,
        postnet_embedding_dim: int,
        postnet_kernel_size: int,
        postnet_n_convolutions: int,
        p_dropout: float = 0.5,
    ):
        """
        Tacotron 2 Postnet. A convolutional network with postnet_n_convolutions number of layers. Each layer has a
        kernel of postnet_kernel_size. Each layer apart from the last outputs postnet_embedding_dim channels, the last
        outputs n_mel_channels channels. After each layer is a dropout layer with p_dropout% drop. The last linear has
        no activation, all intermediate layers have tanh activation.

        Args:
            n_mel_channels (int): Number of mel channels to output from Posnet.
            postnet_embedding_dim (int): Number of channels to output from the intermediate layers.
            postnet_kernel_size (int): The kernel size for the convolution layers.
            postnet_n_convolutions (int): The number of convolutions layers.
            p_dropout (float): Dropout probability. Defaults to 0.5.
        """
        super().__init__()
        self.convolutions = torch.nn.ModuleList()

        self.convolutions.append(
            torch.nn.Sequential(
                ConvNorm(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain='tanh',
                ),
                torch.nn.BatchNorm1d(postnet_embedding_dim),
            )
        )

        for _ in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                torch.nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain='tanh',
                    ),
                    torch.nn.BatchNorm1d(postnet_embedding_dim),
                )
            )

        self.convolutions.append(
            torch.nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain='linear',
                ),
                torch.nn.BatchNorm1d(n_mel_channels),
            )
        )
        self.p_dropout = p_dropout

    @property
    def input_types(self):
        return {
            "mel_spec": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
        }

    @property
    def output_types(self):
        return {
            "mel_spec": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
        }

    @typecheck()
    def forward(self, *, mel_spec):
        mel_spec_out = mel_spec
        for i in range(len(self.convolutions) - 1):
            mel_spec_out = F.dropout(torch.tanh(self.convolutions[i](mel_spec_out)), self.p_dropout, self.training)
        mel_spec_out = F.dropout(self.convolutions[-1](mel_spec_out), self.p_dropout, self.training)

        return mel_spec + mel_spec_out
