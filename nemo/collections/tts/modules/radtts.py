# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import torch.nn.functional as F
from torch import nn

from nemo.collections.tts.modules.attribute_prediction_model import get_attribute_prediction_model
from nemo.collections.tts.modules.common import (
    AffineTransformationLayer,
    BiLSTM,
    ConvAttention,
    ExponentialClass,
    Invertible1x1Conv,
    Invertible1x1ConvLUS,
    LinearNorm,
    get_radtts_encoder,
)
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths, mas_width1, regulate_len
from nemo.core.classes import Exportable, NeuralModule
from nemo.core.neural_types.elements import Index, LengthsType, MelSpectrogramType, TokenDurationType, TokenIndex
from nemo.core.neural_types.neural_type import NeuralType


@torch.jit.script
def pad_dur(dur, txt_enc):
    if dur.shape[-1] < txt_enc.shape[-1]:
        to_pad = txt_enc.shape[-1] - dur.shape[-1]
        dur = F.pad(dur, [0, to_pad])
    return dur


@torch.jit.script
def pad_energy_avg_and_f0(energy_avg, f0, max_out_len):
    to_pad = int(max_out_len - energy_avg.shape[1])
    if to_pad > 0:
        f0 = F.pad(f0[None], [0, to_pad])[0]
        energy_avg = F.pad(energy_avg[None], [0, to_pad])[0]
    to_pad = int(max_out_len - f0.shape[1])
    if to_pad > 0:
        f0 = F.pad(f0[None], [0, to_pad])[0]
    return energy_avg, f0


def adjust_f0(f0, f0_mean, f0_std, vmask_bool, musical_scaling=True):
    if f0_mean > 0.0:
        if musical_scaling:
            f0_mu, f0_sigma = f0[vmask_bool].mean(), f0[vmask_bool].std()
            f0_factor = f0_mean / f0_mu
            f0[vmask_bool] *= f0_factor
        else:
            f0_sigma, f0_mu = torch.std_mean(f0[vmask_bool])
            f0 = ((f0 - f0_mu) / f0_sigma).to(dtype=f0.dtype)
            f0_std = f0_std if f0_std > 0 else f0_sigma
            f0 = (f0 * f0_std + f0_mean).to(dtype=f0.dtype)
            f0 = f0.masked_fill(~vmask_bool, 0.0)
    return f0


class FlowStep(nn.Module):
    def __init__(
        self,
        n_mel_channels,
        n_context_dim,
        n_layers,
        affine_model='simple_conv',
        scaling_fn='exp',
        matrix_decomposition='',
        affine_activation='softplus',
        use_partial_padding=False,
    ):
        super(FlowStep, self).__init__()
        if matrix_decomposition == 'LUS':
            self.invtbl_conv = Invertible1x1ConvLUS(n_mel_channels)
        else:
            self.invtbl_conv = Invertible1x1Conv(n_mel_channels)

        self.affine_tfn = AffineTransformationLayer(
            n_mel_channels,
            n_context_dim,
            n_layers,
            affine_model=affine_model,
            scaling_fn=scaling_fn,
            affine_activation=affine_activation,
            use_partial_padding=use_partial_padding,
        )

    def forward(self, z, context, inverse=False, seq_lens=None):
        if inverse:  # for inference z-> mel
            z = self.affine_tfn(z, context, inverse, seq_lens=seq_lens)
            z = self.invtbl_conv(z, inverse)
            return z
        else:  # training mel->z
            z, log_det_W = self.invtbl_conv(z)
            z, log_s = self.affine_tfn(z, context, seq_lens=seq_lens)
            return z, log_det_W, log_s


class RadTTSModule(NeuralModule, Exportable):
    """
    Takes model parameters (modelConfig) from config file to initialize radtts module.
    Specify the type of training in the include_modules parameter. "decatnvpred" for decoder training. and "decatnunvbiasdpmvpredapm" for feature training
    n_speakers (int): Number of speakers
    n_speaker_dim (int): number of speakers dimension
    n_text (int): Symbols embedding size
    n_text_dim (int):
    n_flows (int):
    n_conv_layers_per_step (int): number of convolution layers per step
    dummy_speaker_embedding (bool):
    include_modules (string): A string that describes what to train. "decatnvpred" for decoder training. and "decatnunvbiasdpmvpredapm" for feature training.
    scaling_fn (string): scaling function
    decoder_use_partial_padding (Bool): Set this to True to add partial padding
    learn_alignments (Bool): set this to true to learn alignments
    attn_use_CTC (Bool): set True to use CTC
    n_f0_dims (int): number of Pitch dimension
    n_early_size (int):
    n_early_every (int):
    n_group_size (int):
    decoder_use_unvoiced_bias (bool):
    context_lstm_w_f0_and_energy (bool):
    use_first_order_features (bool):
    ap_pred_log_f0 (bool):
    dur_model_config: model configuration for duration
    f0_model_config: model configuration for Pitch
    energy_model_config: model configuration for energy
    """

    def __init__(
        self,
        n_speakers,
        n_speaker_dim,
        n_text,
        n_text_dim,
        n_flows,
        n_conv_layers_per_step,
        n_mel_channels,
        dummy_speaker_embedding,
        n_early_size,
        n_early_every,
        n_group_size,
        affine_model,
        dur_model_config,
        f0_model_config,
        energy_model_config,
        v_model_config=None,
        include_modules='dec',
        scaling_fn='exp',
        matrix_decomposition='',
        learn_alignments=False,
        affine_activation='softplus',
        attn_use_CTC=True,
        use_context_lstm=False,
        context_lstm_norm=None,
        n_f0_dims=0,
        n_energy_avg_dims=0,
        context_lstm_w_f0_and_energy=True,
        use_first_order_features=False,
        unvoiced_bias_activation='',
        ap_pred_log_f0=False,
        **kwargs,
    ):
        super(RadTTSModule, self).__init__()
        assert n_early_size % 2 == 0
        self.n_mel_channels = n_mel_channels
        self.n_f0_dims = n_f0_dims  # >= 1 to trains with f0
        self.n_energy_avg_dims = n_energy_avg_dims  # >= 1 trains with energy
        self.decoder_use_partial_padding = kwargs['decoder_use_partial_padding']
        self.n_speaker_dim = n_speaker_dim
        assert self.n_speaker_dim % 2 == 0
        self.speaker_embedding = torch.nn.Embedding(n_speakers, self.n_speaker_dim)
        self.embedding = torch.nn.Embedding(n_text, n_text_dim)
        self.flows = torch.nn.ModuleList()
        self.encoder = get_radtts_encoder(encoder_embedding_dim=n_text_dim)
        self.dummy_speaker_embedding = dummy_speaker_embedding
        self.learn_alignments = learn_alignments
        self.affine_activation = affine_activation
        self.include_modules = include_modules
        self.attn_use_CTC = bool(attn_use_CTC)
        self.use_context_lstm = bool(use_context_lstm)
        self.context_lstm_norm = context_lstm_norm
        self.context_lstm_w_f0_and_energy = context_lstm_w_f0_and_energy
        self.use_first_order_features = bool(use_first_order_features)
        self.decoder_use_unvoiced_bias = kwargs['decoder_use_unvoiced_bias']
        self.ap_pred_log_f0 = ap_pred_log_f0
        self.ap_use_unvoiced_bias = kwargs['ap_use_unvoiced_bias']

        if 'atn' in include_modules or 'dec' in include_modules:
            if self.learn_alignments:
                self.attention = ConvAttention(n_mel_channels, self.n_speaker_dim, n_text_dim)

            self.n_flows = n_flows
            self.n_group_size = n_group_size

            n_flowstep_cond_dims = self.n_speaker_dim + (n_text_dim + n_f0_dims + n_energy_avg_dims) * n_group_size

            if self.use_context_lstm:
                n_in_context_lstm = self.n_speaker_dim + n_text_dim * n_group_size
                n_context_lstm_hidden = int((self.n_speaker_dim + n_text_dim * n_group_size) / 2)

                if self.context_lstm_w_f0_and_energy:
                    n_in_context_lstm = n_f0_dims + n_energy_avg_dims + n_text_dim
                    n_in_context_lstm *= n_group_size
                    n_in_context_lstm += self.n_speaker_dim
                    n_flowstep_cond_dims = self.n_speaker_dim + n_text_dim * n_group_size

                self.context_lstm = BiLSTM(
                    input_size=n_in_context_lstm, hidden_size=n_context_lstm_hidden, num_layers=1,
                )

            if self.n_group_size > 1:
                self.unfold_params = {
                    'kernel_size': (n_group_size, 1),
                    'stride': n_group_size,
                    'padding': 0,
                    'dilation': 1,
                }
                self.unfold_mod = nn.Unfold(**self.unfold_params)

            self.exit_steps = []
            self.n_early_size = n_early_size
            n_mel_channels = n_mel_channels * n_group_size

            for i in range(self.n_flows):
                if i > 0 and i % n_early_every == 0:  # early exiting
                    n_mel_channels -= self.n_early_size
                    self.exit_steps.append(i)

                self.flows.append(
                    FlowStep(
                        n_mel_channels,
                        n_flowstep_cond_dims,
                        n_conv_layers_per_step,
                        affine_model,
                        scaling_fn,
                        matrix_decomposition,
                        affine_activation=affine_activation,
                        use_partial_padding=self.decoder_use_partial_padding,
                    )
                )

        if 'dpm' in include_modules:
            dur_model_config['hparams']['n_speaker_dim'] = n_speaker_dim
            self.dur_pred_layer = get_attribute_prediction_model(dur_model_config)

        self.use_unvoiced_bias = False
        self.use_vpred_module = False
        self.ap_use_voiced_embeddings = kwargs['ap_use_voiced_embeddings']

        if self.decoder_use_unvoiced_bias or self.ap_use_unvoiced_bias:
            assert unvoiced_bias_activation in {'relu', 'exp'}
            self.use_unvoiced_bias = True
            if unvoiced_bias_activation == 'relu':
                unvbias_nonlin = nn.ReLU()
            elif unvoiced_bias_activation == 'exp':
                unvbias_nonlin = ExponentialClass()
            else:
                exit(1)  # we won't reach here anyway due to the assertion
            self.unvoiced_bias_module = nn.Sequential(LinearNorm(n_text_dim, 1), unvbias_nonlin)

        # all situations in which the vpred module is necessary
        if self.ap_use_voiced_embeddings or self.use_unvoiced_bias or 'vpred' in include_modules:
            self.use_vpred_module = True

        if self.use_vpred_module:
            v_model_config['hparams']['n_speaker_dim'] = n_speaker_dim
            self.v_pred_module = get_attribute_prediction_model(v_model_config)
            # 4 embeddings, first two are scales, second two are biases
            if self.ap_use_voiced_embeddings:
                self.v_embeddings = torch.nn.Embedding(4, n_text_dim)
            self.v_pred_threshold = 0.5

        if 'apm' in include_modules:
            f0_model_config['hparams']['n_speaker_dim'] = n_speaker_dim
            energy_model_config['hparams']['n_speaker_dim'] = n_speaker_dim
            if self.use_first_order_features:
                f0_model_config['hparams']['n_in_dim'] = 2
                energy_model_config['hparams']['n_in_dim'] = 2
                if (
                    'spline_flow_params' in f0_model_config['hparams']
                    and f0_model_config['hparams']['spline_flow_params'] is not None
                ):
                    f0_model_config['hparams']['spline_flow_params']['n_in_channels'] = 2
                if (
                    'spline_flow_params' in energy_model_config['hparams']
                    and energy_model_config['hparams']['spline_flow_params'] is not None
                ):
                    energy_model_config['hparams']['spline_flow_params']['n_in_channels'] = 2
            else:
                if (
                    'spline_flow_params' in f0_model_config['hparams']
                    and f0_model_config['hparams']['spline_flow_params'] is not None
                ):
                    f0_model_config['hparams']['spline_flow_params']['n_in_channels'] = f0_model_config['hparams'][
                        'n_in_dim'
                    ]
                if (
                    'spline_flow_params' in energy_model_config['hparams']
                    and energy_model_config['hparams']['spline_flow_params'] is not None
                ):
                    energy_model_config['hparams']['spline_flow_params']['n_in_channels'] = energy_model_config[
                        'hparams'
                    ]['n_in_dim']

            self.f0_pred_module = get_attribute_prediction_model(f0_model_config)
            self.energy_pred_module = get_attribute_prediction_model(energy_model_config)

    def encode_speaker(self, spk_ids):
        spk_ids = spk_ids * 0 if self.dummy_speaker_embedding else spk_ids
        spk_vecs = self.speaker_embedding(spk_ids)
        return spk_vecs

    def encode_text(self, text, in_lens):
        # text_embeddings: b x len_text x n_text_dim
        text_embeddings = self.embedding(text).transpose(1, 2)
        # text_enc: b x n_text_dim x encoder_dim (512)
        text_enc = self.encoder(text_embeddings, in_lens).transpose(1, 2)
        return text_enc, text_embeddings

    def preprocess_context(self, context, speaker_vecs, out_lens, f0, energy_avg, assume_padded=False):
        if self.n_group_size > 1:
            context = self.unfold(context, assume_padded=assume_padded)

            if f0 is not None:
                f0 = self.unfold(f0[:, None, :], assume_padded=assume_padded)
            if energy_avg is not None:
                energy_avg = self.unfold(energy_avg[:, None, :], assume_padded=assume_padded)
        speaker_vecs = speaker_vecs[..., None].expand(-1, -1, context.shape[2])
        context_w_spkvec = torch.cat((context, speaker_vecs), 1)

        if self.use_context_lstm:
            if self.context_lstm_w_f0_and_energy:
                if f0 is not None:
                    context_w_spkvec = torch.cat((context_w_spkvec, f0), 1)

                if energy_avg is not None:
                    context_w_spkvec = torch.cat((context_w_spkvec, energy_avg), 1)

            unfolded_out_lens = out_lens // self.n_group_size
            context_lstm_padded_output = self.context_lstm(context_w_spkvec.transpose(1, 2), unfolded_out_lens)
            context_w_spkvec = context_lstm_padded_output.transpose(1, 2)

        if not self.context_lstm_w_f0_and_energy:
            if f0 is not None:
                context_w_spkvec = torch.cat((context_w_spkvec, f0), 1)

            if energy_avg is not None:
                context_w_spkvec = torch.cat((context_w_spkvec, energy_avg), 1)

        return context_w_spkvec

    def fold(self, mel):
        """Inverse of the self.unfold() operation used for the
        grouping or "squeeze" operation on input

        Args:
            mel: B x C x T tensor of temporal data
        """
        b, d, t = mel.shape
        mel = mel.reshape(b, -1, self.n_group_size, t).transpose(2, 3)
        return mel.reshape(b, -1, t * self.n_group_size)

    def unfold(self, mel, assume_padded=False):
        """operation used for the
        grouping or "squeeze" operation on input

        Args:
            mel: B x C x T tensor of temporal data
        """
        # for inference, mel is being padded beforehand
        if assume_padded:
            b, d, t = mel.shape
            mel = mel.reshape(b, d, -1, self.n_group_size).transpose(2, 3)
            return mel.reshape(b, d * self.n_group_size, -1)
        else:
            return self.unfold_mod(mel.unsqueeze(-1))

    def binarize_attention(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS. These will
        no longer receive a gradient
        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        b_size = attn.shape[0]
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = torch.zeros_like(attn)
            for ind in range(b_size):
                hard_attn = mas_width1(attn_cpu[ind, 0, : out_lens[ind], : in_lens[ind]])
                attn_out[ind, 0, : out_lens[ind], : in_lens[ind]] = torch.tensor(hard_attn, device=attn.get_device())
        return attn_out

    def get_first_order_features(self, feats, dilation=1):
        """
        feats: b x max_length
        out_lens: b-dim
        """
        # add an extra column
        feats_extended_R = torch.cat((feats, torch.zeros_like(feats[:, 0:dilation])), dim=1)
        feats_extended_L = torch.cat((torch.zeros_like(feats[:, 0:dilation]), feats), dim=1)
        dfeats_R = feats_extended_R[:, dilation:] - feats
        dfeats_L = feats - feats_extended_L[:, 0:-dilation]

        return (dfeats_R + dfeats_L) * 0.5

    def apply_voice_mask_to_text(self, text_enc, voiced_mask):
        """
        text_enc: b x C x N
        voiced_mask: b x N
        """
        voiced_mask = voiced_mask.unsqueeze(1)
        voiced_embedding_s = self.v_embeddings.weight[0:1, :, None]
        unvoiced_embedding_s = self.v_embeddings.weight[1:2, :, None]
        voiced_embedding_b = self.v_embeddings.weight[2:3, :, None]
        unvoiced_embedding_b = self.v_embeddings.weight[3:4, :, None]
        scale = torch.sigmoid(voiced_embedding_s * voiced_mask + unvoiced_embedding_s * (1 - voiced_mask))
        bias = 0.1 * torch.tanh(voiced_embedding_b * voiced_mask + unvoiced_embedding_b * (1 - voiced_mask))
        return text_enc * scale + bias

    def forward(
        self,
        mel,
        speaker_ids,
        text,
        in_lens,
        out_lens,
        binarize_attention=False,
        attn_prior=None,
        f0=None,
        energy_avg=None,
        voiced_mask=None,
    ):
        speaker_vecs = self.encode_speaker(speaker_ids)
        text_enc, text_embeddings = self.encode_text(text, in_lens)

        log_s_list, log_det_W_list, z_mel = [], [], []
        attn_hard = None
        if 'atn' in self.include_modules or 'dec' in self.include_modules:
            # make sure to do the alignments before folding
            attn_mask = ~get_mask_from_lengths(in_lens)[..., None]
            # attn_mask should be 1 for unsd t-steps in text_enc_w_spkvec tensor
            attn_soft, attn_logprob = self.attention(
                mel, text_embeddings, out_lens, attn_mask, key_lens=in_lens, attn_prior=attn_prior
            )

            if binarize_attention:
                attn = self.binarize_attention(attn_soft, in_lens, out_lens)
                attn_hard = attn
            else:
                attn = attn_soft

            context = torch.bmm(text_enc, attn.squeeze(1).transpose(1, 2))
        else:
            raise ValueError(
                f"Something unexpected happened. Both 'atn' and 'dec' are not included in 'self.include_modules'. Please double-check."
            )

        f0_bias = 0
        # unvoiced bias forward pass
        voiced_mask_bool = voiced_mask.bool()
        if self.use_unvoiced_bias:
            f0_bias = self.unvoiced_bias_module(context.permute(0, 2, 1))
            f0_bias = -f0_bias[..., 0]
            f0_bias.masked_fill_(voiced_mask_bool, 0.0)

        # mel decoder forward pass
        if 'dec' in self.include_modules:
            if self.n_group_size > 1:
                # might truncate some frames at the end, but that's ok
                # sometimes referred to as the "squeeze" operation
                # invert this by calling self.fold(mel_or_z)
                mel = self.unfold(mel)
            # where context is folded
            # mask f0 in case values are interpolated
            context_w_spkvec = self.preprocess_context(
                context, speaker_vecs, out_lens, f0 * voiced_mask + f0_bias, energy_avg
            )

            log_s_list, log_det_W_list, z_out = [], [], []
            unfolded_seq_lens = out_lens // self.n_group_size
            for i, flow_step in enumerate(self.flows):
                if i in self.exit_steps:
                    z = mel[:, : self.n_early_size]
                    z_out.append(z)
                    mel = mel[:, self.n_early_size :]
                mel, log_det_W, log_s = flow_step(mel, context_w_spkvec, seq_lens=unfolded_seq_lens)
                log_s_list.append(log_s)
                log_det_W_list.append(log_det_W)

            z_out.append(mel)
            z_mel = torch.cat(z_out, 1)

        # duration predictor forward pass
        duration_model_outputs = None
        if 'dpm' in self.include_modules:
            if attn_hard is None:
                attn_hard = self.binarize_attention(attn_soft, in_lens, out_lens)

            # convert hard attention to durations
            attn_hard_reduced = attn_hard.sum(2)[:, 0, :]
            duration_model_outputs = self.dur_pred_layer(
                torch.detach(text_enc), torch.detach(speaker_vecs), torch.detach(attn_hard_reduced.float()), in_lens
            )

        # f0, energy, vpred predictors forward pass
        f0_model_outputs = None
        energy_model_outputs = None
        vpred_model_outputs = None
        if 'apm' in self.include_modules:
            if attn_hard is None:
                attn_hard = self.binarize_attention(attn_soft, in_lens, out_lens)

            # convert hard attention to durations
            if binarize_attention:
                text_enc_time_expanded = context.clone()
            else:
                text_enc_time_expanded = torch.bmm(text_enc, attn_hard.squeeze(1).transpose(1, 2))

            if self.use_vpred_module:
                # unvoiced bias requires  voiced mask prediction
                vpred_model_outputs = self.v_pred_module(
                    torch.detach(text_enc_time_expanded),
                    torch.detach(speaker_vecs),
                    torch.detach(voiced_mask),
                    out_lens,
                )

                # affine transform context using voiced mask
                if self.ap_use_voiced_embeddings:
                    text_enc_time_expanded = self.apply_voice_mask_to_text(text_enc_time_expanded, voiced_mask)
            if self.ap_use_unvoiced_bias:  # whether to use the unvoiced bias in the attribute predictor
                f0_target = torch.detach(f0 * voiced_mask + f0_bias)
            else:
                f0_target = torch.detach(f0)
            # fit to log f0 in f0 predictor
            f0_target[voiced_mask_bool] = torch.log(f0_target[voiced_mask_bool])
            f0_target = f0_target / 6  # scale to ~ [0, 1] in log space
            energy_avg = energy_avg * 2 - 1  # scale to ~ [-1, 1]

            if self.use_first_order_features:
                df0 = self.get_first_order_features(f0_target)
                denergy_avg = self.get_first_order_features(energy_avg)

                f0_voiced = torch.cat((f0_target[:, None], df0[:, None]), dim=1)
                energy_avg = torch.cat((energy_avg[:, None], denergy_avg[:, None]), dim=1)

                f0_voiced = f0_voiced * 3  # scale to ~ 1 std
                energy_avg = energy_avg * 3  # scale to ~ 1 std
            else:
                f0_voiced = f0_target * 2  # scale to ~ 1 std
                energy_avg = energy_avg * 1.4  # scale to ~ 1 std
            f0_model_outputs = self.f0_pred_module(
                text_enc_time_expanded, torch.detach(speaker_vecs), f0_voiced, out_lens
            )

            energy_model_outputs = self.energy_pred_module(
                text_enc_time_expanded, torch.detach(speaker_vecs), energy_avg, out_lens
            )

        outputs = {
            'z_mel': z_mel,
            'log_det_W_list': log_det_W_list,
            'log_s_list': log_s_list,
            'duration_model_outputs': duration_model_outputs,
            'f0_model_outputs': f0_model_outputs,
            'energy_model_outputs': energy_model_outputs,
            'vpred_model_outputs': vpred_model_outputs,
            'attn_soft': attn_soft,
            'attn': attn,
            'text_embeddings': text_embeddings,
            'attn_logprob': attn_logprob,
        }

        return outputs

    def infer(
        self,
        speaker_id,
        text,
        sigma=0.7,
        speaker_id_text=None,
        speaker_id_attributes=None,
        pace=None,
        token_duration_max=100,
        in_lens=None,
        dur=None,
        f0=None,
        f0_mean=0.0,
        f0_std=0.0,
        energy_avg=None,
        voiced_mask=None,
        pitch_shift=None,
    ):

        batch_size = text.shape[0]
        if in_lens is None:
            in_lens = text.new_ones((batch_size,), dtype=torch.int64) * text.shape[1]
            txt_len_pad_removed = text.shape[1]
        else:
            txt_len_pad_removed = torch.max(in_lens)
            # borisf : this should not be needed as long as we have properly formed input batch
            text = text[:, :txt_len_pad_removed]

        spk_vec = self.encode_speaker(speaker_id)

        if speaker_id_text is None:
            speaker_id_text = speaker_id
        if speaker_id_attributes is None:
            speaker_id_attributes = speaker_id
        spk_vec_text = self.encode_speaker(speaker_id_text)
        spk_vec_attributes = self.encode_speaker(speaker_id_attributes)
        txt_enc, _ = self.encode_text(text, in_lens)

        if dur is None:
            # get token durations
            dur = self.dur_pred_layer.infer(txt_enc, spk_vec_text, lens=in_lens)
            dur = pad_dur(dur, txt_enc)
            dur = dur[:, 0]
            dur = dur.clamp(0, token_duration_max)

        if pace is None:
            pace = txt_enc.new_ones((batch_size, txt_len_pad_removed))
        else:
            pace = pace[:, :txt_len_pad_removed]

        txt_enc_time_expanded, out_lens = regulate_len(
            dur, txt_enc.transpose(1, 2), pace, group_size=self.n_group_size, dur_lens=in_lens,
        )
        n_groups = torch.div(out_lens, self.n_group_size, rounding_mode='floor')
        max_out_len = torch.max(out_lens)

        txt_enc_time_expanded.transpose_(1, 2)
        if voiced_mask is None:
            if self.use_vpred_module:
                # get logits
                voiced_mask = self.v_pred_module.infer(txt_enc_time_expanded, spk_vec_attributes, lens=out_lens)
                voiced_mask_bool = torch.sigmoid(voiced_mask[:, 0]) > self.v_pred_threshold
                voiced_mask = voiced_mask_bool.to(dur.dtype)
            else:
                voiced_mask_bool = None
        else:
            voiced_mask_bool = voiced_mask.bool()

        ap_txt_enc_time_expanded = txt_enc_time_expanded
        # voice mask augmentation only used for attribute prediction
        if self.ap_use_voiced_embeddings:
            ap_txt_enc_time_expanded = self.apply_voice_mask_to_text(txt_enc_time_expanded, voiced_mask)

        f0_bias = 0
        # unvoiced bias forward pass
        if self.use_unvoiced_bias:
            f0_bias = self.unvoiced_bias_module(txt_enc_time_expanded.permute(0, 2, 1))
            f0_bias = -f0_bias[..., 0]

        if f0 is None:
            f0 = self.infer_f0(ap_txt_enc_time_expanded, spk_vec_attributes, voiced_mask_bool, out_lens)[:, 0]

        f0 = adjust_f0(f0, f0_mean, f0_std, voiced_mask_bool, musical_scaling=False)

        if energy_avg is None:
            energy_avg = self.infer_energy(ap_txt_enc_time_expanded, spk_vec, out_lens)[:, 0]

        # replication pad, because ungrouping with different group sizes
        # may lead to mismatched lengths
        # FIXME: use replication pad
        (energy_avg, f0) = pad_energy_avg_and_f0(energy_avg, f0, max_out_len)

        if pitch_shift is not None:
            pitch_shift_spec_len, _ = regulate_len(
                dur,
                pitch_shift[:, :txt_len_pad_removed].unsqueeze(-1),
                pace,
                group_size=self.n_group_size,
                dur_lens=in_lens,
            )
            f0_bias = pitch_shift_spec_len.squeeze(-1) + f0_bias

        context_w_spkvec = self.preprocess_context(
            txt_enc_time_expanded, spk_vec, out_lens, (f0 + f0_bias) * voiced_mask, energy_avg, assume_padded=True,
        )

        residual = txt_enc.new_zeros(batch_size, 80 * self.n_group_size, torch.max(n_groups))
        if sigma > 0.0:
            residual = torch.normal(residual) * sigma

        # map from z sample to data
        num_steps_to_exit = len(self.exit_steps)
        split = num_steps_to_exit * self.n_early_size
        mel = residual[:, split:]
        residual = residual[:, :split]

        for i, flow_step in enumerate(reversed(self.flows)):
            curr_step = self.n_flows - i - 1
            mel = flow_step(mel, context_w_spkvec, inverse=True, seq_lens=n_groups)
            if num_steps_to_exit > 0 and curr_step == self.exit_steps[num_steps_to_exit - 1]:
                # concatenate the next chunk of z
                num_steps_to_exit = num_steps_to_exit - 1
                split = num_steps_to_exit * self.n_early_size
                residual_to_add = residual[:, split:]
                residual = residual[:, :split]
                mel = torch.cat((residual_to_add, mel), 1)

        if self.n_group_size > 1:
            mel = self.fold(mel)

        return {'mel': mel, 'out_lens': out_lens, 'dur': dur, 'f0': f0, 'energy_avg': energy_avg}

    def infer_f0(self, txt_enc_time_expanded, spk_vec, voiced_mask=None, lens=None):
        f0 = self.f0_pred_module.infer(txt_enc_time_expanded, spk_vec, lens)

        # constants
        if self.ap_pred_log_f0:
            if self.use_first_order_features:
                f0 = f0[:, 0:1, :] / 3
            else:
                f0 = f0 / 2
            f0 = f0 * 6
        else:
            f0 = f0 / 6
            f0 = f0 / 640

        if voiced_mask is None:
            voiced_mask = f0 > 0.0
        else:
            if len(voiced_mask.shape) == 2:
                voiced_mask = voiced_mask[:, None]
                # due to grouping, f0 might be 1 frame short
                voiced_mask = voiced_mask[:, :, : f0.shape[-1]]

        if self.ap_pred_log_f0:
            # if variable is set, decoder sees linear f0
            f0 = torch.exp(f0).to(dtype=f0.dtype)
        f0.masked_fill_(~voiced_mask, 0.0)
        return f0

    def infer_energy(self, txt_enc_time_expanded, spk_vec, lens):
        energy = self.energy_pred_module.infer(txt_enc_time_expanded, spk_vec, lens)

        # magic constants
        if self.use_first_order_features:
            energy = energy / 3
        else:
            energy = energy / 1.4
        energy = (energy + 1) / 2
        return energy

    def remove_norms(self):
        """Removes spectral and weightnorms from model. Call before inference
        """
        dev = next(self.parameters()).device
        for name, module in self.named_modules():
            try:
                nn.utils.remove_spectral_norm(module, name='weight_hh_l0')
                print("Removed spectral norm from {}".format(name))
            except:
                pass
            try:
                nn.utils.remove_spectral_norm(module, name='weight_hh_l0_reverse')
                print("Removed spectral norm from {}".format(name))
            except:
                pass
            try:
                nn.utils.remove_weight_norm(module)
                print("Removed wnorm from {}".format(name))
            except:
                pass
        self.to(device=dev)

    @property
    def input_types(self):
        return {
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "lens": NeuralType(('B'), LengthsType(), optional=True),
            "speaker_id": NeuralType(('B'), Index()),
            "speaker_id_text": NeuralType(('B'), Index()),
            "speaker_id_attributes": NeuralType(('B'), Index()),
        }

    @property
    def output_types(self):
        return {
            "spect": NeuralType(('B', 'D', 'T_spec'), MelSpectrogramType()),
            "num_frames": NeuralType(('B'), TokenDurationType()),
            "durs_predicted": NeuralType(('B', 'T_text'), TokenDurationType()),
        }
