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
#
# BSD 3-Clause License
#
# Copyright (c) 2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from enum import Enum
from typing import Optional, Tuple

import librosa
import matplotlib.pylab as plt
import numpy as np
import torch
from einops import rearrange
from numba import jit, prange

from nemo.collections.tts.torch.tts_data_types import DATA_STR2DATA_CLASS, MAIN_DATA_TYPES, WithLens
from nemo.utils import logging
from nemo.utils.decorators import deprecated

HAVE_WANDB = True
try:
    import wandb
except ModuleNotFoundError:
    HAVE_WANDB = False

try:
    from pytorch_lightning.utilities import rank_zero_only
except ModuleNotFoundError:
    from functools import wraps

    def rank_zero_only(fn):
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            logging.error(
                f"Function {fn} requires lighting to be installed, but it was not found. Please install lightning first"
            )
            exit(1)


class OperationMode(Enum):
    """Training or Inference (Evaluation) mode"""

    training = 0
    validation = 1
    infer = 2


def get_batch_size(train_dataloader):
    if train_dataloader.batch_size is not None:
        return train_dataloader.batch_size
    elif train_dataloader.batch_sampler is not None:
        if train_dataloader.batch_sampler.micro_batch_size is not None:
            return train_dataloader.batch_sampler.micro_batch_size
        else:
            raise ValueError(f'Could not find batch_size from batch_sampler: {train_dataloader.batch_sampler}')
    else:
        raise ValueError(f'Could not find batch_size from train_dataloader: {train_dataloader}')


def get_num_workers(trainer):
    return trainer.num_devices * trainer.num_nodes


def binarize_attention(attn, in_len, out_len):
    """Convert soft attention matrix to hard attention matrix.

    Args:
        attn (torch.Tensor): B x 1 x max_mel_len x max_text_len. Soft attention matrix.
        in_len (torch.Tensor): B. Lengths of texts.
        out_len (torch.Tensor): B. Lengths of spectrograms.

    Output:
        attn_out (torch.Tensor): B x 1 x max_mel_len x max_text_len. Hard attention matrix, final dim max_text_len should sum to 1.
    """
    b_size = attn.shape[0]
    with torch.no_grad():
        attn_cpu = attn.data.cpu().numpy()
        attn_out = torch.zeros_like(attn)
        for ind in range(b_size):
            hard_attn = mas(attn_cpu[ind, 0, : out_len[ind], : in_len[ind]])
            attn_out[ind, 0, : out_len[ind], : in_len[ind]] = torch.tensor(hard_attn, device=attn.device)
    return attn_out


def binarize_attention_parallel(attn, in_lens, out_lens):
    """For training purposes only. Binarizes attention with MAS.
           These will no longer receive a gradient.

        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
    with torch.no_grad():
        log_attn_cpu = torch.log(attn.data).cpu().numpy()
        attn_out = b_mas(log_attn_cpu, in_lens.cpu().numpy(), out_lens.cpu().numpy(), width=1)
    return torch.from_numpy(attn_out).to(attn.device)


def get_mask_from_lengths(lengths: Optional[torch.Tensor] = None, x: Optional[torch.Tensor] = None,) -> torch.Tensor:
    """Constructs binary mask from a 1D torch tensor of input lengths

    Args:
        lengths: Optional[torch.tensor] (torch.tensor): 1D tensor with lengths
        x: Optional[torch.tensor] = tensor to be used on, last dimension is for mask 
    Returns:
        mask (torch.tensor): num_sequences x max_length x 1 binary tensor
    """
    if lengths is None:
        assert x is not None
        return torch.ones(x.shape[-1], dtype=torch.bool, device=x.device)
    else:
        if x is None:
            max_len = torch.max(lengths)
        else:
            max_len = x.shape[-1]
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = ids < lengths.unsqueeze(1)
    return mask


def sort_tensor(
    context: torch.Tensor, lens: torch.Tensor, dim: Optional[int] = 0, descending: Optional[bool] = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sorts elements in context by the dim lengths specified in lens
    Args:
        context:  source tensor, sorted by lens
        lens: lengths of elements of context along the dimension dim
        dim: Optional[int] : dimension to sort by
    Returns:
        context: tensor sorted by lens along dimension dim
        lens_sorted: lens tensor, sorted
        ids_sorted: reorder ids to be used to restore original order
    
    """
    lens_sorted, ids_sorted = torch.sort(lens, descending=descending)
    context = torch.index_select(context, dim, ids_sorted)
    return context, lens_sorted, ids_sorted


def unsort_tensor(ordered: torch.Tensor, indices: torch.Tensor, dim: Optional[int] = 0) -> torch.Tensor:
    """Reverses the result of sort_tensor function:
       o, _, ids = sort_tensor(x,l) 
       assert unsort_tensor(o,ids) == x
    Args:
        ordered: context tensor, sorted by lengths
        indices: torch.tensor: 1D tensor with 're-order' indices returned by sort_tensor
    Returns:
        ordered tensor in original order (before calling sort_tensor)  
    """
    return torch.index_select(ordered, dim, indices.argsort(0))


@jit(nopython=True)
def mas(attn_map, width=1):
    # assumes mel x text
    opt = np.zeros_like(attn_map)
    attn_map = np.log(attn_map)
    attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]):  # for each text dim
            prev_j = np.arange(max(0, j - width), j + 1)
            prev_log = np.array([log_p[i - 1, prev_idx] for prev_idx in prev_j])

            ind = np.argmax(prev_log)
            log_p[i, j] = attn_map[i, j] + prev_log[ind]
            prev_ind[i, j] = prev_j[ind]

    # now backtrack
    curr_text_idx = attn_map.shape[1] - 1
    for i in range(attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1

    assert opt.sum(0).all()
    assert opt.sum(1).all()

    return opt


@jit(nopython=True)
def mas_width1(log_attn_map):
    """mas with hardcoded width=1"""
    # assumes mel x text
    neg_inf = log_attn_map.dtype.type(-np.inf)
    log_p = log_attn_map.copy()
    log_p[0, 1:] = neg_inf
    for i in range(1, log_p.shape[0]):
        prev_log1 = neg_inf
        for j in range(log_p.shape[1]):
            prev_log2 = log_p[i - 1, j]
            log_p[i, j] += max(prev_log1, prev_log2)
            prev_log1 = prev_log2

    # now backtrack
    opt = np.zeros_like(log_p)
    one = opt.dtype.type(1)
    j = log_p.shape[1] - 1
    for i in range(log_p.shape[0] - 1, 0, -1):
        opt[i, j] = one
        if log_p[i - 1, j - 1] >= log_p[i - 1, j]:
            j -= 1
            if j == 0:
                opt[1:i, j] = one
                break
    opt[0, j] = one
    return opt


@jit(nopython=True, parallel=True)
def b_mas(b_log_attn_map, in_lens, out_lens, width=1):
    assert width == 1
    attn_out = np.zeros_like(b_log_attn_map)

    for b in prange(b_log_attn_map.shape[0]):
        out = mas_width1(b_log_attn_map[b, 0, : out_lens[b], : in_lens[b]])
        attn_out[b, 0, : out_lens[b], : in_lens[b]] = out
    return attn_out


def griffin_lim(magnitudes, n_iters=50, n_fft=1024):
    """
    Griffin-Lim algorithm to convert magnitude spectrograms to audio signals
    """
    phase = np.exp(2j * np.pi * np.random.rand(*magnitudes.shape))
    complex_spec = magnitudes * phase
    signal = librosa.istft(complex_spec)
    if not np.isfinite(signal).all():
        logging.warning("audio was not finite, skipping audio saving")
        return np.array([0])

    for _ in range(n_iters):
        _, phase = librosa.magphase(librosa.stft(signal, n_fft=n_fft))
        complex_spec = magnitudes * phase
        signal = librosa.istft(complex_spec)
    return signal


@rank_zero_only
def log_audio_to_tb(
    swriter,
    spect,
    name,
    step,
    griffin_lim_mag_scale=1024,
    griffin_lim_power=1.2,
    sr=22050,
    n_fft=1024,
    n_mels=80,
    fmax=8000,
):
    filterbank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=fmax)
    log_mel = spect.data.cpu().numpy().T
    mel = np.exp(log_mel)
    magnitude = np.dot(mel, filterbank) * griffin_lim_mag_scale
    audio = griffin_lim(magnitude.T ** griffin_lim_power)
    swriter.add_audio(name, audio / max(np.abs(audio)), step, sample_rate=sr)


@rank_zero_only
def tacotron2_log_to_tb_func(
    swriter,
    tensors,
    step,
    tag="train",
    log_images=False,
    log_images_freq=1,
    add_audio=True,
    griffin_lim_mag_scale=1024,
    griffin_lim_power=1.2,
    sr=22050,
    n_fft=1024,
    n_mels=80,
    fmax=8000,
):
    _, spec_target, mel_postnet, gate, gate_target, alignments = tensors
    if log_images and step % log_images_freq == 0:
        swriter.add_image(
            f"{tag}_alignment", plot_alignment_to_numpy(alignments[0].data.cpu().numpy().T), step, dataformats="HWC",
        )
        swriter.add_image(
            f"{tag}_mel_target", plot_spectrogram_to_numpy(spec_target[0].data.cpu().numpy()), step, dataformats="HWC",
        )
        swriter.add_image(
            f"{tag}_mel_predicted",
            plot_spectrogram_to_numpy(mel_postnet[0].data.cpu().numpy()),
            step,
            dataformats="HWC",
        )
        swriter.add_image(
            f"{tag}_gate",
            plot_gate_outputs_to_numpy(gate_target[0].data.cpu().numpy(), torch.sigmoid(gate[0]).data.cpu().numpy(),),
            step,
            dataformats="HWC",
        )

        if add_audio:
            filterbank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=fmax)
            log_mel = mel_postnet[0].data.cpu().numpy().T
            mel = np.exp(log_mel)
            magnitude = np.dot(mel, filterbank) * griffin_lim_mag_scale
            audio = griffin_lim(magnitude.T ** griffin_lim_power)
            swriter.add_audio(f"audio/{tag}_predicted", audio / max(np.abs(audio)), step, sample_rate=sr)

            log_mel = spec_target[0].data.cpu().numpy().T
            mel = np.exp(log_mel)
            magnitude = np.dot(mel, filterbank) * griffin_lim_mag_scale
            audio = griffin_lim(magnitude.T ** griffin_lim_power)
            swriter.add_audio(f"audio/{tag}_target", audio / max(np.abs(audio)), step, sample_rate=sr)


def tacotron2_log_to_wandb_func(
    swriter,
    tensors,
    step,
    tag="train",
    log_images=False,
    log_images_freq=1,
    add_audio=True,
    griffin_lim_mag_scale=1024,
    griffin_lim_power=1.2,
    sr=22050,
    n_fft=1024,
    n_mels=80,
    fmax=8000,
):
    _, spec_target, mel_postnet, gate, gate_target, alignments = tensors
    if not HAVE_WANDB:
        return
    if log_images and step % log_images_freq == 0:
        alignments = []
        specs = []
        gates = []
        alignments += [
            wandb.Image(plot_alignment_to_numpy(alignments[0].data.cpu().numpy().T), caption=f"{tag}_alignment",)
        ]
        alignments += [
            wandb.Image(plot_spectrogram_to_numpy(spec_target[0].data.cpu().numpy()), caption=f"{tag}_mel_target",),
            wandb.Image(plot_spectrogram_to_numpy(mel_postnet[0].data.cpu().numpy()), caption=f"{tag}_mel_predicted",),
        ]
        gates += [
            wandb.Image(
                plot_gate_outputs_to_numpy(
                    gate_target[0].data.cpu().numpy(), torch.sigmoid(gate[0]).data.cpu().numpy(),
                ),
                caption=f"{tag}_gate",
            )
        ]

        swriter.log({"specs": specs, "alignments": alignments, "gates": gates})

        if add_audio:
            audios = []
            filterbank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=fmax)
            log_mel = mel_postnet[0].data.cpu().numpy().T
            mel = np.exp(log_mel)
            magnitude = np.dot(mel, filterbank) * griffin_lim_mag_scale
            audio_pred = griffin_lim(magnitude.T ** griffin_lim_power)

            log_mel = spec_target[0].data.cpu().numpy().T
            mel = np.exp(log_mel)
            magnitude = np.dot(mel, filterbank) * griffin_lim_mag_scale
            audio_true = griffin_lim(magnitude.T ** griffin_lim_power)

            audios += [
                wandb.Audio(audio_true / max(np.abs(audio_true)), caption=f"{tag}_wav_target", sample_rate=sr,),
                wandb.Audio(audio_pred / max(np.abs(audio_pred)), caption=f"{tag}_wav_predicted", sample_rate=sr,),
            ]

            swriter.log({"audios": audios})


def plot_alignment_to_numpy(alignment, title='', info=None, phoneme_seq=None, vmin=None, vmax=None):
    if phoneme_seq:
        fig, ax = plt.subplots(figsize=(15, 10))
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower', interpolation='none', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    if phoneme_seq != None:
        # for debugging of phonemes and durs in maps. Not used by def in training code
        ax.set_yticks(np.arange(len(phoneme_seq)))
        ax.set_yticklabels(phoneme_seq)
        ax.hlines(np.arange(len(phoneme_seq)), xmin=0.0, xmax=max(ax.get_xticks()))

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_pitch_to_numpy(pitch, ylim_range=None):
    fig, ax = plt.subplots(figsize=(12, 3))
    plt.plot(pitch)
    if ylim_range is not None:
        plt.ylim(ylim_range)
    plt.xlabel("Frames")
    plt.ylabel("Pitch")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_multipitch_to_numpy(pitch_gt, pitch_pred, ylim_range=None):
    fig, ax = plt.subplots(figsize=(12, 3))
    plt.plot(pitch_gt, label="Ground truth")
    plt.plot(pitch_pred, label="Predicted")
    if ylim_range is not None:
        plt.ylim(ylim_range)
    plt.xlabel("Frames")
    plt.ylabel("Pitch")
    plt.legend()
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_spectrogram_to_numpy(spectrogram):
    spectrogram = spectrogram.astype(np.float32)
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def create_plot(data, x_axis, y_axis, output_filepath=None):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(data, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.tight_layout()

    if output_filepath:
        plt.savefig(output_filepath, format="png")

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_gate_outputs_to_numpy(gate_targets, gate_outputs):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(
        range(len(gate_targets)), gate_targets, alpha=0.5, color='green', marker='+', s=1, label='target',
    )
    ax.scatter(
        range(len(gate_outputs)), gate_outputs, alpha=0.5, color='red', marker='.', s=1, label='predicted',
    )

    plt.xlabel("Frames (Green target, Red predicted)")
    plt.ylabel("Gate State")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


@rank_zero_only
def waveglow_log_to_tb_func(
    swriter, tensors, step, tag="train", n_fft=1024, hop_length=256, window="hann", mel_fb=None,
):
    _, audio_pred, spec_target, mel_length = tensors
    mel_length = mel_length[0]
    spec_target = spec_target[0].data.cpu().numpy()[:, :mel_length]
    swriter.add_image(
        f"{tag}_mel_target", plot_spectrogram_to_numpy(spec_target), step, dataformats="HWC",
    )
    if mel_fb is not None:
        mag, _ = librosa.core.magphase(
            librosa.core.stft(
                np.nan_to_num(audio_pred[0].cpu().detach().numpy()), n_fft=n_fft, hop_length=hop_length, window=window,
            )
        )
        mel_pred = np.matmul(mel_fb.cpu().numpy(), mag).squeeze()
        log_mel_pred = np.log(np.clip(mel_pred, a_min=1e-5, a_max=None))
        swriter.add_image(
            f"{tag}_mel_predicted", plot_spectrogram_to_numpy(log_mel_pred[:, :mel_length]), step, dataformats="HWC",
        )


def remove(conv_list):
    new_conv_list = torch.nn.ModuleList()
    for old_conv in conv_list:
        old_conv = torch.nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(old_conv)
    return new_conv_list


def regulate_len(
    durations, enc_out, pace=1.0, mel_max_len=None, group_size=1, dur_lens: torch.tensor = None,
):
    """A function that takes predicted durations per encoded token, and repeats enc_out according to the duration.
    NOTE: durations.shape[1] == enc_out.shape[1]

    Args:
        durations (torch.tensor): A tensor of shape (batch x enc_length) that represents how many times to repeat each
            token in enc_out.
        enc_out (torch.tensor): A tensor of shape (batch x enc_length x enc_hidden) that represents the encoded tokens.
        pace (float): The pace of speaker. Higher values result in faster speaking pace. Defaults to 1.0.        max_mel_len (int): The maximum length above which the output will be removed. If sum(durations, dim=1) >
            max_mel_len, the values after max_mel_len will be removed. Defaults to None, which has no max length.
        group_size (int): replicate the last element specified by durations[i, in_lens[i] - 1] until the
            full length of the sequence is the next nearest multiple of group_size
        in_lens (torch.tensor): input sequence length specifying valid values in the durations input tensor (only needed if group_size >1)
    """

    dtype = enc_out.dtype
    reps = durations.float() / pace
    reps = (reps + 0.5).floor().long()
    dec_lens = reps.sum(dim=1)
    if group_size > 1:
        to_pad = group_size * (torch.div(dec_lens + 1, group_size, rounding_mode='floor')) - dec_lens
        reps.index_put_(
            indices=[torch.arange(dur_lens.shape[0], dtype=torch.long), dur_lens - 1], values=to_pad, accumulate=True
        )
        dec_lens = reps.sum(dim=1)

    max_len = dec_lens.max()
    reps_cumsum = torch.cumsum(torch.nn.functional.pad(reps, (1, 0, 0, 0), value=0.0), dim=1)[:, None, :]
    reps_cumsum = reps_cumsum.to(dtype=dtype, device=enc_out.device)

    range_ = torch.arange(max_len).to(enc_out.device)[None, :, None]
    mult = (reps_cumsum[:, :, :-1] <= range_) & (reps_cumsum[:, :, 1:] > range_)
    mult = mult.to(dtype)
    enc_rep = torch.matmul(mult, enc_out)

    if mel_max_len is not None:
        enc_rep = enc_rep[:, :mel_max_len]
        dec_lens = torch.clamp_max(dec_lens, mel_max_len)

    return enc_rep, dec_lens


def split_view(tensor, split_size: int, dim: int = 0):
    if dim < 0:  # Support negative indexing
        dim = len(tensor.shape) + dim
    # If not divisible by split_size, we need to pad with 0
    if tensor.shape[dim] % split_size != 0:
        to_pad = split_size - (tensor.shape[dim] % split_size)
        padding = [0] * len(tensor.shape) * 2
        padding[dim * 2 + 1] = to_pad
        padding.reverse()
        tensor = torch.nn.functional.pad(tensor, padding)
    cur_shape = tensor.shape
    new_shape = cur_shape[:dim] + (tensor.shape[dim] // split_size, split_size) + cur_shape[dim + 1 :]
    return tensor.reshape(*new_shape)


def slice_segments(x, ids_str, segment_size=4):
    """
    Time-wise slicing (patching) of bathches for audio/spectrogram
    [B x C x T] -> [B x C x segment_size]
    """
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        x_i = x[i]
        if idx_end >= x.size(2):
            # pad the sample if it is shorter than the segment size
            x_i = torch.nn.functional.pad(x_i, (0, (idx_end + 1) - x.size(2)))
        ret[i] = x_i[:, idx_str:idx_end]
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    """
    Chooses random indices and slices segments from batch
    [B x C x T] -> [B x C x segment_size]
    """
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str_max = ids_str_max.to(device=x.device)
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)

    ret = slice_segments(x, ids_str, segment_size)

    return ret, ids_str


def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = get_mask_from_lengths(cum_duration_flat, torch.Tensor(t_y).reshape(1, 1, -1)).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


def process_batch(batch_data, sup_data_types_set):
    batch_dict = {}
    batch_index = 0
    for name, datatype in DATA_STR2DATA_CLASS.items():
        if datatype in MAIN_DATA_TYPES or datatype in sup_data_types_set:
            batch_dict[name] = batch_data[batch_index]
            batch_index = batch_index + 1
            if issubclass(datatype, WithLens):
                batch_dict[name + "_lens"] = batch_data[batch_index]
                batch_index = batch_index + 1
    return batch_dict


def to_device_recursive(e, device: torch.device):
    """
    Use .to(device) on all tensors within nested lists, tuples, values ofdicts
    Returns a new structure with tensors moved to target device, leaving other data intact.

    The intended use is to move collections of tensors to a device while:
        - avoiding calling specific movers like .cpu() or .cuda()
        - avoiding stuff like .to(torch.device("cuda:{some_variable}"))
    """
    if isinstance(e, (list, tuple)):
        return [to_device_recursive(elem, device) for elem in e]
    elif isinstance(e, dict):
        return {key: to_device_recursive(value, device) for key, value in e.items()}
    elif isinstance(e, torch.Tensor):
        return e.to(device)
    else:
        return e


def mask_sequence_tensor(tensor: torch.Tensor, lengths: torch.Tensor):
    """
    For tensors containing sequences, zero out out-of-bound elements given lengths of every element in the batch.

    tensor: tensor of shape (B, D, L) or (B, D1, D2, L),
    lengths: LongTensor of shape (B,)
    """
    batch_size, *_, max_lengths = tensor.shape

    if len(tensor.shape) == 2:
        mask = torch.ones(batch_size, max_lengths).cumsum(dim=-1).type_as(lengths)
        mask = mask <= rearrange(lengths, "b -> b 1")
    elif len(tensor.shape) == 3:
        mask = torch.ones(batch_size, 1, max_lengths).cumsum(dim=-1).type_as(lengths)
        mask = mask <= rearrange(lengths, "b -> b 1 1")
    elif len(tensor.shape) == 4:
        mask = torch.ones(batch_size, 1, 1, max_lengths).cumsum(dim=-1).type_as(lengths)
        mask = mask <= rearrange(lengths, "b -> b 1 1 1")
    else:
        raise ValueError("Can only mask tensors of shape B x D x L and B x D1 x D2 x L")

    return tensor * mask


@torch.jit.script
def batch_from_ragged(
    text: torch.Tensor,
    pitch: torch.Tensor,
    pace: torch.Tensor,
    batch_lengths: torch.Tensor,
    padding_idx: int = -1,
    volume: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    batch_lengths = batch_lengths.to(dtype=torch.int64)
    max_len = torch.max(batch_lengths[1:] - batch_lengths[:-1])

    index = 1
    num_batches = batch_lengths.shape[0] - 1
    texts = torch.zeros(num_batches, max_len, dtype=torch.int64, device=text.device) + padding_idx
    pitches = torch.ones(num_batches, max_len, dtype=torch.float32, device=text.device)
    paces = torch.zeros(num_batches, max_len, dtype=torch.float32, device=text.device) + 1.0
    volumes = torch.zeros(num_batches, max_len, dtype=torch.float32, device=text.device) + 1.0
    lens = torch.zeros(num_batches, dtype=torch.int64, device=text.device)
    last_index = index - 1
    while index < batch_lengths.shape[0]:
        seq_start = batch_lengths[last_index]
        seq_end = batch_lengths[index]
        cur_seq_len = seq_end - seq_start
        lens[last_index] = cur_seq_len
        texts[last_index, :cur_seq_len] = text[seq_start:seq_end]
        pitches[last_index, :cur_seq_len] = pitch[seq_start:seq_end]
        paces[last_index, :cur_seq_len] = pace[seq_start:seq_end]
        if volume is not None:
            volumes[last_index, :cur_seq_len] = volume[seq_start:seq_end]
        last_index = index
        index += 1

    return texts, pitches, paces, volumes, lens


def sample_tts_input(
    export_config, device, max_batch=1, max_dim=127,
):
    """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
    sz = (max_batch * max_dim,) if export_config["enable_ragged_batches"] else (max_batch, max_dim)
    inp = torch.randint(*export_config["emb_range"], sz, device=device, dtype=torch.int64)
    pitch = torch.randn(sz, device=device, dtype=torch.float32) * 0.5
    pace = torch.clamp(torch.randn(sz, device=device, dtype=torch.float32) * 0.1 + 1.0, min=0.2)
    inputs = {'text': inp, 'pitch': pitch, 'pace': pace}
    if export_config["enable_ragged_batches"]:
        batch_lengths = torch.zeros((max_batch + 1), device=device, dtype=torch.int32)
        left_over_size = sz[0]
        batch_lengths[0] = 0
        for i in range(1, max_batch):
            equal_len = (left_over_size - (max_batch - i)) // (max_batch - i)
            length = torch.randint(equal_len // 2, equal_len, (1,), device=device, dtype=torch.int32)
            batch_lengths[i] = length + batch_lengths[i - 1]
            left_over_size -= length.detach().cpu().numpy()[0]
        batch_lengths[-1] = left_over_size + batch_lengths[-2]

        sum = 0
        index = 1
        while index < len(batch_lengths):
            sum += batch_lengths[index] - batch_lengths[index - 1]
            index += 1
        assert sum == sz[0], f"sum: {sum}, sz: {sz[0]}, lengths:{batch_lengths}"
    else:
        batch_lengths = torch.randint(max_dim // 2, max_dim, (max_batch,), device=device, dtype=torch.int32)
        batch_lengths[0] = max_dim
    inputs['batch_lengths'] = batch_lengths

    if export_config["enable_volume"]:
        volume = torch.clamp(torch.randn(sz, device=device, dtype=torch.float32) * 0.1 + 1, min=0.01)
        inputs['volume'] = volume

    if "num_speakers" in export_config:
        inputs['speaker'] = torch.randint(
            0, export_config["num_speakers"], (max_batch,), device=device, dtype=torch.int64
        )
    return inputs


@deprecated(
    explanation="But it will not be removed until a further notice. G2P object root directory "
    "`nemo_text_processing.g2p` has been replaced with `nemo.collections.tts.g2p`. "
    "Please use the latter instead as of NeMo 1.18.0."
)
def g2p_backward_compatible_support(g2p_target: str) -> str:
    # for backward compatibility
    g2p_target_new = g2p_target.replace("nemo_text_processing.g2p", "nemo.collections.tts.g2p")
    return g2p_target_new
