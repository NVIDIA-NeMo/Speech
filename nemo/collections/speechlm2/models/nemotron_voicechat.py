# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import os
import random
import tempfile

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from torch import Tensor, nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel,
    parallelize_module,
)
from transformers import DynamicCache

from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.common.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.models.duplex_s2s_model import replace_control_speech_codes, tokens_to_str
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.metrics.asr_bleu import ASRBLEU
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.precision import fp32_precision
from safetensors.torch import load_file
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_hf,
    set_model_dict_for_partial_init,
    setup_audio_codec,
    setup_speech_encoder,
)
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging


from nemo.collections.speechlm2.models.duplex_ear_tts import DuplexEARTTS

from nemo.collections.speechlm2.models.duplex_stt_model import DuplexSTTModel

def delay_eos(tokens, eos_token_id, pad_token_id, shift=10):
    """
    Delays each EOS token by `shift` steps forward. Replaces original EOS with PAD.
    Skips move if it would go out of bounds or overwrite another EOS/PAD.
    Safe for GPU execution.
    """
    B, T = tokens.shape
    tokens = tokens.clone()
    device = tokens.device

    # Find all EOS positions
    eos_mask = tokens == eos_token_id
    if not eos_mask.any():
        return tokens

    # Flattened indices of EOS tokens
    eos_indices = eos_mask.nonzero(as_tuple=False)  # [N, 2]
    b_idx = eos_indices[:, 0]  # [N]
    eos_pos = eos_indices[:, 1]  # [N]
    new_pos = eos_pos + shift  # [N]

    # Filter: new position must be in bounds and not overwrite EOS or PAD
    valid = (new_pos < T)
    if valid.any():
        b_idx = b_idx[valid]
        old_pos = eos_pos[valid]
        new_pos = new_pos[valid]

        # Now, check overwrite safety in new positions
        target_vals = tokens[b_idx, new_pos]
        safe = (target_vals != eos_token_id)

        if safe.any():
            b_idx = b_idx[safe]
            old_pos = old_pos[safe]
            new_pos = new_pos[safe]
            # Move EOS token: clear original, set new
            tokens[b_idx, old_pos] = pad_token_id
            tokens[b_idx, new_pos] = eos_token_id
    return tokens


def generate_multiturn_speaking_mask(input_ids: torch.Tensor, bos_token_id: int = 0, eos_token_id: int = 1):
    """
    Efficient, batched speaking mask generator that marks 1 between <bos> and <eos> pairs.
    If <eos> is missing after a <bos>, mask continues to end. Handles multiple turns.

    Args:
        input_ids (torch.Tensor): LongTensor of shape (B, T)
        bos_token_id (int): Token ID for <bos>
        eos_token_id (int): Token ID for <eos>

    Returns:
        torch.Tensor: FloatTensor of shape (B, T), with 1.0 for speaking, 0.0 for silence.

    Note BOS is considered as speaking (1) and EOS as non speaking 0
    """
    B, T = input_ids.shape
    device = input_ids.device
    bos_mask = (input_ids == bos_token_id).to(torch.int32).to(device)
    eos_mask = (input_ids == eos_token_id).to(torch.int32).to(device)
    bos_cumsum = torch.cumsum(bos_mask, dim=1)
    eos_cumsum = torch.cumsum(eos_mask, dim=1)
    speaking_mask = (bos_cumsum > eos_cumsum).to(torch.float32)
    return speaking_mask.long()


def add_structured_noise_preserve_tail(
    mask: torch.Tensor,
    span_prob: float = 0.05,
    min_len: int = 2,
    max_len: int = 3,
    min_preserve: int = 4,
):
    """
    Adds structured noise to a binary mask by flipping random spans (2–3 tokens at a time),
    while preserving the last `min_preserve` tokens of each speaking region (1s).

    Args:
        mask (torch.Tensor): Binary mask of shape (B, T), values in {0, 1}
        span_prob (float): Probability of inserting a noisy span per token
        min_len (int): Minimum span length to flip
        max_len (int): Maximum span length to flip
        min_preserve (int): Number of 1s at the end of each span to protect from flipping

    Returns:
        torch.Tensor: Noised mask (same shape)
    """
    B, T = mask.shape
    noised_mask = mask.clone()

    for b in range(B):
        i = 0
        while i < T:
            if mask[b, i] == 1:
                # Start of a speaking region
                start = i
                while i < T and mask[b, i] == 1:
                    i += 1
                end = i  # exclusive
                span_len = end - start

                if span_len > min_preserve:
                    allowed_start = start
                    allowed_end = end - min_preserve
                    j = allowed_start
                    while j < allowed_end:
                        if random.random() < span_prob:
                            flip_len = random.randint(min_len, max_len)
                            flip_end = min(j + flip_len, allowed_end)
                            noised_mask[b, j:flip_end] = (noised_mask[b, j:flip_end] + 1) % 2
                            j = flip_end
                        else:
                            j += 1
            else:
                i += 1
    return noised_mask


class NemotronVoiceChat(LightningModule, HFHubMixin):
    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to DuplexS2SModel as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        # convert dict to config
        cfg = DictConfig(cfg)
        self.full_cfg = cfg
        self.cfg = cfg.model
        self.target_sample_rate = cfg.data.target_sample_rate
        self.source_sample_rate = cfg.data.source_sample_rate
        self.validation_save_path = os.path.join(cfg.exp_manager.explicit_log_dir, "validation_logs")

        # Load Duplex STT model
        self.stt_model = DuplexSTTModel(OmegaConf.to_container(self.cfg.stt, resolve=True))

        # Load Duplex TTS model
        # delete old config name for old version compatibility
        if (
            hasattr(self.cfg, "speech_generation")
            and hasattr(self.cfg.speech_generation, "model")
            and hasattr(self.cfg.speech_generation.model, "tts_config")
            and hasattr(self.cfg.speech_generation.model.tts_config, "cas_config")
            and hasattr(self.cfg.speech_generation.model.tts_config.cas_config, "pretrained_tokenizer_name")
        ):
            del self.cfg.speech_generation.model.tts_config.cas_config.pretrained_tokenizer_name

        self.tts_model = DuplexEARTTS(OmegaConf.to_container(self.cfg.speech_generation, resolve=True))
        # reset silence tokens to avoid inference issues
        self.tts_model.codec_silence_tokens = self.tts_model.get_codec_silence_frame()

        self.target_fps = self.tts_model.target_fps
        # compute source fps
        self.source_fps = self.source_sample_rate / (
            self.source_sample_rate * cfg.data.frame_length
        )  # conver frame rate in fps

        if self.cfg.get("pretrained_s2s_model", None):
            logging.info(f"Loading pretrained s2s model from {self.cfg.pretrained_s2s_model}")
            if os.path.isdir(self.cfg.pretrained_s2s_model):
                # Hugging Face format
                from safetensors import safe_open
                import gc
                
                # Load tensors incrementally to avoid OOM
                model_state_dict = self.state_dict()
                loaded_keys = []
                missing_keys = []
                
                with safe_open(os.path.join(self.cfg.pretrained_s2s_model, "model.safetensors"), framework="pt", device="cpu") as f:
                    available_keys = f.keys()
                    for key in available_keys:
                        if key in model_state_dict:
                            # Load tensor and copy to model parameter
                            tensor = f.get_tensor(key)
                            model_state_dict[key].copy_(tensor)
                            loaded_keys.append(key)
                            del tensor  # Free memory immediately
                        else:
                            missing_keys.append(key)
                        
                        # Periodic garbage collection for very large models
                        if len(loaded_keys) % 100 == 0:
                            gc.collect()

                logging.info(f"Loaded {len(loaded_keys)} tensors from pretrained model")
                if missing_keys:
                    logging.warning(f"Keys in checkpoint but not in model: {len(missing_keys)} keys")
                
                del model_state_dict
                gc.collect()
            else:
                self.init_from_model_from_ckpt(self.cfg.pretrained_s2s_model)

        self._use_fsdp = False
        self._use_tp = False

    def init_from_model_from_ckpt(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                    checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            # partial initialization support
            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.state_dict())
            self.load_state_dict(checkpoint_state, strict=True)

    def training_step(self, batch: dict, batch_idx: int):

        return None

    def on_train_epoch_start(self) -> None:
        self.tts_model.on_train_epoch_start()
        self.stt_model.on_train_epoch_start()

    def on_validation_epoch_start(self) -> None:
        self.on_train_epoch_start()
        self.results_logger = ResultsLogger(self.validation_save_path).reset()
        self.asr_bleu = ASRBLEU(self.cfg.scoring_asr).reset()
        self.bleu = BLEU().reset()

    def on_validation_epoch_end(self, prefix="val") -> None:
        asr_bleu = self.asr_bleu.compute()
        for k, m in asr_bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        bleu = self.bleu.compute()
        for k, m in bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)

    def validation_step(self, batch: dict, batch_idx: int):

        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted

            dataset_batch = dataset_batch["audio_data"]

            prompt_tokens = dataset_batch.get("prompt_tokens", None)
            prompt_token_lens = dataset_batch.get("prompt_token_lens", None)

            results = self.offline_inference(
                dataset_batch["source_audio"],
                dataset_batch["source_audio_lens"],
                prompt_tokens=prompt_tokens,
                prompt_token_lens=prompt_token_lens,
            )

            with fp32_precision():  # resample is fragile to bfloat16 default dtype
                asr_hyps = self.asr_bleu.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    pred_audio=resample(results["audio"], 22050, 16000),
                    pred_audio_lens=(results["audio_len"] / 22050 * 16000).to(torch.long),
                )

                self.results_logger.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    hyps=results["text"],
                    asr_hyps=asr_hyps,
                    samples_id=dataset_batch['sample_id'],
                    pred_audio=results["audio"],
                    pred_audio_sr=self.target_sample_rate,
                    user_audio=dataset_batch["source_audio"],
                    user_audio_sr=self.source_sample_rate,
                    eou_pred=(
                        results["gen_eou"]
                        if "gen_eou" in results
                        else None
                    ),
                    fps=self.source_fps,
                    results=results if self.cfg.get("dump_tokens_text", False) else None,
                    tokenizer=self.stt_model.tokenizer,
                )

            self.bleu.update(name=name, refs=dataset_batch["target_texts"], hyps=results["text"])

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(prefix="test")

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    @torch.no_grad()
    def offline_inference(
        self,
        input_signal: torch.Tensor,
        input_signal_lens: torch.Tensor,
        prompt_tokens: torch.Tensor = None,
        prompt_token_lens: torch.Tensor = None,
        input_pad_len: int = 0,
        force_bos_positions=None,
        decode_audio: bool = True,
        incremental_audio_decoding: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Autoregressive prediction.

        Args:
            input_signal: a batch of waveforms with shape (B, T) with source sampling rate.
            input_signal_lens: example lengths as number of samples of shape (B,).
            decode_audio: bool, whether to decode audio codes to waveform.

        Returns:
            A dict with keys:
                * "text": generated text, de-tokenized to strings, properly skipping text_pad_id; list of length B.
                * "tokens_text": generated text tokens of shape (B, T2).
                * "tokens_audio": generated audio codes of shape (B, T2, K) where `K=num_codebooks`.
                * "tokens_len" output lengths as number of tokens of shape (B,).
                * "audio": generated waveform of shape (B, T3) (`decode_audio=True`).
                * "audio_len" output lengths as number of waveform samples of shape (B,) (when `decode_audio=True`).
        """

        inference_state = self.stt_model._init_inference(
            input_signal, input_signal_lens, input_pad_len,
            force_bos_positions, prompt_tokens, prompt_token_lens
        )

        ans, inference_state = self.stt_model._step_zero(inference_state)

        B = inference_state["B"]
        T = inference_state["T"]

        # Init external Duplex TTS model
        generation_config = None
        guidance_enabled = True

        # create speaker audio for init
        speaker_audio, sr = torchaudio.load(self.cfg.inference_speaker_reference)
        speaker_audio = resample(speaker_audio, sr, self.tts_model.target_sample_rate)
        speaker_audio = speaker_audio.repeat(B, 1).to(self.device)

        # lengths -> [B]
        speaker_audio_lens = torch.tensor([speaker_audio.size(1)]).long().repeat(B).to(self.device) 

        #  init tts_model
        self.tts_model.set_init_inputs(
            speaker_audio=speaker_audio,
            speaker_audio_lens=speaker_audio_lens,
        )
        init_inputs = self.tts_model.get_init_inputs(B=B)

        if generation_config is None:
            generation_config = self.tts_model._get_generation_config(guidance_enabled)

        init_inputs.update({"use_cache": True, "past_key_values": None, "guidance_enabled": guidance_enabled})
        # warmup the model and generate the very first audio token
        outputs = self.tts_model.tts_model(**init_inputs)
        code = init_inputs["code"][:, -1:]

        past_key_values = outputs.past_key_values
        gen_codes = torch.zeros(B, T, self.tts_model.tts_model.config.num_quantizers, device=self.device, dtype=torch.long)
        first_context_subword_id = init_inputs["subword_ids"][:, -1].unsqueeze(-1)
        subword_mask = torch.ones(B, T, device=self.device, dtype=torch.bool)

        # init
        audio_pred = None
        audio_pred_len = torch.zeros(B, device=self.device, dtype=torch.long)

        # Autoregressive loop
        for t in range(1, T):
            # do one step inference on Duplex STT model
            _  = self.stt_model._step_inference(t, inference_state, ans, force_bos_positions)

            # do one step inference on Duplex TTS model
            # current subword id is always seem
            current_subword_id = inference_state["gen_text"][:, t].unsqueeze(-1)
            if t == 1:
                prev_subword_id = first_context_subword_id
            else:
                prev_subword_id = inference_state["gen_text"][:, t - 1].unsqueeze(-1)

            # create subword_mask
            current_subword_mask = subword_mask[:, t].unsqueeze(-1)

            code, past_key_values = self.tts_model.infer_codes_one_step(
                current_subword_id=current_subword_id,
                prev_subword_id=prev_subword_id,
                current_subword_mask=current_subword_mask,
                prev_audio_tokens=code,
                past_key_values=past_key_values,
                guidance_enabled=guidance_enabled,
                generation_config=generation_config,
                ignore_eos_flag_stop=True,
            )

            gen_codes[:, t] = code.squeeze(1)

            if decode_audio and incremental_audio_decoding:
                audio_pred_i, audio_pred_i_len = self.tts_model.decode_one_audio_step(
                    gen_codes[:, : t + 1],
                    number_prev_tokens=self.cfg.get("inference_codec_decoding_prev_tokens_number", None),
                )
                if audio_pred is None:
                    audio_pred = audio_pred_i
                else:
                    audio_pred = torch.cat([audio_pred, audio_pred_i], dim=1)
                audio_pred_len += audio_pred_i_len

            logging.info(f"Autoregressive inference step: {t} of {T} !")

        # Trim back to local length if padded
        if self._use_fsdp and T > inference_state["T_local"]:
            gen_codes = gen_codes[:, :inference_state["T_local"]]

        ans = self.stt_model._post_inference(inference_state, prompt_token_lens)

        if decode_audio:
            gen_codes = gen_codes[:, :inference_state["T_local"]]
            if not incremental_audio_decoding:
                gen_audio_codes_lens = torch.tensor([gen_codes.shape[1]] * gen_codes.shape[0]).to(self.device)
                gen_audio_codes = gen_codes
                with fp32_precision(), torch.no_grad():
                    audio_pred, audio_pred_len = self.tts_model.audio_codec.decode(
                        gen_audio_codes, gen_audio_codes_lens
                    )
            ans["audio"] = audio_pred.squeeze(1)
            ans["audio_len"] = audio_pred_len

        return ans

    def load_state_dict(self, state_dict, strict: bool = True):
        try:
            return super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            logging.info(f"Error loading model state_dict !! Retrying with partial initialization!")
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            return super().load_state_dict(model_dict, strict=False)
