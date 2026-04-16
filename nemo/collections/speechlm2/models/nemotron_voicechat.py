# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
import gc
import json
import os
import warnings
from pathlib import Path
import torch
from huggingface_hub import CONFIG_NAME
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from safetensors import safe_open
from transformers.utils import cached_file

from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.speechlm2.models.duplex_ear_tts import DuplexEARTTS, load_audio_librosa
from nemo.collections.speechlm2.models.duplex_stt_model import DuplexSTTModel
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.metrics.asr_bleu import ASRBLEU
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.pretrained import set_model_dict_for_partial_init
from nemo.utils import logging


class NemotronVoiceChat(LightningModule, HFHubMixin):
    """
    NemotronVoiceChat: End-to-End Duplex Speech-to-Speech Model.

    This class integrates:

        • DuplexSTTModel  — Duplex speech-to-text (STT) model
        • DuplexEARTTS    — autoregressive speech decoder (TTS)

    The module is evaluation-oriented (no training_step implemented) and
    supports:

        • BLEU computation (text vs reference)
        • ASR-BLEU computation (ASR transcription of generated speech vs reference)
        • Audio + text result logging
        • HuggingFace-compatible checkpoint loading/saving
        • Partial checkpoint initialization
        • Speaker prompt conditioning

    Configuration
    -------------

    The class required config fields:
            model:
                scoring_asr: str
                stt:
                    model: ...
                speech_generation:
                    model: ...

            data:
                target_sample_rate: int
                source_sample_rate: int
                frame_length: float
                validation_ds:
                    datasets:
                        evaluation_set:
                        shar_path: ...

            exp_manager:
                explicit_log_dir: str

    Validation artifacts are stored under:
        cfg.exp_manager.explicit_log_dir + "/validation_logs"

    Metrics
    -------

    During validation/test:

        • BLEU
        • ASRBLEU (using model.scoring_asr from config)

    Results are logged via:

        ResultsLogger

    Notes
    -----
    • Designed for inference, validation, and model export workflows.
    • Supports both standard checkpoints and HuggingFace safetensors format.

    """

    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to NemotronVoiceChat as a Python dict to support hyperparameter serialization "
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
        self.stt_model = DuplexSTTModel(OmegaConf.to_container(self.cfg.stt.model, resolve=True))

        # Load Duplex TTS model
        self.tts_model = DuplexEARTTS(OmegaConf.to_container(self.cfg.speech_generation, resolve=True))

        # reset silence tokens to avoid inference issues (skip when codec
        # has random weights — the buffer will be loaded from checkpoint)
        if self.tts_model.cfg.get('pretrained_codec_model', None) is not None:
            self.tts_model.codec_silence_tokens = self.tts_model.get_codec_silence_frame()
        self.target_fps = self.tts_model.target_fps
        # compute source fps
        self.source_fps = self.source_sample_rate / (
            self.source_sample_rate * cfg.data.frame_length
        )  # conver frame rate in fps

        self._use_fsdp = False
        self._use_tp = False

    def save_pretrained(
        self,
        save_directory: str | Path,
        **kwargs,
    ) -> str | None:
        """Save model and export LLM artifacts (tokenizer + perception config) for offline inference."""
        result = super().save_pretrained(save_directory, **kwargs)

        # Save tokenizer for offline loading
        try:
            llm_dir = Path(save_directory) / "llm_artifacts"
            llm_dir.mkdir(parents=True, exist_ok=True)
            self.stt_model.tokenizer.tokenizer.save_pretrained(str(llm_dir))
            logging.info(f"Saved LLM tokenizer to {llm_dir}")
        except Exception as e:
            warnings.warn(f"Failed to save LLM tokenizer: {e}. Inference will fall back to downloading from HF.")

        # Save full perception config at the top level of config.json so that
        # resolve_pretrained_config() can skip pretrained ASR/LLM downloads.
        try:
            config_path = Path(save_directory) / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                config["perception"] = OmegaConf.to_container(self.stt_model.cfg.perception, resolve=True)
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                logging.info(f"Saved perception config to {config_path}")
        except Exception as e:
            warnings.warn(f"Failed to save perception config: {e}")

        return result

    def init_from_model_from_ckpt(self, checkpoint_path):
        if checkpoint_path is not None:
            checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            # partial initialization support
            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.state_dict())
            self.load_state_dict(checkpoint_state, strict=True)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        proxies: dict | None,
        resume_download: bool | None,
        local_files_only: bool,
        token: str | bool | None,
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        """
        Load Pytorch pretrained weights and return the loaded model.
        Wrapper over PyTorchModelHubMixin that auto-handles config and uses our
        custom memory-efficient safetensors streaming loader to prevent OOM.
        """
        # Fetch the Config
        resolved_config_file = cached_file(
            model_id,
            CONFIG_NAME,  # Ensure CONFIG_NAME is defined in your file (e.g., "config.yaml" or "config.json")
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
        if resolved_config_file is None:
            raise RuntimeError(f"Missing {CONFIG_NAME} file for {model_id=}")

        model_kwargs['cfg'] = OmegaConf.to_container(OmegaConf.load(resolved_config_file))

        # Skip loading child module weights natively
        model_kwargs['cfg']['pretrained_weights'] = False

        # Propagate pretrained_weights=False into nested configs so child
        # modules skip downloading pretrained ASR, LLM, and codec models.
        cfg = model_kwargs['cfg']
        try:
            stt_model_cfg = cfg['model']['stt']['model']
            stt_model_cfg['pretrained_weights'] = False
            stt_model_cfg['use_meta_device'] = True
            if 'perception' in cfg:
                stt_model_cfg['perception'] = cfg['perception']
                logging.info("Injected saved perception config into STT model config")
        except (KeyError, TypeError):
            logging.warning("Could not propagate pretrained_weights=False into nested STT config")
        try:
            tts_model_cfg = cfg['model']['speech_generation']['model']
            tts_model_cfg['pretrained_model'] = None
            tts_model_cfg['pretrained_codec_model'] = None
        except (KeyError, TypeError):
            logging.warning("Could not nullify pretrained TTS/codec paths in nested TTS config")

        # Instantiate the empty model skeleton
        model = cls(model_kwargs['cfg'])

        # Fetch the Safetensors weights
        resolved_weights_file = cached_file(
            model_id,
            "model.safetensors",
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
        if resolved_weights_file is None:
            raise RuntimeError(f"Missing model.safetensors file for {model_id=}")

        # Stream the weights from safetensors
        ckpt_dir = os.path.dirname(resolved_weights_file)
        model.init_from_safetensors_ckpt(ckpt_dir)

        return model

    def _replace_tensor(self, full_key, value):
        """Replace a parameter or buffer on its parent module.

        Meta-device tensors (from torch.device('meta')) have no storage, so the
        usual ``target.data.copy_(tensor)`` raises an error.  Instead we walk
        the module tree to find the parent and swap the entry directly in
        ``module._parameters`` or ``module._buffers``.
        """
        parts = full_key.split(".")
        module = self
        for part in parts[:-1]:
            module = getattr(module, part)
        name = parts[-1]
        if name in module._parameters:
            module._parameters[name] = torch.nn.Parameter(
                value, requires_grad=module._parameters[name].requires_grad
            )
        elif name in module._buffers:
            module._buffers[name] = value

    def init_from_safetensors_ckpt(self, ckpt_path, prefix=""):
        """
        Memory-efficient streaming safetensors loader with dynamic
        audio_prompt_latents recreation support.

        Safe for large models and distributed training (if called before DDP/FSDP wrap).
        """

        loaded_keys = []
        missing_keys = []

        # Build fast lookup tables once
        param_dict = dict(self.named_parameters())
        buffer_dict = dict(self.named_buffers())

        with safe_open(
            os.path.join(ckpt_path, "model.safetensors"),
            framework="pt",
            device="cpu",
        ) as f:

            for key in f.keys():

                try:
                    tensor = f.get_tensor(key)
                except Exception as e:
                    logging.warning(f"Failed loading tensor {key}: {e}")
                    continue

                # recreates the audio_prompt_latents if needed
                if "audio_prompt_latents." in key:
                    self.tts_model.maybe_recreate_cached_audio_prompt_latents_structure({key: tensor})

                    # refresh param dict since new param may exist now
                    param_dict = dict(self.named_parameters())

                if prefix + key in param_dict:
                    target = param_dict[prefix + key]

                    if target.shape != tensor.shape:
                        logging.warning(f"Shape mismatch for {key}: " f"model {target.shape} vs ckpt {tensor.shape}")
                    elif target.is_meta:
                        self._replace_tensor(prefix + key, tensor)
                    else:
                        target.data.copy_(tensor)

                    loaded_keys.append(key)

                elif prefix + key in buffer_dict:
                    target = buffer_dict[prefix + key]

                    if target.shape != tensor.shape:
                        logging.warning(
                            f"Buffer shape mismatch for {key}: " f"model {target.shape} vs ckpt {tensor.shape}"
                        )
                    elif target.is_meta:
                        self._replace_tensor(prefix + key, tensor)
                    else:
                        target.data.copy_(tensor)

                    loaded_keys.append(key)

                else:
                    missing_keys.append(key)

                del tensor

                if len(loaded_keys) % 100 == 0:
                    gc.collect()

        logging.info(f"Loaded {len(loaded_keys)} tensors from pretrained model")

        if missing_keys:
            logging.warning(f"{len(missing_keys)} keys in checkpoint not found in model")

        # Fail if any *parameters* are still on meta device — those genuinely
        # need weights from the checkpoint and their absence is an error.
        meta_params = [n for n, p in self.named_parameters() if p.is_meta]
        if meta_params:
            raise RuntimeError(
                f"{len(meta_params)} parameters still on meta device after checkpoint load "
                f"(missing from checkpoint): {meta_params[:20]}"
            )

        # Buffers on meta device are typically non-persistent computed values
        # (e.g. rotary_emb.inv_freq registered with persistent=False) that
        # save_pretrained / safetensors intentionally omit.  Reinitialize
        # them by moving the owning module to CPU then back, which triggers
        # the buffer's factory function.
        meta_buffers = [(n, b) for n, b in self.named_buffers() if b.is_meta]
        if meta_buffers:
            logging.info(
                f"Reinitializing {len(meta_buffers)} non-persistent meta buffer(s): "
                f"{[n for n, _ in meta_buffers[:10]]}"
            )
            for buf_name, buf in meta_buffers:
                parts = buf_name.split(".")
                module = self
                for part in parts[:-1]:
                    module = getattr(module, part)
                module.to_empty(device="cpu")
                module.to(device="cpu")
            logging.info("Meta buffers reinitialised on CPU")

        gc.collect()

    def training_step(self, batch: dict, batch_idx: int):
        raise NotImplementedError(
            "NemotronVoiceChat.training_step is not implemented yet - for now, this class is inference-only."
        )

    def on_train_epoch_start(self) -> None:
        self.tts_model.on_train_epoch_start()
        self.stt_model.on_train_epoch_start()

    def on_validation_epoch_start(self) -> None:
        """
        Initializes evaluation metrics and result buffers.

        This method:

            • Resets ResultsLogger
            • Initializes ASRBLEU using config parameter:
                model.scoring_asr
            • Resets BLEU metric

        All validation artifacts are saved under:

            self.validation_save_path
            = cfg.exp_manager.explicit_log_dir + "/validation_logs"
        """
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

        self.results_logger.compute_and_save()

    def validation_step(
        self,
        batch: dict,
        batch_idx: int,
        speaker_audio: torch.Tensor = None,
        speaker_audio_lens: torch.Tensor = None,
    ):
        """
        Runs one validation step for NemotronVoiceChat.

        This method performs:

            1. Offline speech-to-speech inference
            2. ASR transcription of generated audio
            3. BLEU and ASR-BLEU metric updates
            4. Audio and text logging to JSON

        Args:
            batch (dict):
                Dictionary of dataset batches.
                Each entry contains an "audio_data" field with:

                    • "source_audio"        — user waveform (B, T)
                    • "source_audio_lens"   — lengths (B,)
                    • "target_texts"        — reference text
                    • "sample_id"           — unique sample identifier
                    • optional prompt tokens

            batch_idx (int):
                Batch index.

            speaker_audio (torch.Tensor, optional):
                Explicit speaker reference waveform (B, T_ref).

            speaker_audio_lens (torch.Tensor, optional):
                Lengths of speaker reference audio.

        Behavior:
            • Calls offline_inference()
            • Resamples generated speech to 16kHz for ASR scoring
            • Updates ASRBLEU using model.scoring_asr
            • Logs per-sample JSON entries
            • Updates text BLEU

        Returns:
            None. Metrics are accumulated and logged on epoch end.
        """
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
                speaker_audio=speaker_audio,
                speaker_audio_lens=speaker_audio_lens,
            )

            with fp32_precision():  # resample is fragile to bfloat16 default dtype
                asr_hyps = self.asr_bleu.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    pred_audio=resample(results["audio"], self.target_sample_rate, 16000),
                    pred_audio_lens=(results["audio_len"] / self.target_sample_rate * 16000).to(torch.long),
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
                    eou_pred=(results["gen_eou"] if "gen_eou" in results else None),
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
        speaker_audio: torch.Tensor = None,
        speaker_audio_lens: torch.Tensor = None,
        input_pad_len: int = 0,
        decode_audio: bool = True,
        incremental_audio_decoding: bool = False,
        generation_config: dict = None,
        guidance_enabled: bool = True,
        return_logits: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Runs full offline duplex speech-to-speech inference.

        This method performs:

            1. Streaming STT inference
            2. Autoregressive text generation
            3. Autoregressive audio codec generation
            4. Optional incremental waveform decoding

        Args:
            input_signal (torch.Tensor):
                Input user waveform of shape (B, T_source),
                sampled at self.source_sample_rate.

            input_signal_lens (torch.Tensor):
                Lengths of input waveforms in samples, shape (B,).

            prompt_tokens (torch.Tensor, optional):
                Optional text prompt tokens.

            prompt_token_lens (torch.Tensor, optional):
                Lengths of prompt tokens.

            speaker_audio (torch.Tensor, optional):
                Explicit speaker reference waveform.

            speaker_audio_lens (torch.Tensor, optional):
                Lengths of speaker reference audio.

            input_pad_len (int, optional):
                Padding length used during streaming inference.

            decode_audio (bool, optional):
                Whether to decode codec tokens into waveform.

            incremental_audio_decoding (bool, optional):
                If True, waveform decoding happens during generation.
                If False, decoding occurs after all tokens are generated.

            generation_config (dict, optional):
                Generation parameters (sampling, noise, EOS rules).
                If None, defaults are obtained from TTS model.

            guidance_enabled (bool, optional):
                Enables classifier-free guidance.

            return_logits (bool, optional):
                When True, collect per-step text and ASR logits and
                include them in the returned dict as ``"text_logits"``
                (B, T, V_text) and ``"asr_logits"`` (B, T, V_asr).
                Useful for parity testing against incremental inference.

        Returns:
            dict[str, torch.Tensor]:

                • "text":
                    List[str] — generated text per sample.

                • "tokens_text":
                    Tensor (B, T_text) — generated text tokens.

                • "tokens_audio":
                    Tensor (B, T_audio, K) — audio codec tokens.

                • "tokens_len":
                    Tensor (B,) — number of generated tokens.

                • "audio":
                    Tensor (B, T_wave) — generated waveform
                    (if decode_audio=True).

                • "audio_len":
                    Tensor (B,) — waveform lengths in samples
                    (if decode_audio=True).

                • "text_logits" (only when return_logits=True):
                    Tensor (B, T, V_text) — per-step text head logits.

                • "asr_logits" (only when return_logits=True):
                    Tensor (B, T, V_asr) — per-step ASR head logits.

        Notes:
            • Uses streaming inference backend of DuplexSTTModel.
            • Uses autoregressive codec generation from DuplexEARTTS.
            • Speaker reference may be loaded automatically from:
                cfg.inference_speaker_reference
            • Supports cached audio prompt latents.
        """

        inference_state = self.stt_model.streaming_inference._init_inference(
            input_signal, input_signal_lens, input_pad_len, prompt_tokens, prompt_token_lens
        )

        ans, inference_state = self.stt_model.streaming_inference._step_zero(inference_state)

        B = inference_state["B"]
        T = inference_state["T"]

        if return_logits:
            _text_logits = [ans["text_logits"][:, -1].detach()]
            _asr_logits = [ans["asr_logits"][:, -1].detach()] if "asr_logits" in ans else []

        # if speaker_name is provided uses it, if not uses the speaker_audio provided, if speaker_audio is None load it from inference_speaker_reference
        if speaker_audio is None:
            speaker_name = self.cfg.get("inference_speaker_name", None)
            if speaker_name is not None:
                speaker_audio = None
                speaker_audio_lens = None
            else:
                # create speaker audio for init
                speaker_audio, sr = load_audio_librosa(self.cfg.inference_speaker_reference)
                speaker_audio = resample(speaker_audio, sr, self.tts_model.target_sample_rate)
                speaker_audio = speaker_audio.repeat(B, 1).to(self.device)

                # lengths -> [B]
                speaker_audio_lens = torch.tensor([speaker_audio.size(1)]).long().repeat(B).to(self.device)
        else:
            speaker_name = None

        #  init tts_model
        self.tts_model.set_init_inputs(
            speaker_audio=speaker_audio,
            speaker_audio_lens=speaker_audio_lens,
            speaker_name=speaker_name,
        )
        init_inputs = self.tts_model.get_init_inputs(B=B)

        if generation_config is None:
            generation_config = self.tts_model._get_generation_config(guidance_enabled)

        init_inputs.update({"use_cache": True, "past_key_values": None, "guidance_enabled": guidance_enabled})
        # warmup the model and generate the very first audio token
        outputs = self.tts_model.tts_model(**init_inputs)
        code = init_inputs["code"][:, -1:]

        past_key_values = outputs.past_key_values
        gen_codes = torch.zeros(
            B, T, self.tts_model.tts_model.config.num_quantizers, device=self.device, dtype=torch.long
        )
        first_context_subword_id = init_inputs["subword_ids"][:, -1].unsqueeze(-1)
        subword_mask = torch.ones(B, T, device=self.device, dtype=torch.bool)

        # init
        audio_pred = None
        audio_pred_len = torch.zeros(B, device=self.device, dtype=torch.long)

        # Autoregressive loop
        for t in range(1, T):
            # do one step inference on Duplex STT model
            ans = self.stt_model.streaming_inference._step_inference(t, inference_state, ans)

            if return_logits:
                _text_logits.append(ans["text_logits"][:, -1].detach())
                if "asr_logits" in ans:
                    _asr_logits.append(ans["asr_logits"][:, -1].detach())

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

            logging.debug(f"Autoregressive inference step: {t} of {T} !")

        # Trim back to local length if padded
        if self._use_fsdp and T > inference_state["T_local"]:
            gen_codes = gen_codes[:, : inference_state["T_local"]]

        ans = self.stt_model.streaming_inference._post_inference(inference_state, prompt_token_lens)

        if decode_audio:
            gen_codes = gen_codes[:, : inference_state["T_local"]]
            if not incremental_audio_decoding:
                gen_audio_codes_lens = torch.tensor([gen_codes.shape[1]] * gen_codes.shape[0]).to(self.device)
                gen_audio_codes = gen_codes
                with fp32_precision(), torch.no_grad():
                    audio_pred, audio_pred_len = self.tts_model.audio_codec.decode(
                        gen_audio_codes, gen_audio_codes_lens
                    )
            ans["audio"] = audio_pred.squeeze(1)
            ans["audio_len"] = audio_pred_len

        if return_logits:
            ans["text_logits"] = torch.stack(_text_logits, dim=1)
            if _asr_logits:
                ans["asr_logits"] = torch.stack(_asr_logits, dim=1)

        return ans

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Loads model weights with audio prompt latent compatibility.

        This method:

            1. Recreates cached audio prompt latent structures if required
            2. Attempts strict loading
            3. Falls back to partial initialization if necessary

        Args:
            state_dict (dict):
                Model state dictionary.

            strict (bool, optional):
                Whether to enforce strict key matching.

        Returns:
            IncompatibleKeys:
                As returned by LightningModule.load_state_dict.
        """
        # recreate audio prompt latent entries if needed
        self.tts_model.maybe_recreate_cached_audio_prompt_latents_structure(state_dict)
        try:
            return super().load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            logging.info("Error loading model state_dict !! Retrying with partial initialization!")
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            return super().load_state_dict(model_dict, strict=False)
