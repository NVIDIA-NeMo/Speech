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

from dataclasses import dataclass

import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel, parse_multitask_prompt
from nemo.collections.asr.parts.submodules.multitask_decoding import AEDStreamingDecodingConfig
from nemo.collections.asr.parts.utils.eval_utils import compute_laal
from nemo.collections.asr.parts.utils.streaming_utils import ContextSize
from nemo.utils import logging


@dataclass
class AEDStreamingState:
    decoder_input_ids: torch.Tensor | None = None  # tokens ids of initial AED model prompt
    pred_tokens_ids: torch.Tensor | None = None  # buffer with predicted tokens ids
    decoding_step: int = -1  # current decoding step
    decoder_mems_list: list | None = None  # decoder caches, helps to reduce the memory usage
    is_last_chunk_batch: torch.Tensor = False  # whether the current chunk is the last speech chunk in the audio
    max_generation_length: int = (
        256  # maximum number of tokens to be generated for each sample (can be bigger for long audio)
    )
    max_tokens_per_one_second: int = 10  # maximum number of tokens to be generated per one second of audio
    max_tokens_per_alignatt_step: int | None = (
        None  # maximum number of tokens to be generated for each step of alignatt decoding policy
    )
    use_avgpool_for_alignatt: bool = False  # use avgpooling for alignatt decoding policy
    tokens_frame_alignment: torch.Tensor | None = (
        None  # frame alignment of the predicted tokens (used for LAAL calculation in alignatt)
    )
    prev_encoder_shift: int = 0  # previous encoder shift (used for LAAL calculation in alignatt)
    device: torch.device | None = None


class GreedyBatchedStreamingAEDComputer:
    """
    Batched streaming AED decoding with support for waitk and alignatt decoding policies.
    """

    def __init__(
        self,
        asr_model: EncDecMultiTaskModel,
        frame_chunk_size: int,
        decoding_cfg: AEDStreamingDecodingConfig,
    ):
        """
        Init method.
        Args:
            asr_model: instance of ASR model (Canary)
            frame_chunk_size: size of the frame chunk
            decoding_cfg: decoding configuration
        """

        self.asr_model = asr_model
        self.frame_chunk_size = frame_chunk_size
        self.decoding_cfg = decoding_cfg

        # define the decoding method once during initialization
        if decoding_cfg.streaming_policy == "waitk":
            self._run_decoding_step = self.run_waitk_decoding_step
        elif decoding_cfg.streaming_policy == "alignatt":
            self._run_decoding_step = self.run_alignatt_decoding_step
        else:
            raise ValueError("Canary streaming decoding supports only alignatt or waitk decoding policy")

    def __call__(
        self,
        encoder_output: torch.Tensor,
        encoder_output_len: torch.Tensor,
        encoder_input_mask: torch.Tensor,
        prev_batched_state: AEDStreamingState,
    ) -> AEDStreamingState:

        self.state = prev_batched_state
        self.state.encoder_output_len = encoder_output_len

        # initial waitk lagging. Applicable for Wait-k and AlignAtt decoding policies. Control the start of the decoding process.
        if encoder_output_len.max() // self.frame_chunk_size < self.decoding_cfg.waitk_lagging and torch.any(
            torch.logical_not(self.state.is_last_chunk_batch)
        ):
            # need to wait for more speech
            return self.state

        # call the pre-determined decoding method
        self._run_decoding_step(encoder_output, encoder_input_mask)

        return self.state

    def run_waitk_decoding_step(self, encoded_speech, encoder_input_mask):
        """
        Run a decoding step for waitk streaming policy.
        """
        if self.state.decoding_step < 0:
            # first decoding step
            pred_tokens_ids, batch_size, _ = self.asr_model.decoding.decoding.greedy_search._prepare_for_search(
                self.state.decoder_input_ids,
                encoded_speech,
            )
            input_ids = pred_tokens_ids
        else:
            input_ids = self.state.pred_tokens_ids[
                self.state.batch_idxs, self.state.current_context_lengths - 1
            ].unsqueeze(-1)

        self.state.active_samples_inner_loop = (
            torch.ones(self.state.batch_size, dtype=torch.bool, device=self.state.device) * self.state.active_samples
        )
        decoder_mems_list = self.state.decoder_mems_list

        # define start and max generation lengths
        start_from = self.state.decoding_step + 1
        if torch.any(torch.logical_not(self.state.is_last_chunk_batch)):
            # predict only one token per speech chunk if not the last one
            max_generation_length = start_from + 1
        else:
            max_generation_length = self.decoding_cfg.max_generation_length

        # inner decoding loop (with same speech chunk)
        for i in range(start_from, max_generation_length):

            if not decoder_mems_list:
                positional_indexes = torch.zeros_like(self.state.current_context_lengths)
            else:
                positional_indexes = self.state.current_context_lengths - 1

            logits, decoder_mems_list, xatt_scores_list = (
                self.asr_model.decoding.decoding.greedy_search._one_step_forward(
                    input_ids,
                    encoded_speech,
                    encoder_input_mask,
                    decoder_mems_list,
                    positional_indexes,
                    return_scores=False,
                )
            )
            next_tokens = torch.argmax(logits[:, -1], dim=-1)

            # compute eos tokens mask
            is_eos_tokens = next_tokens == self.asr_model.tokenizer.eos
            # rearange active samples (inner loop) depends on eos prediction
            self.state.active_samples_inner_loop *= torch.logical_not(is_eos_tokens)
            # disable samples (upper loop) with eos and end of speech
            eos_and_end_speech_mask = is_eos_tokens * self.state.is_last_chunk_batch
            self.state.active_samples = self.state.active_samples * torch.logical_not(eos_and_end_speech_mask)

            if not torch.any(self.state.active_samples_inner_loop):
                break

            # write predicted tokens to the pred_tokens_ids tensor
            torch.where(self.state.active_samples_inner_loop, next_tokens, self.state.eos_tokens, out=next_tokens)
            self.state.pred_tokens_ids[self.state.batch_idxs, self.state.current_context_lengths] = next_tokens

            self.state.decoding_step += input_ids.size(-1)

            # check for hallucinations
            if self.decoding_cfg.hallucinations_detector:
                hallucination_mask = self.detect_hallucinations(
                    self.state.pred_tokens_ids, self.state.batch_idxs, self.state.current_context_lengths
                )
                self.state.active_samples *= torch.logical_not(hallucination_mask)
                self.state.active_samples_inner_loop *= torch.logical_not(hallucination_mask)

            self.state.current_context_lengths += self.state.active_samples_inner_loop
            input_ids = self.state.pred_tokens_ids[
                self.state.batch_idxs, self.state.current_context_lengths - 1
            ].unsqueeze(-1)

            # disable samples with maximum context length
            samples_with_max_context_length = (
                self.state.current_context_lengths == self.decoding_cfg.max_generation_length - 1
            )
            if torch.any(samples_with_max_context_length * self.state.active_samples):
                logging.info("!!! maximum context length reached !!!")
                self.state.active_samples *= torch.logical_not(samples_with_max_context_length)
                self.state.active_samples_inner_loop *= torch.logical_not(samples_with_max_context_length)

            # zero out decoder_mems_list for non active samples
            if torch.any(torch.logical_not(self.state.active_samples_inner_loop)):
                for j in range(len(decoder_mems_list)):
                    decoder_mems_list[j][:, -1] *= self.state.active_samples_inner_loop.unsqueeze(-1)
            self.state.decoder_mems_list = decoder_mems_list

    def run_alignatt_decoding_step(self, encoded_speech, encoder_input_mask):
        """
        Run a decoding step for alignatt streaming policy.
        """
        if self.state.decoding_step < 0:
            # first decoding step
            pred_tokens_ids, batch_size, _ = self.asr_model.decoding.decoding.greedy_search._prepare_for_search(
                self.state.decoder_input_ids,
                encoded_speech,
            )
            input_ids = pred_tokens_ids
            start_from = 0
        else:
            input_ids = self.state.pred_tokens_ids[
                self.state.batch_idxs, self.state.current_context_lengths - 1
            ].unsqueeze(-1)
            start_from = torch.min(self.state.current_context_lengths).item() - 1

        decoder_mems_list = self.state.decoder_mems_list
        self.state.steps_per_inner_loop = torch.zeros(
            self.state.batch_size, dtype=torch.long, device=self.state.device
        )
        self.state.active_samples_inner_loop = (
            torch.ones(self.state.batch_size, dtype=torch.bool, device=self.state.device) * self.state.active_samples
        )

        for i in range(start_from, self.state.max_generation_length):
            # prepare positional indexes offset for attention decoder
            if not decoder_mems_list:
                positional_indexes = torch.zeros_like(self.state.current_context_lengths)
            else:
                positional_indexes = self.state.current_context_lengths - 1

            logits, decoder_mems_list, xatt_scores_list = (
                self.asr_model.decoding.decoding.greedy_search._one_step_forward(
                    input_ids,
                    encoded_speech,
                    encoder_input_mask,
                    decoder_mems_list,
                    positional_indexes,
                    return_scores=False,
                )
            )
            next_tokens = torch.argmax(logits[:, -1], dim=-1)

            # compute the most attended encoder token
            xatt_scores = xatt_scores_list[self.decoding_cfg.xatt_scores_layer]
            xatt_scores = torch.mean(xatt_scores, 1)
            if i == 0 and xatt_scores.shape[-1] <= self.decoding_cfg.exclude_sink_frames:
                exclude_sink_frames = xatt_scores.shape[-1] // 2
            else:
                exclude_sink_frames = (
                    self.decoding_cfg.exclude_sink_frames if self.state.prev_encoder_shift == 0 else 0
                )
            most_attended_idxs = torch.argmax(xatt_scores[:, :, exclude_sink_frames:], dim=-1) + exclude_sink_frames

            # we can try to smooth peaky xatt scores with avgpooling
            if self.decoding_cfg.use_avgpool_for_alignatt:
                average_pooling_xatt_scores = self.state.avgpool2d(xatt_scores[:, :, exclude_sink_frames:])
                most_attended_idxs_avgpool = torch.argmax(average_pooling_xatt_scores, dim=-1) + exclude_sink_frames
                most_attended_idxs = most_attended_idxs_avgpool

            # select the last attended token for each sample
            if most_attended_idxs.size(-1) > 1:
                most_attended_idxs = most_attended_idxs[:, -1]
            else:
                most_attended_idxs = most_attended_idxs.squeeze(-1)

            # aligatt condition (True -- continue decoding, False -- wait for more speech)
            alignatt_condition = (
                self.state.encoder_output_len - (most_attended_idxs + 1) >= self.decoding_cfg.alignatt_thr
            )
            # alignatt condition is always True for the last speech chunk
            alignatt_condition += self.state.is_last_chunk_batch

            # applay alignatt condition for inner loop
            self.state.active_samples_inner_loop *= alignatt_condition

            # increase speech chunk if no active samples in the inner loop
            if not torch.any(self.state.active_samples_inner_loop):
                break

            # compute eos tokens mask
            # TODO add a case of "." + EOS prediction for models with PC support?
            is_eos_tokens = next_tokens == self.asr_model.tokenizer.eos
            # rearange active samples (inner loop) depends on eos prediction
            self.state.active_samples_inner_loop *= torch.logical_not(is_eos_tokens)
            # disable samples (upper loop) with eos and end of speech
            eos_and_end_speech_mask = is_eos_tokens * self.state.is_last_chunk_batch
            self.state.active_samples *= torch.logical_not(eos_and_end_speech_mask)

            if not torch.any(self.state.active_samples_inner_loop):
                break

            # write predicted tokens to the pred_tokens_ids tensor
            torch.where(self.state.active_samples_inner_loop, next_tokens, self.state.eos_tokens, out=next_tokens)
            self.state.pred_tokens_ids[self.state.batch_idxs, self.state.current_context_lengths] = next_tokens

            # update tokens frame alignment based on current encoder step (this alignment is used for LAAL calculation)
            self.state.tokens_frame_alignment[self.state.batch_idxs, self.state.current_context_lengths] = (
                self.state.encoder_output_len
                + self.state.prev_encoder_shift  # we need to add the real frame position in the audio signal
            )

            self.state.decoding_step += input_ids.size(-1)

            # check for hallucinations
            if self.decoding_cfg.hallucinations_detector:
                hallucination_mask = self.detect_hallucinations(
                    self.state.pred_tokens_ids, self.state.batch_idxs, self.state.current_context_lengths
                )
                if torch.any(hallucination_mask):
                    self.state.active_samples *= torch.logical_not(hallucination_mask)
                    self.state.active_samples_inner_loop *= torch.logical_not(hallucination_mask)

            # disable samples with maximum context length
            samples_with_max_context_length = (
                self.state.current_context_lengths == self.state.max_generation_length - 1
            )
            if torch.any(samples_with_max_context_length * self.state.active_samples):
                logging.info("!!! maximum context length reached !!!")
                self.state.active_samples *= torch.logical_not(samples_with_max_context_length)
                self.state.active_samples_inner_loop *= torch.logical_not(samples_with_max_context_length)

            # zero out decoder_mems_list for non active samples
            # TODO batched decoding works wrong if first token was EOS for one of the samples
            if torch.any(torch.logical_not(self.state.active_samples_inner_loop)):
                for j in range(len(decoder_mems_list)):
                    decoder_mems_list[j][:, -1] *= self.state.active_samples_inner_loop.unsqueeze(-1)

            self.state.decoder_mems_list = decoder_mems_list
            self.state.current_context_lengths += self.state.active_samples_inner_loop
            # TODO model does not predicts any real tokens in the case of first EOS prediction (rare case for batched decoding)
            input_ids = self.state.pred_tokens_ids[
                self.state.batch_idxs, self.state.current_context_lengths - 1
            ].unsqueeze(-1)

            # limit number of steps per inner loop if not end of speech
            if self.state.max_tokens_per_alignatt_step is not None:
                self.state.steps_per_inner_loop += self.state.active_samples_inner_loop
                disable_samples_mask = self.state.steps_per_inner_loop >= self.state.max_tokens_per_alignatt_step
                disable_samples_mask *= torch.logical_not(self.state.is_last_chunk_batch)
                self.state.active_samples_inner_loop *= torch.logical_not(disable_samples_mask)

            if not torch.any(self.state.active_samples_inner_loop):
                break

    def detect_hallucinations(self, pred_tokens_ids, batch_idxs, current_context_lengths):

        # we need to have at least 8 tokens to run hallucinations detector
        if torch.any(current_context_lengths < 8):
            return torch.zeros(pred_tokens_ids.shape[0], dtype=torch.bool, device=pred_tokens_ids.device)

        ccl = current_context_lengths
        # pattern 1: four consecutive tokens are the same: "a a a a"
        hallucination_mask_1 = (
            (pred_tokens_ids[batch_idxs, ccl] == pred_tokens_ids[batch_idxs, ccl - 1])
            * (pred_tokens_ids[batch_idxs, ccl] == pred_tokens_ids[batch_idxs, ccl - 2])
            * (pred_tokens_ids[batch_idxs, ccl] == pred_tokens_ids[batch_idxs, ccl - 3])
            * (pred_tokens_ids[batch_idxs, ccl] == pred_tokens_ids[batch_idxs, ccl - 4])
        )
        if torch.any(hallucination_mask_1):
            logging.info("!!! hallucination 'a a a a' detected !!!")
        # pattern 2: "a b a b a b"
        hallucination_mask_2 = (
            (pred_tokens_ids[batch_idxs, ccl] == pred_tokens_ids[batch_idxs, ccl - 2])
            * (pred_tokens_ids[batch_idxs, ccl - 1] == pred_tokens_ids[batch_idxs, ccl - 3])
            * (pred_tokens_ids[batch_idxs, ccl] == pred_tokens_ids[batch_idxs, ccl - 4])
            * (pred_tokens_ids[batch_idxs, ccl - 1] == pred_tokens_ids[batch_idxs, ccl - 5])
        )
        if torch.any(hallucination_mask_2):
            logging.info("!!! hallucination 'a b a b a b' detected !!!")
        # pattern 3: "a b c a b c a b c"
        hallucination_mask_3 = (
            (pred_tokens_ids[batch_idxs, ccl] == pred_tokens_ids[batch_idxs, ccl - 3])
            * (pred_tokens_ids[batch_idxs, ccl - 1] == pred_tokens_ids[batch_idxs, ccl - 4])
            * (pred_tokens_ids[batch_idxs, ccl - 2] == pred_tokens_ids[batch_idxs, ccl - 5])
            * (pred_tokens_ids[batch_idxs, ccl] == pred_tokens_ids[batch_idxs, ccl - 6])
            * (pred_tokens_ids[batch_idxs, ccl - 1] == pred_tokens_ids[batch_idxs, ccl - 7])
            * (pred_tokens_ids[batch_idxs, ccl - 2] == pred_tokens_ids[batch_idxs, ccl - 8])
        )
        if torch.any(hallucination_mask_3):
            logging.info("!!! hallucination 'a b c a b c a b c' detected !!!")
        hallucination_mask = hallucination_mask_1 + hallucination_mask_2 + hallucination_mask_3
        return hallucination_mask

    def compute_alignatt_lagging(
        self,
        records,
        predicted_token_ids,
        tokens_frame_alignment,
        context_encoder_frames,
        audio_encoder_fs,
        BOW_PREFIX="\u2581",
    ):
        # import ipdb; ipdb.set_trace()
        tokens_idx_shift = self.state.decoder_input_ids.size(-1)
        target_length_word = [len(item['text'].split()) for item in records]
        audio_signal_lengths = [float(item['duration']) * 1000 for item in records]
        # import ipdb; ipdb.set_trace()
        tokenizer_vocab = self.asr_model.tokenizer.vocab
        eos_token = tokenizer_vocab[self.asr_model.tokenizer.eos_id]
        laal_list = []
        for i, tokens in enumerate(predicted_token_ids):
            if len(tokens) == 0:
                laal_list.append(5000)
                continue
            audio_signal_length = audio_signal_lengths[i]
            # obtain lagging for alignatt
            lagging = []
            for j, cur_t in enumerate(tokens):
                pred_idx = (
                    tokens_frame_alignment[i][tokens_idx_shift + j] + context_encoder_frames.right
                )  # TODO: check right_context
                cur_t = tokenizer_vocab[cur_t.item()]
                if (cur_t.startswith(BOW_PREFIX) and cur_t != BOW_PREFIX) or cur_t == eos_token:  # word boundary
                    lagging.append(pred_idx * audio_encoder_fs)
                if cur_t == eos_token:
                    break
            if len(lagging) == 0:
                lagging.append(0)
            laal = compute_laal(lagging, audio_signal_length, target_length_word[i])
            if torch.is_tensor(laal):
                laal_list.append(laal.item())
            else:
                laal_list.append(laal)
        return laal_list

    def compute_waitk_lagging(
        self, records, predicted_token_ids, context_encoder_frames, audio_encoder_fs, BOW_PREFIX="\u2581"
    ):
        waitk_lagging = self.decoding_cfg.waitk_lagging
        pre_decision_ratio = context_encoder_frames.chunk
        target_length_word = [len(item['text'].split()) for item in records]
        audio_signal_lengths = [float(item['duration']) * 1000 for item in records]
        tokenizer_vocab = self.asr_model.tokenizer.vocab
        laal_list = []
        for i, tokens in enumerate(predicted_token_ids):
            lagging = []
            audio_signal_length = audio_signal_lengths[i]
            for j, cur_t in enumerate(tokens):
                cur_src_len = (j + waitk_lagging) * pre_decision_ratio + context_encoder_frames.right
                cur_src_len *= audio_encoder_fs  # to ms
                cur_src_len = min(audio_signal_length, cur_src_len)
                spm = tokenizer_vocab[cur_t.item()]
                # reach word boundary
                if (
                    spm.startswith(BOW_PREFIX) and spm != BOW_PREFIX
                ) or cur_t == self.asr_model.tokenizer.eos_id:  # word boundary
                    lagging.append(cur_src_len)
                if cur_t == self.asr_model.tokenizer.eos_id:
                    break
            if len(lagging) == 0:
                lagging.append(0)
            laal = compute_laal(lagging, audio_signal_length, target_length_word[i])
            laal_list.append(laal)
        return laal_list

    @classmethod
    def initialize_aed_model_state(
        cls,
        asr_model,
        decoder_input_ids: torch.Tensor,
        batch_size: int,
        context_encoder_frames: ContextSize,
        chunk_secs: float,
        right_context_secs: float,
    ) -> AEDStreamingState:
        """
        Initialize AED model state for streaming inference.

        Args:
            asr_model: ASR model instance (used for tokenizer and device)
            decoder_input_ids: Prompt tensor for decoder input
            batch_size: Batch size for inference
            context_encoder_frames: Context size configuration

        Returns:
            Initialized AEDStreamingState object
        """
        # initialize AED model state
        model_state = AEDStreamingState(decoder_input_ids=decoder_input_ids, device=asr_model.device)

        model_state.frame_chunk_size = context_encoder_frames.chunk
        model_state.batch_idxs = torch.arange(batch_size, dtype=torch.long, device=asr_model.device)
        model_state.current_context_lengths = torch.zeros_like(model_state.batch_idxs) + decoder_input_ids.size(-1)
        model_state.decoder_input_ids = decoder_input_ids[:batch_size]
        model_state.pred_tokens_ids = torch.full(
            [batch_size, model_state.max_generation_length],
            asr_model.tokenizer.eos,
            dtype=torch.long,
            device=asr_model.device,
        )
        model_state.pred_tokens_ids[:, : model_state.decoder_input_ids.size(-1)] = model_state.decoder_input_ids
        model_state.tokens_frame_alignment = torch.zeros_like(model_state.pred_tokens_ids)
        model_state.active_samples = torch.ones(batch_size, dtype=torch.bool, device=asr_model.device)
        model_state.active_samples_inner_loop = torch.ones(batch_size, dtype=torch.bool, device=asr_model.device)
        model_state.right_context = context_encoder_frames.right
        model_state.eos_tokens = torch.full(
            [batch_size], asr_model.tokenizer.eos, dtype=torch.long, device=asr_model.device
        )
        model_state.avgpool2d = torch.nn.AvgPool2d(5, stride=1, padding=2, count_include_pad=False)
        model_state.batch_size = batch_size
        model_state.max_tokens_per_alignatt_step = model_state.max_tokens_per_one_second * int(
            chunk_secs + right_context_secs
        )

        return model_state


def return_decoder_input_ids(decoding_config, asr_model) -> torch.Tensor:
    """
    Obtain decoder input ids for decoding config.
    """
    override_cfg = asr_model.get_transcribe_config()
    override_cfg.prompt = parse_multitask_prompt(OmegaConf.to_container(decoding_config.prompt))

    default_turns = asr_model.prompt.get_default_dialog_slots()
    if not decoding_config.prompt:
        # No turns were provided, use defaults.
        turns = default_turns
    else:
        # Turns were provided, iterate over them and fill missing slot values using defaults..
        turns = override_cfg.prompt.copy()  # shallow copy #1: don't override the config
        for turn in turns:
            role = turn["role"]
            # Check if we have defaults for this role.
            # There shouldn't be more than a single turn for a given role, but if there are,
            # we'll emit a warning.
            if default_turns_for_role := [t for t in default_turns if t["role"] == role]:
                if len(default_turns_for_role) > 1:
                    logging.warning(
                        f"More than one default turn detected for {role=}. "
                        f"We'll be using default slot values for the first turn of {role=} only."
                    )
                default_slots = default_turns_for_role[0]["slots"]
                turn["slots"] = turn["slots"].copy()  # shallow copy #1: don't override the config
                # fill missing slots using defaults
                for slot, val in default_slots.items():
                    if turn["slots"].get(slot) is None:
                        turn["slots"][slot] = val

    decoder_input_ids = (
        asr_model.prompt.encode_dialog(turns=turns)["context_ids"]
        .unsqueeze(0)
        .repeat(decoding_config.batch_size, 1)
        .to(asr_model.device)
    )

    return decoder_input_ids
