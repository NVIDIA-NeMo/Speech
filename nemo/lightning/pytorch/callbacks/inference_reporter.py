"""
Inference reporter callback for validation-time text generation.

Runs Megatron Core inference during validation and logs results (tokens, logprobs,
logits, text) as artifacts using Lightning logger interface.
"""

import json
import os
import tempfile
import time
from contextlib import nullcontext
from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from megatron.core import parallel_state
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.inference_request import InferenceRequest, Status
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import TextGenerationController


class InferenceReporter(L.Callback):
    """
    Runs inference during validation and logs results as artifacts.

    Args:
        checkpoint_name: Identifier for checkpoint in artifact paths
        dataset_name: Identifier for dataset in artifact paths
        inference_batch_times_seqlen_threshold: Memory threshold for inference batching
        inference_max_seq_length: Maximum sequence length for inference
        inference_params_dtype: Data type for inference parameters (defaults to model dtype)
        output_dir: Base directory for outputs
        max_batch_size: Maximum batch size (currently only 1 supported)
        random_seed: Random seed for reproducibility
        sampling_params: Dictionary of sampling parameters for SamplingParams
    """

    def __init__(
        self,
        checkpoint_name: str,
        dataset_name: str,
        inference_batch_times_seqlen_threshold: int,
        inference_max_seq_length: int,
        inference_params_dtype: torch.dtype | None = None,
        output_dir: str = "./",
        max_batch_size: int | None = None,
        random_seed: int = 0,
        sampling_params: dict[str, Any] | None = None,
    ) -> None:
        self.checkpoint_name = checkpoint_name
        self.dataset_name = dataset_name
        self.output_dir = os.path.join(output_dir, f"{checkpoint_name}-{dataset_name}")
        self.inference_batch_times_seqlen_threshold = inference_batch_times_seqlen_threshold
        self.inference_params_dtype = inference_params_dtype
        self.inference_max_seq_length = inference_max_seq_length
        self.max_batch_size = max_batch_size
        self.random_seed = random_seed
        self.sampling_params = SamplingParams(**(sampling_params or {}))
        self.sample_idx = 0
        self.text_generation_controller: TextGenerationController | None = None

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        pl_module.tokenizer.detokenize = pl_module.tokenizer.ids_to_text

        # Add noop methods to avoid exceptions - we don't need text processing
        if not hasattr(pl_module.tokenizer, "offsets"):
            pl_module.tokenizer.offsets = lambda tokens, text: []

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        L.seed_everything(self.random_seed)

        prompt_tokens = self._get_prompt_tokens(batch)
        generated_tokens, prompt_logprobs, prompt_logits = self._run_inference(pl_module, prompt_tokens)

        input_text = pl_module.tokenizer.detokenize(prompt_tokens)
        generated_text = pl_module.tokenizer.detokenize(generated_tokens)

        self._log_artifacts(
            trainer,
            batch_idx,
            generated_tokens,
            prompt_logprobs,
            prompt_logits,
            prompt_tokens,
            input_text,
            generated_text,
        )
        self.sample_idx += 1

    def _get_prompt_tokens(self, batch: Any) -> list[int]:
        assert len(batch["tokens"]) == 1, "Only one sample at a time generation supported at the moment"
        tokens = batch["tokens"][0]

        # Add the label token (last token from original sequence) to prompt_tokens
        if torch.distributed.get_rank() == 0 and "labels" in batch and len(batch["labels"]) > 0:
            last_label = batch["labels"][0][-1].item()
            tokens = torch.cat([tokens, torch.tensor([last_label], device=tokens.device)])

        device = "cuda" if torch.cuda.is_available() else "cpu"
        seq_len = torch.tensor(
            [tokens.size(0) if torch.distributed.get_rank() == 0 else 0],
            dtype=torch.long,
            device=device,
        )
        torch.distributed.broadcast(seq_len, src=0)

        if torch.distributed.get_rank() == 0:
            tokens = tokens.cuda() if torch.cuda.is_available() else tokens
        else:
            tokens = torch.empty(int(seq_len.item()), dtype=torch.long, device=device)

        torch.distributed.broadcast(tokens, src=0)
        return tokens.cpu().tolist()

    def _get_inference_engine(self, pl_module: L.LightningModule) -> TextGenerationController:
        if self.text_generation_controller is not None:
            return self.text_generation_controller

        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=pl_module.config.hidden_size,
            inference_batch_times_seqlen_threshold=self.inference_batch_times_seqlen_threshold,
            inference_max_requests=1,
            fp32_residual_connection=False,
            params_dtype=self.inference_params_dtype or pl_module.dtype,
            padded_vocab_size=pl_module.module.module.vocab_size,
            inference_max_seq_length=self.inference_max_seq_length,
        )

        inference_context = StaticInferenceContext.from_config(inference_wrapper_config)
        inference_wrapped_model = GPTInferenceWrapper(pl_module.module, inference_wrapper_config, inference_context)

        self.text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model,
            tokenizer=pl_module.tokenizer,
        )
        return self.text_generation_controller

    def _run_inference(
        self, pl_module: L.LightningModule, prompt_tokens: list[int]
    ) -> tuple[list[int], list[float] | None, Any | None]:
        inference_request = InferenceRequest(
            request_id=(request_id := "0"),
            prompt="",
            sampling_params=self.sampling_params,
            arrival_time=time.time(),
            prompt_tokens=prompt_tokens,
            status=Status.ACTIVE_BUT_NOT_GENERATING_TOKENS,
        )

        results = self._get_inference_engine(pl_module).generate_all_output_tokens_static_batch(
            {request_id: inference_request}
        )

        result = results[request_id]
        generated_tokens = result.generated_tokens.tolist()
        prompt_logprobs = result.prompt_log_probs
        prompt_logits = result.logits if hasattr(result, "prompt_logits") else None

        return generated_tokens, prompt_logprobs, prompt_logits

    def _log_artifacts(
        self,
        trainer: L.Trainer,
        batch_idx: int,
        generated_tokens: list[int],
        prompt_logprobs: list[float] | None,
        prompt_logits: Any | None,
        prompt_tokens: list[int],
        input_text: str,
        generated_text: str,
    ) -> None:
        if not (
            generated_tokens
            and parallel_state.get_tensor_model_parallel_rank() == 0
            and parallel_state.get_data_parallel_rank() == 0
        ):
            return

        artifact_path = f"inference/validation/step_{trainer.global_step}/batch_{batch_idx}"
        data_map = {
            "token_ids": generated_tokens,
            "prompt_logprobs": prompt_logprobs,
            "token_logits": prompt_logits,
            "prompt_tokens": prompt_tokens,
            "input_text": input_text,
            "generated_text": generated_text,
        }

        has_logger = (
            trainer.logger
            and hasattr(trainer.logger, "experiment")
            and hasattr(trainer.logger.experiment, "log_artifact")
        )

        ctx = (
            tempfile.TemporaryDirectory() if has_logger else nullcontext(os.path.join(self.output_dir, artifact_path))
        )
        with ctx as base_dir:
            for dir_name, data in data_map.items():
                if data:
                    dir_path = os.path.join(base_dir, dir_name)
                    os.makedirs(dir_path, exist_ok=True)
                    file_path = os.path.join(dir_path, f"{dir_name}_{self.sample_idx}.json")
                    with open(file_path, "w") as f:
                        json.dump(data, f)
                    if has_logger:
                        trainer.logger.experiment.log_artifact(file_path, f"{artifact_path}/{dir_name}")
