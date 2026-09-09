# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
"""EasyMagpieTTS pipeline topologies for vLLM-Omni.

EasyMagpie LM reports the generic ``model_type="nemotron_h"``, so routing uses
its HF architecture or an explicit ``pipeline: …`` deployment setting.

* :data:`EASYMAGPIE_PIPELINE` — two-stage text → acoustic codes → waveform.
* :data:`EASYMAGPIE_LM_PIPELINE` — single-stage acoustic-token
  prediction only (no in-engine Code2Wav). Use this to benchmark and develop
  EasyMagpie LM in isolation; select it with ``pipeline: easymagpie_lm``.
"""
from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

_PROC = "easymagpie_vllm_omni.stage_processors"

# EasyMagpie LM repurposes a 2-wide dummy backbone vocab as a continue/stop signal;
# the last index is the audio-EOS stop token (see
# ``EasyMagpieTTSForConditionalGeneration.audio_eos_stop_token_id``).
_AUDIO_EOS_STOP_TOKEN_ID = 1

EASYMAGPIE_PIPELINE = PipelineConfig(
    model_type="easymagpie",
    model_arch="EasyMagpieTTSForConditionalGeneration",
    hf_architectures=(
        "EasyMagpieTTSForConditionalGeneration",
        "EasyMagpieTTS",
    ),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="easymagpie",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            # Surface stage 0 as a (latent) final output so incremental WebSocket
            # requests can pace their next text chunk against the acoustic frames
            # already produced. This is safe for the plain /v1/audio/speech path:
            # in two-stage (latent) mode stage-0 codes are emitted under the
            # ``audio_codes``/``codes`` keys (see EasyMagpieTTS.make_omni_output),
            # which the HTTP handler's ``_extract_audio_output`` does not treat as
            # audio (it only keys on ``audio``/``model_outputs``), so those deltas
            # are skipped and only the stage-1 waveform is returned.
            final_output=True,
            final_output_type="latent",
            engine_output_type="latent",
            # Resumable/segment-stop scheduling for paced streaming; a no-op for
            # non-resumable single-shot HTTP requests.
            scheduler_cls="easymagpie_vllm_omni.scheduler.EasyMagpieARAsyncScheduler",
            async_chunk_process_next_stage_input_func=f"{_PROC}.talker2code2wav_async_chunk",
            custom_process_next_stage_input_func=f"{_PROC}.talker2code2wav_full_payload",
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [_AUDIO_EOS_STOP_TOKEN_ID],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="easymagpie_codec",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="EasyMagpieCodecForConditionalGeneration",
            model_subdir="codec_native",
            scheduler_cls="easymagpie_vllm_omni.scheduler.EasyMagpieCodecScheduler",
            # Sync mode uses one placeholder per frame; the connector carries codes.
            sync_process_input_func=f"{_PROC}.talker2code2wav_token_only",
            sampling_constraints={"detokenize": True},
        ),
    ),
)

EASYMAGPIE_LM_PIPELINE = PipelineConfig(
    model_type="easymagpie_lm",
    model_arch="EasyMagpieTTSForConditionalGeneration",
    hf_architectures=(
        "EasyMagpieTTSForConditionalGeneration",
        "EasyMagpieTTS",
    ),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="easymagpie",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            final_output=True,
            # A single stage that is *also* the final stage needs
            # ``engine_output_type="audio"`` for two reasons:
            #  1. vLLM-Omni's AR runner force-includes single-stage requests in
            #     the client pooler payload only for ``engine_output_type ==
            #     "audio"`` (see GPUARModelRunner._resolve_pooler_payload_req_ids).
            #     With "latent" there is no downstream stage to consume the codes
            #     and no client payload is built, so the codes get filtered out.
            #  2. It drives the output modality to "audio", so the model's
            #     ``model_outputs`` codes key (see EasyMagpieTTS.make_omni_output,
            #     which keys off engine_output_type) is remapped to the DRAINABLE
            #     ``audio`` modality — the client then streams per-step code
            #     deltas instead of a growing cumulative payload every step.
            final_output_type="audio",
            engine_output_type="audio",
            scheduler_cls="easymagpie_vllm_omni.scheduler.EasyMagpieARAsyncScheduler",
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [_AUDIO_EOS_STOP_TOKEN_ID],
            },
        ),
    ),
)
