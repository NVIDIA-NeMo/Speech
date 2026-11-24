# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NDJSON audio and attention-alignment streaming for a complete V2 target text."""
from __future__ import annotations

import base64
import copy
import json
import logging
from contextlib import aclosing

import numpy as np
from easymagpie_vllm_omni.audio_output import extract_audio_from_stage_output
from easymagpie_vllm_omni.timestamps import StreamingAlignmentDecoder, unpack_alignment_packets
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from vllm.sampling_params import RequestOutputKind
from vllm.utils import random_uuid
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.utils import coerce_param_message_types


logger = logging.getLogger(__name__)


async def create_aligned_speech(request: OpenAICreateSpeechRequest, raw_request: Request):
    """Stream PCM chunks and causal timestamp deltas; finish with complete timestamp dictionaries."""
    service = raw_request.app.state.openai_serving_speech
    if service is None or getattr(service, '_tts_model_type', None) != 'easymagpie':
        return JSONResponse({'error': 'This endpoint requires an EasyMagpie V2 speech server'}, status_code=400)
    if request.speed not in (None, 1.0):
        return JSONResponse({'error': 'Attention timestamps require speed=1.0'}, status_code=400)
    extra = dict(request.extra_params or {}, return_timestamps=True)
    request = request.model_copy(update={'extra_params': extra, 'stream': True, 'response_format': 'pcm'})
    adapter = service._get_tts_adapter()
    error = adapter.validate(request)
    if error:
        return JSONResponse({'error': error}, status_code=400)
    model_error = await service._check_model(request) if request.model is not None else None
    if model_error is not None:
        return JSONResponse({'error': getattr(model_error, 'message', str(model_error))}, status_code=400)
    params = coerce_param_message_types(
        copy.deepcopy(service.engine_client.default_sampling_params_list), is_streaming=True
    )
    params[0].max_tokens = request.max_new_tokens or params[0].max_tokens
    for value in params:
        value.output_kind = RequestOutputKind.DELTA
    prepared = await adapter.build(request, params, False)
    alignment = StreamingAlignmentDecoder(
        adapter._model_path(), tokenizer=adapter._model_tokenizer(), text=request.input
    )
    request_id = 'speech-aligned-' + random_uuid()

    async def stream():
        total_samples = 0
        sample_rate = None
        completed = False
        observed = False
        try:
            yield _line(
                {
                    'type': 'speech.start',
                    'request_id': request_id,
                    'format': 'pcm_s16le',
                    'timings_are_estimates': True,
                }
            )
            generator = service.engine_client.generate(
                prompt=prepared.prompt,
                request_id=request_id,
                sampling_params_list=params,
                output_modalities=['audio'],
            )
            async with aclosing(generator):
                async for output in generator:
                    if await raw_request.is_disconnected():
                        return
                    for packet in unpack_alignment_packets(output):
                        delta = alignment.update(packet)
                        if delta is not None:
                            observed = True
                            if extra.get('timestamp_debug') is True:
                                delta['attention_observation'] = packet
                            yield _line(delta)
                    extracted = extract_audio_from_stage_output(output)
                    if extracted is None:
                        continue
                    waveform, rate = extracted
                    if sample_rate is not None and rate != sample_rate:
                        raise ValueError('The audio sample rate changed within an aligned stream')
                    sample_rate = rate
                    pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype('<i2').tobytes()
                    yield _line(
                        {
                            'type': 'audio.chunk',
                            'sample_rate': rate,
                            'sample_offset': total_samples,
                            'format': 'pcm_s16le',
                            'audio_base64': base64.b64encode(pcm).decode('ascii'),
                        }
                    )
                    total_samples += len(pcm) // 2
            if not observed:
                raise RuntimeError('The V2 worker returned no attention observations')
            if not total_samples:
                raise RuntimeError('The V2 worker returned no audio samples')
            result = alignment.finalize(total_samples / sample_rate if sample_rate else 0.0)
            result.update(request_id=request_id, audio_sample_rate=sample_rate, audio_samples=total_samples)
            yield _line(result)
            yield _line(
                {
                    'type': 'speech.done',
                    'request_id': request_id,
                    'audio_samples': total_samples,
                    'sample_rate': sample_rate,
                }
            )
            completed = True
        except Exception as error:
            yield _line({'type': 'error', 'request_id': request_id, 'message': str(error)})
        finally:
            if not completed:
                try:
                    await service.engine_client.abort(request_id)
                except Exception:
                    logger.debug('Failed to abort aligned request %s', request_id, exc_info=True)

    return StreamingResponse(stream(), media_type='application/x-ndjson', headers={'X-Accel-Buffering': 'no'})


def register_alignment_endpoint() -> None:
    """Attach the optional endpoint during the existing serving-plugin registration."""
    from vllm_omni.entrypoints.openai import api_server

    endpoint = '/v1/audio/speech/alignments'
    if not any(getattr(route, 'path', None) == endpoint for route in api_server.router.routes):
        api_server.router.add_api_route(endpoint, create_aligned_speech, methods=['POST'])


def _line(value):
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
