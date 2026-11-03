# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
from nemo.collections.speechlm2.parts.ctc_timestamp_utils import (
    TransformerCTCDecoder,
    load_ctc_timestamp_artifact,
    save_ctc_timestamp_artifact,
)


@pytest.mark.unit
@pytest.mark.parametrize('mode', ['conv_transpose', 'repeat', 'repeat_phase'])
@pytest.mark.parametrize('factor', [1, 2])
def test_temporal_upsampling_shapes_padding_and_artifact_roundtrip(tmp_path, mode, factor):
    tokenizer = SentencePieceTokenizer(
        str(Path(__file__).resolve().parents[2] / '.data/asr/tokenizers/an4_spe_128/tokenizer.model')
    )
    config = dict(
        feat_in=32,
        num_classes=tokenizer.vocab_size,
        upsampling_factor=factor,
        upsampling_mode=mode,
        use_transformer=mode != 'conv_transpose',
        n_heads=2,
        n_layers=1,
        drop_rate=0.0,
        sync_max_audio_length=False,
    )
    decoder = TransformerCTCDecoder(**config).eval()
    states, lengths = torch.randn(2, 32, 5), torch.tensor([5, 3])
    changed = states.clone()
    changed[1, :, 3:] = 100 * torch.randn_like(changed[1, :, 3:])
    with torch.no_grad():
        if factor > 1:
            torch.testing.assert_close(decoder._upsample(states), states.repeat_interleave(factor, dim=-1))
        output = decoder(states, encoded_lengths=lengths)
        perturbed = decoder(changed, encoded_lengths=lengths)
    assert output.shape == (2, 5 * factor, tokenizer.vocab_size + 1)
    torch.testing.assert_close(output[1, : 3 * factor], perturbed[1, : 3 * factor])
    torch.testing.assert_close(decoder.output_lengths(lengths), lengths * factor)
    if factor == 1:
        assert not any(key.startswith('upsample') or key.startswith('phase') for key in decoder.state_dict())
    path = save_ctc_timestamp_artifact(tmp_path / 'head.safetensors', decoder, tokenizer, config)
    restored = load_ctc_timestamp_artifact(path).decoder
    with torch.no_grad():
        torch.testing.assert_close(restored(states, encoded_lengths=lengths), output)
    torch.testing.assert_close(restored.output_lengths(lengths), lengths * factor)


@pytest.mark.unit
@pytest.mark.parametrize(
    'mode,use_transformer,attention',
    [('repeat', False, 'rope'), ('repeat_phase', False, 'rope'), ('repeat', True, 'no_pos')],
)
def test_temporal_upsampling_rejects_invalid_bridge(mode, use_transformer, attention):
    with pytest.raises(ValueError, match='needs|unless'):
        TransformerCTCDecoder(
            feat_in=8,
            num_classes=3,
            upsampling_factor=2,
            upsampling_mode=mode,
            use_transformer=use_transformer,
            self_attention_model=attention,
        )


@pytest.mark.unit
@pytest.mark.parametrize('model_class', [EncDecCTCModel, EncDecCTCModelBPE])
@pytest.mark.parametrize('factor', [None, 1, 2])
def test_native_transcription_uses_ctc_output_frame_rate(model_class, factor):
    hyp = Hypothesis(score=0.0, y_sequence=torch.tensor([1]), text='hello')
    hyp.timestamp = {key: [dict(start_offset=10, end_offset=11)] for key in ('char', 'word', 'segment')}
    prediction = Mock(return_value=[hyp])
    model = SimpleNamespace(
        decoder=SimpleNamespace(**({} if factor is None else {'upsampling_factor': factor})),
        encoder=SimpleNamespace(subsampling_factor=8),
        cfg={'preprocessor': {'window_stride': 0.01}},
        decoding=SimpleNamespace(ctc_decoder_predictions_tensor=prediction),
    )
    logits, lengths = torch.randn(1, 24, 4), torch.tensor([12])
    results = model_class._transcribe_output_processing(
        model, {'logits': logits, 'logits_len': lengths}, SimpleNamespace(timestamps=True, return_hypotheses=True)
    )
    assert results[0].text == 'hello'
    assert results[0].y_sequence.shape == (12, 4)
    assert prediction.call_args.kwargs['decoder_lengths'] is lengths
    step = 0.08 / (factor or 1)
    for key in ('char', 'word', 'segment'):
        assert results[0].timestamp[key][0]['start'] == pytest.approx(10 * step)
        assert results[0].timestamp[key][0]['end'] == pytest.approx(11 * step)
