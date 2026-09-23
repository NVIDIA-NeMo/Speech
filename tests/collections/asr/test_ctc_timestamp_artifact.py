# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch

from nemo.collections.common.tokenizers.sentencepiece_tokenizer import (
    SentencePieceTokenizer,
)
from nemo.collections.speechlm2.parts.ctc_timestamp_utils import (
    CTC_TIMESTAMP_ARTIFACT_FORMAT,
    TransformerCTCDecoder,
    export_ctc_timestamp_artifact,
    load_ctc_timestamp_artifact,
    save_ctc_timestamp_artifact,
)


@pytest.mark.unit
def test_lightweight_artifact_round_trip_contains_decoder_and_tokenizer(tmp_path):
    tokenizer_path = Path(__file__).resolve().parents[2] / ".data/asr/tokenizers/an4_spe_128/tokenizer.model"
    tokenizer = SentencePieceTokenizer(str(tokenizer_path))
    vocabulary = tokenizer.ids_to_tokens(list(range(tokenizer.vocab_size)))
    config = {
        "feat_in": 8,
        "num_classes": tokenizer.vocab_size,
        "vocabulary": vocabulary,
        "use_transformer": False,
    }
    decoder = TransformerCTCDecoder(**config)
    artifact_path = save_ctc_timestamp_artifact(tmp_path / "timestamp.pt", decoder, tokenizer, config)

    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    assert set(payload) == {
        "format",
        "decoder_config",
        "decoder_state_dict",
        "tokenizer",
    }
    assert payload["format"] == CTC_TIMESTAMP_ARTIFACT_FORMAT
    assert "encoder" not in payload

    restored = load_ctc_timestamp_artifact(artifact_path)
    assert restored.tokenizer.text_to_ids("hello world") == tokenizer.text_to_ids("hello world")
    assert restored.decoder_config == config
    for name, value in decoder.state_dict().items():
        assert torch.equal(restored.decoder.state_dict()[name], value)


@pytest.mark.unit
def test_loader_rejects_decoder_only_legacy_payload(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"format": "transformer_ctc_decoder_state_dict_v1"}, path)

    with pytest.raises(ValueError, match="Convert the original .nemo adapter"):
        load_ctc_timestamp_artifact(path)


@pytest.mark.unit
def test_export_temporarily_registers_legacy_decoder_target(tmp_path, monkeypatch):
    import nemo.collections.asr.modules as asr_modules
    from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE

    source = tmp_path / "adapter.nemo"
    source.touch()
    tokenizer_path = Path(__file__).resolve().parents[2] / ".data/asr/tokenizers/an4_spe_128/tokenizer.model"
    tokenizer = SentencePieceTokenizer(str(tokenizer_path))
    config = {
        "feat_in": 8,
        "num_classes": tokenizer.vocab_size,
        "vocabulary": tokenizer.ids_to_tokens(list(range(tokenizer.vocab_size))),
        "use_transformer": False,
    }
    decoder = TransformerCTCDecoder(**config)

    def restore_from(*args, **kwargs):
        assert asr_modules.TransformerCTCDecoder is TransformerCTCDecoder
        cfg = type("Config", (), {"decoder": config})()
        return type("Adapter", (), {"decoder": decoder, "tokenizer": tokenizer, "cfg": cfg})()

    monkeypatch.delattr(asr_modules, "TransformerCTCDecoder", raising=False)
    monkeypatch.setattr(EncDecCTCModelBPE, "restore_from", restore_from)

    output = export_ctc_timestamp_artifact(source, tmp_path / "timestamp.pt")

    assert output.is_file()
    assert not hasattr(asr_modules, "TransformerCTCDecoder")
