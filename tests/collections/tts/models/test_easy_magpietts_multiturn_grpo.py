# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel
from nemo.collections.tts.models.easy_magpietts_preference_optimization import EasyMagpieTTSModelOnlinePO
from tests.collections.tts.models.test_easy_magpietts import (
    _codec_restore_uses_random_initialized_audio_codec,
    _padded_token_tensor,
    _toy_codes,
    tiny_easy_magpie_cfg,
)


pytestmark = pytest.mark.unit


class _RolloutHarness(EasyMagpieTTSModel):
    _is_user_turn = staticmethod(EasyMagpieTTSModelOnlinePO._is_user_turn)
    _is_assistant_turn = staticmethod(EasyMagpieTTSModelOnlinePO._is_assistant_turn)
    _reset_streaming_turn_state = staticmethod(EasyMagpieTTSModelOnlinePO._reset_streaming_turn_state)
    _encode_turn_text = EasyMagpieTTSModelOnlinePO._encode_turn_text
    _prepare_user_turn_embedding = EasyMagpieTTSModelOnlinePO._prepare_user_turn_embedding
    _run_streaming_assistant_turn = EasyMagpieTTSModelOnlinePO._run_streaming_assistant_turn
    _generate_multiturn_target_rollouts = EasyMagpieTTSModelOnlinePO._generate_multiturn_target_rollouts

    def _encode_turn_phonemes(self, ipa, num_repeats):
        del ipa
        return (
            torch.empty(num_repeats, 0, dtype=torch.long, device=self.device),
            torch.zeros(num_repeats, dtype=torch.long, device=self.device),
        )

    def _set_gt_turn_phonemes(self, state, phoneme_tokens, phoneme_lens):
        del state, phoneme_tokens, phoneme_lens

    def _set_gt_assistant_audio(self, state, audio, audio_lens):
        self.gt_prefix_audio_calls = getattr(self, "gt_prefix_audio_calls", 0) + 1
        return EasyMagpieTTSModelOnlinePO._set_gt_assistant_audio(self, state, audio, audio_lens)

    def _prefill_user_turn(self, state, user_audio, user_audio_lens, turn_text, turn_text_lens):
        self.user_prefill_calls = getattr(self, "user_prefill_calls", 0) + 1
        return EasyMagpieTTSModelOnlinePO._prefill_user_turn(
            self, state, user_audio, user_audio_lens, turn_text, turn_text_lens
        )


def _selection_stub(max_prefix_seconds=float("inf")):
    stub = SimpleNamespace(cfg={"multiturn_grpo_max_prefix_seconds": max_prefix_seconds})
    stub._is_user_turn = EasyMagpieTTSModelOnlinePO._is_user_turn
    stub._is_assistant_turn = EasyMagpieTTSModelOnlinePO._is_assistant_turn
    return stub


def test_select_multiturn_target_uses_last_valid_turn_for_eval():
    turns = [
        {"role": "user", "text": "u0", "start": 0.0},
        {"role": "assistant", "text": "a0", "start": 1.0},
        {"role": "user", "text": "u1", "start": 2.0},
        {"role": "assistant", "text": "a1", "start": 3.0},
    ]
    target = EasyMagpieTTSModelOnlinePO._select_multiturn_target(
        _selection_stub(), {"turn_metadata": [turns]}, mode="val"
    )

    assert target["metadata_idx"] == 3
    assert target["assistant_turn_idx"] == 1
    assert target["user_turn_idx"] == 2
    assert target["text"] == "a1"


def test_select_multiturn_target_respects_prefix_limit():
    turns = [
        {"role": "user", "text": "u0", "start": 0.0},
        {"role": "assistant", "text": "a0", "start": 1.0},
        {"role": "user", "text": "u1", "start": 2.0},
        {"role": "assistant", "text": "a1", "start": 3.0},
    ]
    target = EasyMagpieTTSModelOnlinePO._select_multiturn_target(
        _selection_stub(max_prefix_seconds=1.5), {"turn_metadata": [turns]}, mode="val"
    )
    assert target["metadata_idx"] == 1


def test_action_mask_limits_grpo_gradient_to_sampled_turn():
    stub = SimpleNamespace(
        loss_type="grpo",
        max_decoder_steps=8,
        cfg={"grpo_beta": 0.0},
        reference_free=True,
    )
    stub._get_per_token_logps = EasyMagpieTTSModelOnlinePO._get_per_token_logps.__get__(stub)
    stub._apply_teacher_forced_transform = EasyMagpieTTSModelOnlinePO._apply_teacher_forced_transform.__get__(stub)

    logits = torch.zeros(1, 3, 4, requires_grad=True)
    targets = torch.tensor([[[0, 1, 2]]], dtype=torch.long)
    loss, _, _ = EasyMagpieTTSModelOnlinePO._compute_action_po_components(
        stub,
        logits=logits,
        reference_logits=None,
        targets=targets,
        target_lens=torch.tensor([3]),
        vocab_size=4,
        advantages=torch.tensor([1.0]),
        group_validities=torch.tensor([1.0]),
        action_mask=torch.tensor([[False, True, False]]),
    )
    loss.backward()

    assert torch.count_nonzero(logits.grad[:, 0]).item() == 0
    assert torch.count_nonzero(logits.grad[:, 1]).item() > 0
    assert torch.count_nonzero(logits.grad[:, 2]).item() == 0


def test_build_multiturn_replay_keeps_gt_prefix_and_masks_generated_span():
    stub = SimpleNamespace(frame_stacking_factor=2)
    stub.repeat_items_in_batch = EasyMagpieTTSModelOnlinePO.repeat_items_in_batch.__get__(stub)
    user_prefix = torch.randn(1, 7, 3, requires_grad=True)
    stub._prepare_user_audio_embeddings_for_po = lambda batch: user_prefix

    gt_codes = torch.arange(24, dtype=torch.long).reshape(1, 2, 12)
    batch = {
        "audio_codes": gt_codes,
        "audio_codes_lens": torch.tensor([12]),
        "text": torch.arange(7, dtype=torch.long).reshape(1, 7),
        "text_lens": torch.tensor([7]),
        "phoneme_tokens": torch.arange(7, dtype=torch.long).reshape(1, 7),
        "phoneme_tokens_lens": torch.tensor([7]),
        "agent_mask": torch.ones(1, 7, dtype=torch.bool),
        "agent_mask_lens": torch.tensor([7]),
        "raw_texts": ["all turns"],
        "task": ["duplex"],
    }
    target_codes = torch.tensor(
        [
            [[101, 102, 103, 0, 0], [111, 112, 113, 0, 0]],
            [[201, 202, 203, 204, 205], [211, 212, 213, 214, 215]],
        ]
    )
    target_lens = torch.tensor([3, 5])
    target = {"text": "selected", "start_frame": 2, "end_frame": 5}

    replay_batch, replay_codes, replay_lens = EasyMagpieTTSModelOnlinePO._build_multiturn_po_replay(
        stub,
        batch=batch,
        target_turn=target,
        target_codes=target_codes,
        target_codes_lens=target_lens,
        num_generations=2,
    )

    torch.testing.assert_close(replay_codes[:, :, :4], gt_codes[:, :, :4].repeat(2, 1, 1))
    torch.testing.assert_close(replay_codes[0, :, 4:7], target_codes[0, :, :3])
    torch.testing.assert_close(replay_codes[1, :, 4:9], target_codes[1, :, :5])
    assert replay_codes.shape == (2, 2, 10)
    assert replay_lens.tolist() == [10, 10]
    assert replay_batch["raw_texts"] == ["selected", "selected"]
    assert replay_batch["user_audio_embedded"].shape == (2, 7, 3)
    assert not replay_batch["user_audio_embedded"].requires_grad
    assert replay_batch["audio_action_mask"].tolist() == [
        [False, False, True, True, False],
        [False, False, True, True, True],
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Streaming rollout harness requires CUDA.")
def test_gt_prefix_streaming_rollout_generates_only_selected_turn():
    cfg = tiny_easy_magpie_cfg(
        {
            "use_multiturn_dataset": True,
            "condition_on_user_speech": True,
            "use_user_speaking_token": True,
            "use_user_speaking_end_token": True,
        }
    )
    with _codec_restore_uses_random_initialized_audio_codec():
        model = _RolloutHarness(cfg)
    model = model.cuda()
    model.eval()
    model.audio_sampling_temperature = 0.7
    model.audio_sampling_topk = model.num_all_tokens_per_codebook
    model.phoneme_sampling_temperature = 0.7
    model.phoneme_sampling_topk = 2
    model.max_decoder_steps = 32

    steps_per_turn = 8
    samples_per_step = model.codec_model_samples_per_frame * model.frame_stacking_factor
    turn_samples = steps_per_turn * samples_per_step
    total_samples = turn_samples * 4
    frame_seconds = samples_per_step / model.sample_rate
    context_text, context_text_lens = _padded_token_tensor(model, ["[EN]"])

    turns = [
        {
            "role": "user",
            "text": "u0",
            "ipa": "",
            "start": 0.0,
            "end": steps_per_turn * frame_seconds,
            "start_frame": 0,
            "end_frame": steps_per_turn,
        },
        {
            "role": "assistant",
            "text": "a0",
            "ipa": "",
            "start": steps_per_turn * frame_seconds,
            "end": 2 * steps_per_turn * frame_seconds,
            "start_frame": steps_per_turn,
            "end_frame": 2 * steps_per_turn,
            "metadata_idx": 1,
        },
        {
            "role": "user",
            "text": "u1",
            "ipa": "",
            "start": 2 * steps_per_turn * frame_seconds,
            "end": 3 * steps_per_turn * frame_seconds,
            "start_frame": 2 * steps_per_turn,
            "end_frame": 3 * steps_per_turn,
        },
        {
            "role": "assistant",
            "text": "a1",
            "ipa": "",
            "start": 3 * steps_per_turn * frame_seconds,
            "end": 4 * steps_per_turn * frame_seconds,
            "start_frame": 3 * steps_per_turn,
            "end_frame": 4 * steps_per_turn,
            "metadata_idx": 3,
        },
    ]
    batch = {
        "audio": torch.zeros(1, total_samples, device=model.device),
        "context_audio_codes": _toy_codes(model, 1, 2).to(model.device),
        "context_audio_codes_lens": torch.tensor([2], device=model.device),
        "context_text_tokens": context_text.to(model.device),
        "context_text_tokens_lens": context_text_lens.to(model.device),
        "turn_metadata": [turns],
        "user_audio_turn_splitted": torch.zeros(2, turn_samples, device=model.device),
        "user_audio_turn_splitted_lens": torch.tensor([turn_samples, turn_samples], device=model.device),
    }

    target_turn = {
        **turns[-1],
        "assistant_turn_idx": 1,
        "user_turn_idx": 2,
    }
    output = model._generate_multiturn_target_rollouts(
        batch=batch,
        target_turn=target_turn,
        num_generations=2,
        use_local_transformer_for_inference=False,
    )

    assert model.gt_prefix_audio_calls == 1
    assert model.user_prefill_calls == 2
    assert target_turn["prefix_assistant_turns"] == 1
    assert target_turn["prefix_user_turns"] == 2
    assert output.audio.size(0) == 2
    assert output.audio_codes.size(0) == 2
    assert torch.all(output.audio_codes_len <= steps_per_turn * model.frame_stacking_factor)
