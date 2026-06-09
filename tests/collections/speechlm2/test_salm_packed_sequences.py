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
import ast
from pathlib import Path

import pytest
import torch

from nemo.collections.speechlm2.parts.packed_sequences import pack_audio_into_text_embeds, prepare_packed_llm_inputs

PAD = 0
AUDIO = 100
REPO_ROOT = Path(__file__).parents[3]


def test_packed_sequences_does_not_import_speechlm2_models_globally():
    source = REPO_ROOT / "nemo/collections/speechlm2/parts/packed_sequences.py"
    tree = ast.parse(source.read_text())
    bad_imports = []
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("nemo.collections.speechlm2.models")
        ):
            bad_imports.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("nemo.collections.speechlm2.models"):
                    bad_imports.append((node.lineno, alias.name))
    assert bad_imports == []


def _basic_batch():
    """Mirrors `test_audio_placeholders.py::test_replace_placeholders`.

    Two utterances, three audio replacements of length 4, 3, 2.
    """
    input_ids = torch.tensor(
        [
            [7, AUDIO, 1, 2, AUDIO, 1],
            [PAD, PAD, 3, AUDIO, 4, 5],  # left-padded
        ]
    )
    loss_mask = torch.tensor(
        [
            [False, False, False, False, False, True],
            [False, False, False, False, True, True],
        ]
    )
    embeds = torch.ones(2, 6, 2)
    embeds[1, :2] = 0  # zero left-pad slots
    replacements = [
        torch.full((4, 2), fill_value=2.0),
        torch.full((3, 2), fill_value=3.0),
        torch.full((2, 2), fill_value=4.0),
    ]
    target_ids = input_ids.where(loss_mask, -100)
    return input_ids, embeds, target_ids, replacements


def test_basic_pack_shapes_and_cu_seqlens():
    input_ids, embeds, target_ids, replacements = _basic_batch()
    out = pack_audio_into_text_embeds(
        input_ids=input_ids,
        embeds=embeds,
        target_ids=target_ids,
        replacements=replacements,
        padding_id=PAD,
        placeholder_id=AUDIO,
    )
    # Real per-utterance flat lengths:
    #   utt0: 1 text + 4 audio + 2 text + 3 audio + 1 text = 11
    #   utt1: 1 text + 2 audio + 2 text                    = 5  (left-pad of 2 stripped)
    assert out["seq_lens"].squeeze(-1).tolist() == [11, 5]
    # cp_size=1, tp_size=1 ⇒ no rounding
    assert out["seq_lens_padded"].squeeze(-1).tolist() == [11, 5]
    # cu_seqlens = [0] + cumsum(seq_lens_padded)
    assert out["cu_seqlens"].dtype == torch.int32
    assert out["cu_seqlens"].tolist() == [0, 11, 16]
    assert out["max_seqlen"].item() == 11
    assert out["qkv_format"] == "thd"

    T_total = 11 + 5
    assert out["inputs_embeds"].shape == (T_total, 2)
    assert out["labels"].shape == (T_total,)
    assert out["position_ids"].shape == (T_total,)


def test_position_ids_reset_per_utt():
    input_ids, embeds, target_ids, replacements = _basic_batch()
    out = pack_audio_into_text_embeds(
        input_ids=input_ids,
        embeds=embeds,
        target_ids=target_ids,
        replacements=replacements,
        padding_id=PAD,
        placeholder_id=AUDIO,
    )
    pos = out["position_ids"]
    cu = out["cu_seqlens"].tolist()
    for start, end in zip(cu[:-1], cu[1:]):
        assert pos[start].item() == 0
        assert torch.equal(pos[start:end], torch.arange(end - start, dtype=torch.long))


def test_audio_frame_labels_are_ignored():
    """Audio-frame slots must be -100 in `labels` regardless of loss_mask."""
    input_ids, embeds, target_ids, replacements = _basic_batch()
    out = pack_audio_into_text_embeds(
        input_ids=input_ids,
        embeds=embeds,
        target_ids=target_ids,
        replacements=replacements,
        padding_id=PAD,
        placeholder_id=AUDIO,
    )
    labels = out["labels"]
    cu = out["cu_seqlens"].tolist()
    # Utterance 0 layout: [t, a, a, a, a, t, t, a, a, a, t]
    #                   pos 0  1  2  3  4  5  6  7  8  9  10
    # Audio slots before shift: 1..4 and 7..9. After per-utt next-token shift,
    # the *previous* slot's label becomes the audio target → also ignored once
    # the original slot's label was -100. Verify all original audio slots map
    # to -100 in the shifted output (audio at t means lab_shift[t-1] gets
    # what was at t, which is -100 from the audio fill).
    utt0 = labels[cu[0] : cu[1]]
    # Shifted: original audio at positions 1-4 → label[0..3] should be -100;
    # audio at 7-9 → label[6..8] should be -100; last slot (10) is -100.
    assert (utt0[0:4] == -100).all()
    assert (utt0[6:9] == -100).all()
    assert utt0[-1].item() == -100


def test_labels_shifted_per_utt():
    """`labels[t]` should equal the original `target_ids` at position t+1
    *within the utterance*, with the last slot set to -100."""
    input_ids, embeds, target_ids, replacements = _basic_batch()
    out = pack_audio_into_text_embeds(
        input_ids=input_ids,
        embeds=embeds,
        target_ids=target_ids,
        replacements=replacements,
        padding_id=PAD,
        placeholder_id=AUDIO,
    )
    labels = out["labels"]
    cu = out["cu_seqlens"].tolist()
    # Utt 0 last slot is the trailing text "1" with loss_mask=True (target=1),
    # which after per-utt shift becomes the label of position 9 (the last
    # audio frame). Since the shift puts orig[t+1] at slot t, the second-to-
    # last slot of utt0 holds target_ids of the trailing "1".
    utt0 = labels[cu[0] : cu[1]]
    assert utt0[-2].item() == 1  # trailing text token "1" with loss_mask=True
    assert utt0[-1].item() == -100  # last position of every utterance is -100

    # Utt 1 last two text tokens (4, 5) had loss_mask=True. After per-utt
    # shift, label[L-3] = 4, label[L-2] = 5, label[L-1] = -100.
    utt1 = labels[cu[1] : cu[2]]
    assert utt1[-3].item() == 4
    assert utt1[-2].item() == 5
    assert utt1[-1].item() == -100


def test_no_audio_utterance():
    """Utterance without any audio placeholders still packs correctly."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    loss_mask = torch.tensor([[False, False, True, True, True]])
    embeds = torch.full((1, 5, 2), 1.0)
    target_ids = input_ids.where(loss_mask, -100)
    out = pack_audio_into_text_embeds(
        input_ids=input_ids,
        embeds=embeds,
        target_ids=target_ids,
        replacements=[],
        padding_id=PAD,
        placeholder_id=AUDIO,
    )
    assert out["seq_lens"].squeeze(-1).tolist() == [5]
    assert out["seq_lens_padded"].squeeze(-1).tolist() == [5]
    assert out["cu_seqlens"].tolist() == [0, 5]
    labels = out["labels"]
    # Original target_ids = [-100, -100, 3, 4, 5]; after per-utt shift:
    # [-100, 3, 4, 5, -100]
    assert labels.tolist() == [-100, 3, 4, 5, -100]


def test_b_one():
    """Single-utterance batch produces valid `cu_seqlens=[0, L]`."""
    input_ids = torch.tensor([[1, AUDIO, 2]])
    loss_mask = torch.tensor([[False, False, True]])
    embeds = torch.full((1, 3, 2), 1.0)
    target_ids = input_ids.where(loss_mask, -100)
    replacements = [torch.full((3, 2), 7.0)]
    out = pack_audio_into_text_embeds(
        input_ids=input_ids,
        embeds=embeds,
        target_ids=target_ids,
        replacements=replacements,
        padding_id=PAD,
        placeholder_id=AUDIO,
    )
    assert out["seq_lens"].squeeze(-1).tolist() == [5]  # 1 + 3 audio + 1
    assert out["cu_seqlens"].tolist() == [0, 5]
    assert out["inputs_embeds"].shape == (5, 2)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_cp_divisibility(cp_size):
    """Each per-utt padded length is a multiple of 2*cp_size."""
    input_ids, embeds, target_ids, replacements = _basic_batch()
    out = pack_audio_into_text_embeds(
        input_ids=input_ids,
        embeds=embeds,
        target_ids=target_ids,
        replacements=replacements,
        padding_id=PAD,
        placeholder_id=AUDIO,
        cp_size=cp_size,
    )
    mult = 2 * cp_size
    for L in out["seq_lens_padded"].squeeze(-1).tolist():
        assert L % mult == 0


def test_tp_divisibility():
    """`T_total` is a multiple of tp_size (last utterance gets the bump)."""
    input_ids, embeds, target_ids, replacements = _basic_batch()
    # real_lens = [11, 5], total = 16; pick tp_size=3 to force a bump.
    out = pack_audio_into_text_embeds(
        input_ids=input_ids,
        embeds=embeds,
        target_ids=target_ids,
        replacements=replacements,
        padding_id=PAD,
        placeholder_id=AUDIO,
        tp_size=3,
    )
    T_total = out["seq_lens_padded"].sum().item()
    assert T_total % 3 == 0
    # First utterance untouched (only the last gets the TP bump).
    assert out["seq_lens_padded"].squeeze(-1).tolist()[0] == 11
    # Last utterance bumped from 5 → 7 (next multiple of 3 after 16 is 18).
    assert out["seq_lens_padded"].squeeze(-1).tolist()[1] == 7


def test_tp_and_cp_combined():
    input_ids, embeds, target_ids, replacements = _basic_batch()
    out = pack_audio_into_text_embeds(
        input_ids=input_ids,
        embeds=embeds,
        target_ids=target_ids,
        replacements=replacements,
        padding_id=PAD,
        placeholder_id=AUDIO,
        cp_size=2,
        tp_size=8,
    )
    padded = out["seq_lens_padded"].squeeze(-1).tolist()
    real = out["seq_lens"].squeeze(-1).tolist()
    # Every padded length ≥ real and divisible by 4.
    for r, p in zip(real, padded):
        assert p >= r
        assert p % 4 == 0
    # Total divisible by tp_size.
    assert sum(padded) % 8 == 0


def test_tp_bump_preserves_cp_alignment_when_tp_is_not_cp_multiple():
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    loss_mask = torch.ones_like(input_ids, dtype=torch.bool)
    embeds = torch.ones(2, 5, 2)
    target_ids = input_ids.where(loss_mask, -100)
    out = pack_audio_into_text_embeds(
        input_ids=input_ids,
        embeds=embeds,
        target_ids=target_ids,
        replacements=[],
        padding_id=PAD,
        placeholder_id=AUDIO,
        cp_size=2,
        tp_size=6,
    )
    padded = out["seq_lens_padded"].squeeze(-1).tolist()
    assert padded == [8, 16]
    for L in padded:
        assert L % 4 == 0
    assert sum(padded) % 6 == 0


def test_cu_seqlens_matches_padded_cumsum():
    input_ids, embeds, target_ids, replacements = _basic_batch()
    out = pack_audio_into_text_embeds(
        input_ids=input_ids,
        embeds=embeds,
        target_ids=target_ids,
        replacements=replacements,
        padding_id=PAD,
        placeholder_id=AUDIO,
        cp_size=2,
        tp_size=8,
    )
    expected = [0]
    for L in out["seq_lens_padded"].squeeze(-1).tolist():
        expected.append(expected[-1] + L)
    assert out["cu_seqlens"].tolist() == expected
    assert out["max_seqlen"].item() == max(out["seq_lens_padded"].squeeze(-1).tolist())


def test_loss_mask_propagates_to_minus_100():
    """Positions where loss_mask=False end up as -100 in the shifted labels."""
    input_ids = torch.tensor([[1, 2, 3, 4]])
    loss_mask = torch.tensor([[False, False, False, False]])  # nothing supervised
    embeds = torch.full((1, 4, 2), 1.0)
    target_ids = input_ids.where(loss_mask, -100)
    out = pack_audio_into_text_embeds(
        input_ids=input_ids,
        embeds=embeds,
        target_ids=target_ids,
        replacements=[],
        padding_id=PAD,
        placeholder_id=AUDIO,
    )
    assert (out["labels"] == -100).all()


def test_prepare_packed_llm_inputs_attention_kwargs_reach_te_preprocessor():
    """End-to-end contract: ``prepare_packed_llm_inputs`` → Automodel's TE
    attention preprocessor must yield ``max_seqlen_q``/``max_seqlen_kv`` and
    ``cu_seqlens_q``/``cu_seqlens_kv`` populated for the THD path.

    Regression for: ``prepare_packed_llm_inputs`` previously emitted
    ``max_seqlen_q``/``max_seqlen_kv`` (plural), but the preprocessor only
    inspects the singular ``max_seqlen`` key in the ``cu_seqlens`` branch
    (``Automodel/.../attention/utils.py``). The plural keys were silently
    dropped, TE fell back to a degenerate varlen path, and step-1 backward
    produced NaN gradients that poisoned every subsequent step.
    """
    automodel_attn = pytest.importorskip(
        "nemo_automodel.components.attention.utils",
        reason="Automodel attention preprocessor required for the contract test.",
    )
    input_ids, embeds, target_ids, replacements = _basic_batch()
    out = prepare_packed_llm_inputs(
        input_ids=input_ids,
        text_embs=embeds,
        audio_embs=replacements,
        target_ids=target_ids,
        padding_id=PAD,
        placeholder_id=AUDIO,
        device_mesh=None,  # CP=1, TP=1 path
    )
    llm_kwargs = out["llm_kwargs"]
    assert out["attention_mask"] is None
    assert llm_kwargs["qkv_format"] == "thd"
    assert "cu_seqlens" in llm_kwargs
    assert "max_seqlen" in llm_kwargs, (
        "Automodel's preprocessor only checks the singular `max_seqlen` key in the "
        "cu_seqlens THD branch; pre-split `max_seqlen_q`/`max_seqlen_kv` would be dropped."
    )
    assert "cu_seqlens_padded" not in llm_kwargs, (
        "Standard Automodel pipeline emits only `cu_seqlens`, never both `cu_seqlens` "
        "and `cu_seqlens_padded`. Passing both activates the `pad_between_seqs=True` "
        "branch in Automodel/.../attention/utils.py, routing TE down a different path."
    )

    # Run the LLM kwargs through the preprocessor exactly as the attention
    # layer does: 4D BSHD-shaped Q/K/V plus attention_mask=None plus the
    # llm_kwargs splatted in. ``input_embeds`` is now 2D ``[T, H]`` per the
    # canonical Automodel THD shape contract.
    B, T, H = 1, int(out["input_embeds"].shape[0]), 2
    nh, hd = 2, 4
    q = torch.zeros(B, T, nh, hd)
    k = torch.zeros(B, T, nh, hd)
    v = torch.zeros(B, T, nh, hd)
    _, _, _, te_attn_kwargs = automodel_attn.preprocess_args_and_kwargs_for_attn(
        q, k, v, attention_mask=None, attn_impl="te", **llm_kwargs
    )
    assert te_attn_kwargs.get("qkv_format") == "thd"
    assert te_attn_kwargs.get("attn_mask_type") == "padding_causal"
    assert "cu_seqlens_q" in te_attn_kwargs and "cu_seqlens_kv" in te_attn_kwargs
    assert "max_seqlen_q" in te_attn_kwargs and "max_seqlen_kv" in te_attn_kwargs, (
        "TE DotProductAttention requires max_seqlen_q/kv for qkv_format='thd'; "
        "missing keys cause silent degenerate-path fallback and NaN gradients."
    )
    assert te_attn_kwargs["max_seqlen_q"] == llm_kwargs["max_seqlen"]
    assert te_attn_kwargs["max_seqlen_kv"] == llm_kwargs["max_seqlen"]


def _bshd_supervised_pairs(input_ids, embeds, target_ids, replacements):
    """Run the BSHD path (``replace_placeholders_and_build_targets`` + the
    ``[:-1] / [1:]`` next-token shift used in
    ``SALMAutomodel.prepare_inputs``) and return the ordered list of
    supervised ``(input_embedding, target_token_id)`` pairs.
    """
    from nemo.collections.speechlm2.models.salm import replace_placeholders_and_build_targets

    bshd_embs, bshd_targets, _ = replace_placeholders_and_build_targets(
        input_ids=input_ids,
        embeds=embeds,
        padding_id=PAD,
        placeholder_id=AUDIO,
        replacements=[r.clone() for r in replacements],
        target_ids=target_ids,
    )
    bshd_embs = bshd_embs[:, :-1]
    bshd_targets = bshd_targets[:, 1:]

    pairs = []
    B, T = bshd_targets.shape
    for b in range(B):
        for t in range(T):
            tgt = bshd_targets[b, t].item()
            if tgt != -100:
                pairs.append((bshd_embs[b, t].clone(), tgt))
    return pairs


def _thd_supervised_pairs(input_ids, embeds, target_ids, replacements):
    """Run the THD path (``pack_audio_into_text_embeds`` with the per-utt
    next-token shift) and return the ordered list of supervised
    ``(input_embedding, target_token_id)`` pairs.
    """
    out = pack_audio_into_text_embeds(
        input_ids=input_ids,
        embeds=embeds,
        target_ids=target_ids,
        replacements=[r.clone() for r in replacements],
        padding_id=PAD,
        placeholder_id=AUDIO,
    )
    embs = out["inputs_embeds"]  # [T_total, H]
    labels = out["labels"]  # [T_total]
    pairs = []
    for t in range(labels.shape[0]):
        tgt = labels[t].item()
        if tgt != -100:
            pairs.append((embs[t].clone(), tgt))
    return pairs


def _assert_pairs_equivalent(bshd_pairs, thd_pairs, *, atol=1e-6):
    assert len(bshd_pairs) == len(thd_pairs), (
        f"BSHD has {len(bshd_pairs)} supervised pairs, THD has {len(thd_pairs)}. "
        f"Both paths must yield the same ordered set of (input, target) pairs."
    )
    for i, ((e_b, t_b), (e_t, t_t)) in enumerate(zip(bshd_pairs, thd_pairs)):
        assert t_b == t_t, (
            f"Pair {i}: target_id mismatch — BSHD={t_b}, THD={t_t}. "
            f"Per-utt next-token shift must align with global [:-1]/[1:] shift."
        )
        assert torch.allclose(e_b, e_t, atol=atol), (
            f"Pair {i} (target={t_b}): input embedding mismatch between BSHD and THD. " f"BSHD={e_b}, THD={e_t}"
        )


def test_thd_and_bshd_supervised_pairs_match_basic():
    """First-principles invariant: BSHD and THD are different *layouts* of the
    same data. The set of supervised ``(input_embedding, target_token_id)``
    pairs that contribute to the cross-entropy loss must be identical between
    paths. Any divergence in this set means the THD path is feeding the model
    something the BSHD path is not (or vice-versa).
    """
    input_ids, embeds, target_ids, replacements = _basic_batch()
    bshd_pairs = _bshd_supervised_pairs(input_ids, embeds, target_ids, replacements)
    thd_pairs = _thd_supervised_pairs(input_ids, embeds, target_ids, replacements)
    _assert_pairs_equivalent(bshd_pairs, thd_pairs)


def test_thd_and_bshd_supervised_pairs_match_no_audio_utt():
    """Pure-text utterance (no audio_locator)."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    loss_mask = torch.tensor([[False, False, True, True, True]])
    embeds = torch.randn(1, 5, 4)
    target_ids = input_ids.where(loss_mask, -100)
    bshd_pairs = _bshd_supervised_pairs(input_ids, embeds, target_ids, replacements=[])
    thd_pairs = _thd_supervised_pairs(input_ids, embeds, target_ids, replacements=[])
    _assert_pairs_equivalent(bshd_pairs, thd_pairs)


def test_thd_and_bshd_supervised_pairs_match_left_padded():
    """Left-padded utterances must yield the same supervised pairs."""
    input_ids = torch.tensor(
        [
            [PAD, PAD, PAD, 1, 2, AUDIO, 3, 4],
            [PAD, 5, 6, AUDIO, 7, AUDIO, 8, 9],
        ]
    )
    loss_mask = torch.tensor(
        [
            [False, False, False, False, False, True, True, True],
            [False, False, False, True, True, True, True, True],
        ]
    )
    embeds = torch.randn(2, 8, 4)
    embeds[0, :3] = 0  # zero left-pad slots
    embeds[1, :1] = 0
    target_ids = input_ids.where(loss_mask, -100)
    replacements = [
        torch.randn(3, 4),  # utt0 audio
        torch.randn(2, 4),  # utt1 first audio
        torch.randn(4, 4),  # utt1 second audio
    ]
    bshd_pairs = _bshd_supervised_pairs(input_ids, embeds, target_ids, replacements)
    thd_pairs = _thd_supervised_pairs(input_ids, embeds, target_ids, replacements)
    _assert_pairs_equivalent(bshd_pairs, thd_pairs)


def test_thd_and_bshd_supervised_pairs_match_b1():
    """Single-utterance batch."""
    input_ids = torch.tensor([[1, AUDIO, 2, 3, AUDIO, 4]])
    loss_mask = torch.tensor([[False, False, True, True, True, True]])
    embeds = torch.randn(1, 6, 4)
    target_ids = input_ids.where(loss_mask, -100)
    replacements = [torch.randn(2, 4), torch.randn(5, 4)]
    bshd_pairs = _bshd_supervised_pairs(input_ids, embeds, target_ids, replacements)
    thd_pairs = _thd_supervised_pairs(input_ids, embeds, target_ids, replacements)
    _assert_pairs_equivalent(bshd_pairs, thd_pairs)


def test_padded_slots_have_zero_embed_and_ignored_label():
    """Inter-utt padding (added for cp_size rounding) gets zero embedding,
    -100 label, and contiguous position_ids."""
    input_ids, embeds, target_ids, replacements = _basic_batch()
    out = pack_audio_into_text_embeds(
        input_ids=input_ids,
        embeds=embeds,
        target_ids=target_ids,
        replacements=replacements,
        padding_id=PAD,
        placeholder_id=AUDIO,
        cp_size=2,  # rounds 11→12 and 6→6
    )
    embs = out["inputs_embeds"]
    labels = out["labels"]
    pos = out["position_ids"]
    # Utt 0 had real_len=11 and padded to 12 (next multiple of 4). Slot 11 is
    # the pad slot.
    assert torch.equal(embs[11], torch.zeros(2))
    assert labels[11].item() == -100
    assert pos[11].item() == 11  # contiguous with the utt's real positions
