# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from nemo.collections.tts.models.easy_magpietts_preference_optimization import OnlinePreferenceOptimizationMixin
from nemo.collections.tts.models.nemotron_tts.v2 import base as v2_base
from nemo.collections.tts.models.nemotron_tts.v2.model import NemotronTTSModel
from nemo.collections.tts.models.nemotron_tts.v2.post_training import NemotronTTSModelOnlinePO
from tests.collections.tts.modules.nemotron_tts.v2.test_acoustic_codes_predictor import (
    AUDIO_EOS_ID,
    CODEBOOK_SIZE,
    D_MODEL,
    MASK_TOKEN_ID,
    NUM_CODES,
    NUM_SPECIAL_TOKENS,
    _backbone_config,
    _make_predictor,
)

pytestmark = pytest.mark.unit


class _TinyPolicy(NemotronTTSModelOnlinePO):
    def __init__(self):
        nn.Module.__init__(self)
        self._cfg = OmegaConf.create(
            {'inference_temperature': 0.7, 'inference_topk': 80, 'inference_phoneme_sampling_method': 'sample'}
        )
        self._device = torch.device('cpu')
        self._dtype = torch.float32
        self.decoder = nn.Linear(D_MODEL, D_MODEL)
        self.acoustic_codes_predictor = _make_predictor()
        self.num_audio_codebooks = NUM_CODES
        self.num_all_tokens_per_codebook = CODEBOOK_SIZE + NUM_SPECIAL_TOKENS
        self.codebook_size = CODEBOOK_SIZE
        self.audio_eos_id = AUDIO_EOS_ID
        self.mask_token_id = MASK_TOKEN_ID
        self.frame_stacking_factor = 1
        self.phoneme_stacking_factor = 2
        self.phoneme_vocab_size = 7
        self.phoneme_tokenizer = SimpleNamespace(bos_token_id=5, eos_token_id=6)
        self.tokenizer = None
        self.phoneme_final_proj = nn.Linear(D_MODEL, 2 * self.phoneme_vocab_size)
        self.audio_out_projection = nn.Linear(D_MODEL, NUM_CODES * self.num_all_tokens_per_codebook)
        self.text_aligner = nn.Linear(D_MODEL, D_MODEL)
        self._eval_asr_model = nn.Linear(D_MODEL, D_MODEL)
        self.default_inference_mode = 'test_mode'
        self.mode_name_to_mode = {'test_mode': object()}
        self.reference_free = True
        self.loss_type = 'grpo'
        self.max_decoder_steps = 4
        self.aux_phoneme_loss_weight = 0.0
        self.aux_phoneme_po_loss_weight = 0.2
        self.entropy_coeff = 0.0
        self.inputs = torch.randn(2, 4, D_MODEL)

    def process_batch(self, audio_codes, audio_codes_lens, **kwargs):
        assert kwargs['return_policy_logits'] and kwargs['audio_codes_are_stacked']
        hidden = self.decoder(self.inputs)
        _, logits = self.acoustic_codes_predictor.compute_loss(
            hidden, audio_codes.transpose(1, 2), audio_codes_lens, return_logits=True
        )
        phoneme_logits = self.phoneme_final_proj(hidden)
        phoneme_targets = phoneme_logits.reshape(2, 4, 2, self.phoneme_vocab_size).argmax(-1).transpose(1, 2)
        return SimpleNamespace(
            acoustic_codes_predictor_logits=logits,
            logits=self.audio_out_projection(hidden),
            audio_codes_target=audio_codes,
            audio_codes_lens_target=audio_codes_lens,
            phoneme_logits=phoneme_logits,
            phoneme_tokens_target=phoneme_targets,
            phoneme_tokens_lens_target=audio_codes_lens,
            phoneme_loss=None,
        )


@pytest.mark.parametrize('text_input_mode', ['streaming', 'full'])
def test_post_training_constructor_preserves_training_target_and_requires_streaming(monkeypatch, text_input_mode):
    def initialize_policy(self, cfg, trainer=None):
        nn.Module.__init__(self)
        self._cfg = cfg
        self.acoustic_codes_predictor = nn.Linear(2, 2)
        self.final_proj = nn.Linear(2, 2)
        self.text_aligner = nn.Linear(2, 2)
        self.default_inference_mode = 'test_mode'
        self.mode_name_to_mode = {'test_mode': SimpleNamespace(text_input_mode=text_input_mode)}

    monkeypatch.setattr(OnlinePreferenceOptimizationMixin, '__init__', initialize_policy)
    original_target = f'{NemotronTTSModelOnlinePO.__module__}.{NemotronTTSModelOnlinePO.__name__}'
    cfg = OmegaConf.create({'target': original_target, 'run_val_inference': True})
    if text_input_mode != 'streaming':
        with pytest.raises(ValueError, match='requires a streaming inference mode'):
            NemotronTTSModelOnlinePO(cfg)
        return

    model = NemotronTTSModelOnlinePO(cfg)
    assert model.cfg.target == original_target
    assert model.cfg.run_val_inference is False
    assert cfg.target == original_target and cfg.run_val_inference is True
    assert model.acoustic_codes_predictor.sampling_in_fp32 is True
    assert all(not parameter.requires_grad for parameter in model.final_proj.parameters())
    assert all(not parameter.requires_grad for parameter in model.text_aligner.parameters())


@pytest.mark.parametrize('save_fails', [False, True])
def test_inference_export_restores_live_training_config(monkeypatch, save_fails):
    model = _TinyPolicy()
    model._cfg.target = f'{NemotronTTSModelOnlinePO.__module__}.{NemotronTTSModelOnlinePO.__name__}'
    model._cfg.run_val_inference = True
    model._cfg.use_utmos = True
    training_cfg = model.cfg
    original = OmegaConf.to_container(training_cfg, resolve=False)
    sentinel = object()

    def save_inference_archive(self, save_path):
        assert save_path == 'policy.nemo'
        assert self.cfg is not training_cfg
        assert self.cfg.target == f'{NemotronTTSModel.__module__}.{NemotronTTSModel.__name__}'
        assert self.cfg.run_val_inference is False and self.cfg.use_utmos is False
        if save_fails:
            raise RuntimeError('archive write failed')
        return sentinel

    monkeypatch.setattr(NemotronTTSModel, 'save_to', save_inference_archive)
    if save_fails:
        with pytest.raises(RuntimeError, match='archive write failed'):
            model.save_to('policy.nemo')
    else:
        assert model.save_to('policy.nemo') is sentinel

    assert model.cfg is training_cfg
    assert OmegaConf.to_container(model.cfg, resolve=False) == original


def test_frozen_reference_is_excluded_from_export_and_preserved_on_restore(monkeypatch):
    model = _TinyPolicy()
    model._reference_model = nn.Linear(D_MODEL, D_MODEL).requires_grad_(False)
    state = model.state_dict()
    assert not any('_reference_model' in key for key in state)
    restored = _TinyPolicy()
    restored._reference_model = nn.Linear(D_MODEL, D_MODEL).requires_grad_(False)
    original_reference = {key: value.clone() for key, value in restored._reference_model.state_dict().items()}
    monkeypatch.setattr(v2_base, 'check_text_embedding_matches_tokenizer', lambda *args, **kwargs: None)

    restored.load_state_dict(state)

    for key, value in restored._reference_model.state_dict().items():
        torch.testing.assert_close(value, original_reference[key])
    torch.testing.assert_close(restored.decoder.weight, model.decoder.weight)


def test_singleturn_prompt_loader_preserves_multiturn_architecture(monkeypatch):
    model = _TinyPolicy()
    model._cfg.use_multiturn_dataset = True
    dataset_cfg = OmegaConf.create({'load_normalized_text_percent': 1.0})
    sentinel = object()

    def get_prompt_loader(self, cfg, mode='train', use_multiturn_dataset=None):
        assert cfg is dataset_cfg and mode == 'train'
        assert use_multiturn_dataset is False
        assert self.cfg.use_multiturn_dataset is True
        return sentinel

    monkeypatch.setattr(NemotronTTSModel, 'get_lhotse_dataloader', get_prompt_loader)
    assert model.get_lhotse_dataloader(dataset_cfg, mode='train') is sentinel
    assert model.cfg.use_multiturn_dataset is True


@pytest.mark.parametrize('has_reference', [False, True])
def test_rollout_cer_scores_normalized_reference_without_changing_raw_prompt(monkeypatch, has_reference):
    model = _TinyPolicy()
    model.output_sample_rate = 24000
    model.max_decoder_steps = 10
    model.use_utmos = False
    model.best_cer_threshold = model.worst_cer_threshold = 1.0
    model.scale_rewards = True
    batch = {'raw_texts': ['1000']}
    if has_reference:
        batch['cer_reference_texts'] = ['one thousand']

    def infer_batch(batch, **kwargs):
        assert batch['raw_texts'] == ['1000', '1000']
        return SimpleNamespace(
            predicted_audio=torch.zeros(2, 2400),
            predicted_audio_lens=torch.tensor([2400, 2400]),
            predicted_codes=torch.zeros(2, NUM_CODES, 5, dtype=torch.long),
            predicted_codes_lens=torch.tensor([5, 5]),
        )

    monkeypatch.setattr(model, 'infer_batch', infer_batch)
    monkeypatch.setattr(model, '_prepare_rollout_batch', lambda *args: None)
    monkeypatch.setattr(model, '_save_waveforms_to_paths', lambda **kwargs: ['audio-0.wav', 'audio-1.wav'])
    monkeypatch.setattr(model, '_compute_pred_transcripts', lambda *args: ['one thousand', 'one thousand'])
    monkeypatch.setattr(model, '_compute_speaker_embeddings_parallel', lambda *args: (None, None))
    monkeypatch.setattr(model, '_print_group_cer_wer_table', lambda **kwargs: None)

    result = model.generate_and_reward(batch, num_generations_per_item=2)

    assert len(result['metrics']) == 2
    for metric in result['metrics']:
        assert metric['gt_transcript'] == ('one thousand' if has_reference else '1000')
        if has_reference:
            assert metric['cer_gt'] == 0.0
        else:
            assert metric['cer_gt'] > 0.0
    assert result['batch_repeated']['raw_texts'] == ['1000', '1000']
    assert batch['raw_texts'] == ['1000']


def test_grpo_updates_sampled_policy_and_round_trips_weights(monkeypatch):
    torch.manual_seed(7)
    model = _TinyPolicy()
    with torch.no_grad():
        model.acoustic_codes_predictor.blocks[0].codebook_projection.bias[CODEBOOK_SIZE] = 20.0
        codes = model.acoustic_codes_predictor.predict_codes(
            model.decoder(model.inputs), temperature=0.7, topk=80
        ).transpose(1, 2)
    lengths = torch.tensor([4, 2])
    batch = {
        'text': torch.ones(2, 2, dtype=torch.long),
        'text_lens': torch.tensor([2, 2]),
        'context_text_tokens': torch.ones(2, 1, dtype=torch.long),
        'context_text_tokens_lens': torch.ones(2, dtype=torch.long),
        'context_audio_codes': torch.zeros(2, NUM_CODES, 1, dtype=torch.long),
        'context_audio_codes_lens': torch.ones(2, dtype=torch.long),
        '_po_audio_codes_stacked': codes,
        '_po_audio_codes_lens': lengths,
    }
    output = model._run_easy_process_batch(model, batch, codes, lengths, mode='val')
    logits = output.local_transformer_logits.reshape(2, 4, NUM_CODES, -1)
    assert model.training
    assert codes[0, 0, 0] == AUDIO_EOS_ID
    assert logits[..., AUDIO_EOS_ID].isfinite().all()
    assert (logits[..., CODEBOOK_SIZE] == torch.finfo(torch.float32).min).all()
    losses = model._compute_po_losses_from_outputs(
        output, None, torch.tensor([1.0, -0.5]), torch.ones(2), rollout_phoneme_input_type='pred'
    )
    assert all(torch.isfinite(value).all() for value in losses.values() if isinstance(value, torch.Tensor))
    losses['loss'].backward()
    assert model.decoder.weight.grad.abs().sum() > 0
    assert model.phoneme_final_proj.weight.grad.abs().sum() > 0
    assert all(
        block.codebook_projection.weight.grad.abs().sum() > 0 for block in model.acoustic_codes_predictor.blocks
    )
    assert model.audio_out_projection.weight.grad is None
    assert model.text_aligner.weight.grad is None
    model.setup_optimizer_param_groups()
    optimized = {id(parameter) for group in model._optimizer_param_groups for parameter in group['params']}
    assert id(model.phoneme_final_proj.weight) in optimized
    assert id(model.audio_out_projection.weight) not in optimized
    assert id(model.text_aligner.weight) not in optimized
    before = model.decoder.weight.detach().clone()
    torch.optim.SGD(model._optimizer_param_groups, lr=0.01).step()
    assert not torch.equal(model.decoder.weight, before)

    state = model.state_dict()
    assert not any('_eval_asr_model' in key for key in state)
    restored = _TinyPolicy()
    monkeypatch.setattr(v2_base, 'check_text_embedding_matches_tokenizer', lambda *args, **kwargs: None)
    restored.load_state_dict(state)
    for name, value in state.items():
        torch.testing.assert_close(restored.state_dict()[name], value)


def test_bf16_policy_trains_with_fp32_residuals_and_frozen_reference():
    model = _TinyPolicy()
    config = _backbone_config()
    config.residual_in_fp32 = True
    model.acoustic_codes_predictor = _make_predictor(backbone_config=config)
    reference = _TinyPolicy()
    reference.acoustic_codes_predictor = _make_predictor(backbone_config=config)
    reference.load_state_dict(model.state_dict())
    reference.inputs = model.inputs.clone()
    reference.requires_grad_(False)
    model._reference_model = reference
    model.reference_free = False
    model._cfg.grpo_beta = 0.1
    model.to(dtype=torch.bfloat16)
    model.inputs = model.inputs.bfloat16()
    model.acoustic_codes_predictor.sampling_in_fp32 = True

    assert model.decoder.weight.dtype == model.phoneme_final_proj.weight.dtype == torch.bfloat16
    assert all(block.config.residual_in_fp32 for block in model.acoustic_codes_predictor.blocks)
    assert all(parameter.dtype == torch.float32 for parameter in reference.parameters())
    assert model.text_aligner.weight.dtype == model._eval_asr_model.weight.dtype == torch.float32
    with torch.no_grad():
        codes = model.acoustic_codes_predictor.predict_codes(
            model.decoder(model.inputs), temperature=0.7, topk=80
        ).transpose(1, 2)
    lengths = torch.tensor([4, 2])
    batch = {
        'text': torch.ones(2, 2, dtype=torch.long),
        'text_lens': torch.tensor([2, 2]),
        'context_text_tokens': torch.ones(2, 1, dtype=torch.long),
        'context_text_tokens_lens': torch.ones(2, dtype=torch.long),
        'context_audio_codes': torch.zeros(2, NUM_CODES, 1, dtype=torch.long),
        'context_audio_codes_lens': torch.ones(2, dtype=torch.long),
        '_po_audio_codes_stacked': codes,
        '_po_audio_codes_lens': lengths,
    }
    output = model._run_easy_process_batch(model, batch, codes, lengths, mode='val')
    with torch.no_grad():
        reference_output = model._run_easy_process_batch(reference, batch, codes, lengths, mode='val')
    probabilities = output.local_transformer_logits.reshape(2, 4, NUM_CODES, -1).softmax(-1)
    assert probabilities.dtype == torch.float32 and torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, 4, NUM_CODES))
    losses = model._compute_po_losses_from_outputs(
        output, reference_output, torch.tensor([1.0, -0.5]), torch.ones(2), rollout_phoneme_input_type='pred'
    )

    for name in ('loss', 'po_loss', 'kl_loss', 'entropy'):
        assert losses[name].dtype == torch.float32 and torch.isfinite(losses[name])
    losses['loss'].backward()
    assert model.decoder.weight.grad.dtype == torch.bfloat16
    assert model.decoder.weight.grad.abs().sum() > 0
    assert model.phoneme_final_proj.weight.grad.abs().sum() > 0
    assert all(
        block.codebook_projection.weight.grad.abs().sum() > 0 for block in model.acoustic_codes_predictor.blocks
    )
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in reference.parameters())
    model.setup_optimizer_param_groups()
    before = model.decoder.weight.detach().clone()
    torch.optim.SGD(model._optimizer_param_groups, lr=0.05).step()
    assert not torch.equal(model.decoder.weight, before)
    assert all(parameter.dtype == torch.float32 for parameter in reference.parameters())


def test_rollout_keeps_terminal_actions_and_generated_phoneme_trajectory(monkeypatch):
    model = _TinyPolicy()
    model._capture_policy_rollout = True
    model.frame_stacking_factor = 2
    codes = torch.arange(2 * NUM_CODES * 6).reshape(2, NUM_CODES, 6) % CODEBOOK_SIZE
    codes[0, 0, 4] = AUDIO_EOS_ID
    phonemes = torch.tensor([[[1, 2, 6], [3, 4, 6]], [[2, 6, 0], [4, 6, 0]]])
    state = SimpleNamespace(
        all_predictions=list(codes.split(2, dim=-1)),
        audio_steps=torch.tensor([2, 1]),
        audio_prediction_start_idx=torch.tensor([2, 0]),
        all_phoneme_predictions=list(phonemes.unbind(-1)),
        phoneme_prediction_start_idx=torch.tensor([1, 0]),
        phoneme_prediction_end_idx=torch.tensor([3, 2]),
    )
    sentinel = object()
    monkeypatch.setattr(NemotronTTSModel, 'streaming_finalize', lambda *args, **kwargs: sentinel)
    assert model.streaming_finalize(state) is sentinel
    batch = {'phoneme_tokens': torch.zeros(2, 3, dtype=torch.long)}
    model._prepare_rollout_batch(None, batch, phoneme_input_type='pred')

    expected = codes[0, :, 2:6].reshape(NUM_CODES, 2, 2).permute(0, 2, 1).reshape(2 * NUM_CODES, 2)
    torch.testing.assert_close(batch['_po_audio_codes_stacked'][0], expected)
    assert batch['_po_audio_codes_stacked'][0, 0, 1] == AUDIO_EOS_ID
    assert batch['_po_audio_codes_stacked'][0, 2, 1] != AUDIO_EOS_ID
    assert batch['_po_audio_codes_lens'].tolist() == [2, 1]
    assert (batch['_po_audio_codes_stacked'][1, :, 1] == model.mask_token_id).all()
    assert batch['phoneme_tokens'].tolist() == [[5, 2, 4, 6, 6], [5, 2, 4, 6, 6]]
    assert batch['phoneme_tokens_lens'].tolist() == [5, 5]
    ground_truth = torch.zeros(2, 3, dtype=torch.long)
    batch = {'phoneme_tokens': ground_truth}
    model._prepare_rollout_batch(None, batch, phoneme_input_type='gt')
    assert batch['phoneme_tokens'] is ground_truth


def test_sampling_logits_use_fp32_temperature_and_finite_topk_mask():
    logits = torch.tensor([[1.0, 4.0, 3.0, -1.0]], dtype=torch.bfloat16)
    actual = NemotronTTSModelOnlinePO._sampling_logits(logits, temperature=0.7, topk=2)

    assert actual.dtype == torch.float32 and actual.isfinite().all()
    torch.testing.assert_close(actual[:, 1:3], torch.tensor([[4.0, 3.0]]) / 0.7)
    assert (actual[:, [0, 3]] == torch.finfo(torch.float32).min).all()
    assert actual.log_softmax(-1).exp()[0, [0, 3]].sum() == 0


def test_phoneme_rollouts_sample_in_fp32_and_bound_topk(monkeypatch):
    model = _TinyPolicy()
    captured = []
    multinomial = torch.multinomial

    def record_probabilities(probabilities, *args, **kwargs):
        captured.append(probabilities.detach().clone())
        return multinomial(probabilities, *args, **kwargs)

    monkeypatch.setattr(torch, 'multinomial', record_probabilities)
    tokens = model.sample_codes_from_logits_phoneme(torch.randn(2, 14, dtype=torch.bfloat16), topk=80)

    assert tokens.shape == (2, 2) and ((tokens >= 0) & (tokens < model.phoneme_vocab_size)).all()
    assert len(captured) == model.phoneme_stacking_factor
    assert all(probabilities.dtype == torch.float32 and probabilities.isfinite().all() for probabilities in captured)


def test_sampled_audio_targets_keep_terminal_frame_and_delay(monkeypatch):
    model = _TinyPolicy()
    model._cfg.embedding_dim = D_MODEL
    model.audio_bos_id = CODEBOOK_SIZE
    model.frame_stacking_factor = 2
    model.acoustic_codes_predictor.num_audio_codebooks = 2 * NUM_CODES
    codes = torch.randint(0, CODEBOOK_SIZE, (2, 2 * NUM_CODES, 3))
    codes[0, 0, -1] = AUDIO_EOS_ID
    lengths = torch.tensor([3, 2])
    delays = torch.tensor([3, 1])
    embedded_inputs = []

    def embed_codes(inputs):
        embedded_inputs.append(inputs.clone())
        return inputs.float().mean(1).unsqueeze(-1).expand(-1, -1, D_MODEL)

    monkeypatch.setattr(model, 'embed_audio_tokens', embed_codes)
    embeddings, channel_lens, targets, target_lens, _ = model.prepare_audio_channel_embeddings(
        codes, lengths, delay=delays, audio_codes_are_stacked=True
    )
    torch.testing.assert_close(targets, codes)
    torch.testing.assert_close(target_lens, lengths)
    assert channel_lens.tolist() == [6, 3]
    assert (embedded_inputs[0][:, :, 0] == model.audio_bos_id).all()
    torch.testing.assert_close(embedded_inputs[0][:, :, 1:], codes[:, :, :-1])
    assert embeddings[0, :3].count_nonzero() == 0
    assert (embeddings[0, 3] == model.audio_bos_id).all()
    timeline, _ = model._acoustic_codes_predictor_targets(7, targets, target_lens, delays)
    torch.testing.assert_close(timeline[0, 3:6], codes[0].T)
    torch.testing.assert_close(timeline[1, 1:3], codes[1, :, :2].T)
    assert (timeline[0, :3] == model.mask_token_id).all()


@pytest.mark.parametrize('policy_scoring', [False, True])
def test_real_process_batch_returns_delayed_acoustic_predictions(monkeypatch, policy_scoring):
    model = _TinyPolicy()
    model._cfg.embedding_dim = D_MODEL
    model._cfg.use_multiturn_dataset = False
    model.phoneme_tokenizer = None
    model.use_attention_alignment = False
    model.feature_masking = None
    model.pad_id = 0
    model.audio_bos_id = CODEBOOK_SIZE
    model._codec_converter = None
    model.parallel_codebook_loss_scale = 1.0
    model.acoustic_codes_predictor_loss_scale = 1.0
    model.label_smoothing = 0.0
    model.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
    model.audio_out_projection = nn.Identity()
    model.final_proj = nn.Linear(D_MODEL, NUM_CODES * model.num_all_tokens_per_codebook)
    mode = SimpleNamespace(
        text_input_mode='streaming', streaming_speech_delay=2, streaming_phonemes_delay=1, name='test_mode'
    )
    codes = torch.randint(0, CODEBOOK_SIZE, (2, NUM_CODES, 4))
    lengths = torch.tensor([4, 2])
    if policy_scoring:
        codes[0, 0, -1] = AUDIO_EOS_ID

    def prepare_context(**kwargs):
        return (
            torch.zeros(2, 2, D_MODEL),
            torch.tensor([2, 1]),
            kwargs['context_audio_codes'],
            kwargs['context_audio_codes_lens'],
        )

    monkeypatch.setattr(model, 'prepare_context_tensors', prepare_context)
    monkeypatch.setattr(
        model, 'embed_text_tokens', lambda text, **kwargs: text.float().unsqueeze(-1).expand(-1, -1, D_MODEL)
    )
    monkeypatch.setattr(model, 'embed_audio_tokens', lambda codes: model.acoustic_codes_predictor.embed_codes(codes))
    monkeypatch.setattr(
        model,
        'forward',
        lambda inputs_embeds, **kwargs: SimpleNamespace(last_hidden_state=model.decoder(inputs_embeds)),
    )
    output = NemotronTTSModel.process_batch(
        model,
        text=torch.ones(2, 2, dtype=torch.long),
        text_lens=torch.tensor([2, 2]),
        context_text_tokens=torch.ones(2, 1, dtype=torch.long),
        context_text_tokens_lens=torch.tensor([1, 1]),
        audio_codes=codes,
        audio_codes_lens=lengths,
        context_audio_codes=torch.zeros(2, NUM_CODES, 2, dtype=torch.long),
        context_audio_codes_lens=torch.tensor([2, 1]),
        mode='val',
        training_mode=mode,
        return_policy_logits=policy_scoring,
        audio_codes_are_stacked=policy_scoring,
    )

    torch.testing.assert_close(output.audio_codes_lens_target, lengths if policy_scoring else lengths + 1)
    assert torch.isfinite(output.loss)
    if policy_scoring:
        torch.testing.assert_close(output.audio_codes_target, codes)
        assert output.logits is None
        assert output.acoustic_codes_predictor_logits.shape == (2, 4, NUM_CODES, CODEBOOK_SIZE + 1)
    else:
        assert output.logits.shape == (2, 5, NUM_CODES * model.num_all_tokens_per_codebook)
        assert output.acoustic_codes_predictor_logits is None
        for item, length in enumerate(lengths.tolist()):
            torch.testing.assert_close(output.audio_codes_target[item, :, :length], codes[item, :, :length])
            assert (output.audio_codes_target[item, :, length] == AUDIO_EOS_ID).all()
    output.loss.backward()
    assert model.decoder.weight.grad.abs().sum() > 0
    assert all(
        block.codebook_projection.weight.grad.abs().sum() > 0 for block in model.acoustic_codes_predictor.blocks
    )
