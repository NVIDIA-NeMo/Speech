# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import os
from math import ceil
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer
from torch.nn import Linear
from tqdm import tqdm

from nemo.collections.common.losses.cross_entropy import CrossEntropyLoss
from nemo.collections.nlp.models.token_classification.punctuation_capitalization_model import (
    PunctuationCapitalizationModel,
)
from nemo.collections.nlp.modules.common.transformer import TransformerDecoder
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.classes.mixins import adapter_mixins
from nemo.utils import logging

try:
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.submodules.conformer_modules import ConformerLayer

    ASR_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    ASR_AVAILABLE = False

__all__ = ['PunctuationCapitalizationLexicalAudioModel']


def update_model_config_to_support_adapter(model_cfg):
    with open_dict(model_cfg):
        adapter_metadata = adapter_mixins.get_registered_adapter(model_cfg.encoder._target_)
        if adapter_metadata is not None:
            model_cfg.encoder._target_ = adapter_metadata.adapter_class_path

    return model_cfg


class PunctuationCapitalizationLexicalAudioModel(PunctuationCapitalizationModel):
    """
        A model for restoring punctuation and capitalization in text using lexical and audio features.

        The model consists of a language model and two multilayer perceptrons (MLP) on top the fusion of LM and AM. The first
        MLP serves for punctuation prediction and the second is for capitalization prediction. You can use only BERT-like
        HuggingFace language models (model ``forward`` method accepts ``input_ids``, ``token_types_ids``,
        ``attention_mask`` arguments). See more about model config options :ref:`here<model-config-label>`.
        And any :class:`~nemo.collections.asr.models.EncDecCTCModel` which has encoder module which is used as an AM.

        For training and testing use dataset
        :class:`~nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset.BertPunctuationCapitalizationDataset` with parameter ``use_audio`` set to ``True``,
        for training on huge amounts of data which cannot be loaded into memory simultaneously use
        :class:`~nemo.collections.nlp.data.token_classification.punctuation_capitalization_tarred_dataset.BertPunctuationCapitalizationTarredDataset` with parameter ``use_audio`` set to ``True``.

        Args:
            cfg: a model configuration. It should follow dataclass
                :class:`~nemo.collections.nlp.models.token_classification.punctuation_capitalization_config.PunctuationCapitalizationLexicalAudioModelConfig`
                See an example of full config in
                `nemo/examples/nlp/token_classification/conf/punctuation_capitalization_lexical_audio_config.yaml
                <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/conf/punctuation_capitalization_lexical_audio_config.yaml>`_
            trainer: an instance of a PyTorch Lightning trainer
        """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None) -> None:
        super().__init__(cfg, trainer)
        if not ASR_AVAILABLE:
            raise ModuleNotFoundError(
                'Nemo ASR was not installed, see https://github.com/NVIDIA/NeMo#installation for installation instructions'
            )
        if os.path.exists(cfg.audio_encoder.pretrained_model):
            audio_cfg = nemo_asr.models.ASRModel.restore_from(cfg.audio_encoder.pretrained_model, return_config=True)
        else:
            audio_cfg = nemo_asr.models.ASRModel.from_pretrained(
                cfg.audio_encoder.pretrained_model, return_config=True
            )

        if cfg.audio_encoder.get('adapter', None):
            if cfg.audio_encoder.adapter.enable:
                audio_cfg = update_model_config_to_support_adapter(audio_cfg)

        if os.path.exists(cfg.audio_encoder.pretrained_model):
            self.audio_encoder = nemo_asr.models.ASRModel.restore_from(
                cfg.audio_encoder.pretrained_model, override_config_path=audio_cfg
            )
        else:
            self.audio_encoder = nemo_asr.models.ASRModel.from_pretrained(
                cfg.audio_encoder.pretrained_model, override_config_path=audio_cfg
            )

        if cfg.audio_encoder.adapter.get('enable', False):
            with open_dict(cfg):
                cfg.audio_encoder.adapter.config.in_features = self.audio_encoder.cfg.decoder.feat_in
            self.audio_encoder.add_adapter(name='audio_adapter', cfg=cfg.audio_encoder.adapter.config)
            self.audio_encoder.set_enabled_adapters(enabled=True)
            self.audio_encoder.freeze()
            self.audio_encoder.unfreeze_enabled_adapters()

        self.fusion = TransformerDecoder(
            num_layers=cfg.audio_encoder.fusion.num_layers,
            hidden_size=self.bert_model(**self.bert_model.input_example()[0]).size()[-1],
            inner_size=cfg.audio_encoder.fusion.inner_size,
            num_attention_heads=cfg.audio_encoder.fusion.num_attention_heads,
        )

        if hasattr(self.audio_encoder.cfg, 'decoder.feat_in'):
            self.audio_proj = Linear(
                self.audio_encoder.cfg.decoder.feat_in,
                self.bert_model(**self.bert_model.input_example()[0]).size()[-1],
            )
        else:
            self.audio_proj = Linear(
                self.audio_encoder.cfg.encoder.d_model,
                self.bert_model(**self.bert_model.input_example()[0]).size()[-1],
            )

        if cfg.audio_encoder.freeze.get('is_enabled', False):
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            for i in range(cfg.audio_encoder.freeze.get('num_layers')):
                self.audio_encoder.add_module(
                    f'conf_encoder_{i}',
                    ConformerLayer(
                        d_model=cfg.audio_encoder.freeze.get('d_model'), d_ff=cfg.audio_encoder.freeze.get('d_ff')
                    ),
                )

        if cfg.get('restore_lexical_encoder_from', None) and not self._is_model_being_restored():
            if os.path.exists(cfg.get('restore_lexical_encoder_from')):
                self.bert_model = (
                    PunctuationCapitalizationModel.restore_from(cfg.restore_lexical_encoder_from)
                    .to(self.device)
                    .bert_model
                )
            else:
                raise ValueError(f'Provided path {cfg.get("restore_lexical_encoder_from")} does not exists')

        if hasattr(self.audio_encoder, 'decoder'):
            del self.audio_encoder.decoder
        if hasattr(self.audio_encoder, '_wer'):
            del self.audio_encoder._wer
        if hasattr(self.audio_encoder, 'loss'):
            del self.audio_encoder.loss
        if hasattr(self.audio_encoder, 'decoder_losses'):
            del self.audio_encoder.decoder_losses

        if cfg.get('use_weighted_loss', False):
            punct_freq = torch.tensor(
                list(self.train_dataloader().dataset.punct_label_frequencies.values()), dtype=torch.float
            )
            punct_weight = 1 - (punct_freq - punct_freq.min()) / punct_freq.max()

            capit_freq = torch.tensor(
                list(self.train_dataloader().dataset.capit_label_frequencies.values()), dtype=torch.float
            )
            capit_weight = 1 - (capit_freq - capit_freq.min()) / capit_freq.max()

            self.loss_punct = CrossEntropyLoss(logits_ndim=3, weight=punct_weight)
            self.loss_capit = CrossEntropyLoss(logits_ndim=3, weight=capit_weight)
        else:
            self.loss_punct = self.loss
            self.loss_capit = self.loss

        self.set_max_audio_length(1024)

    def _make_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        punct_logits, capit_logits = self(
            input_ids=batch['input_ids'],
            token_type_ids=batch['segment_ids'],
            attention_mask=batch['input_mask'],
            features=batch['features'],
            features_length=batch['features_length'],
        )

        punct_loss = self.loss_punct(logits=punct_logits, labels=batch['punct_labels'], loss_mask=batch['loss_mask'])
        capit_loss = self.loss_capit(logits=capit_logits, labels=batch['capit_labels'], loss_mask=batch['loss_mask'])
        loss = self.agg_loss(loss_1=punct_loss, loss_2=capit_loss)
        return loss, punct_logits, capit_logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        features: torch.Tensor = None,
        features_length: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Executes a forward pass through the model. For more details see ``forward`` method of :class:`~nemo.collections.nlp.models.token_classification.punctuation_capitalization_config.PunctuationCapitalizationLexicalAudioModelConfig`
                and ``forward`` method of :class:'~nemo.collections.asr.models.EncDecCTCModel'

                Args:
                    input_ids (:obj:`torch.Tensor`): an integer torch tensor of shape ``[Batch, Time]``. Contains encoded
                        source tokens.
                    attention_mask (:obj:`torch.Tensor`): a boolean torch tensor of shape ``[Batch, Time]``. Contains an
                        attention mask for excluding paddings.
                    token_type_ids (:obj:`torch.Tensor`): an integer torch Tensor of shape ``[Batch, Time]``. Contains an index
                        of segment to which a token belongs. If ``token_type_ids`` is not ``None``, then it should be a zeros
                        tensor.
                    features (:obj:`torch.Tensor`): tensor that represents a batch of raw audio signals,
                        of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                        sample_rate number of floating point values.
                    features_length (:obj:`torch.Tensor`): Vector of length B, that contains the individual lengths of the audio
                        sequences.

                Returns:
                    :obj:`Tuple[torch.Tensor, torch.Tensor]`: a tuple containing

                        - ``punct_logits`` (:obj:`torch.Tensor`): a float torch tensor of shape
                          ``[Batch, Time, NumPunctuationLabels]`` containing punctuation logits
                        - ``capit_logits`` (:obj:`torch.Tensor`): a float torch tensor of shape
                          ``[Batch, Time, NumCapitalizationLabels]`` containing capitalization logits
                """
        self.update_max_seq_length(seq_length=features.size(1), device=features.device)
        lexical_hidden_states = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        if isinstance(lexical_hidden_states, tuple):
            lexical_hidden_states = lexical_hidden_states[0]

        processed_signal, processed_signal_length = self.audio_encoder.preprocessor(
            input_signal=features, length=features_length,
        )

        if self.audio_encoder.spec_augmentation is not None and self.training:
            processed_signal = self.audio_encoder.spec_augmentation(
                input_spec=processed_signal, length=processed_signal_length
            )

        audio_hidden_states, audio_hidden_states_length = self.audio_encoder.encoder(
            audio_signal=processed_signal, length=processed_signal_length
        )
        audio_hidden_states = audio_hidden_states.permute(0, 2, 1)
        audio_hidden_states = self.audio_proj(audio_hidden_states)

        fused = self.fusion(
            lexical_hidden_states,
            attention_mask,
            audio_hidden_states,
            self.make_pad_mask(audio_hidden_states.size(1), audio_hidden_states_length),
        )

        punct_logits = self.punct_classifier(hidden_states=fused)
        capit_logits = self.capit_classifier(hidden_states=fused)

        return punct_logits, capit_logits

    def make_pad_mask(self, max_audio_length, seq_lens):
        """Make masking for padding."""
        mask = self.seq_range[:max_audio_length].expand(seq_lens.size(0), -1) < seq_lens.unsqueeze(-1)
        return mask

    def update_max_seq_length(self, seq_length: int, device):
        if torch.distributed.is_initialized():
            global_max_len = torch.tensor([seq_length], dtype=torch.float32, device=device)

            # Update across all ranks in the distributed system
            torch.distributed.all_reduce(global_max_len, op=torch.distributed.ReduceOp.MAX)

            seq_length = global_max_len.int().item()

        if seq_length > self.max_audio_length:
            self.set_max_audio_length(seq_length)

    def set_max_audio_length(self, max_audio_length):
        """
        Sets maximum input length.
        Pre-calculates internal seq_range mask.
        """
        self.max_audio_length = max_audio_length
        device = next(self.parameters()).device
        seq_range = torch.arange(0, self.max_audio_length, device=device)
        if hasattr(self, 'seq_range'):
            self.seq_range = seq_range
        else:
            self.register_buffer('seq_range', seq_range, persistent=False)

    def add_punctuation_capitalization(
        self,
        queries: List[str],
        batch_size: int = None,
        max_seq_length: int = 64,
        step: int = 8,
        margin: int = 16,
        return_labels: bool = False,
        dataloader_kwargs: Dict[str, Any] = None,
        audio_queries: Optional[Union[List[bytes], List[str]]] = None,
        target_sr: Optional[int] = None,
    ) -> List[str]:
        """
               Adds punctuation and capitalization to the queries. Use this method for inference.

               Parameters ``max_seq_length``, ``step``, ``margin`` are for controlling the way queries are split into segments
               which are processed by the model. Parameter ``max_seq_length`` is a length of a segment after tokenization
               including special tokens [CLS] in the beginning and [SEP] in the end of a segment. Parameter ``step`` is a
               shift between consequent segments. Parameter ``margin`` is used to exclude negative effect of subtokens near
               borders of segments which have only one side context.

               If segments overlap, probabilities of overlapping predictions are multiplied and then the label with
               corresponding to the maximum probability is selected.

               Args:
                   queries (:obj:`List[str]`): lower cased text without punctuation.
                   batch_size (:obj:`List[str]`, `optional`): batch size to use during inference. If ``batch_size`` parameter
                       is not provided, then it will be equal to length of ``queries`` list.
                   max_seq_length (:obj:`int`, `optional`, defaults to :obj:`64`): maximum sequence length of a segment after
                       tokenization including :code:`[CLS]` and :code:`[SEP]` tokens.
                   step (:obj:`int`, `optional`, defaults to :obj:`8`): relative shift of consequent segments into which long
                       queries are split. Long queries are split into segments which can overlap. Parameter ``step`` controls
                       such overlapping. Imagine that queries are tokenized into characters, ``max_seq_length=5``, and
                       ``step=2``. In such case, query ``"hello"`` is tokenized into segments
                       ``[['[CLS]', 'h', 'e', 'l', '[SEP]'], ['[CLS]', 'l', 'l', 'o', '[SEP]']]``.
                   margin (:obj:`int`, `optional`, defaults to :obj:`16`): number of subtokens in the beginning and the end of
                       segments which are not used for prediction computation. The first segment does not have left margin and
                       the last segment does not have right margin. For example, if an input sequence is tokenized into
                       characters, ``max_seq_length=5``, ``step=1``, and ``margin=1``, then query ``"hello"`` will be
                       tokenized into segments ``[['[CLS]', 'h', 'e', 'l', '[SEP]'], ['[CLS]', 'e', 'l', 'l', '[SEP]'],
                       ['[CLS]', 'l', 'l', 'o', '[SEP]']]``. These segments are passed to the model. Before final predictions
                       computation, margins are removed. In the next list, subtokens which logits are not used for final
                       predictions computation are marked with asterisk: ``[['[CLS]'*, 'h', 'e', 'l'*, '[SEP]'*],
                       ['[CLS]'*, 'e'*, 'l', 'l'*, '[SEP]'*], ['[CLS]'*, 'l'*, 'l', 'o', '[SEP]'*]]``.
                   return_labels (:obj:`bool`, `optional`, defaults to :obj:`False`): whether to return labels in NeMo format
                       (see :ref:`nlp/punctuation_and_capitalization/NeMo Data Format`) instead of queries with restored
                       punctuation and capitalization.
                   dataloader_kwargs (:obj:`Dict[str, Any]`, `optional`): an optional dictionary with parameters of PyTorch
                       data loader. May include keys: ``'num_workers'``, ``'pin_memory'``, ``'worker_init_fn'``,
                       ``'prefetch_factor'``, ``'persistent_workers'``.
                   audio_queries (:obj:`List[str]`, `optional`): paths to audio files.
                   target_sr (:obj:`int`, `optional`): target sample rate for audios.
               Returns:
                   :obj:`List[str]`: a list of queries with restored capitalization and punctuation if
                   ``return_labels=False``, else a list of punctuation and capitalization labels strings for all queries
               """

        if len(queries) == 0:
            return []
        if batch_size is None:
            batch_size = len(queries)
            logging.info(f'Using batch size {batch_size} for inference')
        result: List[str] = []
        mode = self.training
        try:
            self.eval()
            infer_datalayer = self._setup_infer_dataloader(
                queries, batch_size, max_seq_length, step, margin, dataloader_kwargs, audio_queries, target_sr
            )
            # Predicted labels for queries. List of labels for every query
            all_punct_preds: List[List[int]] = [[] for _ in queries]
            all_capit_preds: List[List[int]] = [[] for _ in queries]
            # Accumulated probabilities (or product of probabilities acquired from different segments) of punctuation
            # and capitalization. Probabilities for words in a query are extracted using `subtokens_mask`. Probabilities
            # for newly processed words are appended to the accumulated probabilities. If probabilities for a word are
            # already present in `acc_probs`, old probabilities are replaced with a product of old probabilities
            # and probabilities acquired from new segment. Segments are processed in an order they appear in an
            # input query. When all segments with a word are processed, a label with the highest probability
            # (or product of probabilities) is chosen and appended to an appropriate list in `all_preds`. After adding
            # prediction to `all_preds`, probabilities for a word are removed from `acc_probs`.
            acc_punct_probs: List[Optional[np.ndarray]] = [None for _ in queries]
            acc_capit_probs: List[Optional[np.ndarray]] = [None for _ in queries]
            d = self.device
            for batch_i, batch in tqdm(
                enumerate(infer_datalayer), total=ceil(len(infer_datalayer.dataset) / batch_size), unit="batch"
            ):
                (
                    inp_ids,
                    inp_type_ids,
                    inp_mask,
                    subtokens_mask,
                    start_word_ids,
                    query_ids,
                    is_first,
                    is_last,
                    features,
                    features_length,
                ) = batch
                punct_logits, capit_logits = self.forward(
                    input_ids=inp_ids.to(d),
                    token_type_ids=inp_type_ids.to(d),
                    attention_mask=inp_mask.to(d),
                    features=features.to(d),
                    features_length=features_length.to(d),
                )
                _res = self._transform_logit_to_prob_and_remove_margins_and_extract_word_probs(
                    punct_logits, capit_logits, subtokens_mask, start_word_ids, margin, is_first, is_last
                )
                punct_probs, capit_probs, start_word_ids = _res
                for i, (q_i, start_word_id, bpp_i, bcp_i) in enumerate(
                    zip(query_ids, start_word_ids, punct_probs, capit_probs)
                ):
                    for all_preds, acc_probs, b_probs_i in [
                        (all_punct_preds, acc_punct_probs, bpp_i),
                        (all_capit_preds, acc_capit_probs, bcp_i),
                    ]:
                        if acc_probs[q_i] is None:
                            acc_probs[q_i] = b_probs_i
                        else:
                            all_preds[q_i], acc_probs[q_i] = self._move_acc_probs_to_token_preds(
                                all_preds[q_i], acc_probs[q_i], start_word_id - len(all_preds[q_i]),
                            )
                            acc_probs[q_i] = self._update_accumulated_probabilities(acc_probs[q_i], b_probs_i)
            for all_preds, acc_probs in [(all_punct_preds, acc_punct_probs), (all_capit_preds, acc_capit_probs)]:
                for q_i, (pred, prob) in enumerate(zip(all_preds, acc_probs)):
                    if prob is not None:
                        all_preds[q_i], acc_probs[q_i] = self._move_acc_probs_to_token_preds(pred, prob, len(prob))
            for i, query in enumerate(queries):
                result.append(
                    self._get_labels(all_punct_preds[i], all_capit_preds[i])
                    if return_labels
                    else self._apply_punct_capit_predictions(query, all_punct_preds[i], all_capit_preds[i])
                )
        finally:
            # set mode back to its original value
            self.train(mode=mode)
        return result

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []
