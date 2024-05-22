from typing import Any, List

import torch
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from omegaconf import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.information_retrieval.megatron_bert_embedding_model import MegatronBertEmbeddingModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy


class NeMoBertEmbeddings(BaseEmbedding):
    _model: MegatronBertEmbeddingModel = PrivateAttr()
    _model_cfg: DictConfig = PrivateAttr()

    def __init__(
        self,
        model_path: str = None,
        cfg: Any = None,
        embed_batch_size: int = 16,
        **kwargs: Any,
    ) -> None:

        # set up trainer
        trainer_config = {
            "devices": cfg.trainer.devices,
            "num_nodes": 1,
            "accelerator": "gpu",
            "logger": False,
            "precision": cfg.trainer.precision,
        }
        trainer = Trainer(strategy=NLPDDPStrategy(), **trainer_config)

        # setup/override model config
        model_cfg = MegatronBertEmbeddingModel.restore_from(
            restore_path=model_path, trainer=trainer, return_config=True
        )
        model_cfg.micro_batch_size = 1
        model_cfg.global_batch_size = cfg.trainer.devices
        self._model_cfg = model_cfg
        print("self._model_cfg: ", self._model_cfg)

        # restore model
        model = MegatronBertEmbeddingModel.restore_from(
            restore_path=model_path, trainer=trainer, override_config_path=model_cfg, strict=True
        )
        model.freeze()
        self._model = model

        super().__init__(
            embed_batch_size=embed_batch_size,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "nemo_bert_embeddings"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _construct_forward_input(self, texts: List[str]):
        # this method construct model's forward input arguments from texts, following the constructing step in nemo/collections/nlp/data/information_retrieval/bert_embedding_dataset.py

        # retrieve arguments from model_config
        max_seq_length = self._model_cfg.encoder_seq_length

        # tokenize text
        input_ids = [self._model.tokenizer.text_to_ids(text) for text in texts]

        # truncate input_ids
        input_ids = [item[: (max_seq_length - 1)] for item in input_ids]

        # add bos and eos
        input_ids = [([self._model.tokenizer.bos_id] + item + [self._model.tokenizer.eos_id]) for item in input_ids]

        # pad input_ids
        def _ceil_to_nearest(n, m):
            return (n + m - 1) // m * m

        lengths = [len(item) for item in input_ids]
        max_length = min(max_seq_length, _ceil_to_nearest(max(lengths), 16))
        assert max_length <= max_seq_length
        input_ids = [item + [self._model.tokenizer.pad_id] * (max_length - len(item)) for item in input_ids]
        input_ids = torch.LongTensor(input_ids)

        # construct attention_mask
        def _create_attention_mask2(max_length, item_lengh):
            """Create `attention_mask`.
            Args:
                input_ids: A 1D tensor that holds the indices of tokens.
            """
            # seq_length = len(input_ids)
            # `attention_mask` has the shape of [1, seq_length, seq_length]
            attention_mask = torch.zeros(max_length)
            attention_mask[:item_lengh] = 1
            return attention_mask

        attention_mask = [_create_attention_mask2(max_length, len) for len in lengths]
        attention_mask = torch.stack(attention_mask)

        # construct token_type_ids
        token_type_ids = torch.zeros_like(input_ids)

        processed_batch = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
        }

        return processed_batch

    def _get_query_embedding(self, query: str) -> List[float]:
        constructed_forward_input = self._construct_forward_input([query])
        for key in constructed_forward_input.keys():
            constructed_forward_input[key] = constructed_forward_input[key].to(self._model.device)

        embeddings = self._model.forward(**constructed_forward_input)
        embeddings = embeddings.transpose(0, 1)  # reshape tensor shape [hidden_dim, bs] to [bs, hidden_dim]

        return embeddings[0].tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        constructed_forward_input = self._construct_forward_input([text])
        for key in constructed_forward_input.keys():
            constructed_forward_input[key] = constructed_forward_input[key].to(self._model.device)

        embeddings = self._model.forward(**constructed_forward_input)
        embeddings = embeddings.transpose(0, 1)  # reshape tensor shape [hidden_dim, bs] to [bs, hidden_dim]

        return embeddings[0].tolist()

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        constructed_forward_input = self._construct_forward_input(texts)
        for key in constructed_forward_input.keys():
            constructed_forward_input[key] = constructed_forward_input[key].to(self._model.device)

        embeddings = self._model.forward(**constructed_forward_input)
        embeddings = embeddings.transpose(0, 1)  # reshape tensor shape [hidden_dim, bs] to [bs, hidden_dim]

        return embeddings.tolist()
