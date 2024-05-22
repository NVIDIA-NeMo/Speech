from typing import Any

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.llms import CompletionResponse, CompletionResponseGen, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy


class NeMoGPTLLM(CustomLLM):
    context_window: int = 2048
    num_output: int = 256
    model_name: str = "NeMo LLM"
    dummy_response: str = "My response"

    length_params: LengthParam = {
        "max_length": 500,
        "min_length": 0,
    }

    sampling_params: SamplingParam = {
        "use_greedy": True,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "add_BOS": True,
        "all_probs": False,
        "compute_logprob": False,
        "end_strings": ["<|endoftext|>"],
    }

    _model: Any = PrivateAttr()
    _model_cfg: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(
        self,
        model_path: str = None,
        cfg: Any = None,
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

        tensor_model_parallel_size = 1
        pipeline_model_parallel_size = 1

        # trainer required for restoring model parallel models
        trainer = Trainer(strategy=NLPDDPStrategy(), **trainer_config)
        assert (
            trainer_config["devices"] * trainer_config['num_nodes']
            == tensor_model_parallel_size * pipeline_model_parallel_size
        ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

        # setup/override model config
        model_cfg = MegatronGPTModel.restore_from(restore_path=model_path, trainer=trainer, return_config=True)
        model_cfg.micro_batch_size = 1
        model_cfg.global_batch_size = cfg.trainer.devices
        self._model_cfg = model_cfg
        print("self._model_cfg: ", self._model_cfg)

        # restore model
        model = MegatronGPTModel.restore_from(
            restore_path=model_path, trainer=trainer, override_config_path=model_cfg, strict=True
        )
        model.freeze()
        self._model = model
        super().__init__(**kwargs)

        # update LLM metadata
        self.context_window = self._model_cfg.encoder_seq_length

        # update inference params
        length_params: LengthParam = {
            "max_length": cfg.generating.inference.tokens_to_generate,
            "min_length": cfg.generating.inference.min_tokens_to_generate,
        }

        sampling_params: SamplingParam = {
            "use_greedy": cfg.generating.inference.greedy,
            "temperature": cfg.generating.inference.temperature,
            "top_k": cfg.generating.inference.top_k,
            "top_p": cfg.generating.inference.top_p,
            "repetition_penalty": cfg.generating.inference.repetition_penalty,
            "add_BOS": cfg.generating.inference.add_BOS,
            "all_probs": cfg.generating.inference.all_probs,
            "compute_logprob": cfg.generating.inference.compute_logprob,
            "end_strings": cfg.generating.inference.end_strings,
        }

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        llm_response = self._model.generate(
            inputs=[prompt], length_params=self.length_params, sampling_params=self.sampling_params
        )
        text_response = llm_response['sentences'][0]

        return CompletionResponse(text=text_response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        llm_response = self._model.generate(
            inputs=[prompt], length_params=self.length_params, sampling_params=self.sampling_params
        )
        text_response = llm_response['sentences'][0]

        response = ""
        for token in text_response:
            response += token
            yield CompletionResponse(text=response, delta=token)
