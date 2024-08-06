# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import tempfile
from typing import Any, Callable, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from transformers import CLIPImageProcessor, SiglipImageProcessor

from nemo.collections.multimodal.data.clip.augmentations.augmentations import image_transform
from nemo.collections.multimodal.data.neva.neva_dataset import process_image
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPFSDPStrategy, NLPSaveRestoreConnector
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.utils import AppState, logging
from nemo.utils.model_utils import inject_model_parallel_rank

try:
    import decord
except Exception:
    logging.warning("The package `decord` was not installed in this environment.")

try:
    from megatron.core import dist_checkpointing

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def randn_like(x, generator=None):
    return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)


def extend_instance(obj, mixin):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(
        base_cls_name, (mixin, base_cls), {}
    )  # mixin needs to go first for our forward() logic to work


def getattr_recursive(obj, att):
    """
    Return nested attribute of obj
    Example: getattr_recursive(obj, 'a.b.c') is equivalent to obj.a.b.c
    """
    if att == "":
        return obj
    i = att.find(".")
    if i < 0:
        return getattr(obj, att)
    else:
        return getattr_recursive(getattr(obj, att[:i]), att[i + 1 :])


def setattr_recursive(obj, att, val):
    """
    Set nested attribute of obj
    Example: setattr_recursive(obj, 'a.b.c', val) is equivalent to obj.a.b.c = val
    """
    if "." in att:
        obj = getattr_recursive(obj, ".".join(att.split(".")[:-1]))
    setattr(obj, att.split(".")[-1], val)


def apply_with_stopping_condition(module, apply_fn, apply_condition=None, stopping_condition=None, **other_args):
    if stopping_condition(module):
        return
    if apply_condition(module):
        apply_fn(module, **other_args)
    for child in module.children():
        apply_with_stopping_condition(
            child, apply_fn, apply_condition=apply_condition, stopping_condition=stopping_condition, **other_args
        )


def load_nemo_model_weights(nemo_path, sharded_state_dict=None):
    """
    Shared method to load model weights from a given nemo_path.
    """
    if torch.cuda.is_available():
        map_location = torch.device('cuda')
    else:
        map_location = torch.device('cpu')

    save_restore_connector = NLPSaveRestoreConnector()
    cwd = os.getcwd()
    app_state = AppState()
    is_dist_ckpt = False

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            if os.path.isfile(nemo_path):
                save_restore_connector._unpack_nemo_file(path2file=nemo_path, out_folder=tmpdir)
            else:
                tmpdir = nemo_path
            os.chdir(tmpdir)
            if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
                model_weights = save_restore_connector._inject_model_parallel_rank_for_ckpt(
                    tmpdir, save_restore_connector.model_weights_ckpt
                )
            else:
                model_weights = os.path.join(tmpdir, save_restore_connector.model_weights_ckpt)

            state_dict = save_restore_connector._load_state_dict_from_disk(model_weights, map_location=map_location)

            # distributed checkpointing
            if state_dict is None and sharded_state_dict is not None:

                is_dist_ckpt = True
                checkpoint = dict(state_dict=sharded_state_dict)

                tmp_model_weights_ckpt = os.path.join(tmpdir, save_restore_connector.model_weights_ckpt)
                tmp_model_weights_dir = os.path.splitext(tmp_model_weights_ckpt)[0]
                assert os.path.isdir(tmp_model_weights_dir), f'Expected {tmp_model_weights_dir} to be a directory.'
                checkpoint = dist_checkpointing.load(
                    sharded_state_dict=checkpoint,
                    checkpoint_dir=tmp_model_weights_dir,
                )
                state_dict = checkpoint["state_dict"]

        finally:
            os.chdir(cwd)

    return state_dict, is_dist_ckpt


def setup_trainer_and_models_for_inference(
    model_provider: Any,
    cfg: DictConfig,
    model_cfg_modifier: Callable,
):
    """
    Set up a trainer and NeMo model for inference.

    Args:
        model_provider (Any): An object that provides the NeMo model.
        cfg (DictConfig): The configuration dictionary, containing the
            necessary settings for the trainer and the models.
        model_cfg_modifier (Callable): A function that modifies the model
            configuration for inference.

    Returns:
        Tuple[Trainer, Any]: A tuple containing the trainer and the model.
    """

    # Check if we need to use the TorchElasticEnvironment plugin for the trainer.
    plugins = []
    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    # Use the NLPDDPStrategy for the distributed data parallel strategy.
    # We don't use DDP for async grad allreduce and don't find unused parameters.
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,
        find_unused_parameters=False,
    )

    # Set up the trainer with the specified plugins and strategy.
    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer)

    # Create the NLPSaveRestoreConnector object for model saving and restoring.
    save_restore_connector = NLPSaveRestoreConnector()

    models = []
    for single_model_cfg in cfg.models:
        if not single_model_cfg.restore_from_path:
            continue
        if single_model_cfg.restore_from_path.endswith(".nemo"):
            # Set the model_extracted_dir attribute if the restore path is a directory.
            if os.path.isdir(single_model_cfg.restore_from_path):
                save_restore_connector.model_extracted_dir = single_model_cfg.restore_from_path

            # Restore the model configuration from the specified path and modify it for inference.
            model_cfg = model_provider.restore_from(
                restore_path=single_model_cfg.restore_from_path,
                trainer=trainer,
                save_restore_connector=save_restore_connector,
                return_config=True,
            )
            with open_dict(model_cfg):
                model_cfg_modifier(model_cfg)  # modify the configuration for inference

            # Restore the model from the specified path and configuration, and set it up for inference.
            model = model_provider.restore_from(
                restore_path=single_model_cfg.restore_from_path,
                trainer=trainer,
                override_config_path=model_cfg,
                save_restore_connector=save_restore_connector,
                strict=True,
            )
            models.append(model)

        elif single_model_cfg.restore_from_path.endswith(".ckpt"):
            logging.warning(
                "Loading from .ckpt checkpoint for inference is experimental! It doesn't support models with model parallelism!"
            )

            model = model_provider.load_from_checkpoint(
                single_model_cfg.restore_from_path,
                hparams_file=cfg.model.get("hparams_file"),
                trainer=trainer,
            )
            models.append(model)

        else:
            raise ValueError(f"Unrecognized checkpoint type: {single_model_cfg.restore_from_path}")

    # initialize apex DDP strategy
    def dummy():
        return

    if trainer.strategy.launcher is not None:
        trainer.strategy.launcher.launch(dummy, trainer=trainer)
    trainer.strategy.setup_environment()

    models = [model.cuda() for model in models]  # move the model to the GPU
    for model in models:
        model.eval().requires_grad_(False)  # set the model to evaluation mode and disable gradients

    # Return the trainer and model objects.
    return trainer, models


def setup_trainer_and_model_for_inference(
    model_provider: Any,
    cfg: DictConfig,
    model_cfg_modifier: Callable,
) -> Tuple[Trainer, Any]:
    """
    Set up a trainer and NeMo model for inference.

    Args:
        model_provider (Any): An object that provides the NeMo model.
        cfg (DictConfig): The configuration dictionary, containing the
            necessary settings for the trainer and the model.
        model_cfg_modifier (Callable): A function that modifies the model
            configuration for inference.

    Returns:
        Tuple[Trainer, Any]: A tuple containing the trainer and the model.
    """

    # Check if we need to use the TorchElasticEnvironment plugin for the trainer.
    plugins = []
    plugins.append(TorchElasticEnvironment())

    # Use the NLPDDPStrategy for the distributed data parallel strategy.
    # We don't use DDP for async grad allreduce and don't find unused parameters.
    if not cfg.model.get('fsdp', False):
        logging.info("FSDP is False, using DDP strategy.")
        strategy = NLPDDPStrategy(
            no_ddp_communication_hook=True,
            find_unused_parameters=False,
        )
    else:
        logging.info("Using FSDP strategy.")
        strategy = NLPFSDPStrategy(
            limit_all_gathers=cfg.model.get('fsdp_limit_all_gathers', True),
            sharding_strategy=cfg.model.get('fsdp_sharding_strategy', 'full'),
            cpu_offload=cfg.model.get('fsdp_cpu_offload', True),
            grad_reduce_dtype=cfg.model.get('fsdp_grad_reduce_dtype', 32),
            precision=cfg.trainer.precision,
            # use_orig_params=cfg.model.inductor,
            set_buffer_dtype=cfg.get('fsdp_set_buffer_dtype', None),
        )

    # Set up the trainer with the specified plugins and strategy.
    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer)

    # Create the NLPSaveRestoreConnector object for model saving and restoring.
    save_restore_connector = NLPSaveRestoreConnector()
    if cfg.model.restore_from_path is not None:
        if cfg.model.restore_from_path.endswith(".nemo") or os.path.isdir(cfg.model.restore_from_path):
            # Set the model_extracted_dir attribute if the restore path is a directory.
            if os.path.isdir(cfg.model.restore_from_path):
                save_restore_connector.model_extracted_dir = cfg.model.restore_from_path

            # Restore the model configuration from the specified path and modify it for inference.
            model_cfg = model_provider.restore_from(
                restore_path=cfg.model.restore_from_path,
                trainer=trainer,
                save_restore_connector=save_restore_connector,
                return_config=True,
            )
            with open_dict(model_cfg):
                model_cfg_modifier(model_cfg)  # modify the configuration for inference

            # Restore the model from the specified path and configuration, and set it up for inference.
            model = model_provider.restore_from(
                restore_path=cfg.model.restore_from_path,
                trainer=trainer,
                override_config_path=model_cfg,
                save_restore_connector=save_restore_connector,
                strict=True,
            )

        elif cfg.model.restore_from_path.endswith(".ckpt"):
            logging.warning(
                "Loading from .ckpt checkpoint for inference is experimental! It doesn't support models with model parallelism!"
            )

            model = model_provider.load_from_checkpoint(
                cfg.model.restore_from_path,
                hparams_file=cfg.model.get("hparams_file"),
                trainer=trainer,
            )

    else:
        # load a model from scratch
        logging.warning("Loading a model from scratch for inference. Tread carefully.")
        model = model_provider(cfg=cfg.model, trainer=trainer)

    # initialize apex DDP strategy
    def dummy():
        return

    if trainer.strategy.launcher is not None:
        trainer.strategy.launcher.launch(dummy, trainer=trainer)
    trainer.strategy.setup_environment()

    model = model.cuda()  # move the model to the GPU
    model.eval().requires_grad_(False)  # set the model to evaluation mode and disable gradients

    # Return the trainer and model objects.
    return trainer, model


def create_neva_model_and_processor(cfg):
    from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import MegatronNevaModel

    plugins = []
    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())
    # trainer required for restoring model parallel models
    trainer = Trainer(plugins=plugins, strategy=NLPDDPStrategy(), **cfg.trainer)

    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    if cfg.neva_model_file:
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.neva_model_file):
            save_restore_connector.model_extracted_dir = cfg.neva_model_file

        neva_cfg = MegatronNevaModel.restore_from(
            restore_path=cfg.neva_model_file,
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )
        OmegaConf.set_struct(neva_cfg, True)
        with open_dict(neva_cfg):
            neva_cfg.sequence_parallel = False
            neva_cfg.activations_checkpoint_granularity = None
            neva_cfg.activations_checkpoint_method = None
            neva_cfg.precision = trainer.precision
            neva_cfg.mm_cfg.llm.from_pretrained = cfg.get('base_model_file', None)
            neva_cfg.apply_rope_fusion = False
            neva_cfg.fp8 = False
            neva_cfg.tensor_model_parallel_size = cfg.tensor_model_parallel_size
            neva_cfg.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
        #    neva_cfg.mm_cfg.vision_encoder.from_pretrained = None

        model = MegatronNevaModel.restore_from(
            restore_path=cfg.neva_model_file,
            trainer=trainer,
            override_config_path=neva_cfg,
            save_restore_connector=save_restore_connector,
        )
        if neva_cfg.get('peft') is not None:
            peft_cfg_cls = PEFT_CONFIG_MAP[neva_cfg.peft.peft_scheme]
            if peft_cfg_cls is not None:
                model.load_adapters(cfg.neva_model_file, peft_cfg_cls(neva_cfg))

    elif cfg.checkpoint_dir:
        app_state = AppState()
        if cfg.tensor_model_parallel_size > 1 or cfg.pipeline_model_parallel_size > 1:
            app_state.model_parallel_size = cfg.tensor_model_parallel_size * cfg.pipeline_model_parallel_size
            app_state.tensor_model_parallel_size = cfg.tensor_model_parallel_size
            app_state.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
            (
                app_state.tensor_model_parallel_rank,
                app_state.pipeline_model_parallel_rank,
                app_state.expert_model_parallel_rank,
                app_state.model_parallel_size,
                app_state.data_parallel_size,
                app_state.pipeline_model_parallel_split_rank,
                app_state.virtual_pipeline_model_parallel_rank,
            ) = fake_initialize_model_parallel(
                world_size=app_state.model_parallel_size,
                rank=trainer.global_rank,
                tensor_model_parallel_size_=cfg.tensor_model_parallel_size,
                pipeline_model_parallel_size_=cfg.pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank_=cfg.pipeline_model_parallel_split_rank,
            )
        checkpoint_path = inject_model_parallel_rank(os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name))
        # TODO: This wont work properly (We need to set model.llm.from_pretrained model.vision.from_pretrained to nul)
        model = MegatronNevaModel.load_from_checkpoint(checkpoint_path, hparams_file=cfg.hparams_file, trainer=trainer)
    else:
        raise ValueError("need at least a nemo file or checkpoint dir")

    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass
    try:
        model.model.module.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    def image_processor(maybe_image_path):
        if isinstance(maybe_image_path, str):
            image = Image.open(maybe_image_path).convert('RGB')
        else:
            image = maybe_image_path

        processor = (
            model.model.module.image_processor if hasattr(model.model, "module") else model.model.image_processor
        )
        image = process_image(processor, image, neva_cfg.data.image_aspect_ratio)
        if neva_cfg.precision in [16, '16', '16-mixed']:
            media = image.type(torch.float16)
        elif neva_cfg.precision in [32, '32', '32-true']:
            media = image.type(torch.float32)
        else:
            media = image.type(torch.bfloat16)

        return media.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)

    # add video processor for video neva
    def video_processor(maybe_video_path):

        if isinstance(maybe_video_path, str):
            vr = decord.VideoReader(maybe_video_path)
            if neva_cfg.data.splice_single_frame == 'first':
                frames = [Image.fromarray(vr[0].asnumpy()).convert('RGB')]
            elif neva_cfg.data.splice_single_frame == 'middle':
                frames = [Image.fromarray(vr[len(vr) // 2].asnumpy()).convert('RGB')]
            elif neva_cfg.data.splice_single_frame == 'last':
                frames = [Image.fromarray(vr[-1].asnumpy()).convert('RGB')]
            else:
                if neva_cfg.data.num_frames == -1:
                    frames = [Image.fromarray(frame.asnumpy()).convert('RGB') for frame in vr]
                else:
                    num_frames = min(len(vr), neva_cfg.data.num_frames)
                    indices = np.linspace(0, len(vr) - 1, num_frames, dtype=int)
                    frames = [Image.fromarray(vr[i].asnumpy()).convert('RGB') for i in indices]
                    while len(frames) < neva_cfg.data.num_frames:
                        frames.append(frames[-1])
        else:
            frames = maybe_video_path

        processor = (
            model.model.module.image_processor if hasattr(model.model, "module") else model.model.image_processor
        )
        # support single video inference
        if neva_cfg.data.image_aspect_ratio == 'keep':
            max_hw, min_hw = max(frames.size), min(frames.size)
            aspect_ratio = max_hw / min_hw
            max_len, min_len = 448, 224
            shortest_edge = int(min(max_len / aspect_ratio, min_len))
            frames = processor.preprocess(
                frames, return_tensors='pt', do_center_crop=False, size={"shortest_edge": shortest_edge}
            )['pixel_values']
        elif neva_cfg.data.image_aspect_ratio == 'pad':

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            frames = [expand2square(frame, tuple(int(x * 255) for x in processor.image_mean)) for frame in frames]
            frames = processor.preprocess(frames, return_tensors='pt')['pixel_values']
        else:
            frames = processor.preprocess(frames, return_tensors='pt')['pixel_values']

        media_tensors = frames.type(torch_dtype_from_precision(neva_cfg.precision))
        return media_tensors.unsqueeze(dim=0).unsqueeze(dim=0)

    return model, image_processor, video_processor


def create_image_processor(mm_cfg):
    if mm_cfg.vision_encoder.get("from_hf", False):
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(mm_cfg.vision_encoder.from_pretrained)
        if config.architectures[0] == "CLIPVisionModel" or config.architectures[0] == "CLIPModel":
            image_processor = CLIPImageProcessor.from_pretrained(
                mm_cfg.vision_encoder.from_pretrained, torch_dtype=torch.bfloat16
            )
        elif config.architectures[0] == "SiglipVisionModel" or config.architectures[0] == "SiglipModel":
            image_processor = SiglipImageProcessor.from_pretrained(
                mm_cfg.vision_encoder.from_pretrained, torch_dtype=torch.bfloat16
            )
        else:
            raise (ValueError("Currently only support CLIPImageProcessor and SiglipImageProcessor from Huggingface"))

        crop_size = mm_cfg.vision_encoder.get("crop_size")
        if hasattr(image_processor, 'crop_size') and crop_size is not None:
            assert crop_size == (
                image_processor.crop_size['height'],
                image_processor.crop_size['width'],
            ), f"Crop size {crop_size} does not match the HuggingFace CLIP model's crop size {(image_processor.crop_size['height'], image_processor.crop_size['width'])}"

    else:
        # Corresponds to MegatronCLIPModel
        crop_size = mm_cfg.get("crop_size", (224, 224))
        image_processor = image_transform(
            crop_size,
            is_train=False,
            mean=None,
            std=None,
        )
    return image_processor
