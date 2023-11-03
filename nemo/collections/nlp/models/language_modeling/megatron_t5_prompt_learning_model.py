# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import itertools
from typing import Any, List

import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.t5_prompt_learning_dataset import T5PromptLearningDataset
from nemo.collections.nlp.models.language_modeling.megatron_base_prompt_learning_model import (
    MegatronBasePromptLearningModel,
)
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.models.language_modeling.megatron_t5_sft_model import MegatronT5SFTModel
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_iterator_k_split,
)
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils import AppState, logging

try:
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator,
        get_micro_batch_size,
        get_num_microbatches,
    )

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    from megatron.core import parallel_state
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


__all__ = ['MegatronT5PromptLearningModel']


class MegatronT5PromptLearningModel(MegatronBasePromptLearningModel):
    """
    Model class for prompt-tuning or p-tuning a pretrained Megatron T5 model. 

    Prompt Tuning initalizes virtual prompt embeddings directly from a copy of
    certain token embeddings from the the pretrained T5 model's vocabulary
    and directly tunes these embedding weights. The token embeddings used in 
    initalization are specified by the user in the config file. The model can 
    be prompt-tuned for multiple tasks at once. Virtual prompts are stored in a 
    prompt table and can be added or deleted without disrupting virtual prompts 
    for other tasks. 

    P-tuning initializes an LSTM encoder model that generates virtual prompt
    embeddings for every task. Each task shares the same encoder. After p-tuning
    is compelete, the learned virtual prompts can be saved to the prompt table
    using add_ptuned_prompts_to_prompt_table(). Thus, if a user wants to add a 
    new virtual prompt via p-tuning, they do not need to retrain on all previous 
    tasks. This gives p-tuning the same task flexiblity as prompt-tuning.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        self.model_type = ModelType.encoder_and_decoder

    def first_stage_of_pipeline(self):
        if self.frozen_model.enc_dec_model.pre_process and parallel_state.get_pipeline_model_parallel_rank() == 0:
            return True
        return False

    def forward(
        self, input_ids, dec_input, enc_mask, dec_mask, position_ids, taskname_ids, labels=None, inference=False,
    ):
        """
        Special forward method for p-tuning/prompt-tuning pretrained
        T5 style models.
        """
        batch_size, seq_length = input_ids.shape

        if self.first_stage_of_pipeline():
            # Get embeddings for text tokens and insert virtual token embeddings
            input_embeds = self.embed_input(input_ids, taskname_ids, inference)
            # TODO: This check needs to be revisited with PP support.
            if hasattr(self.frozen_model.enc_dec_model.encoder_embedding, 'position_embeddings'):
                position_embeddings = self.frozen_model.enc_dec_model.encoder_embedding.position_embeddings(
                    position_ids
                )
                encoder_input = input_embeds + position_embeddings
            else:
                encoder_input = input_embeds
        else:
            encoder_input = None

        # If the decoder input starts with <pad> instead of <bos>, which is the case for huggingface T5 models, we don't want to mask the first token.
        # For NeMo-Megatron, the sequence starts with <bos>, which is never masked so we can always set index 0 to be unmasked.
        dec_mask[:, 0] = 1

        # Call forward on T5 model with preprocessed embeddings
        if self.autocast_dtype == torch.float32:
            output = self.frozen_model.enc_dec_model(
                enc_input_ids=None,
                enc_attn_mask=enc_mask,
                dec_input_ids=dec_input,
                dec_attn_mask=dec_mask,
                token_type_ids=None,
                labels=labels,
                output_enc_hidden_only=False,
                enc_input=encoder_input,
            )
        else:
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                output = self.frozen_model.enc_dec_model(
                    enc_input_ids=None,
                    enc_attn_mask=enc_mask,
                    dec_input_ids=dec_input,
                    dec_attn_mask=dec_mask,
                    token_type_ids=None,
                    labels=labels,
                    output_enc_hidden_only=False,
                    enc_input=encoder_input,
                )

        return output, encoder_input

    def load_frozen_model(self, cfg, trainer):
        self.megatron_amp_O2 = cfg.get('megatron_amp_O2', False)

        # TODO: Fix this once apex patches FusedScaledMaskedSoftmax.
        # This is a workaround for the fact that `masked_softmax_fusion` has issues with certain input sizes that may be present while finetuning.
        t5_cfg = MegatronT5Model.restore_from(cfg.get('language_model_path'), trainer=trainer, return_config=True)
        OmegaConf.set_struct(t5_cfg, True)
        with open_dict(t5_cfg):
            if hasattr(t5_cfg, 'encoder') and hasattr(t5_cfg, 'decoder'):
                t5_cfg.encoder.masked_softmax_fusion = False
                t5_cfg.decoder.masked_softmax_fusion = False
            else:
                t5_cfg.masked_softmax_fusion = False
            t5_cfg.megatron_amp_O2 = self.megatron_amp_O2
            # hack to make the _GLOBAL_NUM_MICROBATCHES_CALCULATOR initialize
            t5_cfg.micro_batch_size = cfg.get('micro_batch_size', 4)
            t5_cfg.global_batch_size = cfg.get('global_batch_size', 4)
            t5_cfg.precision = trainer.precision

        self.frozen_model = MegatronT5Model.restore_from(
            cfg.get('language_model_path'),
            trainer=trainer,
            override_config_path=t5_cfg,
            save_restore_connector=NLPSaveRestoreConnector(),
        )

    def fwd_bwd_step(self, dataloader_iter, batch_idx, forward_only):
        """
            Dataloader produces a global batch which is turned into a list of microbatches.
            The list of microbatches is then piped through the pipeline using megatron-core fwd/bwd functions.
        """
        # Get seq length of batch
        batch = next(dataloader_iter)
        _, seq_length = batch[0].shape
        _, dec_seq_length = batch[1].shape
        data_iter = get_iterator_k_split(batch, get_num_microbatches())

        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            data_iterator=data_iter,
            model=[self],
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=seq_length,
            micro_batch_size=get_micro_batch_size(),
            decoder_seq_length=dec_seq_length,
        )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            # average loss across micro batches
            loss_tensors_list = [loss_reduced['loss'] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.concat(loss_tensors_list)
            loss_mean = loss_tensor.mean()
        else:
            # we're not on the last pipeline stage so no losses
            loss_mean = torch.tensor(0.0).cuda()

        return loss_mean

    def get_forward_output_and_loss_func(self):
        # FIXME: consolidate this method into MegatronLMEncoderDecoderModel (or have a common base class)
        def fwd_output_and_loss_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            batch = [x.cuda(non_blocking=True) for x in batch]
            enc_input, dec_input, labels, loss_mask, enc_mask, dec_mask, position_ids, taskname_ids = batch

            output_tensor, encoder_input = model(
                enc_input, dec_input, enc_mask, dec_mask, position_ids, taskname_ids, labels, inference=False
            )
            output_tensor = output_tensor.contiguous()

            def loss_func(output_tensor):
                loss = self.frozen_model.loss_func(loss_mask, output_tensor)
                reduced_loss = average_losses_across_data_parallel_group([loss])
                return loss, {'loss': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def backward(self, *args, **kwargs):
        """ LightningModule hook to do backward.
            We want this to do nothing since we run backward in the fwd/bwd functions from megatron-core.
            No need to call it here.
        """
        return

    def optimizer_zero_grad(self, *args, **kwargs):
        """ LightningModule hook to zero grad.
            We want this to do nothing as we are zeroing grads during the training_step.
        """
        return

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.
        When using pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.frozen_model.enc_dec_model.set_input_tensor(input_tensor)

    def on_train_epoch_start(self) -> None:
        gbs = self.cfg.global_batch_size
        mbs = self.cfg.micro_batch_size
        self._reconfigure_batch_sizes(gbs, mbs)
        return super().on_train_epoch_start()

    def on_validation_epoch_start(self) -> None:
        gbs = self.cfg.get('validation_global_batch_size', self.cfg.global_batch_size)
        mbs = self.cfg.get('validation_micro_batch_size', self.cfg.micro_batch_size)
        self._reconfigure_batch_sizes(gbs, mbs)
        return super().on_validation_epoch_start()

    def training_step(self, dataloader_iter, batch_idx):
        self._optimizer.zero_grad()
        batch = next(dataloader_iter)
        loss_mean = self.fwd_bwd_step(itertools.chain([batch]), batch_idx, forward_only=False)
        self.allreduce_gradients()

        ## logging
        # we can only log on one rank if it is rank zero so we broadcast from last rank
        # we can avoid this broadcast by updating the PTL log function to accept specific ranks
        torch.distributed.broadcast(loss_mean, get_last_rank())

        if self.torch_dtype == torch.float16 and hasattr(self.trainer.precision_plugin.scaler, "_scale"):
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale, batch_size=1)

        self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True, batch_size=1)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, rank_zero_only=True, batch_size=1)
        self.log('global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True, batch_size=1)
        return loss_mean

    def get_predictions(self, input_ids, enc_mask, encoder_input, labels):
        predicted_token_ids, log_probs = self.frozen_model.decode(
            tokens_enc=input_ids,
            enc_mask=enc_mask,
            num_tokens_to_generate=self.decoder_seq_length,
            encoder_input=encoder_input,
            bos_id=self.tokenizer.pad_id
            if self.cfg.data.get('decoder_starts_with_pad', False)
            else self.tokenizer.bos_id,
        )
        # Special ids to text function to handle stripping <eos> and special tokens with sentencepiece tokenizers.
        preds_text = MegatronT5SFTModel.ids_to_text(predicted_token_ids, self.tokenizer)
        labels_text = MegatronT5SFTModel.ids_to_text(labels, self.tokenizer)
        input_text = MegatronT5SFTModel.ids_to_text(input_ids, self.tokenizer)
        return {
            'predicted_token_ids': preds_text,
            'labels': labels_text,
            'enc_inputs': input_text,
        }

    def validation_step(self, batch, batch_idx, inference=False):
        prefix = "test" if self.trainer.testing else "val"
        input_ids, dec_input, labels, loss_mask, enc_mask, dec_mask, position_ids, taskname_ids = batch
        # does not use dataloader_iter due to device placement issues arising from PTL
        mode = self.training
        self.eval()
        gbs = self.cfg.get('validation_global_batch_size', self.cfg.global_batch_size)
        self._reconfigure_and_process_inference_batch(input_ids.size(0), gbs)
        loss_mean = self.fwd_bwd_step(itertools.chain([batch]), batch_idx, forward_only=True)

        if self.first_stage_of_pipeline():
            # Get embeddings for text tokens and insert virtual token embeddings
            input_embeds = self.embed_input(input_ids, taskname_ids, False)

            if hasattr(self.frozen_model.enc_dec_model.encoder_embedding, 'position_embeddings'):
                position_embeddings = self.frozen_model.enc_dec_model.encoder_embedding.position_embeddings(
                    position_ids
                )
                encoder_input = input_embeds + position_embeddings
            else:
                encoder_input = input_embeds
        else:
            encoder_input = None

        if self.cfg.get("report_validation_metric", False):
            metrics = self.get_predictions(input_ids, enc_mask, encoder_input, labels)
            metrics['loss'] = loss_mean
        else:
            metrics = {'loss': loss_mean}

        self.train(mode=mode)
        self.frozen_model.eval()
        self.validation_step_outputs.append(metrics) if prefix == 'val' else self.test_step_outputs.append(metrics)
        return metrics

    def on_validation_epoch_end(self):
        prefix = "test" if self.trainer.testing else "val"
        outputs = self.validation_step_outputs if prefix == 'val' else self.test_step_outputs

        if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
            if parallel_state.is_pipeline_last_stage():
                # only the last pipeline parallel stages return loss
                averaged_loss = torch.stack([i['loss'] for i in outputs]).mean()
            else:
                averaged_loss = torch.tensor(0.0).cuda()

            # we can only log on one rank if it is rank zero so we broadcast from last rank
            torch.distributed.broadcast(averaged_loss, get_last_rank())

            self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True, batch_size=1)
            logging.info(f'Validation loss: {averaged_loss}')

        else:
            averaged_loss = torch.stack([item['loss'] for item in outputs]).mean()
            logging.info(f'Validation loss: {averaged_loss}')
            self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True, batch_size=1)

        if self.cfg.get("report_validation_metric", False):
            gather_results = [None for _ in range(parallel_state.get_data_parallel_world_size())]

            all_preds = list(itertools.chain(*[item['predicted_token_ids'] for item in outputs]))
            all_labels = list(itertools.chain(*[item['labels'] for item in outputs]))
            all_inputs = list(itertools.chain(*[item['enc_inputs'] for item in outputs]))

            assert len(all_preds) == len(all_labels)
            assert len(all_preds) == len(all_inputs)

            # Gather inputs, preds, labels from all workers
            torch.distributed.all_gather_object(
                gather_results,
                [(input, pred, label) for (input, pred, label) in zip(all_inputs, all_preds, all_labels)],
                group=parallel_state.get_data_parallel_group(),
            )

            # Deduplicate sentences that may have been distributed across multiple data parallel ranks.
            if parallel_state.get_data_parallel_rank() == 0:

                gather_results_dedup = list(set(itertools.chain(*gather_results)))

                val_metric_dict = self.validation_metric.get_score(
                    [i[2] for i in gather_results_dedup], [i[1] for i in gather_results_dedup],
                )

                for metric, val in val_metric_dict.items():
                    logging.info(f'Validation {metric}: {val}')
                val_metric = list(val_metric_dict.items())[0][1]
                metric_name = list(val_metric_dict.items())[0][0]
            else:
                val_metric = torch.tensor(0.0).cuda()
                metric_name = ''

            self.log(f'val_{metric_name}', val_metric, prog_bar=True, rank_zero_only=True, batch_size=1)

        gbs = self.cfg.global_batch_size
        mbs = self.cfg.micro_batch_size
        self._reconfigure_batch_sizes(gbs, mbs)

        self.validation_step_outputs.clear() if prefix == 'val' else self.test_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def build_virtual_prompt_dataset(
        self, dataset_paths, batch_size, for_train, drop_last, shuffle, num_workers, pin_memory
    ):
        dataset = T5PromptLearningDataset(
            datasets=dataset_paths,
            tokenizer=self.tokenizer,
            virtual_prompt_source=self.virtual_prompt_source,
            task_templates=self.task_templates,
            pseudo_tokens=self.pseudo_tokens,
            pad_token_id=self.pad_token_id,
            max_seq_length=self.cfg.data.get('max_seq_length', self.frozen_model.cfg.max_position_embeddings),
            min_seq_length=self.cfg.data.get('min_seq_length', 1),
            add_bos=self.cfg.data.get('add_bos', False),
            add_eos=self.cfg.data.get('add_eos', True),
            decoder_starts_with_pad=self.cfg.data.get('decoder_starts_with_pad', False),
            add_eos_to_decoder_output=self.cfg.data.get('add_eos_to_decoder_output', True),
            add_sentinel_to_input=self.cfg.data.get('add_sentinel_to_input', True),
            ul2_prompt_token=self.cfg.data.get('ul2_prompt_token', None),
            for_train=for_train,
        )

        rank = parallel_state.get_data_parallel_rank()
        world_size = parallel_state.get_data_parallel_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=self.cfg.seed
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            sampler=sampler,
            batch_size=batch_size // world_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True
            if num_workers > 0
            else False,  # (@adithyare and @eharper) We need to set this to True to get around issues with spawn=True
        )
        print('build success', len(dataloader), dataset_paths)
        return dataset, dataloader

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:

        input_ids, dec_input, labels, loss_mask, enc_mask, dec_mask, position_ids, taskname_ids = batch

        batch_size, seq_length = input_ids.shape
        if self.first_stage_of_pipeline():
            input_embeds = self.embed_input(input_ids, taskname_ids, use_cached_reps=True)

            # TODO: This check needs to be revisited with PP support.
            if hasattr(self.frozen_model.enc_dec_model.encoder_embedding, 'position_embeddings'):
                position_embeddings = self.frozen_model.enc_dec_model.encoder_embedding.position_embeddings(
                    position_ids
                )
                encoder_input = input_embeds + position_embeddings
            else:
                encoder_input = input_embeds

        else:
            encoder_input = torch.zeros((batch_size, seq_length, self.hidden_size), dtype=self.autocast_dtype).cuda()

        predicted_token_ids, log_probs = self.frozen_model.decode(
            tokens_enc=input_ids,
            enc_mask=enc_mask,
            num_tokens_to_generate=self.decoder_seq_length,
            encoder_input=encoder_input,
            bos_id=self.tokenizer.pad_id
            if self.cfg.data.get('decoder_starts_with_pad', False)
            else self.tokenizer.bos_id,
        )
        # Special ids to text function to handle stripping <eos> and special tokens with sentencepiece tokenizers.
        preds_text = MegatronT5SFTModel.ids_to_text(predicted_token_ids, self.tokenizer)
        input_text = MegatronT5SFTModel.ids_to_text(input_ids, self.tokenizer)

        if labels is not None:
            labels_text = MegatronT5SFTModel.ids_to_text(labels, self.tokenizer)
        else:
            labels_text = [None] * len(preds_text)

        return {
            'input_text': input_text,
            'preds_text': preds_text,
            'labels_text': labels_text,
        }

    def on_predict_epoch_end(self) -> None:
        # PTL 2.0 removes outputs arg from on_predict_epoch_end and is retrived by self.trainer.predict_loop.predictions
        outputs = self.trainer.predict_loop.predictions
        gather_results = [None for _ in range(parallel_state.get_data_parallel_world_size())]
        # In case of single dataloader format of self.trainer.predict_loop.predictions is [{dict1},{dict2}] v/s [[{dict1},{dict2}]] of outputs in 1.9
        # Removing indix [0] to avoid TypeError (https://github.com/Lightning-AI/lightning/pull/16655/files)
        all_preds = list(itertools.chain(*[item['preds_text'] for item in outputs]))
        all_labels = list(itertools.chain(*[item['labels_text'] for item in outputs]))
        all_inputs = list(itertools.chain(*[item['input_text'] for item in outputs]))

        assert len(all_preds) == len(all_labels)
        assert len(all_preds) == len(all_inputs)

        # Gather inputs, predictions, and ground truths from all workers
        torch.distributed.all_gather_object(
            gather_results,
            [(input, pred, label) for (input, pred, label) in zip(all_inputs, all_preds, all_labels)],
            group=parallel_state.get_data_parallel_group(),
        )

        # Deduplicate sentences that may have been distributed across multiple data parallel ranks.
        if parallel_state.get_data_parallel_rank() == 0:
            gather_results_dedup = list(set(itertools.chain(*gather_results)))

            input_prediction_pair = []
            correct = 0
            for (input, pred, label) in gather_results_dedup:
                input_prediction_pair.append((input, pred))
                if label:
                    if pred == label:
                        correct += 1

            acc = correct / len(gather_results_dedup) if all_labels[0] else None
            logging.info(f'Prediction results: {acc}')
            logging.info(f'Test finish')
