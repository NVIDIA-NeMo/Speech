# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import time

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

# from sacrebleu import corpus_bleu


class LogEpochTimeCallback(Callback):
    """Simple callback that logs how long each epoch takes, in seconds, to a pytorch lightning log
    """

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.time()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        curr_time = time.time()
        duration = curr_time - self.epoch_start
        trainer.logger.log_metrics({"epoch_time": duration}, step=trainer.global_step)


# class MachineTranslationLogEvalCallback(Callback):
#     def _on_eval_end(self, trainer, pl_module, mode):
#         counts = np.array(self._non_pad_tokens)
#         eval_loss = np.sum(np.array(self._losses) * counts) / np.sum(counts)
#         sacre_bleu = corpus_bleu(self._translations, [self._ground_truths], tokenize="13a")
#         print(f"{mode} results for process with global rank {pl_module.global_rank}".upper())
#         for i in range(pl_module.num_examples[mode]):
#             print('\u0332'.join(f"EXAMPLE {i}:"))  # Underline output
#             sent_id = np.random.randint(len(self._translations))
#             print(f"Ground truth: {self._ground_truths[sent_id]}\n")
#             print(f"Translation: {self._translations[sent_id]}\n")
#             print()
#         print("-" * 50)
#         print(f"loss: {eval_loss:.3f}")
#         print(f"SacreBLEU: {sacre_bleu}")
#         print("-" * 50)

#     @rank_zero_only
#     def on_test_end(self, trainer, pl_module):
#         self._on_eval_end(trainer, pl_module, "test")

#     @rank_zero_only
#     def on_validation_end(self, trainer, pl_module):
#         self._on_eval_end(trainer, pl_module, "val")

#     @rank_zero_only
#     def on_sanity_check_end(self, trainer, pl_module):
#         self._on_eval_end(trainer, pl_module, "val")

#     def _on_eval_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, mode):
#         self._translations.extend(outputs['translations'])
#         self._ground_truths.extend(outputs['ground_truths'])
#         self._non_pad_tokens.append(outputs['num_non_pad_tokens'])
#         self._losses.append(outputs[f'{mode}_loss'])

#     @rank_zero_only
#     def on_test_batch_end(self, trainer, pl_module, batch, outputs, batch_idx, dataloader_idx):
#         self._on_eval_batch_end(trainer, pl_module, batch, outputs, batch_idx, dataloader_idx, 'test')

#     @rank_zero_only
#     def on_validation_batch_end(self, trainer, pl_module, batch, outputs, batch_idx, dataloader_idx):
#         self._on_eval_batch_end(trainer, pl_module, batch, outputs, batch_idx, dataloader_idx, 'val')

#     def _on_eval_start(self, trainer, pl_module):
#         self._translations = []
#         self._ground_truths = []
#         self._losses = []
#         self._non_pad_tokens = []

#     @rank_zero_only
#     def on_test_start(self, trainer, pl_module):
#         self._on_eval_start(trainer, pl_module)

#     @rank_zero_only
#     def on_validation_start(self, trainer, pl_module):
#         self._on_eval_start(trainer, pl_module)

#     @rank_zero_only
#     def on_sanity_check_start(self, trainer, pl_module):
#         self._on_eval_start(trainer, pl_module)
