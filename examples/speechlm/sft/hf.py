# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import fiddle as fdl
import torch
from lhotse.dataset.collation import collate_matrices, collate_vectors
from omegaconf import OmegaConf

from nemo import lightning as nl
from nemo.collections import speechlm
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.speechlm.models import HFAutoModelForSpeechSeq2Seq

torch.set_float32_matmul_precision("medium")


class LhotseHfNeMoDataset(torch.utils.data.Dataset):
    def __init__(self, processor, tokenizer, decoder_mask_fill=-100):
        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.decoder_mask_fill = decoder_mask_fill

    def __getitem__(self, cuts):
        features = []
        for cut in cuts:
            audio = cut.load_audio()
            features.append(
                self.processor(
                    audio,
                    sampling_rate=cut.sampling_rate,
                    return_tensors="pt",
                    text=cut.supervisions[0].text,
                )
            )

        input_features = collate_matrices(tensors=[f["input_features"].squeeze(0) for f in features])
        labels = collate_vectors(tensors=[c.supervisions[0].tokens for c in cuts])
        decoder_input_ids = labels[:, :-1]
        decoder_input_ids = decoder_input_ids.masked_fill(
            decoder_input_ids == self.decoder_mask_fill, self.tokenizer.pad_id
        )
        labels = labels[:, 1:].reshape(-1)

        return {
            "input_features": input_features,
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
        }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # Models can be one of the supported ones by AutoModelForSpeechSeq2Seq such as
    # openai/whisper-large-v3 and facebook/s2t-small-librispeech-asr
    parser.add_argument('--model', default='openai/whisper-large-v3')
    parser.add_argument('--strategy', type=str, default='auto', choices=['auto', 'ddp', 'fsdp'])
    parser.add_argument('--devices', default=1)
    parser.add_argument('--accelerator', default='gpu', choices=['gpu'])
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--model-save-path', type=str, default=None)
    args = parser.parse_args()

    model = HFAutoModelForSpeechSeq2Seq(model_name=args.model)
    model = model.to(torch.float)
    processor = model.processor
    tokenizer = AutoTokenizer(args.model, include_special_tokens=True)

    config = OmegaConf.create(
        {
            "cuts_path": "/opt/checkpoints/lhotse/libri/libri-train-5.jsonl.gz",
            "sample_rate": 16000,
            "shuffle": True,
            "num_workers": 2,
            "batch_size": 4,
            "shuffle_buffer_size": 100,
        }
    )

    train_dataloader = get_lhotse_dataloader_from_config(
        config,
        global_rank=0,
        world_size=1,
        dataset=LhotseHfNeMoDataset(
            processor=processor,
            tokenizer=tokenizer,
        ),
        tokenizer=tokenizer,
    )

    speechlm.api.finetune(
        model=model,
        data=train_dataloader,
        trainer=nl.Trainer(
            devices=args.devices,
            max_steps=args.max_steps,
            accelerator=args.accelerator,
            strategy=args.strategy,
            precision="bf16-mixed",
            log_every_n_steps=1,
            limit_val_batches=0.0,
            num_sanity_val_steps=0,
            accumulate_grad_batches=10,
            gradient_clip_val=0.5,
            use_distributed_sampler=False,
            callbacks=[],
            logger=None,
        ),
        optim=fdl.build(speechlm.adam.pytorch_adam_with_flat_lr(lr=1e-5)),
        log=None,
    )

    if args.model_save_path is not None:
        model.save_pretrained(args.model_save_path)
