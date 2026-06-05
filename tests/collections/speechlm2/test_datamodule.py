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
import pytest
import torch
from lhotse import CutSet
from lhotse.testing.dummies import DummyManifest
from lightning.pytorch.utilities import CombinedLoader
from omegaconf import DictConfig

from nemo.collections.common.data.lhotse.broadcasting import BroadcastingDataLoader
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model
from nemo.collections.speechlm2.data import DataModule


@pytest.fixture
def data_config(tmp_path):
    ap, cp = tmp_path / "audio", str(tmp_path) + "/{tag}_cuts.jsonl.gz"

    def _assign(k, v):
        def _inner(obj):
            setattr(obj, k, v)
            return obj

        return _inner

    for tag in ("train", "val_set_0", "val_set_1"):
        (
            DummyManifest(CutSet, begin_id=0, end_id=2, with_data=True)
            .map(_assign("tag", tag))
            .save_audios(ap)
            .drop_in_memory_data()
            .to_file(cp.format(tag=tag))
        )

    return DictConfig(
        {
            "train_ds": {
                "input_cfg": [
                    {
                        "type": "lhotse",
                        "cuts_path": cp.format(tag="train"),
                    }
                ],
                "batch_size": 2,
            },
            "validation_ds": {
                "datasets": {
                    "val_set_0": {"cuts_path": cp.format(tag="val_set_0")},
                    "val_set_1": {"cuts_path": cp.format(tag="val_set_1")},
                },
                "batch_size": 2,
            },
        }
    )


@pytest.fixture
def tokenizer(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("tok")
    text_path = tmpdir / "text.txt"
    text_path.write_text("\n".join(chr(i) for i in range(256)))
    create_spt_model(
        text_path,
        vocab_size=512,
        sample_size=-1,
        do_lower_case=False,
        output_dir=str(tmpdir),
        bos=True,
        eos=True,
        remove_extra_whitespaces=True,
    )
    return SentencePieceTokenizer(str(tmpdir / "tokenizer.model"))


class Identity(torch.utils.data.Dataset):
    def __getitem__(self, item):
        return item


def test_datamodule_train_dataloader(data_config, tokenizer):
    data = DataModule(data_config, tokenizer=tokenizer, dataset=Identity())
    dl = data.train_dataloader()
    assert isinstance(dl, (BroadcastingDataLoader, torch.utils.data.DataLoader))
    dli = iter(dl)

    batch = next(dli)
    assert isinstance(batch, CutSet)
    assert len(batch) == 2
    assert all(c.tag == "train" for c in batch)


def test_datamodule_validation_dataloader(data_config, tokenizer):
    val_sets = {"val_set_0", "val_set_1"}
    data = DataModule(data_config, tokenizer=tokenizer, dataset=Identity())
    dl = data.val_dataloader()
    assert isinstance(dl, CombinedLoader)
    dli = iter(dl)

    batch, batch_idx, dataloader_idx = next(dli)
    assert isinstance(batch, dict)
    assert batch.keys() == val_sets
    for vs in val_sets:
        assert len(batch[vs]) == 2
        assert all(c.tag == vs for c in batch[vs])
