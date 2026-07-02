# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""No-op dataset that materializes the per-batch ``cut.id`` list and the
worker subprocess metadata. The sampler/dataloader machinery decides
*which* cuts each call gets, which is exactly the question the
validator answers."""

import torch.utils.data


class CutIdDataset(torch.utils.data.Dataset):
    """Returns per-batch ``cut.id`` list and ``worker_info`` instead of
    realizing audio/features. Bypasses ``SALMDataset`` and the tokenizer
    so the validator can iterate orders of magnitude faster than a real
    training step."""

    def __getitem__(self, cuts):
        info = torch.utils.data.get_worker_info()
        return {
            "cut_ids": [str(cut.id) for cut in cuts],
            "worker_id": int(info.id) if info is not None else 0,
            "num_workers": int(info.num_workers) if info is not None else 1,
        }
