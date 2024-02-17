# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import random
from typing import Dict, List

from torch.utils.data import Dataset


class BertEmbeddingDataset(Dataset):
    """SentenceTransformer tokenizer and MultipleNegativesRankingLoss expects
        a single positive and a single hard-negative (optional) per example.
        This Dataset manages the case where there is more than one positive or negative
        available, in form of a list.
        It uses the list of positives/negatives as a queue, where for each epoch the 
        first positive/negative of the queue is used for training, after which the
        item is moved to the end of the queue.
        If num_hard_negs > 1, multiple negatives will be sampled for each example.

        Args:
            data (List[Dict[str, str]]): A list of Dict whose 
            keys are "question", "pos_doc", "neg_doc"
            num_hard_negs (int): Number of hard-negatives for each query to sample
            shuffled_negs (bool, optional): Whether the negatives per example
            needs to be shuffled in the initialization. Defaults to False.
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        shuffled_negs: bool = False,
        num_hard_negs: int = 1,
        query_prefix: str = "",
        passage_prefix: str = "",
    ):
        self.data = data
        self.num_hard_negs = num_hard_negs
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

        if shuffled_negs:
            for example in self.data:
                random.shuffle(example["neg_doc"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        example = self.data[item]
        question = f'{self.query_prefix} {example["question"]}'.strip()
        texts = [question]

        positive = example["pos_doc"]
        if isinstance(positive, list):

            positive = example["pos_doc"][0]

        positive = f"{self.passage_prefix} {positive}".strip()
        texts.append(positive)

        negative = []
        if "neg_doc" in example:
            negative = example["neg_doc"]
            selected_negs = []
            if isinstance(negative, list):
                for counter in range(self.num_hard_negs):
                    if len(example["neg_doc"]) > 0:

                        negative = example["neg_doc"][counter]
                        selected_negs.append(negative)
                    else:
                        # Providing empty hard-negative, for this example,
                        # so that it matches the number of hard negatives
                        # of the other examples
                        selected_negs.append("")

            else:
                selected_negs = [negative]
            selected_negs = [f"{self.passage_prefix} {neg}".strip() for neg in selected_negs]
            texts.extend(selected_negs)
        return texts
