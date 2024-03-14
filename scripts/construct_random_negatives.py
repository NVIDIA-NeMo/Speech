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

import argparse

import numpy as np
import pandas as pd


def construct_negatives(input_file, output_file, num_passages, num_negatives):
    qrels = pd.read_csv(input_file, delimiter="\t", header=None)
    with open(output_file, "w") as f:
        for i in range(len(qrels)):
            query_id, rel_passage_id = qrels[0][i], qrels[2][i]
            negatives = np.random.randint(num_passages, size=num_negatives)
            output_ids = [query_id, rel_passage_id] + negatives.tolist()
            output_str = [str(id_) for id_ in output_ids]
            print("\t".join(output_str), file=f)


def main():
    parser = argparse.ArgumentParser(description="Negative passages construction")
    parser.add_argument("--data", type=str, default="msmarco_dataset", help="path to folder with data")
    parser.add_argument("--num_passages", type=int, default=8841823, help="total number of passages")
    parser.add_argument("--num_negatives", type=int, default=10, help="number of negatives per positive")
    args = parser.parse_args()

    for mode in ["train", "dev"]:
        construct_negatives(
            input_file=f"{args.data}/qrels.{mode}.tsv",
            output_file=f"{args.data}/query2passages.{mode}.tsv",
            num_passages=args.num_passages,
            num_negatives=args.num_negatives,
        )


if __name__ == '__main__':
    main()
