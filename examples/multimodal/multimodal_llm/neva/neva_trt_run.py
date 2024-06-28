# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from nemo.core.config import hydra_runner
from nemo.export.tensorrt_mm_exporter import TensorRTMMExporter


@hydra_runner(config_path='conf', config_name='neva_trt_infer')
def main(cfg):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    exporter = TensorRTMMExporter(cfg.engine_dir)
    output = exporter.forward(
        input_text=cfg.input_text,
        input_media=cfg.input_media,
        batch_size=cfg.batch_size,
        max_output_len=cfg.infer.max_new_tokens,
        top_k=cfg.infer.top_k,
        top_p=cfg.infer.top_p,
        temperature=cfg.infer.temperature,
        repetition_penalty=cfg.infer.repetition_penalty,
        num_beams=cfg.infer.num_beams,
    )

    print(output)


if __name__ == '__main__':
    main()
