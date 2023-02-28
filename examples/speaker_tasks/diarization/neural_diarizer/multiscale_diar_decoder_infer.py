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

from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from nemo.core.config import hydra_runner


"""
Run the entire speaker diarization pipeline: VAD, clustering diarizer for initializing clustering then Multi-scale Diarization Decoder (MSDD).
python multiscale_diar_decoder_infer.py --config-path='../conf/inference' --config-name='diar_infer_telephonic.yaml' \
    diarizer.vad.model_path=<NeMo VAD model path> \
    diarizer.msdd_model.model_path=<NeMo MSDD model path> \
    diarizer.oracle_vad=False \
    diarizer.manifest_filepath=<test_manifest> \
    diarizer.out_dir=<test_temp_dir> \
"""


@hydra_runner(config_path="../conf/inference", config_name="diar_infer_telephonic.yaml")
def main(cfg):
    diarizer_model = NeuralDiarizer(cfg=cfg).to(cfg.device)
    diarizer_model.diarize()


if __name__ == '__main__':
    main()
