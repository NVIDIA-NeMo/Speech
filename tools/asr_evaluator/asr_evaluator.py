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
import json

import git
from omegaconf import OmegaConf, open_dict
from utils import cal_target_metadata_wer, run_asr_inference

from nemo.collections.asr.parts.utils.eval_utils import cal_write_wer
from nemo.core.config import hydra_runner
from nemo.utils import logging

"""
This script serves as evaluator of ASR models
Usage:
python asr_evaluator.py \
engine.pretrained_name="stt_en_conformer_transducer_large" \
engine.inference.mode="offline" \
engine.test_ds.augmentor.noise.manifest_path=<manifest file for noise data> \
.....

Check out parameters in ./conf/eval.yaml
"""


@hydra_runner(config_path="conf", config_name="eval.yaml")
def main(cfg):
    report = {}
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    # Store git hash for reproducibility
    if cfg.env.save_git_hash:
        repo = git.Repo(search_parent_directories=True)
        report['git_hash'] = repo.head.object.hexsha

    ## Engine
    # Could skip run_asr_inference and use the generated manifest by
    # specifying analyst.metric_calculator.exist_pred_manifest
    if cfg.analyst.metric_calculator.exist_pred_manifest is None:
        # If need to change more parameters for ASR inference, change it in
        # 1) shell script in utils.py
        # 2) TranscriptionConfig on top of the executed scripts such as transcribe_speech.py in examples/asr
        # Note we SKIP calculating wer during asr_inference stage with calculate_wer=False and calculate wer for each sample below
        # for more flexibility and reducing possible redundant inference cost.
        cfg.engine = run_asr_inference(cfg=cfg.engine)

    else:
        logging.info(
            f"Use generated prediction manifest {cfg.analyst.metric_calculator.exist_pred_manifest} and skip enigneer"
        )
        with open_dict(cfg):
            cfg.engine.output_filename = cfg.analyst.metric_calculator.exist_pred_manifest

    ## Analyst
    output_manifest_w_wer, total_res, eval_metric = cal_write_wer(
        pred_manifest=cfg.engine.output_filename,
        clean_groundtruth_text=cfg.analyst.metric_calculator.clean_groundtruth_text,
        langid=cfg.analyst.metric_calculator.langid,
        use_cer=cfg.analyst.metric_calculator.use_cer,
        output_filename=cfg.analyst.metric_calculator.output_filename,
    )
    with open_dict(cfg):
        cfg.analyst.metric_calculator.output_filename = output_manifest_w_wer

    report.update({"res": total_res})

    for target in cfg.analyst.metadata:
        if cfg.analyst.metadata[target].enable:
            occ_avg_wer = cal_target_metadata_wer(
                manifest=cfg.analyst.metric_calculator.output_filename,
                target=target,
                meta_cfg=cfg.analyst.metadata[target],
                eval_metric=eval_metric,
            )
            report[target] = occ_avg_wer

    config_engine = OmegaConf.to_object(cfg.engine)
    report.update(config_engine)

    config_metric_calculator = OmegaConf.to_object(cfg.analyst.metric_calculator)
    report.update(config_metric_calculator)

    pretty = json.dumps(report, indent=4)
    res = "%.3f" % (report["res"][eval_metric] * 100)
    logging.info(pretty)
    logging.info(f"Overall {eval_metric} is {res} %")

    ## Writer
    report_file = "report.json"
    if "report_filename" in cfg.writer and cfg.writer.report_filename:
        report_file = cfg.writer.report_filename

    with open(report_file, "a") as fout:
        json.dump(report, fout)
        fout.write('\n')
        fout.flush()


if __name__ == "__main__":
    main()
