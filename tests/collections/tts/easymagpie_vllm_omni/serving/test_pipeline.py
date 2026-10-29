# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Tests for the standalone EasyMagpie LM pipeline topology."""
from __future__ import annotations

import pytest

pytest.importorskip("vllm_omni")

from easymagpie_vllm_omni.pipeline import (  # noqa: E402
    EASYMAGPIE_CFG_PAIRING_PIPELINE,
    EASYMAGPIE_LM_PIPELINE,
    EASYMAGPIE_PIPELINE,
)


def test_lm_pipeline_is_single_stage():
    assert EASYMAGPIE_LM_PIPELINE.model_type == "easymagpie_lm"
    assert len(EASYMAGPIE_LM_PIPELINE.stages) == 1
    stage = EASYMAGPIE_LM_PIPELINE.stages[0]
    assert stage.stage_id == 0
    assert stage.model_stage == "easymagpie"
    assert stage.final_output is True
    assert stage.final_output_type == "audio"
    assert stage.engine_output_type == "audio"
    assert stage.custom_process_next_stage_input_func is None
    assert stage.async_chunk_process_next_stage_input_func is None


def test_two_stage_pipeline_unchanged():
    assert EASYMAGPIE_PIPELINE.model_type == "easymagpie"
    assert len(EASYMAGPIE_PIPELINE.stages) == 2
    assert EASYMAGPIE_PIPELINE.stages[1].final_output is True
    assert EASYMAGPIE_PIPELINE.stages[1].final_output_type == "audio"


def test_cfg_pairing_pipeline_requires_explicit_selection_and_keeps_standard_pipeline():
    probe = EASYMAGPIE_CFG_PAIRING_PIPELINE
    assert probe.model_type == "easymagpie_cfg_pairing"
    assert probe.hf_architectures == ()
    assert probe.validate() == []
    assert probe.stages[0].scheduler_cls == "easymagpie_vllm_omni.cfg_scheduler.EasyMagpieCFGPairingScheduler"
    assert probe.stages[1] == EASYMAGPIE_PIPELINE.stages[1]
    assert EASYMAGPIE_PIPELINE.stages[0].scheduler_cls.endswith(".EasyMagpieARAsyncScheduler")


def test_cfg_deployment_preserves_dedicated_pipeline_scheduler():
    from conftest import EASYMAGPIE_ROOT
    from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy

    deploy = load_deploy_config(EASYMAGPIE_ROOT / "deploy" / "easymagpie_cfg_pairing.yaml")
    stages = merge_pipeline_deploy(EASYMAGPIE_CFG_PAIRING_PIPELINE, deploy)
    assert stages[0].scheduler_cls == "easymagpie_vllm_omni.cfg_scheduler.EasyMagpieCFGPairingScheduler"
    stage_args = stages[0].yaml_engine_args
    assert stage_args["max_num_seqs"] == 2
    assert stage_args["enforce_eager"] is True
    assert stage_args["async_scheduling"] is False
    assert stage_args["enable_chunked_prefill"] is False


def test_cfg_synthesis_deployment_uses_same_conservative_paired_execution():
    from conftest import EASYMAGPIE_ROOT
    from easymagpie_vllm_omni.pipeline import EASYMAGPIE_CFG_PIPELINE
    from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy

    deploy = load_deploy_config(EASYMAGPIE_ROOT / "deploy" / "easymagpie_cfg.yaml")
    assert deploy.pipeline == "easymagpie_cfg"
    assert EASYMAGPIE_CFG_PIPELINE.validate() == []
    stages = merge_pipeline_deploy(EASYMAGPIE_CFG_PIPELINE, deploy)
    assert stages[0].scheduler_cls.endswith(".EasyMagpieCFGScheduler")
    assert stages[0].yaml_engine_args["max_num_seqs"] == 2


def test_concurrent_cfg_profile_counts_physical_sequences_and_keeps_conservative_execution():
    from conftest import EASYMAGPIE_ROOT
    from easymagpie_vllm_omni.pipeline import EASYMAGPIE_CFG_PIPELINE
    from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy

    deploy = load_deploy_config(EASYMAGPIE_ROOT / 'deploy' / 'easymagpie_cfg_concurrent.yaml')
    stages = merge_pipeline_deploy(EASYMAGPIE_CFG_PIPELINE, deploy)
    assert stages[0].scheduler_cls.endswith('.EasyMagpieCFGScheduler')
    assert stages[0].yaml_engine_args['max_num_seqs'] == 6
    assert stages[1].yaml_engine_args['max_num_seqs'] == 4
    for stage in stages:
        args = stage.yaml_engine_args
        assert args['enforce_eager'] is True
        assert args['async_scheduling'] is False
        assert args['enable_prefix_caching'] is False
        assert args['enable_chunked_prefill'] is False


@pytest.mark.parametrize(
    "profile",
    [
        "easymagpie_cfg.yaml",
        "easymagpie_cfg_pairing.yaml",
        "easymagpie_cfg_concurrent.yaml",
        "easymagpie_cfg_graph.yaml",
        "easymagpie_cfg_compile.yaml",
    ],
)
def test_cfg_codec_history_fits_full_acoustic_generation(profile):
    import yaml
    from conftest import EASYMAGPIE_ROOT

    config = yaml.safe_load((EASYMAGPIE_ROOT / "deploy" / profile).read_text())
    acoustic, codec = config["stages"]
    # Omni's generation runner stores the cumulative codec prompt in InputBatch,
    # even though each forward consumes only the next small streaming chunk.
    assert codec["max_model_len"] >= acoustic["default_sampling_params"]["max_tokens"]


@pytest.mark.parametrize(
    'profile',
    [
        'easymagpie_cfg.yaml',
        'easymagpie_cfg_concurrent.yaml',
        'easymagpie_cfg_graph.yaml',
        'easymagpie_cfg_compile.yaml',
        'easymagpie_cfg_pairing.yaml',
    ],
)
def test_cfg_profiles_match_standard_precision(profile):
    from pathlib import Path

    import yaml

    deploy = Path(__file__).resolve().parents[5] / 'tools/easymagpie_vllm_omni/deploy'
    standard = yaml.safe_load((deploy / 'easymagpie.yaml').read_text())
    lm = yaml.safe_load((deploy / 'easymagpie_lm.yaml').read_text())
    cfg = yaml.safe_load((deploy / profile).read_text())
    # Compare every explicit dtype, including inherited model/attention/conv
    # defaults: missing fields must stay missing on both sides as well.
    for reference in (standard, lm):
        assert {k: v for k, v in cfg.items() if k.endswith('dtype')} == {
            k: v for k, v in reference.items() if k.endswith('dtype')
        }
        assert {k: v for k, v in cfg['stages'][0].items() if k.endswith('dtype')} == {
            k: v for k, v in reference['stages'][0].items() if k.endswith('dtype')
        }
    assert {k: v for k, v in cfg['stages'][1].items() if k.endswith('dtype')} == {
        k: v for k, v in standard['stages'][1].items() if k.endswith('dtype')
    }
