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
import copy
import glob

import lightning.pytorch as ptl
import pytest
import torch
from kaldialign import edit_distance
from omegaconf import DictConfig, open_dict

from nemo.core.config.pytorch_lightning import TrainerConfig
from nemo.core.utils.cuda_python_utils import skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported


# These tests move the model to CUDA before calling transcribe(), so avoid forking DataLoader workers afterwards.
CUDA_GRAPH_TRANSCRIBE_NUM_WORKERS = 0


def test_forced_full_graph_compile_does_not_fallback():
    from nemo.collections.asr.parts.submodules.transducer_decoding.rnnt_label_looping import (
        GreedyBatchedRNNTLabelLoopingComputer,
    )

    accelerator_error = getattr(torch, "AcceleratorError", RuntimeError)
    computer = GreedyBatchedRNNTLabelLoopingComputer.__new__(GreedyBatchedRNNTLabelLoopingComputer)
    computer.cuda_graphs_allow_fallback = False

    with pytest.raises(RuntimeError, match="Full CUDA graph decoding failed"):
        computer._raise_or_warn_no_while_loop_cuda_graphs(accelerator_error("CUDA error: invalid argument"))


@pytest.mark.with_downloads
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA decoder can run only on CUDA")
@pytest.mark.parametrize(
    ("model_name", "batch_size", "enable_bfloat16"),
    [
        ("stt_en_fastconformer_transducer_large", 8, False),
        ("stt_en_fastconformer_transducer_large", 8, True),
    ],
)
@pytest.mark.parametrize("loop_labels", [False, True])
def test_cuda_graph_rnnt_greedy_decoder(model_name, batch_size, enable_bfloat16, loop_labels: bool, request):
    if not loop_labels:
        skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported()
    if enable_bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is not supported")

    device = torch.device("cuda")
    nemo_model = request.getfixturevalue(model_name).to(device)
    decoding_config = copy.deepcopy(nemo_model.cfg.decoding)

    with open_dict(decoding_config):
        decoding_config["greedy"]["max_symbols"] = 5
        decoding_config["greedy"]["loop_labels"] = loop_labels
        decoding_config["greedy"]["use_cuda_graph_decoder"] = False

    nemo_model.change_decoding_strategy(decoding_config)
    audio_filepaths = glob.glob("tests/.data/asr/test/an4/wav/*.wav")

    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=enable_bfloat16):
        actual_hypotheses = nemo_model.transcribe(
            audio_filepaths, batch_size=batch_size, num_workers=CUDA_GRAPH_TRANSCRIBE_NUM_WORKERS
        )

    actual_transcripts = [hyp.text for hyp in actual_hypotheses]
    actual_y_sequences = [hyp.y_sequence for hyp in actual_hypotheses]

    decoding_config["greedy"]["use_cuda_graph_decoder"] = True

    nemo_model.change_decoding_strategy(decoding_config)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=enable_bfloat16):
        fast_hypotheses = nemo_model.transcribe(
            audio_filepaths, batch_size=batch_size, num_workers=CUDA_GRAPH_TRANSCRIBE_NUM_WORKERS
        )

    fast_transcripts = [hyp.text for hyp in fast_hypotheses]
    fast_y_sequences = [hyp.y_sequence for hyp in fast_hypotheses]

    total_dist = sum(
        edit_distance(r.split(), h.split())['total'] for r, h in zip(actual_transcripts, fast_transcripts)
    )
    total_words = sum(len(r.split()) for r in actual_transcripts)
    wer = total_dist / total_words if total_words > 0 else 0.0
    y_sequence_eq = [torch.equal(act_y, fast_y) for (act_y, fast_y) in zip(actual_y_sequences, fast_y_sequences)]

    assert wer <= 1e-3, "Cuda graph greedy decoder should match original decoder implementation."
    assert all(y_sequence_eq), "Cuda graph greedy decoder should match original decoder implementation."

    for actual, fast in zip(actual_transcripts, fast_transcripts):
        if actual != fast:
            print("erroneous samples:")
            print("Original transcript:", actual)
            print("New transcript:", fast)


@pytest.mark.with_downloads
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA decoder can run only on CUDA")
@pytest.mark.parametrize("force_mode", ["no_graphs", "no_while_loops", "full_graph"])
@pytest.mark.parametrize("enable_bfloat16", [False, True])
def test_loop_labels_cuda_graph_rnnt_greedy_decoder_forced_mode(
    stt_en_fastconformer_transducer_large, force_mode: str, enable_bfloat16: bool
):
    """
    Testing Label-Looping algorithm with CUDA graphs in forced mode.
    This test guarantees that we check that the fallback behavior is working.
    NB: Since it is impossible to directly debug CUDA graphs, when making changes,
    start testing and debugging the code with forced "no_graphs" mode.
    """
    if enable_bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is not supported")

    if force_mode == "full_graph":
        skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported()

    batch_size = 16
    device = torch.device("cuda")
    nemo_model = stt_en_fastconformer_transducer_large.to(device)
    decoding_config = copy.deepcopy(nemo_model.cfg.decoding)

    with open_dict(decoding_config):
        decoding_config["greedy"]["max_symbols"] = 5
        decoding_config["greedy"]["loop_labels"] = True
        decoding_config["greedy"]["use_cuda_graph_decoder"] = False
        # test that alignments and confidence do not introduce failures
        decoding_config["greedy"]["preserve_alignments"] = True
        decoding_config["greedy"]["preserve_frame_confidence"] = True

    nemo_model.change_decoding_strategy(decoding_config)
    audio_filepaths = glob.glob("tests/.data/asr/test/an4/wav/*.wav")

    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=enable_bfloat16):
        actual_hypotheses = nemo_model.transcribe(
            audio_filepaths, batch_size=batch_size, num_workers=CUDA_GRAPH_TRANSCRIBE_NUM_WORKERS
        )
    actual_transcripts = [hyp.text for hyp in actual_hypotheses]

    # transcribe with use implementation with cuda graphs
    decoding_config["greedy"]["use_cuda_graph_decoder"] = True
    nemo_model.change_decoding_strategy(decoding_config)
    backup_cuda_graph_mode = nemo_model.decoding.decoding.decoding_computer.cuda_graphs_mode
    try:
        nemo_model.decoding.decoding.decoding_computer.force_cuda_graphs_mode(mode=force_mode)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=enable_bfloat16):
            fast_hypotheses = nemo_model.transcribe(
                audio_filepaths, batch_size=batch_size, num_workers=CUDA_GRAPH_TRANSCRIBE_NUM_WORKERS
            )
        fast_transcripts = [hyp.text for hyp in fast_hypotheses]

        total_dist = sum(
            edit_distance(r.split(), h.split())['total'] for r, h in zip(actual_transcripts, fast_transcripts)
        )
        total_words = sum(len(r.split()) for r in actual_transcripts)
        wer = total_dist / total_words if total_words > 0 else 0.0

        assert wer <= 1e-3, "Cuda graph greedy decoder should match original decoder implementation."

        for actual, fast in zip(actual_transcripts, fast_transcripts):
            if actual != fast:
                print("erroneous samples:")
                print("Original transcript:", actual)
                print("New transcript:", fast)
    finally:
        nemo_model.decoding.decoding.decoding_computer.force_cuda_graphs_mode(mode=backup_cuda_graph_mode)


@pytest.mark.with_downloads
@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="Test requires CUDA device with bf16 support",
)
@pytest.mark.parametrize("is_tdt", [False, True])
def test_loop_labels_cuda_graph_ddp_mixed_precision(
    tmp_path_factory,
    an4_train_manifest_corrected,
    stt_en_fastconformer_transducer_large,
    stt_en_fastconformer_tdt_large,
    is_tdt: bool,
):
    """CUDA graphs with DDP and mixed precision have bugs. We need to test that validation works with these settings."""
    batch_size = 16

    # instantiate trainer with bf16 mixed precision
    trainer_cfg = TrainerConfig(devices=[0], accelerator="cuda", strategy="ddp", max_epochs=1, precision="bf16-mixed")
    trainer = ptl.Trainer(**DictConfig(trainer_cfg))

    model = stt_en_fastconformer_tdt_large if is_tdt else stt_en_fastconformer_transducer_large

    # setup validation data
    val_ds_cfg = model.cfg.validation_ds
    with open_dict(val_ds_cfg):
        val_ds_cfg.manifest_filepath = [str(an4_train_manifest_corrected)]
        val_ds_cfg.batch_size = batch_size
        val_ds_cfg.is_tarred = False
        val_ds_cfg.use_lhotse = False
        if is_tdt:
            # TDT model has config with missing mandatory values, this results in errors when setting up validation data
            # we set all the mandatory values to dummy values
            model.cfg.train_ds.tarred_audio_filepaths = None
            model.cfg.train_ds.manifest_filepath = None
            model.cfg.test_ds.manifest_filepath = None
            model.cfg.tokenizer.dir = None

    model.setup_multiple_validation_data(val_ds_cfg)

    # validate using trainer
    val_results = trainer.validate(model)
    wer = val_results[0]["val_wer"]

    # explicitly free resources, then test conditions
    trainer._teardown()
    # teardown from the trainer is not enough, and problem with CPU will still exist, related issue:
    # https://github.com/Lightning-AI/pytorch-lightning/issues/18803)
    # solution is to destroy torch process group explicitly
    torch.distributed.destroy_process_group()

    assert wer <= 0.1, f"WER is too high: {wer}"


@pytest.mark.with_downloads
@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="Test requires 2 GPUs")
@pytest.mark.parametrize("loop_labels", [False, True])
def test_change_devices(loop_labels: bool, stt_en_fastconformer_transducer_large):
    if not loop_labels:
        skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported()

    first_device = torch.device("cuda:0")
    second_device = torch.device("cuda:1")

    batch_size = 8

    nemo_model = stt_en_fastconformer_transducer_large.to(second_device)
    decoding_config = copy.deepcopy(nemo_model.cfg.decoding)

    with open_dict(decoding_config):
        decoding_config["greedy"]["max_symbols"] = 5
        decoding_config["greedy"]["loop_labels"] = loop_labels
        decoding_config["greedy"]["use_cuda_graph_decoder"] = True

    nemo_model.change_decoding_strategy(decoding_config)

    # Test that the model can run successfully when it is first
    # initialized on second_device and then transferred to
    # true_device
    nemo_model.to(first_device)
    audio_filepaths = glob.glob("tests/.data/asr/test/an4/wav/*.wav")
    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
        second_device_hypotheses = nemo_model.transcribe(
            audio_filepaths, batch_size=batch_size, num_workers=CUDA_GRAPH_TRANSCRIBE_NUM_WORKERS
        )
    second_device_transcripts = [hyp.text for hyp in second_device_hypotheses]

    # Test that the model can run successfully back on second_device
    # after having been first run on first_device. Because the
    # decoder's data structures are lazily initialized, this activates
    # slightly different code than the first case (where the decoder
    # has not run at all), so we want to exercise both cases.
    nemo_model.to(second_device)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
        first_device_hypotheses = nemo_model.transcribe(
            audio_filepaths, batch_size=batch_size, num_workers=CUDA_GRAPH_TRANSCRIBE_NUM_WORKERS
        )
    first_device_transcripts = [hyp.text for hyp in first_device_hypotheses]
    # Sanity check: The device we run on should not change execution
    # output.
    assert first_device_transcripts == second_device_transcripts
