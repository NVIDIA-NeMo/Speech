# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
Tests for MagpieTTS inference.
"""

import csv
import json
import os

import examples.tts.magpietts_inference as magpietts_inference_module
import pytest
from examples.tts.magpietts_inference import main as magpietts_inference_main

import nemo.collections.tts.modules.magpietts_inference.evaluate_generated_audio as evaluate_generated_audio
import nemo.collections.tts.modules.magpietts_inference.utils as magpietts_utils
from nemo.collections.tts.modules.magpietts_inference.evaluate_generated_audio import (
    FILEWISE_METRICS_TO_SAVE,
    _get_record_texts,
    load_evalset_config,
)
from nemo.collections.tts.modules.magpietts_inference.evaluation import (
    EvaluationConfig,
    resolve_evaluation_config_for_dataset,
)
from nemo.collections.tts.modules.magpietts_inference.utils import (
    EXPERIMENT_METRICS_CSV_HEADER,
    _group_multiturn_filewise_metrics_by_sample,
    _write_grouped_multiturn_filewise_metrics_csv,
    append_metrics_to_csv,
    write_csv_header_if_needed,
)


class TestMagpieTTSInferenceCLI:
    """Tests for MagpieTTS inference command-line interface options."""

    @pytest.mark.run_only_on('GPU')
    @pytest.mark.parametrize(
        "disable_flag,metric_key",
        [
            # Test both the --disable_fcd and --disable_utmosv2 flags
            ("--disable_fcd", "frechet_codec_distance"),
            ("--disable_utmosv2", "utmosv2_avg"),
        ],
        # Test names
        ids=["disable_fcd", "disable_utmosv2"],
    )
    def test_disable_metric_produces_nan(self, tmp_path, disable_flag, metric_key):
        """
        Test that disabling a metric via CLI flag:
        1. Does not cause the script to crash
        2. Produces NaN for the corresponding metric
        """

        # Test data paths in CI environment
        codec_model_path = "/home/TestData/tts/AudioCodec_21Hz_no_eliz_without_wavlm_disc.nemo"
        hparams_file = (
            "/home/TestData/tts/2506_ZeroShot/lrhm_short_yt_prioralways_alignement_0.002_priorscale_0.1.yaml"
        )
        checkpoint_file = "/home/TestData/tts/2506_ZeroShot/dpo-T5TTS--val_loss=0.4513-epoch=3.ckpt"
        datasets_json_path = "examples/tts/evalset_config.json"

        # Build command-line arguments
        args = [
            "--codecmodel_path", codec_model_path,
            "--datasets_json_path", datasets_json_path,
            "--datasets", "an4_val_tiny_ci",
            "--out_dir", str(tmp_path),
            "--batch_size", "4",
            "--num_repeats", "1",
            "--temperature", "0.6",
            "--hparams_files", hparams_file,
            "--checkpoint_files", checkpoint_file,
            "--legacy_codebooks",
            "--legacy_text_conditioning",
            "--apply_attention_prior",
            "--run_evaluation",
            disable_flag,
        ]  # fmt: skip

        # Run the main function directly with arguments
        magpietts_inference_main(args)

        # Look for the metrics file
        metrics_file = os.path.join(tmp_path, "all_experiment_metrics_with_ci.csv")
        assert os.path.exists(metrics_file), f"Metrics file not found at {metrics_file}"

        # Load and verify the metrics
        with open(metrics_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0, "No data rows found in metrics CSV"
        metrics = rows[0]  # Get the first data row

        metric_value = metrics.get(metric_key)
        assert metric_value is not None, f"{metric_key} key not found in metrics"
        assert "nan" in metric_value.lower(), f"{metric_key} should be NaN but got: {metric_value}"


@pytest.mark.unit
def test_evaluation_uses_normalized_text_for_metrics():
    record = {
        "text": "July 15th",
        "normalized_text": "july fifteenth",
        "original_text": "legacy original text",
    }

    tts_text_input, dataloader_normalized_text, metric_reference_text = _get_record_texts(record)

    assert tts_text_input == "July 15th"
    assert dataloader_normalized_text == "july fifteenth"
    assert metric_reference_text == "july fifteenth"
    assert "tts_text_input" in FILEWISE_METRICS_TO_SAVE
    assert "dataloader_normalized_text" in FILEWISE_METRICS_TO_SAVE
    # Legacy phonemized manifests keep the orthography in original_text; a JSON null counts as absent.
    assert _get_record_texts({"text": "dʒʊˈlaɪ", "original_text": "July"})[2] == "July"
    assert _get_record_texts({"text": "July 15th", "original_text": None})[2] == "July 15th"


@pytest.mark.unit
def test_grouped_multiturn_exports_input_and_normalized_text(tmp_path):
    grouped_rows = _group_multiturn_filewise_metrics_by_sample(
        [
            {
                "source_sample_idx": 0,
                "turn_id": 0,
                "tts_text_input": "July 15th",
                "dataloader_normalized_text": "july fifteenth",
                "gt_text": "july fifteenth",
                "pred_text": "july fifteenth",
            }
        ]
    )

    assert grouped_rows[0]["tts_text_input"] == ["July 15th"]
    assert grouped_rows[0]["dataloader_normalized_text"] == ["july fifteenth"]

    csv_path = tmp_path / "metrics.csv"
    _write_grouped_multiturn_filewise_metrics_csv(str(csv_path), grouped_rows)
    with csv_path.open(encoding="utf-8") as csv_file:
        csv_row = next(csv.DictReader(csv_file))

    assert json.loads(csv_row["tts_text_input"]) == ["July 15th"]
    assert json.loads(csv_row["dataloader_normalized_text"]) == ["july fifteenth"]


@pytest.mark.unit
def test_resolve_evaluation_config_for_dataset_overrides():
    eval_config = EvaluationConfig(language="en", eou_batch_size=7)
    meta = {
        "manifest_path": "m.json",
        "audio_dir": "a",
        "language": "de",
        "asr_model": {"name": "some/asr", "type": "whisper"},
    }

    resolved = resolve_evaluation_config_for_dataset(eval_config, meta)

    assert resolved.language == "de"
    assert (resolved.asr_model_name, resolved.asr_model_type) == ("some/asr", "whisper")
    assert resolved.eou_batch_size == 7  # fields without an override are inherited
    assert eval_config.language == "en"  # the input is not mutated
    # Absent keys keep the CLI-level values.
    assert resolve_evaluation_config_for_dataset(eval_config, {"manifest_path": "m.json"}) == eval_config


@pytest.mark.unit
def test_load_evalset_config_resolves_relative_paths_against_base_path(tmp_path):
    base = tmp_path / "base"
    (base / "audio").mkdir(parents=True)
    (base / "m.json").write_text("{}\n")
    config_path = tmp_path / "evalset.json"
    config_path.write_text(json.dumps({"ds": {"manifest_path": "m.json", "audio_dir": "audio"}}))

    entry = load_evalset_config(str(config_path), dataset_base_path=base)["ds"]

    assert entry["manifest_path"] == str(base / "m.json")
    assert entry["audio_dir"] == str(base / "audio")


def _filewise_row(gt_text, pred_text, cer, wer):
    nan = float("nan")
    return {
        "gt_text": gt_text,
        "pred_text": pred_text,
        "gt_audio_text": None,
        "cer": cer,
        "wer": wer,
        "cer_pred_gt_audio": nan,
        "wer_pred_gt_audio": nan,
        "pred_gt_ssim": 0.5,
        "pred_context_ssim": 0.5,
        "gt_context_ssim": 0.5,
        "pred_gt_ssim_alternate": 0.5,
        "pred_context_ssim_alternate": 0.5,
        "gt_context_ssim_alternate": 0.5,
        "utmosv2": 3.0,
        "total_gen_audio_seconds": 1.0,
    }


def _write_evalset_config(tmp_path, entry_overrides):
    (tmp_path / "audio").mkdir(exist_ok=True)
    (tmp_path / "m.json").write_text("{}\n")
    entry = {"manifest_path": "m.json", "audio_dir": "audio", **entry_overrides}
    config_path = tmp_path / "evalset.json"
    config_path.write_text(json.dumps({"ds": entry}))
    return config_path


def _patch_below_evaluate(monkeypatch, captured):
    """Stub evaluate_dir/compute_global_metrics so that the real evaluate() wiring (incl. its FCD guard) runs."""

    def fake_evaluate_dir(**kwargs):
        captured.update(kwargs)
        return [_filewise_row("you want", "you want", 0.0, 0.0)]

    def fake_compute_global_metrics(**kwargs):
        captured["global_kwargs"] = kwargs
        return {}

    monkeypatch.setattr(evaluate_generated_audio, "evaluate_dir", fake_evaluate_dir)
    monkeypatch.setattr(evaluate_generated_audio, "compute_global_metrics", fake_compute_global_metrics)


@pytest.mark.unit
def test_standalone_main_applies_evalset_overrides(tmp_path, monkeypatch):
    config_path = _write_evalset_config(
        tmp_path,
        {
            "language": "de",
            "asr_model": {"name": "some/asr", "type": "whisper"},
        },
    )
    captured = {}
    _patch_below_evaluate(monkeypatch, captured)
    argv = [
        "evaluate_generated_audio.py",
        "--evalset", "ds",
        "--datasets_json_path", str(config_path),
        "--datasets_base_path", str(tmp_path),
        "--generated_audio_dir", str(tmp_path),
        "--with_prosody_metrics",  # CLI-level settings without an evalset override are inherited
        "--strip_text_annotations_for_metrics",
    ]  # fmt: skip
    monkeypatch.setattr("sys.argv", argv)

    evaluate_generated_audio.main()

    assert captured["with_prosody_metrics"] is True
    assert captured["strip_text_annotations_for_metrics"] is True
    assert captured["language"] == "de"
    assert (captured["asr_model_name"], captured["asr_model_type"]) == ("some/asr", "whisper")
    assert captured["sv_model_type"] == "wavlm"
    # Pins that the whole EvaluationConfig is forwarded: evaluate()'s own default for eou_model_name is None.
    assert captured["eou_model_name"] == EvaluationConfig().eou_model_name
    assert captured["manifest_path"] == str(tmp_path / "m.json")
    assert captured["audio_dir"] == str(tmp_path / "audio")
    # The standalone script has no codec model argument: FCD is disabled instead of tripping evaluate()'s guard.
    assert captured["global_kwargs"]["codec_model_path"] is None
    assert captured["global_kwargs"]["gt_audio_paths"] is None


@pytest.mark.unit
def test_standalone_main_without_evalset_uses_cli_values(tmp_path, monkeypatch):
    captured = {}
    _patch_below_evaluate(monkeypatch, captured)
    argv = [
        "evaluate_generated_audio.py",
        "--manifest_path", "m.json",
        "--audio_dir", "audio",
        "--generated_audio_dir", "generated",
        "--language", "de",
        "--strip_text_annotations_for_metrics",
    ]  # fmt: skip
    monkeypatch.setattr("sys.argv", argv)

    evaluate_generated_audio.main()

    assert captured["strip_text_annotations_for_metrics"] is True
    assert captured["language"] == "de"
    assert captured["eou_model_name"] == EvaluationConfig().eou_model_name
    assert (captured["manifest_path"], captured["audio_dir"]) == ("m.json", "audio")
    assert captured["generated_audio_dir"] == "generated"


@pytest.mark.unit
@pytest.mark.parametrize(
    "argv",
    [
        ["--evalset", "ds", "--generated_audio_dir", "g"],  # --evalset without --datasets_json_path
        [
            "--evalset",
            "ds",
            "--datasets_json_path",
            "cfg.json",
            "--manifest_path",
            "m.json",
            "--generated_audio_dir",
            "g",
        ],
        ["--datasets_json_path", "cfg.json", "--manifest_path", "m.json", "--generated_audio_dir", "g"],
        ["--datasets_base_path", "base", "--manifest_path", "m.json", "--generated_audio_dir", "g"],
        ["--manifest_path", "m.json", "--audio_dir", "a"],  # missing --generated_audio_dir
        ["--audio_dir", "a", "--generated_audio_dir", "g"],  # neither --evalset nor --manifest_path
    ],
)
def test_standalone_main_rejects_inconsistent_arguments(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["evaluate_generated_audio.py", *argv])
    with pytest.raises(SystemExit):
        evaluate_generated_audio.main()


@pytest.mark.unit
def test_standalone_main_rejects_unknown_evalset(tmp_path, monkeypatch):
    config_path = _write_evalset_config(tmp_path, {})
    argv = [
        "evaluate_generated_audio.py",
        "--evalset", "missing",
        "--datasets_json_path", str(config_path),
        "--datasets_base_path", str(tmp_path),
        "--generated_audio_dir", str(tmp_path),
    ]  # fmt: skip
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        evaluate_generated_audio.main()


@pytest.mark.unit
@pytest.mark.parametrize(
    "entry_overrides, match",
    [
        ({"language": None}, "'language' must be a non-empty string"),
        ({"language": ""}, "'language' must be a non-empty string"),
        ({"language": " en"}, "'language' must be a non-empty string"),
        ({"asr_model": None}, "'asr_model' must be an object"),
        ({"asr_model": {"name": "some/asr"}}, "'asr_model' must be an object"),
        ({"asr_model": {"name": "some/asr", "type": "hf"}}, "'asr_model' must be an object"),
        ({"asr_model": {"name": " ", "type": "nemo"}}, "'asr_model' must be an object"),
    ],
)
def test_malformed_evalset_overrides_are_rejected(tmp_path, entry_overrides, match):
    # Both entry points reject the same malformed values: the resolver (hand-built meta) and the config loader,
    # which prefixes the dataset name.
    with pytest.raises(ValueError, match=match):
        resolve_evaluation_config_for_dataset(EvaluationConfig(), {"manifest_path": "m.json", **entry_overrides})
    config_path = _write_evalset_config(tmp_path, entry_overrides)
    with pytest.raises(ValueError, match=f"Dataset ds: .*{match}"):
        load_evalset_config(str(config_path), dataset_base_path=tmp_path)


@pytest.mark.unit
def test_load_evalset_config_warns_on_unrecognized_keys(tmp_path, monkeypatch):
    warnings_seen = []
    monkeypatch.setattr(
        evaluate_generated_audio.logging, "warning", lambda msg, *args, **kwargs: warnings_seen.append(msg)
    )

    # Keys consumed by the inference/evaluation scripts (e.g. the shipped examples/tts/evalset_config.json) are fine.
    config_path = _write_evalset_config(tmp_path, {"feature_dir": None, "tokenizer_names": ["english_phoneme"]})
    load_evalset_config(str(config_path), dataset_base_path=tmp_path)
    assert warnings_seen == []

    # A misspelled override would silently leave the CLI-level value in force; it is reported instead.
    config_path = _write_evalset_config(tmp_path, {"langauge": "de"})
    load_evalset_config(str(config_path), dataset_base_path=tmp_path)
    assert len(warnings_seen) == 1
    assert "Dataset ds" in warnings_seen[0] and "langauge" in warnings_seen[0]


@pytest.mark.unit
def test_experiment_metrics_csv_header_matches_appended_rows(tmp_path):
    csv_path = tmp_path / "all_experiment_metrics.csv"
    write_csv_header_if_needed(str(csv_path), EXPERIMENT_METRICS_CSV_HEADER)
    metrics = {"cer_filewise_avg": 0.1, "katakana_cer_cumulative": 0.2}
    append_metrics_to_csv(str(csv_path), "ckpt", "ds", metrics)

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert (rows[0]["checkpoint_name"], rows[0]["dataset"]) == ("ckpt", "ds")
    assert rows[0]["cer_filewise_avg"] == "0.1"
    assert rows[0]["katakana_cer_cumulative"] == "0.2"
    assert rows[0]["wer_filewise_avg"] == ""  # absent metrics leave an empty cell
    assert None not in rows[0]  # every value has a header column
    assert len(rows[0]) == len(EXPERIMENT_METRICS_CSV_HEADER.split(","))


@pytest.mark.unit
def test_write_csv_header_if_needed_warns_on_changed_layout(tmp_path, monkeypatch):
    warnings_seen = []
    monkeypatch.setattr(magpietts_utils.logging, "warning", lambda msg, *args, **kwargs: warnings_seen.append(msg))
    csv_path = tmp_path / "all_experiment_metrics_with_ci.csv"

    write_csv_header_if_needed(str(csv_path), EXPERIMENT_METRICS_CSV_HEADER)
    write_csv_header_if_needed(str(csv_path), EXPERIMENT_METRICS_CSV_HEADER)
    assert warnings_seen == []

    # A CSV written before a column was appended keeps its header; the changed layout is reported.
    old_header, dropped_column = EXPERIMENT_METRICS_CSV_HEADER.rsplit(",", 1)
    csv_path.write_text(old_header + "\n")
    write_csv_header_if_needed(str(csv_path), EXPERIMENT_METRICS_CSV_HEADER)
    assert len(warnings_seen) == 1 and dropped_column in warnings_seen[0]
    assert csv_path.read_text() == old_header + "\n"


class _FakeRunner:
    def create_dataset(self, dataset_meta):
        return [0]

    def run_inference_on_dataset(self, **kwargs):
        return [{"rtf": 1.0}], None, []

    def compute_mean_rtf_metrics(self, rtf_metrics_list):
        return {}


class _FakeInferenceConfig:
    def build_identifier(self):
        return "_id"


@pytest.mark.unit
def test_run_inference_and_evaluation_applies_evalset_override(tmp_path, monkeypatch):
    # The example script used to rebuild EvaluationConfig by hand from the evalset entry; it now goes through
    # resolve_evaluation_config_for_dataset, so overrides and inherited CLI-level fields are handled in one place.
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"audio_filepath": "a.wav", "text": "Hello there."}) + "\n")
    captured = {}

    def fake_evaluate_generated_audio_dir(manifest_path, audio_dir, generated_audio_dir, config):
        captured["config"] = config
        return {"cer_cumulative": 0.0, "ssim_pred_context_avg": 1.0}, [{"cer": 0.0}]

    monkeypatch.setattr(magpietts_inference_module, "evaluate_generated_audio_dir", fake_evaluate_generated_audio_dir)
    for name in ("create_violin_plot", "append_metrics_to_csv", "write_csv_header_if_needed"):
        monkeypatch.setattr(magpietts_inference_module, name, lambda *args, **kwargs: None)

    eval_config = EvaluationConfig(language="en", with_fcd=False, with_utmosv2=False)
    dataset_meta_info = {
        "ds": {
            "manifest_path": str(manifest),
            "audio_dir": str(tmp_path),
            "language": "de",
            "asr_model": {"name": "some/asr", "type": "whisper"},
        }
    }

    cer, ssim = magpietts_inference_module.run_inference_and_evaluation(
        runner=_FakeRunner(),
        checkpoint_name="ckpt",
        inference_config=_FakeInferenceConfig(),
        eval_config=eval_config,
        dataset_meta_info=dataset_meta_info,
        datasets=["ds"],
        out_dir=str(tmp_path / "out"),
        flops_per_component={},
        moe_info="",
    )

    assert captured["config"].language == "de"
    assert (captured["config"].asr_model_name, captured["config"].asr_model_type) == ("some/asr", "whisper")
    assert captured["config"].with_utmosv2 is False  # CLI-level settings without an override are inherited
    assert (cer, ssim) == (0.0, 1.0)
