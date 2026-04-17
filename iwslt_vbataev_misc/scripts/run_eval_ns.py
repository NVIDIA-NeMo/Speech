"""
Example: training FastConformer model on LibriSpeech (tarred, speed-perturbed) with BPE tokenizer
You need artifacts (manifests, tokenizer) from https://gitlab-master.nvidia.com/vbataev/exprunner_data (uses git-lfs)

To use this script:
- ensure you correctly set up docker credentials (see https://confluence.nvidia.com/display/ADLR/Enroot+credentials)
- ensure you have access to the appropriate cluster
- if you want to use remote submission to clusters, ensure you correctly set up ssh for this
- fix CLUSTER_2_HOME_DIR dict - use writable directory available to you
- use a project name on WandB that does not intersect with other team projects (if you are using corporate account)
- for local runs - adjust paths to datasets and artifacts
- install nemo-skills
- configure clusters in nemo-skills
- run script

```shell
python .sandbox/experiment_tpl_nemo_skills.py \
    --project "ASR-LibriSpeech-Experiment" \
    --cluster=lepton \
    --num-nodes=2 \
    -bs 128 -ga 1 \
    --name="FastConformer-RNNT-Default-NS-2n"
```
"""

import argparse
from pathlib import Path

from nemo_skills.pipeline.cli import wrap_arguments
from nemo_skills.pipeline.run_cmd import run_cmd

DEFAULT_CONTAINER_TAG = "3b3c6da8aa_iwslt26"
DEFAULT_NGC_REGISTRY = "nvcr.io/nvidian/ac-aiapps/nemo_vb"
# DEFAULT_GITLAB_REGISTRY = "gitlab-master.nvidia.com/vbataev/nemo_containers"

# writable directory
CLUSTER_2_HOME_DIR = {
    "local": "/home/vbataev",
    "lepton": "/llmservice_nemo_speechlm/users/vbataev",
    "draco": "/gpfs/fs1/projects/ent_aiapps/users/vbataev",
    "draco-m3": "/lustre/fs1/ent/aiapps/vbataev",
    "draco-oci": "/lustre/fsw/portfolios/convai/users/vbataev",
    "cs-oci-ord": "/lustre/fsw/portfolios/convai/users/vbataev",
}

# dataset and other artifacts


def clean_cmd(cmd: str) -> str:
    """
    Remove all newline chars + extra spaces from cmd
    :param cmd: string
    :return: cmd without duplicated spaces and newline breaks
    """
    cmd = " ".join(cmd.split())
    return cmd.strip()


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="job name", default="iwslt26-eval-pipeline_v2")
    parser.add_argument("--cluster", required=False, type=str, default="draco-oci")
    opt_args = parser.add_argument_group("additional arguments")
    opt_args.add_argument("--image", required=False, default=None, type=str, help="docker image")
    opt_args.add_argument("--partition", default=None, type=str, help="optional, partition to use")
    opt_args.add_argument("--dry-run", action="store_true", help="do not run, useful for saving command")
    return parser


def main():
    args = get_argparser().parse_args()
    cluster = args.cluster
    left_context, chunk, right_context = 10, 0.96, 0.96
    nmt_model = "Qwen/Qwen3-4B-Instruct-2507"
    asr_model = "nvidia/parakeet-unified-en-0.6b"
    exp_name = f"{args.name}--asr_unified_{left_context:.2g}_{chunk:.2g}_{right_context:.2g}--qwen3_4b--en-de--base"

    script_env = "HYDRA_FULL_ERROR=1 TORCH_CUDNN_V8_API_ENABLED=1"

    tgt_lang_code = "de"
    workspace_mnt = Path("/workspace_mnt")
    data_dir = workspace_mnt / "iwslt26/data"
    exp_dir = workspace_mnt / f"exp/{exp_name}"
    nemo_dir = workspace_mnt / "iwslt26/nemo"
    nemo_config = nemo_dir / "examples/asr/conf/asr_streaming_inference/buffered_rnnt.yaml"
    EVAL_CONFIG = exp_dir / "buffered_rnnt_simulstream.yaml"
    METRICS_LOG_FILE = exp_dir / "en-de_output.jsonl"
    detailed_log_file = exp_dir / "detailed_log.jsonl"
    REFERENCE_FILE = data_dir / f"mcif/raw/ref/{tgt_lang_code}.txt"
    TRANSCRIPT_FILE = data_dir / "mcif/raw/ref/en.txt"
    AUDIO_DEFINITION = data_dir / "mcif/raw/audio-segments.yaml"
    wav_list = data_dir / "mcif/wav_list.txt"
    SACREBLEU_TOKENIZER = "13a"
    MOSES_TOKENIZER = "13a"
    CHAR_LEVEL_FLAG = "--word_level"

    script_path = nemo_dir / "nemo/collections/asr/inference/run_nemo_simulstream.py"


    # per_stream_boosting.phrases_file = {DATA_DIR} / boosting_phrases_v2.json \
    #         per_stream_boosting.alpha = 0.4 \
    cmd = f"""
        echo "*******STARTING********" \
        && echo "---------------" \
        && printenv \
        && nvidia-smi \
        && export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
        && mkdir -p {exp_dir} \
        && cd {exp_dir} \
        && {script_env} python {script_path} \
        --config "{nemo_config}" \
        --wav-list {wav_list} \
        --src-lang "en" \
        --tgt-lang "{tgt_lang_code}" \
        --metrics-log "{METRICS_LOG_FILE}" \
        --use-adapter-v2 \
        streaming.left_padding_size={left_context:.2g} \
        streaming.chunk_size={chunk:.2g} \
        streaming.right_padding_size={right_context:.2g} \
        streaming.decode_temporary=true \
        endpointing.stop_history_eou=1200 \
        pipeline_v2.num_prev_sentences_for_translation=5 \
        detailed_log_path={detailed_log_file} \
        asr.model_name={asr_model} \
        nmt.model_name={nmt_model} \
        && . .evaluation/bin/activate \
        && omnisteval longform \
            --speech_segmentation "{AUDIO_DEFINITION}" \
            --source_sentences_file "{TRANSCRIPT_FILE}" \
            --ref_sentences_file "{REFERENCE_FILE}" \
            --hypothesis_file "{METRICS_LOG_FILE}" \
            --simulstream_config_file "{EVAL_CONFIG}" \
            --hypothesis_format simulstream \
            --comet \
            --comet_model Unbabel/XCOMET-XL \
            --lang "{MOSES_TOKENIZER}" \
            {CHAR_LEVEL_FLAG} \
            --bleu_tokenizer "{SACREBLEU_TOKENIZER}" \
            --output_folder "{exp_dir /'segmentation_output'}"
        """
    # NB: create_tensorboard_logger=false since tensorboard is broken in container
    # TODO: fix tensorboard
    cmd = clean_cmd(cmd)  # remove newline breaks, extra whitespaces

    image = args.image or f"{DEFAULT_NGC_REGISTRY}:{DEFAULT_CONTAINER_TAG}"
    result = run_cmd(
        ctx=wrap_arguments(""),
        cluster=cluster,
        command=cmd,
        container=image,
        expname=exp_name,
        partition="batch_block1",
        num_gpus=1,
        num_nodes=1,
        # num_tasks=1,
        # log_dir (slurm): `nemo-skills` requires log_dir to be defined as inside the container, different to `exprunner`
        log_dir=str(workspace_mnt / f"exp/{exp_name}"),
        run_after=None,
        dependent_jobs=0,
        mount_paths=",".join(
            [
                f"{CLUSTER_2_HOME_DIR[cluster]}:{workspace_mnt}",
            ]
        ),
        dry_run=args.dry_run,
        exclusive=None,
    )
    status = result.status()
    if status is not None:
        print(f"Launched jobs: {status}")
    else:
        print(f"Launched jobs: {result}")


if __name__ == "__main__":
    main()
