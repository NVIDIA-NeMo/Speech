# Performance Recipes

- Scripts defined in `scripts/performance` are recipes optimized for performance. These scripts can launch pre-training experiments on Slurm based clusters.
- You will need a virtual environemnt with NeMo and Nemo-Run related dependencies installed as the experiment configuration is resolved before launching it inside NeMo container.

## Example

The following line shows an example of how you can launch a pre-training experiment-

`python -m scripts.llm.performance.llama3_8b --account <your_slurm_account> -partition <your_slurm_partition>`

## Configuration Options

- Slurm account and partition are mandatory arguments for launching the experiment.
- You can use the following optional arguments as needed-
  - -l/--log_dir: Location to store your experiment artifacts and logs.
    - Make sure the environemnt variable `NEMORUN_HOME=<log_dir>` is accessible and set correctly in your virtual environment.
    - You can run `export NEMORUN_HOME=<log_dir>` in your terminal. You can add it your bashrc file (or equivalent for your OS/Linux distro) for setting it permanently.
  - -t/--time_limit: Maximum time limit for your experiment. Your slurm job will be cancelled after this. Default is 30 minutes.
  - -i/--container_image: The NeMo container you want to use. Defaults to latest dev container- 'nvcr.io/nvidia/nemo:dev'.
  - -c/--compute_dtype: Specifies whether you want to use bf16 or fp8 precision for training. Defaults to 'bf16'. You can choose to use 'fp8'.
  - -ep/--enable_profiling: Enable nsys profiling. It is disabled by default. When enabled, profiling will be enabled for 1 step from step 5 to step 6. You can change the step in the respective recipe script.
  - -tb/--tensorboard: Enable tensorboard logging. It is disabled by default.
    - CAUTION: Tensorboard logging may cause performance overhead.
  - -f/--finetuning: Finetuning scheme to use. Options- 'sft', 'lora'. Defaults is 'lora'.
  - -hf/--hf_token: HuggingFace access token. Defaults to None. Required for accessing tokenizers and checkpoints from HuggingFace.
  - -nh/--nemo_home:  Directory where NeMo searches for models and checkpoints. This saves a lot of time (especially for bigger models) if checkpoints already exist here. Missing files will be downloaded from HuggingFace. Defaults to environment variable DEFAULT_NEMO_CACHE_HOME = ~/.cache/nemo
  - -d/--dryrun: Using this argument will not launch the experiment. It will simply print the sbatch script to stdout. This can be helpful to verify you have set your experiment correctly as needed.
- You don't need to set any value for `--enable_profiling`, `--tensorboard` and `--dryrun`. See the below example for reference-
  `python -m scripts.llm.performance.llama3_8b --account <your_slurm_account> -p <your_slurm_partition> -ep --tensorboard -d`
