# Performance Recipes

- Scripts defined in `scripts/performance` are recipes optimized for performance. These scripts can launch pre-training experiments on Slurm based clusters.
- You will need a virtual environemnt with NeMo and Nemo-Run related dependencies installed as the experiment configuration is resolved before launching it inside NeMo container.

## Example

The following line shows an example of how you can launch a pre-training experiment-

`python -m scripts.performance.llm.llama3_8b --account <your_slurm_account> -partition <your_slurm_partition>`

## Configuration Options

- Slurm account and partition are mandatory arguments for launching the experiment.
- You can use the following optional arguments as needed-
  - -l/--log_dir: Location to store your experiment artifacts and logs.
    - Make sure the environemnt variable `NEMORUN_HOME=<log_dir>` is accessible and set correctly in your virtual environment.
    - You can run `export NEMORUN_HOME=<log_dir>` in your terminal. You can add it your bashrc file (or equivalent for your OS/Linux distro) for setting it permanently.
  - -t/--time_limit: Maximum time limit for your experiment. Your slurm job will be cancelled after this. Default is 30 minutes.
  - -i/--container_image: The NeMo container you want to use. Defaults to latest dev container- 'nvcr.io/nvidia/nemo:dev'.
  - -c/--compute_dtype: Specifies whether you want to use bf16 or fp8 precision for training. Defaults to 'bf16'. You can choose to use 'fp8'.
  - --fp8_recipe: FP8 recipe. Options: 'ds' (per-tensor delayed scaling), 'cs '(per-tensor current scaling), 'mxfp8' (block-level scaling -- 32 values). Defaults to 'ds'.
  - -en/--enable_nsys: Enable nsys profiling. It is disabled by default. When enabled, profiling will be enabled for 1 step from step 5 to step 6. You can change the step in the respective recipe script.
  - -tb/--tensorboard: Enable tensorboard logging. It is disabled by default.
    - CAUTION: Tensorboard logging may cause performance overhead.
  - -wd/--wandb: Enable wandb logging. Disabled by default.
  - -wdk/--wandb_key: Wandb key. Needed for wandb logger projection to server.
  - -wdp/--wandb_prj_name: Wandb project name.
  - -wdj/--wandb_job_name: Wandb job name.
  - -f/--finetuning: Finetuning scheme to use. Options- 'sft', 'lora'. Defaults is 'lora'.
  - -hf/--hf_token: HuggingFace access token. Defaults to None. Required for accessing tokenizers and checkpoints from HuggingFace.
  - -nh/--nemo_home:  Directory where NeMo searches for models and checkpoints. This saves a lot of time (especially for bigger models) if checkpoints already exist here. Missing files will be downloaded from HuggingFace. Defaults to environment variable DEFAULT_NEMO_CACHE_HOME = ~/.cache/nemo
  - -d/--dryrun: Using this argument will not launch the experiment. It will simply print the sbatch script to stdout. This can be helpful to verify you have set your experiment correctly as needed.
  - -tp/--tensor_parallel_size: Intra-layer model parallelism. Splits tensors across GPU ranks.
  - -pp/--pipeline_parallel_size: Inter-layer model parallelism. Splits transformer layers across GPU ranks.
  - -cp/--context_parallel_size: Splits network input along sequence dimension across GPU ranks.
  - -vp/--virtual_pipeline_parallel_size: Number of virtual blocks per pipeline model parallel rank is the virtual model parallel size.
  - -ep/--expert_parallel_size: Distributes Moe Experts across sub data parallel dimension.
  - -et/--expert_tensor_parallel_size: Intra-layer tensor model parallelsm for expert layer. Splits tensors across GPU ranks.
  - -mb/--micro_batch_size: Micro batch size for training.
  - -gb/--global_batch_size: Global batch size for training.
  - -g/--gpu: Target gpu type. Defaults to 'h100'. Options- 'h100', 'b200', 'gb200'.
  - -ng/--num_gpus: Number of gpus.
  - -gn/--gpus_per_node: Number of gpus per node. Defaults to 8.
  - -ms/--max_steps: Number of train steps. Defaults to 100.
  - -cg/--cuda_graphs: Enable CUDA graphs. Options: 'true', '1', 't', 'yes', 'y' to set it to True, 'false', '0', 'f', 'n', 'no' to set it to False. Defaults to None, in which case the program tries to load default values from recommended model configs, if failed defaults to false.
  - -fsdp/--use_mcore_fsdp: Enable megatron-core FSDP.  Options: 'true', '1', 't', 'yes', 'y' to set it to True, 'false', '0', 'f', 'n', 'no' to set it to False. Defaults to None, in which case the program tries to load default values from recommended model configs, if failed defaults to false.
  - -rl/--recompute_layers: Number of transformer layers to recompute activations during training. Defaults to None, in which case the program tries to load default values from recommended model configs, if failed defaults to 0.
  - -ol/--activation_offload_layers: Number of transformer layers to offload activations to CPU during training. Defaults to None, in which case the program tries to load default values from recommended model configs, if failed defaults to 0.
  - -rm/--recompute_modules: Comma separated string of modules in a transformer layer to recompute. If set, program will use selective recompute for all layers. Users should provide zero, one or more than one values. Options are "core_attn", "moe_act", "layernorm", "mla_up_proj", "mlp", "moe". Defaults to None, in which case the program tries to load default values from recommended model configs, if failed defaults to None, which mean no selective recompute. 
  - -cm/--custom_mounts: Comma separated string of mounts.
- You don't need to set any value for `--enable_nsys`, `--tensorboard`, `--wandb`, and `--dryrun`. See the below example for reference-
  `python -m scripts.performance.llm.llama3_8b --account <your_slurm_account> -p <your_slurm_partition> -en --tensorboard -d`
