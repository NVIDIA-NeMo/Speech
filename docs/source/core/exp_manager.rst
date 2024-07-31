
.. _exp-manager-label:

Experiment Manager
==================

NeMo's Experiment Manager leverages PyTorch Lightning for model checkpointing, TensorBoard Logging, Weights and Biases, DLLogger and MLFlow logging. The
Experiment Manager is included by default in all NeMo example scripts.

To use the experiment manager simply call :class:`~nemo.utils.exp_manager.exp_manager` and pass in the PyTorch Lightning ``Trainer``.

.. code-block:: python

    exp_dir = exp_manager(trainer, cfg.get("exp_manager", None))

And is configurable via YAML with Hydra.

.. code-block:: bash

    exp_manager:
        exp_dir: /path/to/my/experiments
        name: my_experiment_name
        create_tensorboard_logger: True
        create_checkpoint_callback: True

Optionally, launch TensorBoard to view the training results in ``./nemo_experiments`` (by default).

.. code-block:: bash

    tensorboard --bind_all --logdir nemo_experiments

..

If ``create_checkpoint_callback`` is set to ``True``, then NeMo automatically creates checkpoints during training
using PyTorch Lightning's `ModelCheckpoint <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html>`_.
We can configure the ``ModelCheckpoint`` via YAML or CLI.

.. code-block:: yaml

    exp_manager:
        ...
        # configure the PyTorch Lightning ModelCheckpoint using checkpoint_call_back_params
        # any ModelCheckpoint argument can be set here

        # save the best checkpoints based on this metric
        checkpoint_callback_params.monitor=val_loss

        # choose how many total checkpoints to save
        checkpoint_callback_params.save_top_k=5

Resume Training
---------------

We can auto-resume training as well by configuring the ``exp_manager``. Being able to auto-resume is important when doing long training
runs that are premptible or may be shut down before the training procedure has completed. To auto-resume training, set the following
via YAML or CLI:

.. code-block:: yaml

    exp_manager:
        ...
        # resume training if checkpoints already exist
        resume_if_exists: True

        # to start training with no existing checkpoints
        resume_ignore_no_checkpoint: True

        # by default experiments will be versioned by datetime
        # we can set our own version with
        exp_manager.version: my_experiment_version


Experiment Loggers
------------------

Alongside Tensorboard, NeMo also supports Weights and Biases, MLFlow, DLLogger, ClearML and NeptuneLogger. To use these loggers, simply set the following
via YAML or :class:`~nemo.utils.exp_manager.ExpManagerConfig`.


Weights and Biases (WandB)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _exp_manager_weights_biases-label:

.. code-block:: yaml

    exp_manager:
        ...
        create_checkpoint_callback: True
        create_wandb_logger: True
        wandb_logger_kwargs:
            name: ${name}
            project: ${project}
            entity: ${entity}
            <Add any other arguments supported by WandB logger here>


MLFlow
~~~~~~

.. _exp_manager_mlflow-label:

.. code-block:: yaml

    exp_manager:
        ...
        create_checkpoint_callback: True
        create_mlflow_logger: True
        mlflow_logger_kwargs:
            experiment_name: ${name}
            tags:
                <Any key:value pairs>
            save_dir: './mlruns'
            prefix: ''
            artifact_location: None
            # provide run_id if resuming a previously started run
            run_id: Optional[str] = None

DLLogger
~~~~~~~~

.. _exp_manager_dllogger-label:

.. code-block:: yaml

    exp_manager:
        ...
        create_checkpoint_callback: True
        create_dllogger_logger: True
        dllogger_logger_kwargs:
            verbose: False
            stdout: False
            json_file: "./dllogger.json"

ClearML
~~~~~~~

.. _exp_manager_clearml-label:

.. code-block:: yaml

    exp_manager:
        ...
        create_checkpoint_callback: True
        create_clearml_logger: True
        clearml_logger_kwargs:
            project: None  # name of the project
            task: None  # optional name of task
            connect_pytorch: False
            model_name: None  # optional name of model
            tags: None  # Should be a list of str
            log_model: False  # log model to clearml server
            log_cfg: False  # log config to clearml server
            log_metrics: False  # log metrics to clearml server

Neptune
~~~~~~~

.. _exp_manager_neptune-label:

.. code-block:: yaml

    exp_manager:
        ...
        create_checkpoint_callback: True
        create_neptune_logger: false
        neptune_logger_kwargs:
            project: ${project}
            name: ${name}
            prefix: train
            log_model_checkpoints: false # set to True if checkpoints need to be pushed to Neptune
            tags: null # can specify as an array of strings in yaml array format
            description: null
            <Add any other arguments supported by Neptune logger here>

Exponential Moving Average
--------------------------

.. _exp_manager_ema-label:

NeMo supports using exponential moving average (EMA) for model parameters. This can be useful for improving model generalization
and stability. To use EMA, simply set the following via YAML or :class:`~nemo.utils.exp_manager.ExpManagerConfig`.

.. code-block:: yaml

    exp_manager:
        ...
        # use exponential moving average for model parameters
        ema:
            enabled: True  # False by default
            decay: 0.999  # decay rate
            cpu_offload: False  # If EMA parameters should be offloaded to CPU to save GPU memory
            every_n_steps: 1  # How often to update EMA weights
            validate_original_weights: False  # Whether to use original weights for validation calculation or EMA weights

Support for Preemption
----------------------

.. _exp_manager_preemption_support-label:

NeMo adds support for a callback upon preemption while running the models on clusters. The callback takes care of saving the current state of training via the ``.ckpt``
file followed by a graceful exit from the run. The checkpoint saved upon preemption has the ``*last.ckpt`` suffix and replaces the previously saved last checkpoints.
This feature is useful to increase utilization on clusters.
The ``PreemptionCallback`` is enabled by default. To disable it simply add ``create_preemption_callback: False`` under exp_manager in the config YAML file. 

Stragglers Detection
----------------------

.. _exp_manager_straggler_det_support-label:

.. note::
    Stragglers Detection feature is included in the optional NeMo resiliency package.

Distributed training can be affected by stragglers, which are slow workers that slow down the overall training process. 
NeMo provides a straggler detection feature that can identify slower GPUs.

This feature is implemented in the ``StragglerDetectionCallback``, which is disabled by default.

The callback computes normalized GPU performance scores, which are scalar values ranging from 0.0 (worst) to 1.0 (best). 
A performance score can be interpreted as the ratio of current performance to reference performance.

There are two types of performance scores provided by the callback:
    - Relative GPU performance score: The best-performing GPU in the workload is used as a reference.
    - Individual GPU performance score: The best historical performance of the GPU is used as a reference.

Examples:
    - If the relative performance score is 0.5, it means that a GPU is twice slower than the fastest GPU.
    - If the individual performance score is 0.5, it means that a GPU is twice slower than its best observed performance.

If a GPU performance score drops below the specified threshold, it is identified as a straggler.

To enable straggler detection, add ``create_straggler_detection_callback: True`` under exp_manager in the config YAML file. 
You might also want to adjust the callback parameters:

.. code-block:: yaml

    exp_manager:
        ...
        create_straggler_detection_callback: True
        straggler_detection_callback_params:
            report_time_interval: 300      # Interval [seconds] of the straggler check
            calc_relative_gpu_perf: True   # Calculate relative GPU performance
            calc_individual_gpu_perf: True # Calculate individual GPU performance
            num_gpu_perf_scores_to_log: 5       # Log 5 best and 5 worst GPU performance scores, even if no stragglers are detected
            gpu_relative_perf_threshold: 0.7    # Threshold for relative GPU performance scores
            gpu_individual_perf_threshold: 0.7  # Threshold for individual GPU performance scores
            stop_if_detected: True              # Terminate the workload if stragglers are detected

Straggler detection might involve inter-rank synchronization, and should be invoked with reasonable frequency (e.g. every few minutes).

Fault Tolerance
---------------

.. _exp_manager_fault_tolerance_support-label:

.. note::
    Fault Tolerance feature is included in the optional NeMo resiliency package.

When training DNN models, faults may occur, hindering the progress of the entire training process. 
This is particularly common in distributed, multi-node training scenarios, with many nodes and GPUs involved. 

NeMo incorporates a fault tolerance mechanism to detect training halts. 
In response, it can terminate a hung workload and, if requested, restart it from the last checkpoint.

Fault tolerance ("FT") relies on a special launcher (``ft_launcher``), which is a modified ``torchrun``. 
The FT launcher runs background processes called rank monitors. **You need to use ft_launcher to start 
your workload if you are using FT**. I.e., `NeMo-Framework-Launcher <https://github.com/NVIDIA/NeMo-Framework-Launcher>`_  
can be used to generate SLURM batch scripts with FT support. 

Each training process (rank) sends `heartbeats` to its monitor during training and validation steps.
If a rank monitor stops receiving `heartbeats`, a training failure is detected.

Fault detection is implemented in the ``FaultToleranceCallback`` and is disabled by default. 
To enable it, add a ``create_fault_tolerance_callback: True`` option under ``exp_manager`` in the 
config YAML file. Additionally, you can customize FT parameters by adding ``fault_tolerance`` section:

.. code-block:: yaml

    exp_manager:
        ...
        create_fault_tolerance_callback: True
        fault_tolerance:
            initial_rank_heartbeat_timeout: 600  # wait for 10 minutes for the initial heartbeat
            rank_heartbeat_timeout: 300  # wait for 5 minutes for subsequent heartbeats
            calculate_timeouts: True # estimate more accurate timeouts based on observed intervals

Timeouts for fault detection need to be adjusted for a given workload:
    * ``initial_rank_heartbeat_timeout`` should be long enough to allow for workload initialization.
    * ``rank_heartbeat_timeout`` should be at least as long as the longest possible interval between steps. 

**Importantly, `heartbeats` are not sent during checkpoint loading and saving**, so time for 
checkpointing related operations should be taken into account.

If ``calculate_timeouts: True`` timeouts will be automatically estimated based on observed intervals. 
Estimated timeouts take precedence over timeouts defined in the config file. **Timeouts are estimated 
at the end of a training run, when checkpoint loading and saving were observed**. Hence, in a multi-part 
training started from scratch, estimated timeouts won't be available during initial two runs. 
Estimated timeouts are stored in a separate JSON file. 

``max_subsequent_job_failures`` allows for the automatic continuation of training on a SLURM cluster. 
This feature requires SLURM job to be scheduled with ``NeMo-Framework-Launcher``. If ``max_subsequent_job_failures`` 
value is `>0` continuation job is prescheduled. It will continue  the work until ``max_subsequent_job_failures`` 
subsequent jobs failed (SLURM job exit code is `!= 0`) or the training is completed successfully 
("end of training" marker file is produced by the ``FaultToleranceCallback``, i.e. due to iters or time limit reached).

All FT configuration items summary:
    * ``workload_check_interval`` (float, default=5.0) Periodic workload check interval [seconds] in the workload monitor.
    * ``initial_rank_heartbeat_timeout`` (Optional[float], default=60.0 * 60.0) Timeout [seconds] for the first heartbeat from a rank. 
    * ``rank_heartbeat_timeout`` (Optional[float], default=45.0 * 60.0) Timeout [seconds] for subsequent heartbeats from a rank. 
    * ``calculate_timeouts`` (bool, default=True) Try to calculate ``rank_heartbeat_timeout`` and ``initial_rank_heartbeat_timeout`` 
      based on the observed heartbeat intervals.
    * ``safety_factor``: (float, default=5.0) When calculating the timeouts, multiply the maximum observed heartbeat interval 
      by this factor to obtain the timeout estimate. Can be made smaller for stable environments and larger for unstable ones.  
    * ``rank_termination_signal`` (signal.Signals, default=signal.SIGKILL) Signal used to terminate the rank when failure is detected.
    * ``log_level`` (str, default='INFO') Log level for the FT client and server(rank monitor).
    * ``max_rank_restarts`` (int, default=0) Used by FT launcher. Max number of restarts for a rank. 
      If ``>0`` ranks will be restarted on existing nodes in case of a failure.
    * ``max_subsequent_job_failures`` (int, default=0) Used by FT launcher. How many subsequent job failures are allowed until stopping autoresuming. 
      ``0`` means do not autoresume.
    * ``additional_ft_launcher_args`` (str, default='') Additional FT launcher params (for advanced use).


.. _nemo_multirun-label:
Hydra Multi-Run with NeMo
-------------------------

When training neural networks, it is common to perform hyper parameter search in order to improve the performance of a model
on some validation data. However, it can be tedious to manually prepare a grid of experiments and management of all checkpoints
and their metrics. In order to simplify such tasks, NeMo integrates with `Hydra Multi-Run support <https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/>`_ in order to provide a unified way to run a set of experiments all
from the config.

There are certain limitations to this framework, which we list below:

* All experiments are assumed to be run on a single GPU, and multi GPU for single run (model parallel models are not supported as of now).
* NeMo Multi-Run supports only grid search over a set of hyper-parameters, but we will eventually add support for advanced hyper parameter search strategies.
* **NeMo Multi-Run only supports running on one or more GPUs** and will not work if no GPU devices are present.

Config Setup
~~~~~~~~~~~~

In order to enable NeMo Multi-Run, we first update our YAML configs with some information to let Hydra know we expect to run multiple experiments from this one config -

.. code-block:: yaml

    # Required for Hydra launch of hyperparameter search via multirun
    defaults:
      - override hydra/launcher: nemo_launcher

    # Hydra arguments necessary for hyperparameter optimization
    hydra:
      # Helper arguments to ensure all hyper parameter runs are from the directory that launches the script.
      sweep:
        dir: "."
        subdir: "."

      # Define all the hyper parameters here
      sweeper:
        params:
          # Place all the parameters you wish to search over here (corresponding to the rest of the config)
          # NOTE: Make sure that there are no spaces between the commas that separate the config params !
          model.optim.lr: 0.001,0.0001
          model.encoder.dim: 32,64,96,128
          model.decoder.dropout: 0.0,0.1,0.2

      # Arguments to the process launcher
      launcher:
        num_gpus: -1  # Number of gpus to use. Each run works on a single GPU.
        jobs_per_gpu: 1  # If each GPU has large memory, you can run multiple jobs on the same GPU for faster results (until OOM).


Next, we will setup the config for ``Experiment Manager``. When we perform hyper parameter search, each run may take some time to complete.
We want to therefore avoid the case where a run ends (say due to OOM or timeout on the machine) and we need to redo all experiments.
We therefore setup the experiment manager config such that every experiment has a unique "key", whose value corresponds to a single
resumable experiment.

Let us see how to setup such a unique "key" via the experiment name. Simply attach all the hyper parameter arguments to the experiment
name as shown below -

.. code-block:: yaml

    exp_manager:
      exp_dir: null  # Can be set by the user.

      # Add a unique name for all hyper parameter arguments to allow continued training.
      # NOTE: It is necessary to add all hyperparameter arguments to the name !
      # This ensures successful restoration of model runs in case HP search crashes.
      name: ${name}-lr-${model.optim.lr}-adim-${model.adapter.dim}-sd-${model.adapter.adapter_strategy.stochastic_depth}

      ...
      checkpoint_callback_params:
        ...
        save_top_k: 1  # Dont save too many .ckpt files during HP search
        always_save_nemo: True # saves the checkpoints as nemo files for fast checking of results later
      ...

      # We highly recommend use of any experiment tracking took to gather all the experiments in one location
      create_wandb_logger: True
      wandb_logger_kwargs:
        project: "<Add some project name here>"

      # HP Search may crash due to various reasons, best to attempt continuation in order to
      # resume from where the last failure case occurred.
      resume_if_exists: true
      resume_ignore_no_checkpoint: true


Running a Multi-Run config
~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the config has been updated, we can now run it just like any normal Hydra script -- with one special flag (``-m``) !

.. code-block:: bash

    python script.py --config-path=ABC --config-name=XYZ -m \
        trainer.max_steps=5000 \  # Any additional arg after -m will be passed to all the runs generated from the config !
        ...

Tips and Tricks
~~~~~~~~~~~~~~~

* Preserving disk space for large number of experiments

Some models may have a large number of parameters, and it may be very expensive to save a large number of checkpoints on
physical storage drives. For example, if you use Adam optimizer, each PyTorch Lightning ".ckpt" file will actually be 3x the
size of just the model parameters - per ckpt file ! This can be exhorbitant if you have multiple runs.

In the above config, we explicitly set ``save_top_k: 1`` and ``always_save_nemo: True`` - what this does is limit the number of
ckpt files to just 1, and also save a NeMo file (which will contain just the model parameters without optimizer state) and
can be restored immediately for further work.

We can further reduce the storage space by utilizing some utility functions of NeMo to automatically delete either
ckpt or NeMo files after a training run has finished. This is sufficient in case you are collecting results in some experiment
tracking tool and can simply rerun the best config after the search is finished.

.. code-block:: python

    # Import `clean_exp_ckpt` along with exp_manager
    from nemo.utils.exp_manager import clean_exp_ckpt, exp_manager

    @hydra_runner(...)
    def main(cfg):
        ...

        # Keep track of the experiment directory
        exp_log_dir = exp_manager(trainer, cfg.get("exp_manager", None))

        ... add any training code here as needed ...

        # Add following line to end of the training script
        # Remove PTL ckpt file, and potentially also remove .nemo file to conserve storage space.
        clean_exp_ckpt(exp_log_dir, remove_ckpt=True, remove_nemo=False)


* Debugging Multi-Run Scripts

When running hydra scripts, you may sometimes face config issues which crash the program. In NeMo Multi-Run, a crash in
any one run will **not** crash the entire program, we will simply take note of it and move onto the next job. Once all
jobs are completed, we then raise the error in the order that it occurred (it will crash the program with the first error's
stack trace).

In order to debug Muti-Run, we suggest to comment out the full hyper parameter config set inside ``sweep.params``
and instead run just a single experiment with the config - which would immediately raise the error.


* Experiment name cannot be parsed by Hydra

Sometimes our hyper parameters include PyTorch Lightning ``trainer`` arguments - such as number of steps, number of epochs
whether to use gradient accumulation or not etc. When we attempt to add these as keys to the expriment manager's ``name``,
Hydra may complain that ``trainer.xyz`` cannot be resolved.

A simple solution is to finalize the hydra config before you call ``exp_manager()`` as follows -

.. code-block:: python

    @hydra_runner(...)
    def main(cfg):
        # Make any changes as necessary to the config
        cfg.xyz.abc = uvw

        # Finalize the config
        cfg = OmegaConf.resolve(cfg)

        # Carry on as normal by calling trainer and exp_manager
        trainer = pl.Trainer(**cfg.trainer)
        exp_log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
        ...


ExpManagerConfig
----------------

.. autoclass:: nemo.utils.exp_manager.ExpManagerConfig
    :show-inheritance:
    :members:
    :member-order: bysource
    :no-index:
