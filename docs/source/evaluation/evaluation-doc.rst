Evaluate NeMo 2.0 Checkpoints
==============================

This guide provides detailed instructions on evaluating NeMo 2.0 checkpoints using the `NVIDIA Evals Factory
<https://pypi.org/project/nvidia-lm-eval/>`__ within the NeMo Framework. Supported benchmarks include:
    * GPQA
    * GSM8K
    * IFEval
    * MGSM
    * MMLU
    * MMLU-Pro
    * MMLU-Redux
    * Wikilingua


Introduction
--------------

The evaluation process employs a server-client approach, comprising two main phases.
In Phase 1, the NeMo 2.0 checkpoint is deployed in-framework on a PyTriton server by exposing
OpenAI API (OAI) compatible endpoints. Both completions (`v1/completions`) and chat-completions
(`v1/chat/completions`) endpoints are exposed, enabling evaluation on both completion and chat benchmarks.
Phase 2 involves running the evaluation on the model using the OAI endpoint and port.

Some of the benchmarks (e.g. GPQA) use a gated dataset. To use them, you must authenticate to the
`Hugging Face Hub <https://huggingface.co/docs/huggingface_hub/quick-start#authentication>`__
before launching the evaluation.

The NVIDIA Evals Factory provides the following predefined configurations for evaluating the completions endpoint:
    * `gsm8k`
    * `mgsm`
    * `mmlu`
    * `mmlu_pro`
    * `mmlu_redux`

It also provides the following configurations for evaluating the chat endpoint:
    * `gpqa_diamond_cot`
    * `gsm8k_cot_instruct`
    * `ifeval`
    * `mgsm_cot`
    * `mmlu_instruct`
    * `mmlu_pro_instruct`
    * `mmlu_redux_instruct`
    * `wikilingua`

Run Evaluations without NeMo-Run
---------------------------------
This section outlines the steps to deploy and evaluate a NeMo 2.0 model directly using Python commands, without using
NeMo-Run. This method is quick and easy, making it ideal for evaluation on a local workstation with GPUs, as it
facilitates easier debugging. However, for running evaluations on clusters, it is recommended to use NeMo-Run for its
ease of use.

The entry point for deployment is the ``deploy`` method defined in ``nemo/collections/llm/api.py``.
Below is an example command for deployment:

.. code-block:: python

    from nemo.collections.llm import deploy

    if __name__ == "__main__":
        deploy(
            nemo_checkpoint='/workspace/hf_llama3_8b_nemo2.nemo',
            max_input_len=4096,
            max_batch_size=4,
            num_gpus=1,)

The entrypoint for evaluation is the ``evaluate`` method defined in ``nemo/collections/llm/api.py``. To run evaluations
on the deployed model, use the following command. Make sure to open a new terminal within the same container to execute
it. For longer evaluations, it is advisable to run both the deploy and evaluate commands in tmux sessions to prevent
the processes from being killed and aborting the runs.

.. code-block:: python

    from nemo.collections.llm import evaluate
    from nemo.collections.llm.evaluation.api import EvaluationConfig, ApiEndpoint, EvaluationTarget, ConfigParams

    api_endpoint = ApiEndpoint()
    eval_target = EvaluationTarget(api_endpoint=api_endpoint)
    eval_params = ConfigParams(top_p=1, temperature=1, limit_samples=2, parallelism=1)
    eval_config = EvaluationConfig(type='mmlu', params=eval_params)

    if __name__ == "__main__":
        evaluate(target_cfg=eval_target, eval_cfg=eval_config)

.. note::
    Please refer to ``deploy`` and ``evaluate`` method in ``nemo/collections/llm/api.py`` to check all the argument options as these are just sample commands and don't share all arguments and their default settings. For more details on arguments in the ApiEndpoint and ConfigParams classes for evaluation, refer to `nemo/collections/llm/evaluation/api.py <https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/evaluation/api.py>`__.

Run Evaluations with NeMo-Run
------------------------------

This section explains how to run evaluations with NeMo-Run. For detailed information about
`NeMo-Run <https://github.com/NVIDIA/NeMo-Run>`__, please refer to its documentation. Below is a concise guide focused
on using NeMo-Run to perform evaluations in NeMo 2.0.

Launch Evaluations with NeMo-Run
#############################

The `evaluation.py <https://github.com/NVIDIA/NeMo/blob/main/scripts/llm/evaluation.py>`__. script serves as a
reference for launching evaluations with NeMo-Run. This script demonstrates how to use NeMo-Run with both local
executors (your local workstation) and Slurm-based executors like clusters. In this setup, the deploy and evaluate
processes are launched as two separate jobs with NeMo-Run. The evaluate method waits until the PyTriton server is
accessible and the model is deployed before starting the evaluations.

.. note::
    Please make sure to update HF_TOKEN in the nemo-run script `evaluation.py <https://github.com/NVIDIA/NeMo/blob/main/scripts/llm/evaluation.py>`__ with your HF_TOKEN.

Run Locally with NeMo-Run
#########################

To run evaluations on your local workstation, use the following command:

.. code-block:: bash

    python scripts/llm/evaluation.py --nemo_checkpoint '/workspace/hf_llama3_8b_nemo2/' --eval_task 'gsm8k' --devices 2

.. note::
    When running locally with NeMo-Run, you will need to manually terminate the deploy process once evaluations are complete.

Run on Slurm-based Clusters
##########################

To run evaluations on Slurm-based clusters, add the ``--slurm`` flag to your command and specify any custom parameters
such as user, host, remote_job_dir, account, mounts, etc. Refer to the evaluation.py script for further details.
Below is an example command:

.. code-block:: bash

    python scripts/llm/evaluation.py --nemo_checkpoint='/workspace/hf_llama3_8b_nemo2.nemo' --slurm --nodes 1 
    --devices 8 --container_image "nvcr.io/nvidia/nemo:25.04" --tensor_parallelism_size 8

By following these commands, you can successfully run evaluations using NeMo-Run on both local and Slurm-based
environments.



Run Legacy Evaluations with lm-evaluation-harness
-----------------------------------------------------------------------------------------------------------

You can also run evaluations of NeMo 2.0 checkpoints using the integrated `lm-evaluation-harness
<https://github.com/EleutherAI/lm-evaluation-harness>`__ within the NeMo Framework. Supported benchmarks include
``MMLU``, ``GSM8k``, ``lambada_openai``, ``winogrande``, ``arc_challenge``, ``arc_easy``, and ``copa``.
Please note that this path is deprecated and will be removed in the NeMo Framework 25.06 release.

The evaluation process employs a server-client approach, comprising two main phases. In Phase 1, the NeMo 2.0
checkpoint is deployed on a PyTriton server by exporting it to TRT-LLM. Phase 2 involves running the evaluation
on the model using the deployed URL and port.


To deploy a model, use the following command. Make sure to pass ``backend="trtllm"``:

.. code-block:: python

    from nemo.collections.llm import deploy

    if __name__ == "__main__":
        deploy(
            nemo_checkpoint='/workspace/hf_llama3_8b_nemo2.nemo',
            max_input_len=4096,
            max_batch_size=4,
            backend="trtllm",
            num_gpus=1,)


The ``evaluate`` method defined in ``nemo/collections/llm/api.py`` supports the legacy way of evaluating the models.
To run evaluations on the deployed model, use the following command. Make sure to pass the `nemo_checkpoint_path` and
the `url` parameters as they are needed to use the legacy evaluation code. Make sure to open a new terminal within the
same container to execute it. For longer evaluations, it is advisable to run both the deploy and evaluate commands
in tmux sessions to prevent the processes from being killed and aborting the runs.

.. code-block:: python

    from nemo.collections.llm import evaluate
    from nemo.collections.llm.evaluation.api import EvaluationConfig, ApiEndpoint, EvaluationTarget, ConfigParams

    nemo_checkpoint = '/workspace/hf_llama3_8b_nemo2.nemo/'
    api_endpoint = ApiEndpoint(nemo_checkpoint_path=nemo_checkpoint, url="http://0.0.0.0:8000")
    eval_target = EvaluationTarget(api_endpoint=api_endpoint)
    eval_params = ConfigParams(top_p=1, temperature=1, top_k=1, limit_samples=2, num_fewshot=5)
    eval_config = EvaluationConfig(type='mmlu', params=eval_params)

    if __name__ == "__main__":
        evaluate(target_cfg=eval_target, eval_cfg=eval_config)
