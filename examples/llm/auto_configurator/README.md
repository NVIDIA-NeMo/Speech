> [!IMPORTANT] 
> This is an early version of the Auto Configurator, and the code base can be modified as it will be integrated into the CLI.

Use Auto Configurator to Find the Optimal Configuration
-------------------------------------------------------

Auto Configurator searches for hyperparameters (HPs) that achieve the maximum highest training throughput when working with Large Language Models (LLMs) utilizing the NeMo Framework.

> [!NOTE] 
> Auto Configurator is supported for Bert, T5, and GPT-based models: GPT3, LLama, Mixtral, Mistral, Gemma, Nemotron, Starcoder, and Qwen.

Auto Configurator Capabilities
------------------------------

Auto Configurator is intended to iterate over different model configurations quickly and find the best configuration, that is, the configuration that minimizes both time and financial expenditure. It offers a range of features to facilitate this, as detailed in the list below.

- **Model size recommendation**: finds the optimal model size if the parameter is not specified.
- **Training time estimation**: estimates model training time based on input parameters.
- **Hyperparameters recommendation**: finds the optimal list of hyperparameters to be trained.
- **Optimal configuration recommendation**: calculates the performance after a short training of candidate configurations and finds the optimal model configuration.

Model Size Recommendation
-------------------------

If you have not decided what model size you want to train, Auto Configurator can recommend a model size for your use case. If you know the number of GPUs, TFLOPS per GPU, the maximum time to train, and the number of tokens to train for, it can recommend a model size that can be trained with the specified hardware and time constraints.

For example, if you had 20 NVIDIA DGX nodes available (in 80 GB GPU memory), and wanted to train a GPT model for a maximum of 5 days, Auto Configurator would recommend using a 5B parameter GPT model.

Training Time Estimation
------------------------

Auto Configurator calculates the estimated training time for your model. It provides a projection of the training time in days, based on the input dataset and parameters you provide.

Hyperparameters Recommendation
------------------------------

After Auto Configurator generates the base configuration, it searches over four critical hyperparameters that have a great impact on training throughput but do not affect model convergence. These hyperparameters include  Tensor Parallelism (TP), Pipeline Parallelism (PP), Context Parallelism (CP), Expert Parallelism (EP), Micro Batch Size (MBS), and Activation Checkpointing Layers (ActCkpt). Auto Configurator will also provide optimal Global Batch Size (GBS) if it's not specified.

Auto Configurator initially applies heuristics to identify suitable candidates for the four key parameters, subsequently generating a grid of candidate configurations. It returns all of the candidate configurations in NeMo 2.0 format.
   
> [!NOTE]
> Some of the candidate configurations may not work due to high-memory usage or other issues.

Once the candidate configurations are generated, you can use NeMo Framework to launch the most promising candidates.
   
When running the candidates on the cluster, you can limit job time and job max steps by using ``max_minutes_per_run`` and ``max_steps_per_run`` parameters. During this search, the jobs will run with the number of nodes specified in the configuration files, using the ``num_nodes`` parameter. Once all of the jobs have finished running, you'll need to run compare_throughput.py to get a ``.csv`` table with performance results for each succeeded job.

Optimal Configuration Recommendation
------------------------------------

After all of the candidate jobs are done, Auto Configurator calculates performance parameters for each of the candidates. 
Auto Configurator generates two ``.csv`` files: one detailing the performance measures of the candidates and another listing the candidates that failed due to out-of-memory errors.

End-To-End Example
------------------

The following list shows the required input parameters for the Auto Configurator runner:

- ``recipe``: model recipe based on NeMo 2.0.
- ``path_to_logs``: path to the directory where the logs will be stored.

The following list shows the optional parameters for the Auto Configurator runner:

- ``tensor_parallel_sizes``: a list, such as ``[1, 2, 4]``.
- ``pipeline_parallel_sizes``: a list, such as ``[1, 2, 4]``.
- ``context_parallel_sizes``: a list, such as ``[1, 2, 4]``.
- ``expert_parallel_sizes``: a list, such as ``[1, 2, 4]``.
- ``micro_batch_sizes``: a list, such as ``[1, 2, 4]``.
- ``min_model_parallel_size``: a value for the minimum desired parallelism.
- ``max_model_parallel_size``: a value for the maximum desired parallelism.

For each of the optional parameters, Auto Configurator will find the optimal value if the parameter is not specified. To view the full list of parameters, please visit [this page](https://github.com/NVIDIA/NeMo/blob/dpykhtar/nemo_autoconf/nemo/collections/llm/tools/auto_configurator/runner.py#L51).

To view an end-to-end example of how to generate candidate configs, train them, and calculate the performance using Auto Configurator with NeMo Framework, please visit [this page](https://github.com/NVIDIA/NeMo/blob/dpykhtar/nemo_autoconf/examples/llm/auto_configurator/auto_config.py).

