.. _drop_layers:

Drop Model Laeyrs
-----------------

To trim the model layers, use the following script:

.. code-block:: bash

  python -m torch.distributed.launch --nproc_per_node=<tensor_model_parallel_size> * <pipeline_model_parallel_size> \
    /NeMo/examples/nlp/language_modeling/megatron_gpt_drop_layers.py \
      --path_to_nemo /path/to/model.nemo \
      --path_to_save /path/to/save/trimmed_model.nemo \
      --tensor_model_parallel_size <tensor_model_parallel_size> \
      --pipeline_model_parallel_size <pipeline_model_parallel_size> \
      --gpus_per_node <gpus_per_node>  \
      --drop_layers 1 2 3 4

**Note:** layer indices start from 1.

To save trimmed model in ``zarr`` checkpoint format, add the following flag to the command above:

.. code-block:: bash

  --zarr

**Note:** the ``zarr`` checkpoint format is deprecated.

Validate Trimmed Model
----------------------

To validate the trimmed model, use the following script:

.. code-block:: bash

  python /NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
    --config-path=/path/to/folder/with/model/config \
    --config-name=model_config.yaml \
    trainer.limit_val_batches=<limit_val_batches> \
    model.restore_from_path=/path/to/trimmed_model.nemo \
    model.skip_train=True \
    model.data.data_impl=mock \
    model.data.data_prefix=[]

To use a specific dataset instead of a mock dataset, modify the ``model.data`` parameters as follows:

.. code-block:: bash

  model.data.data_impl=mmap \
  model.data.data_prefix=["path/to/datafile1", "path/to/datafile2"]

Validate Original Model
-----------------------

To validate the original model without specific layers, use the following script:

.. code-block:: bash

  python /NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
    --config-path=/path/to/folder/with/model/config \
    --config-name=model_config.yaml \
    trainer.limit_val_batches=<limit_val_batches> \
    model.restore_from_path=/path/to/original_model.nemo \
    model.skip_train=True \
    model.data.data_impl=mock \
    model.data.data_prefix=[] \
    model.drop_layers=[1,2,3,4]
