RETRO Model
================

The Retrieval-Enhanced Transformer (RETRO) `(Borgeaud et al., 2022) <https://arxiv.org/abs/2112.04426>`_ is an autoregressive decoder-only language model (LM)
pretrained with retrieval-augmentation.
RETRO features practical scalability to support large-scale pretraining from scratch by retrieving from trillions of
tokens.
Pretraining with retrieval provides a more efficient storage mechanism of factual knowledge, when compared to storing factual knowledge implicitly within the network's parameters. This approach significantly reduces the model's parameter count while achieving lower perplexity than the standard GPT model.
RETRO also provides the flexibility to update the
knowledge stored in LMs `(Wang et al., 2023a) <https://arxiv.org/abs/2304.06762>`_
by updating the retrieval database without training LMs again. 

For the legacy native NeMo RETRO model documentation, please see `NeMo RETRO Model (Legacy) <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/nemo_megatron/retro/retro_model_legacy.rst>`_.

Quick Start
************
The following instructions demonstrate how to preprocess the data as well as train and evaluate a RETRO model.

Data Preprocessing
-------------------

For detailed information on data preprocessing, refer to the `Megatron-LM Github <https://github.com/NVIDIA/Megatron-LM/>`_ repository. This repository contains scripts and comprehensive instructions for the entire preprocessing procedure, specifically focusing on `RETRO Data Preparation <https://github.com/NVIDIA/Megatron-LM/blob/0fecd76e995c136021d478c6c52caa57c2f9aa25/tools/retro/build_db.md>`_. The main stages of the process are summarized below. 

The outcome of the preparation step yields a processed RETRO data directory, fully primed for pre-training. Specifically, this directory encompasses the following key files and subdirectories:

* ``config.json``: contains the hyperparameters used in the data preparation step, which will then be retrieved to use in the pre-training step for consistency. For example: sample length, chunk length, data splits, tokenizer files, etc.
* ``data``: contains the original data before any preprocessing.
* ``tokenizer``: contains tokenizer files used in the preparation step.
* ``db``: contains the chunk database of processed and chunked text used for retrieving neighbors. 
* ``index``: contains the Faiss index of the chunk database for retrieval.
* ``query``: contains the queried neighboring chunks for all training samples.


The data preparation process contains the following main stages:

Build Retrieval Chunk Database
##############################

This stage involves creating a database of text chunks from a corpus such as Wikipedia to be used for retrievals. The chunks are non-overlapping and extracted from the original GPT token dataset, with each chunk traditionally being 64 tokens in length. The database is stored as a 2-D array and is not a relational database. 

The main output of this stage is:

* ``/db/merged/train.hdf5``: the database containing all processed and chunked text.
* ``/db/merged/sampled.hdf5``: the database containing a small portion of all chunks, only used for training the index in the next stage.

Build Index for Similarity Search
#################################

The second stage is to build a search index using Faiss, a library for efficient similarity search. The index is trained on a subset of the chunks ``sampled.hdf5`` from the database. After training, all chunks are added to the index to enable querying. The index accepts 1-D floating point vectors, so chunks must be embedded using Bert embeddings before they can be added to the index. Particularly, the stage is comprised of two sub-stages:

    \- Extract BERT embeddings from the sampled chunk database (``sampled.hdf5``) and use them to train a Faiss index.

    \- Extract BERT embeddings for each chunk in the all chunks database (``train.hdf5``) and add them to the trained Faiss index.

The main output of this stage is:

* ``/index/<RETRO_INDEX_TYPE>/<RETRO_INDEX_STR>/added.faissindex``: the trained index, with all chunks in the database added to it

Query Pretraining Neighbors
###########################

To speed up the RETRO pretraining process, you pre-retrieve neighbors for all training samples instead of retrieving them on-the-fly. In this stage, the pretraining datasets are processed to find and save k-nearest neighbors for each chunk in each sample. The neighbors are saved to disk and labeled with unique properties to ensure they match the pretraining configuration. Query-time hyperparameters can be tuned to improve the quality of the neighbors.

The main output of this stage is:

* ``train_<UNIQUE_HASH>``: directory containing retrieved neighbors for all training samples.
* ``valid_<UNIQUE_HASH>``: directory containing retrieved neighbors for all validating samples.



Train RETRO Model
-----------------------

Once the training samples, pre-retrieved neighbors, and other data are prepared, you are ready to train the RETRO model. The training process will use the output directory from the data preparation step. We set the path to this directory at the ``retro.retro_project_dir`` argument. Many of the data hyperparameters will be retrieved from the ``config.json`` file in this directory, including data splits, sequence length, chunk length, number of training and validating samples, tokenizer, etc.

The table below lists some of the common architecture and optimizer parameters that can be configured for model pre-training. Many of these values are set in ``examples/nlp/language_modeling/conf/megatron_retro_config.yaml``, which is used when training unless being overriden by the running command. Notice unlike other NeMo models, the `model.data.data_prefix` value is set to None, because all data information will be retrieved from `model.retro.retro_project_dir`.

+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| **Parameter**                    | **Default** | **Description**                                                                        |
+==================================+=============+========================================================================================+
| retro_data.retro_chunk_length    | 64          | the chunk size used to retrieve                                                        |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| retro.retro_num_neighbors        | 2           | token sequence length                                                                  |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| retro_encoder_num_layers         | 2           | total number of encoder layers                                                         |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.num_layers                 | 12          | total number of decoder layers                                                         |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.encoder_seq_length         | 2048        | token sequence length                                                                  |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.hidden_size                | 768         | model hidden size                                                                      |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.ffn_hidden_size            | 3072        | model FFN hidden size. Usually 4 * hidden_size                                         |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.num_attention_heads        | 12          | number of attention heads                                                              |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.init_method_std            | 0.023       | standard deviation of the zero mean normal distribution used for weight initialization |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.hidden_dropout             | 0.1         | dropout probability for hidden state transformer                                       |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.attention_dropout          | 0.1         | dropout probability in the attention layer                                             |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.ffn_dropout                | 0.1         | dropout probability in the feed-forward layer                                          |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+

The following example shows a RETRO pre-training script. The rest of the argument values are retrieved from ``examples/nlp/language_modeling/conf/megatron_retro_config.yaml``.

.. code-block:: bash

        python /examples/nlp/language_modeling/megatron_retro_pretraining.py \
            trainer.num_nodes=1 \
            trainer.devices=8 \
            trainer.precision=bf16 \
            trainer.accelerator=gpu \
            trainer.max_steps=750000
            trainer.val_check_interval=10 \
            trainer.precision=16 \
            exp_manager.exp_dir=/path/to/exp_dir \
            model.mcore_gpt=True \
            model.tensor_model_parallel_size=1 \
            model.pipeline_model_parallel_size=1 \
            model.megatron_amp_O2=True \
            model.retro.num_layers=12 \
            model.retro.retro_encoder_num_layers=2 \
            model.retro.retro_num_retrieved_chunks=2 \
            model.retro.retro_project_dir=/path/to/retro_workdir \
            model.micro_batch_size=4 \
            model.data.num_workers=4 \
            model.data.data_prefix=["none"] \
            model.data.shuffle_documents=False \
            model.data.dataloader_type=single \
            model.data.splits_string=\'98,2,0\' \
            model.optim.lr=6.0e-4 \
            model.optim.weight_decay=0.1 \
            model.optim.sched.name=CosineAnnealing \
            model.optim.sched.min_lr=6.0e-5 \
            model.optim.sched.max_steps=650000 \
            model.optim.name=distributed_fused_adam

During the training, we can monitor the process with Weights and Biases (WandB) by setting ``exp_manager.create_wandb_logger=True`` and set relevant wandb arguments.
After training, the model distributed checkpoint directory can be found at the result checkpoint directory.

Run RETRO Model Inference
-------------------------------

Once the RETRO model has been trained, you can put it into inference mode and experiment with it. 
During inference, you are not limited to the indexed corpus to retrieve relevant chunks, but can directly provide any relevant contexts to the prompt through the argument ``neighbors``.
When performing inference, the input for RETRO differs from that used during training structurally. Specifically, the model’s input consists of only two chunks: one for the prompt and another for the answer to be generated. Unlike during training, these chunks do not necessarily have a fixed length of 64 tokens; instead, they match the length of the tokenized prompt. When context neighbors are supplied for a prompt, these neighbors correspond to the first chunk and are processed through the RETRO encoder to generate text for the second chunk.
The following example shows a RETRO inferencing script. The rest of the argument values are retrieved from ``examples/nlp/language_modeling/conf/megatron_retro_inference.yaml``.

.. code-block:: bash

        python /examples/nlp/language_modeling/megatron_retro_eval.py \
            checkpoint_dir=/path/to/checkpoints \
            checkpoint_name=/checkpoint_name \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            trainer.accelerator=gpu \
            trainer.precision=32 \
            megatron_amp_O2=False \
            inference.tokens_to_generate=10 \
            inference.greedy=False \
            inference.add_BOS=False \
            inference.temperature=1.0 \
            inference.retro_inference.retro_num_neighbors=2 \
            prompt="sample prompt" \
            neighbors=["sample neighbor 1","sample neighbor 2"]
