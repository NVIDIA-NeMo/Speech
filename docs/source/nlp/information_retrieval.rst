.. _information_retrieval:

BERT Embedding Models
=====================

Sentence-BERT (SBERT) is a modification of the BERT model that is specifically trained to generate semantically meaningful sentence embeddings. 
The model architecture and pre-training process are detailed in the `Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks <https://aclanthology.org/D19-1410.pdf>`__ paper. Similar to BERT, 
Sentence-BERT utilizes a BERT-based architecture, but it is trained using a Siamese and triplet network structure to derive fixed-sized sentence embeddings that capture semantic information. 
Sentence-BERT is commonly used to generate high-quality sentence embeddings for various downstream natural language processing tasks, such as semantic textual similarity, clustering, and information retrieval

Data Input for the Sentence-BERT model
--------------------------------------

The fine-tuning data for the Sentence-BERT (SBERT) model should consist of data instances, 
each comprising a query, a positive document, and a list of negative documents. Negative mining is 
not supported in NeMo yet; therefore, data preprocessing should be performed offline before training. 
The dataset should be in JSON format. For instance, the dataset should have the following structure:

.. code-block:: python

    [
        {
            "query": "Query",
            "pos_doc": "Positive",
            "neg_doc": ["Negative_1", "Negative_2", ..., "Negative_n"]
        },
        {
            // Next data instance
        },
        ...,
        {
            // Subsequent data instance
        }
    ]

This format ensures that the fine-tuning data is appropriately structured for training the Sentence-BERT model.


Fine-tuning the Sentence-BERT model
-----------------------------------

For fine-tuning Sentence-BERT model, you need to initialize the Sentence-BERT model with BERT model
checkpoint. To do so, you should either have a ``.nemo`` checkpoint or need to convert a HuggingFace
BERT checkpoint to NeMo (mcore) using the following:

.. code-block:: python

     python NeMo/scripts/nlp_language_modeling/convert_bert_hf_to_nemo.py \
            --input_name_or_path "intfloat/e5-large-unsupervised" \
            --output_path /path/to/output/nemo/file.nemo \
            --mcore True \
            --precision 32

Then you can fine-tune the sentence-BERT model using the following script:

.. code-block:: python


    #!/bin/bash

    PROJECT= # wandb project name
    NAME= # wandb run name
    export WANDB_API_KEY= # your_wandb_key

    NUM_DEVICES=1 # number of gpus to train on
    CONFIG_PATH="/NeMo/examples/nlp/information_retrieval/conf/"
    CONFIG_NAME="megatron_bert_embedding_config"
    PATH_TO_NEMO_MODEL= # Path to conveted nemo model from hf
    TRAIN_DATASET_PATH= # Path to json dataset 
    VALIDATION_DATASET_PATH= # Path to validation dataset 
    SAVE_DIR= # where the checkpoint and logs are saved
    mkdir -p $SAVE_DIR
    export NVTE_FLASH_ATTN=0
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
    export NVTE_FUSED_ATTN=0
    
    python NeMo/examples/nlp/information_retrieval/megatron_bert_embedding_finetuning.py \
    --config-path=${CONFIG_PATH} \
    --config-name=${CONFIG_NAME} \
    restore_from_path=${PATH_TO_NEMO_MODEL} \
    trainer.devices=${NUM_DEVICES} \
    trainer.max_steps=10000 \
    trainer.val_check_interval=100 \
    trainer.max_epochs=1 \
    +trainer.num_sanity_val_steps=0 \
    model.mcore_bert=True \
    model.post_process=False \
    model.global_batch_size=8 \ # should be NUM_DEVICES * model.micro_batch_size
    model.micro_batch_size=8 \
    model.optim.lr=0.000005 \
    model.optim.sched.min_lr=0.00000001 \
    model.optim.sched.warmup_steps=100 \
    model.encoder_seq_length=512 \
    model.tokenizer.library="huggingface" \
    model.tokenizer.type="intfloat/e5-large-unsupervised" \
    model.data.data_train=${TRAIN_DATASET_PATH} \
    model.data.data_validation=${VALIDATION_DATASET_PATH} \
    model.data.hard_negatives_to_train=4 \
    exp_manager.explicit_log_dir=${SAVE_DIR} \
    exp_manager.create_wandb_logger=True \
    exp_manager.resume_if_exists=True \
    exp_manager.wandb_logger_kwargs.name=${NAME} \
    exp_manager.wandb_logger_kwargs.project=${PROJECT}
    
