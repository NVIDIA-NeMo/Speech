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

.. code-block:: bash


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
    
GPT Embedding Models
=====================

Recent work has also shown that it is possible to use Decoder-Only (GPT Style) models to train embedding models.
`Improving Text Embeddings with
Large Language Models <https://arxiv.org/pdf/2401.00368.pdf>`__ is one such recent papers which served as inspiration to implement Decoder-only embedding training in Nemo.

Training a GPT Embedding Model
-------------------------------

To train GPT Embedding models we follow a format very similar to the SBERT Embedding training. However, there are a couple of differences. GPT Embedding model training expects a `jsonl` file in which each line is a json object. Here is a truncated example of data jsonl file::

{"query": "What did ... 1952-2002 period?", "pos_doc": "Morning (2008) ... has changed little.", "neg_doc": "Even though ... sapiens.", "query_id": "q103151", "doc_id": "d14755"}
{"query": "What type of ...  passions?", "pos_doc": "Burke was a leading ... upper classes.", "neg_doc": "Writing to a friend ... Government.", "query_id": "q77959", "doc_id": "d11263"}
{"query": "Since 1999, ... progressed at?", "pos_doc": "Commercial solar water ... as of 2007.", "neg_doc": "The potential solar ... acquire.", "query_id": "q16545", "doc_id": "d1883"}


As visible the json object should contain the following fields ``query``, ``pos_doc``, ``neg_doc``, ``query_id`` and ``doc_id``. The ``query_id`` and ``doc_id`` can be any alphanumeric string that uniquely maps to the ``query`` string and ``pos_doc`` string.

During training, the GPT Embedding model employs LoRA (by default) to learn embeddings for the queries and documents, such that similarity of the ``query``-to-``pos_doc`` are maximized while simultaneously minimizing ``query``-to-``neg_doc`` similarity. LoRA allows us to fine-tune large LLMs such as Mistral 7B model with a relatively small number of training parameters.

An example command to launch a training job is

.. code-block:: console

 python3 /NeMo/examples/nlp/information_retrieval/megatron_gpt_embedding_finetuning.py \
    exp_manager.exp_dir="PATH_TO_SAVE_LORA_WEIGHTS" \
    model.global_batch_size=4 \                         # exact choice for global batch size is data dependent typical values are in the range of 32 to 128.
    model.micro_batch_size=4 \                          # exact choice for micro batch size is GPU memory dependent 2 to 8 are reasonable values.
    trainer.devices=1 \                                 # indicates how many GPUs to use during training per node.
    trainer.num_nodes=1 \                               # indicates how many nodes to use if multi-node cluster is available
    trainer.max_steps=20 \                              # how many training steps to run.
    model.restore_from_path="PATH_TO_BASE_NEMO_MODEL" \
    model.peft.lora_tuning.adapter_dim=16 \             # the low-rank size for lora weights.
    model.data.train_ds.file_names=["train.jsonl"]

The full list of possible run arguments is configurable in ``/examples/nlp/information_retrieval/conf/megatron_gpt_embedder_tuning_config.yaml``. By default a trained model file should be generated in here ``PATH_TO_SAVE_LORA_WEIGHTS/megatron_gpt_peft_lora_tuning/checkpoints/`` typically with the extension ``.nemo``.


Inference using a GPT Embedding Model
-------------------------------------

Once trained, the GPT Embedding Model can be used to generate embeddings for queries and corpus documents. We can launch inference using the following command:

.. code-block:: console

 python3 /NeMo/examples/nlp/information_retrieval/megatron_gpt_embedding_generate.py \
    model.global_batch_size=4 \
    model.micro_batch_size=4 \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    model.restore_from_path="PATH_TO_BASE_NEMO_MODEL" \  # Same base model used at training time. 
    model.peft.restore_from_path="PATH_TO_SAVE_LORA_WEIGHTS/megatron_gpt_peft_lora_tuning/checkpoints//megatron_gpt_peft_lora_tuning.nemo" \ 
    model.data.test_ds.query_file_names=["test_query.jsonl"] \
    model.data.test_ds.doc_file_names=\["test_docs.jsonl"] \
    model.data.test_ds.write_embeddings_to_file=True \
    model.data.test_ds.output_file_path_prefix="PATH_TO_SAVE_EMEBDDINGS" 

The contents of ``test_queries.jsonl`` is expected to be in the following format::

{"query": "What do ... quantities?","query_id": "q11600", "doc_id": "d1172"}
{"query": "What are ... subsectors?", "query_id": "q5831", "doc_id": "d577"}
{"query": "Which article ... Government?", "query_id": "q3037", "doc_id": "d336"}

Here, the ``doc_id`` field is expected to be the id of the document/passage which is the correct passage for the query. Note that since we are in inference mode, we don't require query-doc pairs.

The contents of ``test_docs.jsonl`` is expected to be in the following format::

{"pos_doc": "Hormones ... vitamin D.", "doc_id": "d823"}
{"pos_doc": "Historically, Victoria ... October 2016.", "doc_id": "d159"}
{"pos_doc": "Exceptional examples ... Warsaw.", "doc_id": "d1084"}

Once again, we show 3 examples form each file. Typically the ``test_docs.jsonl`` will contain more items than queries in the ``test_queries.jsonl``.

The inference command will result in two folders 

* ``PATH_TO_SAVE_EMBEDDINGS/consumed_samplesX/test_queries`` 
* ``PATH_TO_SAVE_EMBEDDINGS/consumed_samplesX/test_docs``

The ``X`` in the folder ``consumed_samplesX`` is a number denoted number of batches consumed, this is not crucial at test time, but it is useful in training which we will see in the next section. First, let's take a look at the ``test_queries``.

.. code-block:: console

 $> ls PATH_TO_SAVE_EMBEDDINGS/consumed_samplesX/test_queries
 query.ids  query.npy
 $>head -n3 PATH_TO_SAVE_EMBEDDINGS/consumed_samplesX/test_queries/query.ids 
 q11600
 q5831
 q3037

``query.npy`` is a numpy pickled array containing rows of query embeddings and the ``query.ids`` text file list the id of each embedding in the same order.

Similarly let's look into the ``test_docs`` folder

.. code-block:: console

 $> ls PATH_TO_SAVE_EMBEDDINGS/consumed_samplesX/test_doc/
 doc.ids  doc.npy
 $> head -n3 PATH_TO_SAVE_EMBEDDINGS/consumed_samplesX/test_doc/doc.ids 
 d823
 d159
 d1084

We can see that ``test_doc`` has a similar structure to ``test_queries`` but with ids and embeddings of the documents from the ``test_docs.josnl`` file. With this setup it is possible to evaluate the performance using metrics like MRR or NDCG.
