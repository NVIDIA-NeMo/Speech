GPT Model Training
------------------

The Generative Pre-trained Transformer (GPT) is a decoder-only Transformer model. This section demonstrates how to train a GPT-style model with NeMo.






.. note::
    This example is best completed using the latest NeMo Framework Training container `<https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo>`_.

Download and Pre-process Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    The example below will take approximately 3 hours to download data, pre-process it, and train the tokenizer.

1. Download data.

The following step will download approximately 20GB of Wikipedia data, which can take several hours to complete.

.. code-block:: bash

    wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

2. Extract raw data.

.. code-block:: bash

    pip install wikiextractor
    python -m wikiextractor.WikiExtractor enwiki-latest-pages-articles.xml.bz2 --json
    find text -name 'wiki_*' -exec cat {} \; > train_data.jsonl

Now, train_data.jsonl will contain our training data in JSON line format. We are particularly interested in the data within the "text" field.


3. Train tokenizer.

Below, we will consider two options for training data tokenizers: using the pre-built Hugging Face BPE or training and using your own Google Sentencepiece tokenizer.

Note that only the second option allows you to experiment with vocabulary size.

*Option 1:* Use Hugging Face GPT2 tokenizer files.

With this option, we will download a pre-built vocabulary and merge the files for the BPE tokenizer.

.. code-block:: bash

    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt


*Option 2:* Use `Google Sentencepiece <https://github.com/google/sentencepiece>`_ tokenizer library. 

Google Sentencepiece is included as a dependency with NeMo, so if you have installed NeMo, it should already be installed. 
Please note that training the tokenizer model will also take some time.

.. code-block:: bash

   sudo apt install jq
   jq .text train_data.jsonl >> text_for_tokenizer.txt
   spm_train --input=text_for_tokenizer.txt \
        --model_prefix=spm_32k_wiki \
        --vocab_size=32768 \
        --character_coverage=0.9999 \
        --model_type=bpe \
        --byte_fallback=true \
        --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 \
        --split_digits true

Completing this step can take some time. After it is done, you'll have two files: ``spm_32k_wiki.model`` and ``spm_32k_wiki.vocab`` corresponding to the model and vocabulary.

4. Convert training data into memory map format.

The memory map format makes training more efficient, especially with many nodes and GPUs. This step will also tokenize data using the tokenizer model from Step 3.

*Option 1:* Use Hugging Face GPT2 tokenizer files.

.. code-block:: bash

    python <NeMo_ROOT_FOLDER>/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input=train_data.jsonl \
    --json-keys=text \
    --tokenizer-library=megatron \
    --vocab gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file gpt2-merges.txt \
    --output-prefix=hfbpe_gpt_training_data \
    --append-eod \
    --workers=32

*Option 2:* Use `Google Sentencepiece <https://github.com/google/sentencepiece>`_ tokenizer library.

.. code-block:: bash

    python <NeMo_ROOT_FOLDER>/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input=train_data.jsonl \
    --json-keys=text \
    --tokenizer-library=sentencepiece \
    --tokenizer-model=spm_32k_wiki.model \
    --output-prefix=gpt_training_data \
    --append-eod \
    --workers=32


Train a GPT-Style Model
~~~~~~~~~~~~~~~~~~~~~~~

Once you have prepared training data and tokenizer, you are ready to train the model.
The configuration we present below has about 124M parameters and should fit on a single 16GB GPU using float16.
Let's go!

*Option 1:* Use Hugging Face GPT2 tokenizer files.

.. code-block:: bash

    python <NeMo_ROOT_FOLDER>/examples/nlp/language_modeling/megatron_gpt_pretraining.py  \
	--config-path=<NeMo_ROOT_FOLDER>/examples/nlp/language_modeling/conf \
	--config-name=megatron_gpt_config \
	trainer.devices=1 \
	trainer.num_nodes=1 \
	trainer.max_epochs=null \
	trainer.max_steps=300000 \
	trainer.val_check_interval=300 \
	trainer.log_every_n_steps=50 \
	trainer.limit_val_batches=50 \
	trainer.limit_test_batches=50 \
	trainer.accumulate_grad_batches=1 \
	trainer.precision=16 \
	model.micro_batch_size=6 \
	model.global_batch_size=192 \
	model.tensor_model_parallel_size=1 \
	model.pipeline_model_parallel_size=1 \
	model.max_position_embeddings=1024 \
	model.encoder_seq_length=1024 \
	model.hidden_size=768 \
	model.ffn_hidden_size=3072 \
	model.num_layers=12 \
	model.num_attention_heads=12 \
	model.init_method_std=0.021 \
	model.hidden_dropout=0.1 \
	model.layernorm_epsilon=1e-5 \
	model.tokenizer.vocab_file=gpt2-vocab.json \
    model.tokenizer.merge_file=gpt2-merges.txt \
	model.data.data_prefix=[1.0,hfbpe_gpt_training_data_text_document] \
	model.data.num_workers=2 \
	model.data.seq_length=1024 \
	model.data.splits_string=\'980,10,10\' \
	model.optim.name=fused_adam \
	model.optim.lr=6e-4 \
	model.optim.betas=[0.9,0.95] \
	model.optim.weight_decay=0.1 \
	model.optim.sched.name=CosineAnnealing \
	model.optim.sched.warmup_steps=750 \
	model.optim.sched.constant_steps=80000 \
	model.optim.sched.min_lr=6e-5 \
	exp_manager.resume_if_exists=True \
	exp_manager.resume_ignore_no_checkpoint=True \
	exp_manager.create_checkpoint_callback=True \
	exp_manager.checkpoint_callback_params.monitor=val_loss \
	exp_manager.checkpoint_callback_params.save_top_k=3 \
	exp_manager.checkpoint_callback_params.mode=min \
	exp_manager.checkpoint_callback_params.always_save_nemo=False


*Option 2:* Use `Google Sentencepiece <https://github.com/google/sentencepiece>`_ tokenizer library.

.. code-block:: bash

    python <NeMo_ROOT_FOLDER>/examples/nlp/language_modeling/megatron_gpt_pretraining.py  \
	--config-path=<NeMo_ROOT_FOLDER>/examples/nlp/language_modeling/conf \
	--config-name=megatron_gpt_config \
	trainer.devices=1 \
	trainer.num_nodes=1 \
	trainer.max_epochs=null \
	trainer.max_steps=300000 \
	trainer.val_check_interval=300 \
	trainer.log_every_n_steps=50 \
	trainer.limit_val_batches=50 \
	trainer.limit_test_batches=50 \
	trainer.accumulate_grad_batches=1 \
	trainer.precision=16 \
	model.micro_batch_size=6 \
	model.global_batch_size=192 \
	model.tensor_model_parallel_size=1 \
	model.pipeline_model_parallel_size=1 \
	model.max_position_embeddings=1024 \
	model.encoder_seq_length=1024 \
	model.hidden_size=768 \
	model.ffn_hidden_size=3072 \
	model.num_layers=12 \
	model.num_attention_heads=12 \
	model.init_method_std=0.021 \
	model.hidden_dropout=0.1 \
	model.layernorm_epsilon=1e-5 \
	model.tokenizer.library=sentencepiece \
	model.tokenizer.model=spm_32k_wiki.model \
	model.data.data_prefix=[1.0,gpt_training_data_text_document] \
	model.data.num_workers=2 \
	model.data.seq_length=1024 \
	model.data.splits_string=\'980,10,10\' \
	model.optim.name=fused_adam \
	model.optim.lr=6e-4 \
	model.optim.betas=[0.9,0.95] \
	model.optim.weight_decay=0.1 \
	model.optim.sched.name=CosineAnnealing \
	model.optim.sched.warmup_steps=750 \
	model.optim.sched.constant_steps=80000 \
	model.optim.sched.min_lr=6e-5 \
	exp_manager.resume_if_exists=True \
	exp_manager.resume_ignore_no_checkpoint=True \
	exp_manager.create_checkpoint_callback=True \
	exp_manager.checkpoint_callback_params.monitor=val_loss \
	exp_manager.checkpoint_callback_params.save_top_k=3 \
	exp_manager.checkpoint_callback_params.mode=min \
	exp_manager.checkpoint_callback_params.always_save_nemo=False


Next, you can launch Tensorboard to monitor training, as follows:

.. code-block:: bash

    tensorboard --logdir nemo_experiments --bind_all

Next Steps
~~~~~~~~~~

For more information, please refer to:

* :ref:`batching` section for batch size adjustments
* :ref:`parallelisms` section for understanding various types of parallelisms

