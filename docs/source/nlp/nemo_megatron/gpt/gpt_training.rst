GPT Model Training
------------------

The Generative Pre-trained Transformer (GPT) is a decoder-only Transformer model. This section demonstrates how to train a GPT-style model with NeMo.


.. note::
    This example is best completed using the latest NeMo Framework Training container `<https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo>`_.

Download and Preprocess the Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
    The example below will take approximately 3 hours to download data, pre-process it, and train the tokenizer.

1. Download the data.

The following step will download approximately 20GB of Wikipedia data, which can take several hours to complete.

.. code-block:: bash

    wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

2. Extract the raw data.

.. code-block:: bash

    pip install wikiextractor
    python -m wikiextractor.WikiExtractor enwiki-latest-pages-articles.xml.bz2 --json
    find text -name 'wiki_*' -exec cat {} \; > train_data.jsonl

Now, train_data.jsonl will contain our training data in JSON line format. We are particularly interested in the data within the "text" field.


3. Train the tokenizer.

Below, we will consider two options for training data tokenizers: using the pre-built Hugging Face BPE or training and using your own Google Sentencepiece tokenizer.

Note that only the second option allows you to experiment with vocabulary size.

*Option 1:* Use the Hugging Face GPT2 tokenizer files.

With this option, we will download a pre-built vocabulary and merge the files for the BPE tokenizer.

.. code-block:: bash

    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt


*Option 2:* Use the `Google Sentencepiece <https://github.com/google/sentencepiece>`_ tokenizer library. 

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

4. Convert the training data into memory map format.

The memory map format makes training more efficient, especially with many nodes and GPUs. This step will also tokenize data using the tokenizer model from Step 3.

*Option 1:* Use the Hugging Face GPT2 tokenizer files.

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

*Option 2:* Use the `Google Sentencepiece <https://github.com/google/sentencepiece>`_ tokenizer library.

.. code-block:: bash

    python <NeMo_ROOT_FOLDER>/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input=train_data.jsonl \
    --json-keys=text \
    --tokenizer-library=sentencepiece \
    --tokenizer-model=spm_32k_wiki.model \
    --output-prefix=gpt_training_data \
    --append-eod \
    --workers=32


Create a Custom Training Recipe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To train a model with NeMo 2.0, a training recipe is required. You can refer to `this page <https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes/ADD-RECIPE.md>`_ 
to learn how to create a custom training recipe or use an existing one, refer to the `LLM recipes <https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes>`_ developed by NeMo team.


Train a Model
~~~~~~~~~~~~~

Once you have prepared the training data, tokenizer, and recipe, you are ready to train the model. You can follow `this tutorial <https://github.com/NVIDIA/NeMo/blob/main/examples/llm/pretrain/README.md#run-pre-training-with-a-default-recipe>`_ 
To train a model using an existing recipe or a custom one, follow `this example <https://github.com/NVIDIA/NeMo/blob/main/examples/llm/pretrain/README.md#create-and-run-a-custom-recipe>`_ to train a model with a custom recipe.

Next Steps
~~~~~~~~~~

For more information, please refer to:

* `batching <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/nemo_megatron/batching.rst>`_ section for batch size adjustments.
* `parallelisms <https://github.com/NVIDIA/NeMo/blob/main/docs/source/features/parallelisms.rst>`_ section for understanding various types of parallelisms.

