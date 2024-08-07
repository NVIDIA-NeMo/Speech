Sequence Packing
================

Sequence Packing for SFT/PEFT
-----------------------------

Overview
^^^^^^^^

When finetuning a large language model with either full-parameter or parameter-efficient finetuning, GPU
underutilization is a common problem due to an inefficient data pipeline. This is because most finetuning datasets have
a skewed distribution of sequence lengths, with many short sequences and a few long sequences, following Zipf’s Law.
Transformer models can only take in fixed length inputs, so the input has to be padded with many unused pad tokens,
which is inefficient in two ways:

- Computation performed on the pad values is eventually ignored for model output, resulting in wasted FLOPs.
- Micro batch size is often limited by the batch which contains longer sequences, so that most other micro batches have
  underutilized GPU memory.

Sequence packing is a training technique where multiple training sequences (examples) are concatenated together into
one long sequence (pack). This eliminates the need for padding and allows more tokens to be processed in each
micro batch, maximizing both GPU compute and GPU memory.

While sequences for pretraining can be concatenated naively, this is not the case for SFT and instruction fine-tuning
where each input sequence should be treated individually. The conventional solution is to build an extended attention
mask to mark the sequence id each token belongs to, and mask out attention values between sequences. However, this
increases the complexity of attention from :math:`\sum_i {s_i}^2` to :math:`\Big({\sum_i {s_i}}\Big)^2`, where :math:`s_i` is the
length of the ith subsequence. In practice, the conventional solution puts a limit on the length of packing.
Instead, NeMo provides a highly optimized version of sequence packing which makes use of variable-length attention
kernels in FlashAttention and TransformerEngine. With this, attention values between sequences are never calculated,
so the complexity of attention remains at :math:`\sum_i {s_i}^2`. This allows packing sequences to arbitrary lengths so
that GPU memory can be fully utilized.

All things considered, NeMo’s implementation of sequence packing provides [#f1]_

- Up to 10X performance improvement in terms of FLOPs
- Up to 6X performance improvement in terms of training time
- No impact on model convergence



How to run SFT/PEFT with packed sequence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prepare Dataset
"""""""""""""""

We provide a convenient script to pack your SFT or PEFT dataset.
This script assumes that you already have a prepared dataset file for SFT/PEFT training in NeMo. If you do not, please
follow `this <https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/llama2sft.html#prepare-data>`_ to
download and prepare the dolly dataset as an example.
You will get a file named training.jsonl. The rest of this tutorial also assumes you already have a recipe for
training with the unpacked dataset.

Two main steps are run in this script:

1. The online processing code in GPTSFTDataset is run (including prompt template manipulation, sequence length
   truncation, tokenization, etc) and the result is an array of tokenized sequences, represented by indices).
2. The sequences are grouped by length, and a packing algorithm is run.

You can read more about packing algorithms `here <https://en.wikipedia.org/wiki/Bin_packing_problem#Offline_algorithms>`_.
Currently, two variants of *first fit* are supported.
- *first_fit_decreasing* sorts the sequences in decreasing order before applying the first-fit algorithm. It generates a
more optimal packing, but it tends to keep all short sequences together, which may have an impact for convergence.
- *first_fit_shuffle* runs first-fit in a random order. Packing is less optimal but it keeps the dataset order random.
The recommendation is to run *first_fit_shuffle* and check the packed sequence lengths. If they are similar to the
target length (i.e. efficient packing), then use shuffle. Otherwise try *first_fit_decreasing*.

    .. code-block:: bash

        python scripts/nlp_language_modeling/prepare_packed_ft_dataset.py \
           model.data.train_ds.file_names=[/path/to/training.jsonl] \
           model.data.train_ds.max_seq_length=2048 \
           +tokenizer_path=/path/to/tokenizer.model \
           +output_dir=/path/to/output_folder \
           +pack_sizes=[2048,4096,8192] \
        [  +packing_algorithm=first_fit_shuffle \  ]
        [  +seed=0                                 ]

.. note::

    Note 1. If your model or dataset requires non-default configs for conventional SFT/PEFT training in NeMo, you will
    need to pass in the same configs to ``model.data.train_ds`` as you would for training with unpacked dataset.

    Note 2. ``model.data.train_ds.max_seq_length`` is the length to truncate each sequence before packing multiple sequences
    to the size of packed sequence (``pack_size``). ``max_seq_length`` should be set to the same value as unpacked data,
    and can be determined by examining the distribution of sequence lengths in the dataset.

    Note 3. ``pack_sizes`` is a list of packed sequence lengths. In this example, there will be three output files, one for
    each pack size. The output files are named ``<output_folder>/packed_{pack_size}_seed{seed}.npy``.
    This argument is a list because you will likely want to experiment with a few ``pack_sizes`` to find out which length
    can fill the GPU memory without exceeding it. Adjusting ``pack_size`` is analogous to adjusting the micro batch size in
    the unpacked case.


Adjust Training Config
""""""""""""""""""""""

To train with packed sequences, you need to change four items in the SFT/PEFT config file

1. Turn on the packed_sequence flag

    .. code-block:: bash

        ++model.data.train_ds.packed_sequence=True

2. Use the new dataset file instead of the original jsonl file

    .. code-block:: bash

        model.data.train_ds.file_names=output_folder/packed_{pack_size}_seed{seed}.npy

3. Specify the packed sequence length. This should be one of the ``pack_sizes`` you specified during data preparation.

    .. code-block:: bash

        model.data.train_ds.max_seq_length={pack_size}

4. Adjust the batch sizes.

    - Micro batch size has to be set to 1 as a nominal constraint. This is because batches are now concatenated in the
      preprocessing step. You can increase the ``pack_size`` to achieve the same purpose of increasing micro batch size.
    - Global batch size has to be adjusted so that the training recipe is maintained. Because each pack contains
      multiple sequences now, global batch size needs to be reduced by the average number of sequences per pack ``n``,
      where ``n = num_sequences_in_dataset / num_packs``. This ensures that each gradient iteration sees (on
      average) the same number of tokens. The value of ``n`` is printed out when the script is run.

    .. code-block:: bash

        model.micro_batch_size=1
        model.global_batch_size=<GBS divided by n>

Now you are all set to finetune your model with a much improved throughput!

Sequence Packing for NeVA
-------------------------

Sequence packing in NeVA (Multimodal LLMs) differs slightly from the LLM SFT/PEFT approach. For details,
please refer to the documentation below

:doc:`../multimodal/mllm/sequence_packing`

.. rubric:: Footnotes

.. [#f1] Experiments were performed on Llama 7B with Dolly dataset. Actual performance improvement depends on dataset
         and model.