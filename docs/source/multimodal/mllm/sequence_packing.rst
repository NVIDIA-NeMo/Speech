Sequence Packing for NeVA
=========================

Overview
--------
As outlined in the throughput optimizations section, most multimodal LLM datasets, such as the LLaVA datasets, exhibit a skewed distribution of sequence lengths. Many sequences are short, and a few are very long, conforming to Zipf’s Law. Transformer models require fixed-length inputs, necessitating padding with many unused pad tokens, which is inefficient for two reasons:

1. Computation on pad values is disregarded in the final model output, resulting in wasted FLOPs.
2. The micro batch size is often constrained by the batch containing the longest sequences, leading to underutilized GPU memory in most other batches.

Sequence packing is a training technique wherein multiple training sequences (examples) are concatenated into one long sequence (pack). This approach eliminates the need for padding and allows for more tokens to be processed per micro batch, optimizing both GPU compute and memory utilization.

For Sequence Packing in SFT / PEFT for LLMs, NeVA considers the following design:

1. Original Datasets to Sequence Lengths Files

   1.1. **PyTorch Loaders for Dataset Processing Efficiency**
        To efficiently manage large datasets (~700K sequences), the system utilizes PyTorch's DataLoader with multi-worker capabilities, significantly speeding up the data processing phase by parallelizing the loading and pre-processing steps.
   1.2. **Handling Large Datasets**
        The system writes sequence lengths to disk on the fly, ensuring scalability and efficient memory usage, as loading all data into memory is impractical.
   1.3. **Efficient I/O Operations**
        To facilitate efficient I/O operations necessary for parallelized data loading, the system employs IndexedDataset from Megatron-Core, chosen for its ability to dynamically build binary tensor files.

2. Packing Sequences into Bins

   2.1. **Algorithm Choices and Performance**
        The first_fit_decreasing and first_fit_shuffle algorithms initially used for packing sequences into bins showed performance issues due to their O(n^2) complexity, making the processing of NeVA samples time-consuming.
   2.2. **Introduction of shuffle_and_pack**
        To address these inefficiencies, the shuffle_and_pack algorithm was introduced, an O(n) complexity algorithm that shuffles the sequence lengths before packing them into bins sequentially, significantly improving processing time.
   2.3. **Parallelization of Packing Process**
        The system implements a parallelized approach to the first_fit_shuffle algorithm by dividing the samples into chunks (~20K samples each) and processing them separately, effectively mitigating the quadratic complexity problem. The bins from each chunk are then combined in the final step, enhancing overall efficiency.
   2.4. **Efficiency Improvements with completed_bins**
        A minor optimization involves using completed_bins to prevent the algorithm from iterating over bins that cannot accommodate the minimum sequence length, leading to a more efficient packing process.

3. Reading Sequence Lengths and Packing into New Files
   After determining the optimal bins for packing, the system reads the sequence lengths from the generated files and packs these lengths into new files based on the bins' assignments. This final step consolidates the sequences into efficiently packed bins, ready for further processing or analysis.

Performance Improvement
-----------------------
A 40% speed increase was achieved with optimized sequence packing for sequence length w/ Vicuna-1.5 13B (LLaVA 1.5 recipe). Detailed performance metrics across different configurations and stages are provided in the tables below.

Fine-tuning Performance Table:

+--------------+---------------------------+----------------+----+----+-----------+------------------+-----------------+-------------------+---------------+-------------------+
| Stage        | Vision Encoder            | LLM Model      | TP | PP | Precision | Sequence Packing | Step Timing (s) | Global Batch Size | Samples / Sec | Perf Improvement  |
+==============+===========================+================+====+====+===========+==================+=================+===================+===============+===================+
| Fine-tuning  | openai/clip-vit-large-    | Vicuna-1.5 13B | 8  | 1  | BF16      | No               | 2.008           | 128               | 63.745        | 0%                |
|              | patch14-336               |                |    |    |           |                  |                 |                   |               |                   |
+--------------+---------------------------+----------------+----+----+-----------+------------------+-----------------+-------------------+---------------+-------------------+
| Fine-tuning  | openai/clip-vit-large-    | Vicuna-1.5 13B | 4  | 2  | BF16      | No               | 1.889           | 128               | 67.761        | 6%                |
|              | patch14-336               |                |    |    |           |                  |                 |                   |               |                   |
+--------------+---------------------------+----------------+----+----+-----------+------------------+-----------------+-------------------+---------------+-------------------+
| Fine-tuning  | openai/clip-vit-large-    | Vicuna-1.5 13B | 8  | 1  | BF16      | Yes              | 1.302           | 116.08            | 89.155        | 40%               |
|              | patch14-336               |                |    |    |           |                  |                 |                   |               |                   |
+--------------+---------------------------+----------------+----+----+-----------+------------------+-----------------+-------------------+---------------+-------------------+
| Fine-tuning  | openai/clip-vit-large-    | Vicuna-1.5 13B | 4  | 2  | BF16      | Yes              | 1.237           | 116.08            | 93.840        | 47%               |
|              | patch14-336               |                |    |    |           |                  |                 |                   |               |                   |
+--------------+---------------------------+----------------+----+----+-----------+------------------+-----------------+-------------------+---------------+-------------------+

How to Run NeVA with Packed Sequence
------------------------------------
Prepare Dataset
^^^^^^^^^^^^^^^
We provide an easy-to-use script for preprocessing a dataset for the NeMo Multimodal Learning framework. It requires specifying paths for data, images, and the tokenizer model, among other parameters.

.. code-block:: bash

    python examples/multimodal/multimodal_llm/neva/sequence_packing/preprocess_dataset.py \
     --data_path=/path/to/LLaVA-Instruct-150K/llava_v1_5_mix665k_filtered.json \
     --image_folder=/path/to/LLaVA-Instruct-150K/images \
     --tokenizer_path=/path/to/checkpoints/tokenizer_add_special.model \
     --output_dir=/path/to/LLaVA-Instruct-150K/packed_seq_12288_336_v1 \
     --max_seq_length=12288 \
     --packing_algorithm=first_fit_shuffle \
     --hf_vision_encoder=openai/clip-vit-large-patch14-336 \
     --conv_template=v1 \
     --image_aspect_ratio=pad \
     --seed=42

Parameters:
* ``--data_path``: Path to the dataset file in JSON format.
* ``--image_folder``: Directory containing the images referenced in the dataset.
* ``--tokenizer_path``: Path to the tokenizer model.
* ``--output_dir``: Directory where the processed dataset will be stored.
* ``--max_seq_length``: The maximum sequence length of the model.
* ``--packing_algorithm``: Algorithm used for packing sequences. Defaults to 'first_fit_shuffle'.
* ``--hf_vision_encoder``: The Hugging Face vision encoder to use. Default is 'openai/clip-vit-large-patch14-336'.
* ``--conv_template``: Template for data conversion. Default is 'plain', with 'v1' as an alternative.
* ``--image_aspect_ratio``: The aspect ratio for processing images. Defaults to 'square', 'pad' for padding to maintain aspect ratio.
* ``--seed``: Seed for random operations in 'first_fit_shuffle'.
* ``--hparams_file``: Optional path to a YAML file containing additional hyperparameters.

Remarks:
1. The current version of data processing saves processed image tensors in the sequence packing, which may require significant storage. This issue will be addressed in future iterations.
2. The ``max_seq_length`` is crucial for achieving optimal performance. Excessive length can lead to out-of-memory errors, while insufficient length may degrade performance.
3. The conversation prompt template is inserted during this step to ensure accurate sequence length calculation.

Adjust Training Config
""""""""""""""""""""""
To train with packed sequences, modify four items in the SFT/PEFT config file.

1. Enable the ``packed_sequence`` flag:

.. code-block:: bash

    ++model.data.data_prefix=/lustre/fsw/coreai_dlalgo_genai/datasets/LLaVA-Instruct-150K/packed_seq_12288_336_v1/packed_seq_dataset
    ++model.data.crop_size=[224,224]
    ++model.data.packed_sequence=True

2. Use the new dataset file instead of the original JSONL file and ensure the crop sizes are correctly specified since images are now cached:

.. code-block:: bash

    ++model.data.data_prefix=/path/to/datasets/LLaVA-Instruct-150K/packed_seq_12288_336_v1/packed_seq_dataset
    ++model.data.crop_size=[336,336]

4. Adjust batch sizes:

* Micro batch size should be set to 1 due to concatenation in the preprocessing step. Increase ``pack_size`` to achieve a higher micro batch size.
* Global batch size should be adjusted based on the average number of sequences per pack (``n``), calculated as the total number of sequences divided by the number of packs. This maintains the training recipe by ensuring each gradient iteration sees, on average, the same number of tokens.

.. code-block:: bash

    model.micro_batch_size=1
    model.global_batch_size=<GBS divided by n>

Now, you are ready to fine-tune your model with significantly improved throughput!
