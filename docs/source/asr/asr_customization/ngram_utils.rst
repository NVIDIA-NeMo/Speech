.. _ngram-utils:

Scripts for building and merging N-gram Language Models
=======================================================

.. _train-ngram-lm:

Train N-gram LM
===============

NeMo utilizes the KenLM library (`https://github.com/kpu/kenlm`) for building efficient n-gram language models.

.. note::

    KenLM is not installed by default in NeMo.  
    Please see the installation instructions in the script:  
    `scripts/asr_language_modeling/ngram_lm/install_beamsearch_decoders.sh <https://github.com/NVIDIA/NeMo/blob/stable/scripts/asr_language_modeling/ngram_lm/install_beamsearch_decoders.sh>`__.

    Alternatively, you can build a Docker image with all required dependencies using:  
    `scripts/installers/Dockerfile.ngramtools <https://github.com/NVIDIA/NeMo/blob/stable/scripts/installers/Dockerfile.ngramtools>`__.

The script for training an n-gram language model with KenLM is available here:  
`scripts/asr_language_modeling/ngram_lm/train_kenlm.py <https://github.com/NVIDIA/NeMo/blob/stable/scripts/asr_language_modeling/ngram_lm/train_kenlm.py>`__.

This script supports training n-gram LMs on both character-level and BPE-level encodings, which are automatically detected from the model type. The resulting language models can then be used with beam search decoders integrated on top of ASR models.

You can train an n-gram model using the following command:

.. code-block::

    python train_kenlm.py nemo_model_file=<path to the .nemo file of the model> \
                              train_paths=<list of paths to the training text or JSON manifest files> \
                              kenlm_bin_path=<path to the bin folder of KenLM library> \
                              kenlm_model_file=<path to store the binary KenLM model> \
                              ngram_length=<order of N-gram model> \
                              preserve_arpa=true

The `train_paths` parameter allows for various input types, such as a list of text files, JSON manifests, or directories, to be used as the training data.
If the file's extension is anything other than `.json`, it assumes that data format is plain text. For plain text format, each line should contain one
sample. For the JSON manifests, the file must contain JSON-formatted samples per each line like this:

.. code-block::

    {"audio_filepath": "/data_path/file1.wav", "text": "The transcript of the audio file."}

This code extracts the `text` field from each line to create the training text file. After the N-gram model is trained, it is stored at the path specified by `kenlm_model_file`.

The following is the list of the arguments for the training script:

+------------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------------------+
| **Argument**     | **Type**  | **Default** | **Description**                                                                                                                |
+------------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------------------+
| nemo_model_file  | str       | Required    | The path to `.nemo` file of the ASR model, or name of a pretrained NeMo model to extract a tokenizer.                          |
+------------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------------------+
| train_paths      | List[str] | Required    | List of training files or folders. Files can be a plain text file or ".json" manifest or ".json.gz".                           |
+------------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------------------+
| kenlm_model_file | str       | Required    | The path to store the KenLM binary model file.                                                                                 |
+------------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------------------+
| kenlm_bin_path   | str       | Required    | The path to the bin folder of KenLM. It is a folder named `bin` under where KenLM is installed.                                |
+------------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------------------+
| ngram_length**   | int       | Required    | Specifies order of N-gram LM.                                                                                                  |
+------------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------------------+
| ngram_prune      | List[int] | [0]         | List of thresholds to prune N-grams. Example: [0,0,1]. See Pruning section on the https://kheafield.com/code/kenlm/estimation  |
+------------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------------------+
| cache_path       | str       | ``""``      | Cache path to save tokenized files.                                                                                            |
+------------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------------------+
| preserve_arpa    | bool      | ``False``   | Whether to preserve the intermediate ARPA file after construction of the BIN file.                                             |
+------------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------------------+
| verbose          | int       | 1           | Verbose level.                                                                                                                 |
+------------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------------------+
| save_nemo        | bool      | ``False``   | Whether to save LM in .nemo format.                                                                                            |
+------------------+-----------+-------------+--------------------------------------------------------------------------------------------------------------------------------+

..note::
It is recommended that you use 6 as the order of the N-gram model for BPE-based models. Higher orders may require re-compiling KenLM to support them.


Combine N-gram Language Models
==============================

Before combining N-gram LMs, install the required OpenGrm NGram library using `scripts/installers/install_opengrm.sh <https://github.com/NVIDIA/NeMo/blob/stable/scripts/installers/install_opengrm.sh>`__.
Alternatively, you can use Docker image `scripts/installers/Dockerfile.ngramtools <https://github.com/NVIDIA/NeMo/blob/stable/scripts/installers/Dockerfile.ngramtools>`__ with all the necessary dependencies.

Alternatively, you can use the Docker image at:
`scripts/asr_language_modeling/ngram_lm/ngram_merge.py <https://github.com/NVIDIA/NeMo/blob/stable/scripts/asr_language_modeling/ngram_lm/ngram_merge.py>`__, which includes all the necessary dependencies.

This script interpolates two ARPA N-gram language models and creates a KenLM binary file that can be used with the beam search decoders on top of ASR models.
You can specify weights (`--alpha` and `--beta`) for each of the models (`--ngram_a` and `--ngram_b`) correspondingly: `alpha` * `ngram_a` + `beta` * `ngram_b`.
This script supports both character level and BPE level encodings and models which are detected automatically from the type of the model.

To combine two N-gram models, you can use the following command:

.. code-block::

    python ngram_merge.py  --kenlm_bin_path <path to the bin folder of KenLM library> \
                    --ngram_bin_path  <path to the bin folder of OpenGrm Ngram library> \
                    --arpa_a <path to the ARPA N-gram model file A> \
                    --alpha <weight of N-gram model A> \
                    --ar
                    pa_b <path to the ARPA N-gram model file B> \
                    --beta <weight of N-gram model B> \
                    --out_path <path to folder to store the output files>



If you provide `--test_file` and `--nemo_model_file`, This script supports both character-level and BPE-level encodings and models, which are detected automatically based on the type of the model.
Note, the result of each step during the process is cached in the temporary file in the `--out_path`, to speed up further run.
You can use the `--force` flag to discard the cache and recalculate everything from scratch.

.. code-block::

    python ngram_merge.py  --kenlm_bin_path <path to the bin folder of KenLM library> \
                    --ngram_bin_path  <path to the bin folder of OpenGrm Ngram library> \
                    --arpa_a <path to the ARPA N-gram model file A> \
                    --alpha <weight of N-gram model A> \
                    --arpa_b <path to the ARPA N-gram model file B> \
                    --beta <weight of N-gram model B> \
                    --out_path <path to folder to store the output files>
                    --nemo_model_file <path to the .nemo file of the model> \
                    --test_file <path to the test file> \
                    --symbols <path to symbols (.syms) file> \
                    --force <flag to recalculate and rewrite all cached files>


The following is the list of the arguments for the opengrm script:

+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| **Argument**         |**Type**| **Default**      | **Description**                                                                                                 |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| kenlm_bin_path       | str    | Required         | The path to the bin folder of KenLM library. It is a folder named `bin` under where KenLM is installed.         |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| ngram_bin_path       | str    | Required         | The path to the bin folder of OpenGrm Ngram. It is a folder named `bin` under where OpenGrm Ngram is installed. |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| arpa_a               | str    | Required         | Path to the ARPA N-gram model file A.                                                                           |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| alpha                | float  | Required         | Weight of N-gram model A.                                                                                       |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| arpa_b               | int    | Required         | Path to the ARPA N-gram model file B.                                                                           |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| beta                 | float  | Required         | Weight of N-gram model B.                                                                                       |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| out_path             | str    | Required         | Path for writing temporary and resulting files.                                                                 |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| test_file            | str    | None             | Path to test file to count perplexity if provided.                                                              |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| symbols              | str    | None             | Path to symbols (.syms) file. Could be calculated if it is not provided.                                        |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| nemo_model_file      | str    | None             | The path to '.nemo' file of the ASR model, or name of a pretrained NeMo model.                                  |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+
| force                | bool   | ``False``        | Whether to recompile and rewrite all files.                                                                     |
+----------------------+--------+------------------+-----------------------------------------------------------------------------------------------------------------+