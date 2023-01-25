########################################################################
Example: Training Esperanto ASR model using Mozilla Common Voice Dataset
########################################################################

Training an ASR model for a new language can be challenging, especially for low-resource languages (see  `example <https://github.com/NVIDIA/NeMo/blob/main/docs/source/asr/examples/kinyarwanda_asr.rst>`_ for Kinyarwanda ASR model). 
This example describes all basic steps required to build  ASR model for Esperanto:

* Data preparation
* Tokenization
* Training hyper-parameters
* Training from scratch
* Fine-tuning from pretrained models on other languages (English, Spanish, Italian).
* Fine-tuning from pretrained English SSL (`Self-supervised learning <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/ssl/intro.html?highlight=self%20supervised>`_) model
* Model evaluation

****************
Data Preparation
****************
Mozilla Common Voice provides 1400 hours of validated Esperanto speech (see `here <https://arxiv.org/abs/1912.0667>`_). However, the final training dataset consists only of 250 hours because “... the train, test, and development sets are bucketed such that any given speaker may appear in only one. This ensures that contributors seen at train time are not seen at test time, which would skew results. Additionally, repetitions of text sentences are removed from the train, test, and development sets of the corpus”. 

Downloading the Data
####################

You can use the NeMo script to download MCV dataset from Hugging Face and get NeMo data manifests for Esperanto:

.. code-block:: bash
    
    """
    # Setup
    After installation of huggingface datasets (pip install datasets), some datasets might require authentication
    - for example Mozilla Common Voice. You should go to the above link, register as a user and generate an API key.

    ## Authenticated Setup Steps

    Website steps:
    - Visit https://huggingface.co/settings/profile
    - Visit "Access Tokens" on list of items.
    - Create new token - provide a name for the token and "read" access is sufficient.
      - PRESERVE THAT TOKEN API KEY. You can copy that key for next step.
    - Visit the HuggingFace Dataset page for Mozilla Common Voice
      - There should be a section that asks you for your approval.
      - Make sure you are logged in and then read that agreement.
      - If and only if you agree to the text, then accept the terms.

    Code steps:
    - Now on your machine, run `huggingface-cli login`
    - Paste your preserved HF TOKEN API KEY (from above).

    Now you should be logged in. When running the script, dont forget to set `use_auth_token=True` !
    """
    
    
    # Repeat script execution three times for variable SPLIT = test, validation, and train.
    
    python ${NEMO_ROOT}/scripts/speech_recognition/convert_hf_dataset_to_nemo.py \
        output_dir=${OUTPUT_DIR} \
        path="mozilla-foundation/common_voice_11_0" \
        name="eo" \
        split=${SPLIT} \
        ensure_ascii=False \
        use_auth_token=True

You will get the next data structure:

.. code-block:: bash

    .
    └── mozilla-foundation
        └── common_voice_11_0
            └── eo
                ├── test
                ├── train
                └── validation

Dataset Preprocessing
#####################

Next, we must clear the text data from punctuation and various “garbage” characters. In addition to deleting a standard set of elements (as in Kinyarwanda), you can compute  the frequency of characters in the train set and add the rarest (occurring less than ten times) to the list for deletion. 

.. code-block:: python
  
  import json
  from tqdm import tqdm

  dev_manifest = f"{YOUR_DATA_ROOT}/validation/validation_mozilla-foundation_common_voice_11_0_manifest.json"
  test_manifest = f"{YOUR_DATA_ROOT}/test/test_mozilla-foundation_common_voice_11_0_manifest.json"
  train_manifest = f"{YOUR_DATA_ROOT}/train/train_mozilla-foundation_common_voice_11_0_manifest.json"

  def compute_char_counts(manifest):
      char_counts = {}
      with open(manifest, 'r') as fn_in:
          for line in tqdm(fn_in, desc="Compute counts.."):
              line = line.replace("\n", "")
              data = json.loads(line)
              text = data["text"]
              for word in text.split():
                  for char in word:
                      if char not in char_counts:
                          char_counts[char] = 1
                      else:
                          char_counts[char] += 1
      return char_counts

  char_counts = compute_char_counts(train_manifest)

  threshold = 10
  trash_char_list = []

  for char in char_counts:
      if char_counts[char] <= threshold:
          trash_char_list.append(char)

Let's check:

.. code-block:: python

  print(trash_char_list)

  ['é', 'ǔ', 'á', '¨', 'Ŭ', 'ﬁ', '=', 'y', '`', 'q', 'ü', '♫', '‑', 'x', '¸', 'ʼ', '‹', '›', 'ñ']

Next we will check the data for anomalies in audio file (for example,  audio file with noise only). For this end, we check character rate (number of chars per second). For example, If the char rate is too high (more than 15 chars per second), then something is wrong with the audio file. It is better to filter such data from the training dataset in advance. Other problematic files can be filtered out after receiving the first trained model. We will consider this method at the end of our example.

.. code-block:: python

  import re
  import json
  from tqdm import tqdm

  def clear_data_set(manifest, char_rate_threshold=None):

      chars_to_ignore_regex = "[\.\,\?\:\-!;()«»…\]\[/\*–‽+&_\\½√>€™$•¼}{~—=“\"”″‟„]"
      addition_ignore_regex = f"[{''.join(trash_char_list)}]"

      manifest_clean = manifest + '.clean'
      war_count = 0
      with open(manifest, 'r') as fn_in, \
          open(manifest_clean, 'w', encoding='utf-8') as fn_out:
          for line in tqdm(fn_in, desc="Cleaning manifest data"):
              line = line.replace("\n", "")
              data = json.loads(line)
              text = data["text"]
              if char_rate_threshold and len(text.replace(' ', '')) / float(data['duration']) > char_rate_threshold:
                  print(f"[WARNING]: {data['audio_filepath']} has char rate > 15 per sec: {len(text)} chars, {data['duration']} duration")
                  war_count += 1
                  continue
              text = re.sub(chars_to_ignore_regex, "", text)
              text = re.sub(addition_ignore_regex, "", text)
              data["text"] = text.lower()
              data = json.dumps(data, ensure_ascii=False)
              fn_out.write(f"{data}\n")
      print(f"[INFO]: {war_count} files were removed from manifest")

  clear_data_set(dev_manifest)
  clear_data_set(test_manifest)
  clear_data_set(train_manifest, char_rate_threshold=15)


Creating the Tarred Training Dataset
####################################

The tarred dataset allows storing the dataset as large *.tar files instead of small separate audio files. It may speed up the training and minimizes the load when data is moved from storage to GPU nodes.

The NeMo toolkit provides a `script <https://github.com/NVIDIA/NeMo/blob/main/scripts/speech_recognition/convert_to_tarred_audio_dataset.py>`_ to get tarred dataset.

.. code-block:: bash

    TRAIN_MANIFEST=${YOUR_DATA_ROOT}/train/train_mozilla-foundation_common_voice_11_0_manifest.json.clean

    python ${NEMO_ROOT}/scripts/speech_recognition/convert_to_tarred_audio_dataset.py \
      --manifest_path=${TRAIN_MANIFEST} \
      --target_dir=${YOUR_DATA_ROOT}/train_tarred_1bk \
      --num_shards=1024 \
      --max_duration=15.0 \
      --min_duration=1.0 \
      --shuffle \
      --shuffle_seed=1 \
      --sort_in_shards \
      --workers=-1

*****************
Text Tokenization
*****************

We use the standard `Byte-pair <https://en.wikipedia.org/wiki/Byte_pair_encoding>`_ encoding algorithm with 128, 512, and 1024 vocabulary size. We found that 128 works best for relatively small Esperanto dataset (~250 hours). For larger datasets, one can get better results with larger vocabulary size (512…1024 BPE tokens).

.. code-block:: bash

    VOCAB_SIZE=128
    
    python ${NEMO_ROOT}/scripts/tokenizers/process_asr_text_tokenizer.py \
      --manifest=${TRAIN_MANIFEST} \
      --vocab_size=${VOCAB_SIZE} \
      --data_root=${YOUR_DATA_ROOT}/esperanto/tokenizers \
      --tokenizer="spe" \
      --spe_type=bpe \  

*************************
Training hyper-parameters
*************************

The training parameters are defined in the `config file <https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/conformer/conformer_ctc_bpe.yaml>`_ (general description of the `ASR configuration file <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/configs.html>`_). As an encoder, the `Conformer model <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html#conformer-ctc>`_ is used here, the training parameters for which are already well configured based on the training English models. However, the set of optimal parameters may differ for a new language. In this section, we will look at the set of simple parameters that can improve recognition quality for a new language without digging into the details of the Conformer model too much.

Select Training Batch Size
##########################

We trained model on server with 16 V100 GPUs with 32 GB. We use a local batch size = 32 per GPU V100), so global batch size is 32x16=512. In general, we observed, that  global batch between 512 and 2048 works well for Conformer-CTC-Large model. One can  use   the `accumulate_grad_batches <https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/conformer/conformer_ctc_bpe.yaml#L173>`_ parameter to increase the size of the global batch, which is equal  to *local_batch * num_gpu * accumulate_grad_batches*.

Selecting Optimizer and Learning Rate Scheduler
###############################################

The model was trained with AdamW optimizer and CosineAnealing Learning Rate (LR) scheduler. We use Learning Rate warmup when LR goes from 0 to maximum LR to stabilize initial phase of training. The number of warmup steps determines how quickly the scheduler will reach the peak learning rate during model training. The recommended number of steps is approximately 10-20% of total training duration. We used 8,000-10,000 warmup steps.

Now we can plot our learning rate for CosineAnnealing schedule:

.. code-block:: python

    import nemo
    import torch
    import matplotlib.pyplot as plt

    # params:
    train_files_num = 144000     # number of training audio_files
    global_batch_size = 1024     # local_batch * gpu_num * accum_gradient
    num_epoch = 300
    warmup_steps = 10000
    config_learning_rate = 1e-3

    steps_num = int(train_files_num / global_batch_size * num_epoch)
    print(f"steps number is: {steps_num}")

    optimizer = torch.optim.SGD(model.parameters(), lr=config_learning_rate)
    scheduler = nemo.core.optim.lr_scheduler.CosineAnnealing(optimizer,
                                                             max_steps=steps_num,
                                                             warmup_steps=warmup_steps,
                                                             min_lr=1e-6)
    lrs = []

    for i in range(steps_num):
        optimizer.step()
        lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)
        scheduler.step()

    plt.plot(lrs)

.. image:: ./images/CosineAnnealing_scheduler.png
    :align: center
    :alt: NeMo CosineAnnealing scheduler.
    :width: 500px
        
Numerical Precision
###################

By default, it is recommended to use half-precision float (FP16 for V100 and BF16 for A100 GPU) to speed up the training process. However, training with  half-precision may  affect the convergence of the model, for example training loss  can explode. In this case, we recommend to decrease LR or switch to float32. 

********
Training
********

We use three main scenarios to train Espearnto ASR model:

* Training from scratch.
* Fine-tuning from ASR models  for other languages (English, Spanish, Italian).
* Fine-tuning from an English SSL (`Self-supervised learning <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/ssl/intro.html?highlight=self%20supervised>`_) model.

For the training of the `Conformer-CTC <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/models.html#conformer-ctc>`_ model, we use `speech_to_text_ctc_bpe.py <https://github.com/NVIDIA/NeMo/tree/stable/examples/asr/asr_ctc/speech_to_text_ctc_bpe.py>`_ with the default config `conformer_ctc_bpe.yaml <https://github.com/NVIDIA/NeMo/tree/stable/examples/asr/conf/conformer/conformer_ctc_bpe.yaml>`_. Here you can see the example of how to run this training:

.. code-block:: bash

    TOKENIZER=${YOUR_DATA_ROOT}/esperanto/tokenizers/tokenizer_spe_bpe_v128
    TRAIN_MANIFEST=${YOUR_DATA_ROOT}/train_tarred_1bk/tarred_audio_manifest.json
    TARRED_AUDIO_FILEPATHS=${YOUR_DATA_ROOT}/train_tarred_1bk/audio__OP_0..1023_CL_.tar # "_OP_0..1023_CL_" is the range for the banch of files audio_0.tar, audio_1.tar, ..., audio_1023.tar
    DEV_MANIFEST=${YOUR_DATA_ROOT}/validation/validation_mozilla-foundation_common_voice_11_0_manifest.json.clean
    TEST_MANIFEST=${YOUR_DATA_ROOT}/test/test_mozilla-foundation_common_voice_11_0_manifest.json.clean

    python ${NEMO_ROOT}/examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
      --config-path=../conf/conformer/ \
      --config-name=conformer_ctc_bpe \
      exp_manager.name="Name of our experiment" \
      exp_manager.resume_if_exists=true \
      exp_manager.resume_ignore_no_checkpoint=true \
      exp_manager.exp_dir=results/ \
      ++model.encoder.conv_norm_type=layer_norm \
      model.tokenizer.dir=$TOKENIZER \
      model.train_ds.is_tarred=true \
      model.train_ds.tarred_audio_filepaths=$TARRED_AUDIO_FILEPATHS \
      model.train_ds.manifest_filepath=$TRAIN_MANIFEST \
      model.validation_ds.manifest_filepath=$DEV_MANIFEST \
      model.test_ds.manifest_filepath=$TEST_MANIFEST

Main training parameters:

* Tokenization: BPE 128/512/1024
* Model: Conformer-CTC-large with Layer Normalization
* Optimizer: AdamW, weight_decay 1e-3, LR 1e-3
* Scheduler: CosineAnnealing, warmup_steps 10000, min_lr 1e-6
* Batch: 32 local, 1024 global (2 grad accumulation)
* Precision: FP16
* GPUs: 16 V100

The following table provides the results for training Esperanto Conformer-CTC-large model from scratch with different BPE vocabulary size.

+----------------------------------+----------+------------+-------------+
| Training mode                    | BPE size | DEV, WER % | TEST, WER % |
+==================================+==========+============+=============+
|                                  |    128   |   **3.96** |   **6.48**  |
+                                  +----------+------------+-------------+
| From scratch                     |    512   |     4.62   |     7.31    |
+                                  +----------+------------+-------------+
|                                  |   1024   |     5.81   |     8.56    |
+----------------------------------+----------+------------+-------------+

BPE vocabulary with 128 size provides the lowest WER since our training dataset is l (~250 hours) is insufficient to small to train models with larger BPE vocabulary sizes.

For fine-tuning from already trained ASR models, we use three different models:

* English `stt_en_conformer_ctc_large <https://huggingface.co/nvidia/stt_en_conformer_ctc_large>`_ (several thousand hours of English speech).
* Spanish `stt_es_conformer_ctc_large <https://huggingface.co/nvidia/stt_es_conformer_ctc_large>`_ (1340 hours of Spanish speech).
* Italian `stt_it_conformer_ctc_large <https://huggingface.co/nvidia/stt_it_conformer_ctc_large>`_ (487 hours of Italian speech).

To finetune a model with the same vocabulary size, just set the desired model via the *init_from_pretrained_model* parameter:

.. code-block:: bash

    +init_from_pretrained_model=${PRETRAINED_MODEL_NAME}

If the size of the vocabulary differs from the one presented in the pretrained model, you need to change the vocabulary manually as done in the `finetuning tutorial <https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/ASR_CTC_Language_Finetuning.ipynb>`_:

.. code-block:: python

    model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(f"nvidia/{PRETRAINED_MODEL_NAME}", map_location='cpu')
    model.change_vocabulary(new_tokenizer_dir=TOKENIZER, new_tokenizer_type="bpe")
    model.encoder.unfreeze()
    model.save_to(f"{save_path}")


There is no need to change anything for the SSL model, it will replace the vocabulary itself. However, you will need to first download this model and set it through another parameter *init_from_nemo_model*:

.. code-block:: bash

    ++init_from_nemo_model=${PRETRAINED_MODEL} \

As the SSL model, we use `ssl_en_conformer_large <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/ssl_en_conformer_large>`_ which is trained using LibriLight corpus (~56k hrs of unlabeled English speech).
All models for fine-tuning are available on `Nvidia Hugging Face <https://huggingface.co/nvidia>`_ or `NGC <https://catalog.ngc.nvidia.com/models>`_ repo.

The following table shows all results for fine-tuning from pretrained models for the Conformer-CTC-large model and compares them with the model that was obtained by training from scratch (here we use BPE size 128 for all the models because it gives the best results).

+----------------------------------+------------+-------------+
| Training mode                    | DEV, WER % | TEST, WER % |
+==================================+============+=============+
| From scratch                     |     3.96   |     6.48    |
+----------------------------------+------------+-------------+
| Finetuning (English)             |     3.45   |     5.45    |
+----------------------------------+------------+-------------+
| Finetuning (Spanish)             |     3.40   |     5.52    |
+----------------------------------+------------+-------------+
| Finetuning (Italian)             |     3.29   |     5.36    |
+----------------------------------+------------+-------------+
| Finetuning (SSL English)         |  **2.90**  |   **4.76**  |
+----------------------------------+------------+-------------+

We can also monitor test WER behavior during training process using wandb plots (X - global step, Y - test WER):

.. image:: ./images/test_wer_wandb.png
    :align: center
    :alt: Test WER.
    :width: 800px

As you can see, the best way to get the Esperanto ASR model (the model can be found on `NGC <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_eo_conformer_ctc_large>`_ and `Hugging Face <https://huggingface.co/nvidia/stt_eo_conformer_ctc_large>`_) is finetuning from the pretrained SSL model for English.


********
Decoding
********

At the end of the training, several checkpoints (usually 5) and the best model (not always from the latest epoch) are stored in the model folder. Checkpoint averaging (script) can help to improve the final decoding accuracy. In our case, this did not improve the CTC models. However, it was possible to get an improvement in the range of 0.1-0.2% WER for some RNNT models. To make averaging, use the following command:

.. code-block:: bash

    python ${NEMO_ROOT}/scripts/checkpoint_averaging/checkpoint_averaging.py <your_trained_model.nemo>

For decoding you can use:

.. code-block:: bash

    python ${NEMO_ROOT}/examples/asr/speech_to_text_eval.py \
        model_path=${MODEL} \
        pretrained_name=null \
        dataset_manifest=${TEST_MANIFEST} \
        batch_size=${BATCH_SIZE} \
        output_filename=${OUTPUT_MANIFEST} \
        amp=False \
        use_cer=False)

You can use the Speech Data Explorer to analyze recognition errors, similar to the Kinyarwanda example.
We listened to files with an anomaly high WER (>50%) and found many problematic files. They have wrong transcriptions and cut or empty audio files in the dev and test sets.

.. code-block:: bash

    python ${NEMO_ROOT}/tools/speech_data_explorer/data_explorer.py <your_decoded_manifest_file>


**********************
Training data analysis
**********************

For an additional analysis of the training dataset, you can decode it using an already trained model. Train examples with a high error rate (WER > 50%) are likely to be problematic files. Removing them from the training set is preferred because a model can train text even for almost empty audio. We do not want this behavior from the ASR model.


