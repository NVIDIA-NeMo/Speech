Checkpoints
===========

There are three main ways to load pretrained checkpoints in NeMo:

* Using the :code:`restore_from()` method to load a local checkpoint file (``.nemo``), or
* Converting a partially trained ``.ckpt`` (intermediate) checkpoint to ``.nemo`` format.
* Converting HuggingFace public checkpoints to ``.nemo`` format.

Refer to the following sections for instructions and examples for each.

Note that these instructions are for loading fully trained checkpoints for evaluation or fine-tuning.

Loading ``.nemo`` Checkpoints
-------------------------

NeMo automatically saves checkpoints of a model that is trained in a ``.nemo`` format. Alternatively, to manually save the model at any 
point, issue :code:`model.save_to(<checkpoint_path>.nemo)`.

If there is a local ``.nemo`` checkpoint that you'd like to load, use the :code:`restore_from()` method:

.. code-block:: python

  import nemo.collections.multimodal as nemo_multimodal
  model = nemo_multimodal.models.<MODEL_BASE_CLASS>.restore_from(restore_path="<path/to/checkpoint/file.nemo>")

Where the model base class is the MM model class of the original checkpoint.

Converting Intermediate Checkpoints
---------------------------
To evaluate a partially trained checkpoint, you may need to convert it to ``.nemo`` format.
`script to convert the checkpoint <ADD convert_ckpt_to_nemo.py PATH>`.

.. code-block:: python

  python -m torch.distributed.launch --nproc_per_node=<tensor_model_parallel_size> * <pipeline_model_parallel_size> \
    convert_ckpt_to_nemo.py \
    --checkpoint_folder <path_to_PTL_checkpoints_folder> \
    --checkpoint_name <checkpoint_name> \
    --nemo_file_path <path_to_output_nemo_file> \
    --tensor_model_parallel_size <tensor_model_parallel_size> \
    --pipeline_model_parallel_size <pipeline_model_parallel_size>


Converting HuggingFace Checkpoints
---------------------------------

To fully utilize the optimized training pipeline and framework/TRT inference pipeline
of NeMo, we provide scripts to convert popular checkpoints on HuggingFace into NeMo format.
Once converted, you can perform fine-tuning or inference on such checkpoints.

Stable Diffusion & ControlNet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We provide `script to convert the Huggingface checkpoint <ADD convert_hf_ckpt_to_nemo.py PATH>` to ``.nemo`` format, which can then be used within our inference pipeline.


.. code-block:: python

  python convert_hf_ckpt_to_nemo.py \
    --ckpt_path <path_to_HF_checkpoints> \
    --hparams_file <path_to_hparams_file> \
    --nemo_file_path <path_to_output_nemo_file> \
    --model_type <model_to_be_converted> \
    --nemo_clip_path <clip_ckpt_in_nemo_format>


- ``hparams_file``: Config file to be combined with model weights to generate ``.nemo`` checkpoint. It can be generated from a dummy run and can be found at, for example, ``nemo_experiments/stable-diffusion-train/version_0/hparams.yaml``.

- ``model_type``: We support converting `stable_diffusion` and `controlnet` checkpoint in this script.

- ``nemo_clip_path``: It's required only when the ``cond_stage_config`` in ``hparams_file`` refer to a NeMo CLIP model. It will be ignored when ``cond_stage_config`` refer to Hugginface CLIP. See :ref:`sd-config-section` for more details.


Imagen
^^^^^^^^^^^^^^

We will provide conversion script if Imagen research team releases their checkpoint
in the future. Conversion script for DeepFloyd IF models will be provided in the
next release.