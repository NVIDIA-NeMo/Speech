Video NeVA
==========

Model Introduction
------------------

Video NeVa adds support for video modality in NeVa by representing video as multiple image frames. 

There is only a minor change done to :class:`~nemo.collections.multimodal.models.multimodal_llm.neva.neva_model.MegatronNevaModel` class in order to support pretraining on video input data.

Representing video input as a series of images is done in :class:`~nemo.collections.multimodal.data.neva.TarOrFolderVideoLoader` class, using Decord which provides convenient video slicing methods. 


Video Neva Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

  data:
    media_type: video
    splice_single_frame: null
    num_frames: 8
    image_token_len: 256
    image_folder: null
    video_folder: null

- ``media_type``: If set to `video`, NeVa's dataloader goes through the additional preprocessing steps to represent the input video data as a series of image frames.
- ``splice_single_frame``: Can either be set as `first`, `middle` or `last`. This will result in only a single frame in that specific location of the video being selected.
- ``image_token_len``: The NeVa dataloader calculates `image_token_len` based on the height and width of the preprocessed image frame and the patch size of the CLIP model being used. 

.. code-block:: python

  image_token_len = (224 // 14) * (224 // 14) = 16 * 16 = 256

- ``num_frames``: This is used to select the number of image frames that will be used to represent the video.
- ``video_folder``: This specifies the directory where the video files are located. This follows the same format as NeVa's `image_folder`.



Inference with Video NeVA
=========================

We can run ``neva_evaluation.py`` located in ``NeMo/examples/multimodal/multimodal_llm/neva`` to generate inference results from the Video NeVA model.
Currently, video NeVA supports both image and video inference by changing the config attribute ``inference.media_type`` in ``NeMo/examples/multimodal/multimodal_llm/neva/conf/neva_inference.yaml`` to either ``image`` or ``video``, and adding the corresponding media path ``inference.media_base_path``.

Inference with Pretrained Projectors with Base LM Model
-------------------------------------------------------

An example of an inference script execution:

For running video inference::

    CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 /path/to/neva_evaluation.py \
    --config-path=/path/to/conf/ \
    --config-name=neva_inference.yaml \
    tensor_model_parallel_size=4 \
    pipeline_model_parallel_size=1 \
    neva_model_file=/path/to/projector/checkpoint \
    base_model_file=/path/to/base/lm/checkpoint \
    trainer.devices=4 \
    trainer.precision=bf16 \
    prompt_file=/path/to/prompt/file \
    inference.media_base_path=/path/to/videos \
    inference.media_type=video \
    output_file=/path/for/output/file/ \
    inference.temperature=0.2 \
    inference.top_k=0 \
    inference.top_p=0.9 \
    inference.greedy=False \
    inference.add_BOS=False \
    inference.all_probs=False \
    inference.repetition_penalty=1.2 \
    inference.insert_media_token=right \
    inference.tokens_to_generate=256 \
    quantization.algorithm=awq \
    quantization.enable=False

Example format of ``.jsonl`` prompt_file::

    {"video": "video_test.mp4", "text": "Can you describe the scene?", "category": "conv", "question_id": 0}

input video file:: video_test.mp4

Output::

    <extra_id_0>System
    A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

    <extra_id_1>User
    Can you describe the scene?<video>
    <extra_id_1>Assistant
    <extra_id_2>quality:4,toxicity:0,humor:0,creativity:0,helpfulness:4,correctness:4,coherence:4,complexity:4,verbosity:4
    CLEAN RESPONSE: Hand with a robot arm


Inference with Finetuned Video NeVA Model (No Need to Specify Base LM)
----------------------------------------------------------------------

An example of an inference script execution:

For running video inference::

    CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 /path/to/neva_evaluation.py \
    --config-path=/path/to/conf/ \
    --config-name=neva_inference.yaml \
    tensor_model_parallel_size=4 \
    pipeline_model_parallel_size=1 \
    neva_model_file=/path/to/video/neva/model \
    trainer.devices=4 \
    trainer.precision=bf16 \
    prompt_file=/path/to/prompt/file \
    inference.media_base_path=/path/to/videos \
    inference.media_type=video \
    output_file=/path/for/output/file/ \
    inference.temperature=0.2 \
    inference.top_k=0 \
    inference.top_p=0.9 \
    inference.greedy=False \
    inference.add_BOS=False \
    inference.all_probs=False \
    inference.repetition_penalty=1.2 \
    inference.insert_media_token=right \
    inference.tokens_to_generate=256 \
    quantization.algorithm=awq \
    quantization.enable=False

References
----------

.. bibliography:: ../mm_all.bib
    :style: plain
    :filter: docname in docnames
    :labelprefix: MM-MODELS
    :keyprefix: mm-models-
