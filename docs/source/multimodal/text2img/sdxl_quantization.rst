Stable Diffusion XL Int8 Quantization
=======================================

This example shows how to use ModelOpt to calibrate and quantize the UNet part of the SDXL. The UNet part typically consumes
>95% of the e2e Stable Diffusion latency.

We also provide instructions on deploying and running E2E SDXL pipeline
with ModelOpt quantized int8 UNet to generate images and measure latency on target GPUs.

To get started, it is required to have a pretrained SDXL checkpoint in ``nemo`` format. The example training configs are provided in NeMo,
which is located in ``NeMo/examples/multimodal/text2img/stable_diffusion``.

Calibration
---------------
The first step is to run quantization script with default config, and finally the script will export the quantized unet to onnx file.
Here is the default config for ``NeMo/examples/multimodal/text2img/stable_diffusion/sd_xl_quantize.py``.


.. code-block:: yaml

    quantize
      exp_name: nemo
      n_steps: 20          # number of inference steps
      format: 'int8'       # only int8 quantization is supported now
      percentile: 1.0      # Control quantization scaling factors (amax) collecting range, meaning that we will collect the minimum amax in the range of `(n_steps * percentile)` steps. Recommendation: 1.0
      batch_size: 1        # batch size calling sdxl inference pipeline during calibration
      calib_size: 32       # For SDXL, we recommend 32, 64 or 128
      quant_level: 2.5     #Which layers to be quantized, 1: `CNNs`, 2: `CNN + FFN`, 2.5: `CNN + FFN + QKV`, 3: `CNN + Linear`. Recommendation: 2, 2.5 and 3, depending on the requirements for image quality & speedup.
      alpha: 0.8           # A parameter in SmoothQuant, used for linear layers only. Recommendation: 0.8 for SDXL



Important Parameters
^^^^^^^^^^^^^^^^^^^^
- percentile: Control quantization scaling factors (amax) collecting range, meaning that we will collect the minimum amax in the range of (n_steps * percentile) steps. Recommendation: 1.0
- alpha: A parameter in SmoothQuant, used for linear layers only. Recommendation: 0.8 for SDXL, 1.0 for SD 1.5
- quant-level: Which layers to be quantized, 1: CNNs, 2: CNN + FFN, 2.5: CNN + FFN + QKV, 3: CNN + Linear. Recommendation: 2, 2.5 and 3, depending on the requirements for image quality & speedup.
- calib-size: For SDXL, we recommend 32, 64 or 128, for SD 1.5, set to 512 or 1024.


Build the TRT engine for the Quantized ONNX UNet
------------------------------------------------------------

.. code-block:: bash

    trtexec --onnx=./nemo_onnx/unet.onnx --shapes=x:8x4x128x128,timesteps:8,context:8x80x2048,y:8x2816 --fp16 --int8 --builderOptimizationLevel=4 --saveEngine=nemo_unet_xl.plan


Important Parameters
^^^^^^^^^^^^^^^^^^^^
Input shape has to be provided here when building TRT engine.
- x: Input image latent shape (B * C * H * W)
- context: Input text conditioning (B * S * hidden_dimention)
- y: Additional embedding (B * adm_in_channels)

Build End-to-end Stable Diffusion XL Pipeline with NeMo
-----------------------------------------------------------

We provide a script to build end to end TRT inference pipeline with NeMo backend, which is located at `NeMo/examples/multimodal/text2img/stable_diffusion/sd_xl_export.py`.

.. code-block:: yaml

    infer:
        out_path: sdxl_export
        width: 1024
        height: 1024
        batch_size: 2

    trt:
      static_batch: False
      min_batch_size: 1
      max_batch_size: 8

Important Parameters
^^^^^^^^^^^^^^^^^^^^
- out_path: Directory to save onnx file and TRT engine files
- width and height: Image resolution of inference output
- batch_size: Only used for dummy input generation and onnx sanity check
- {min,max}_batch_size: The input batch size of TRT engine along its dynamic axis


Run End-to-end Stable Diffusion XL TRT Pipeline
-----------------------------------------------------------

The inference script can be found at `NeMo/examples/multimodal/text2img/stable_diffusion/sd_xl_trt_inference.py`.

.. code-block:: yaml

    unet_xl: sdxl_export/plan/unet_xl.plan
    vae: sdxl_export/plan/vae.plan
    clip1: sdxl_export/plan/clip1.plan
    clip2: sdxl_export/plan/clip2.plan

    out_path: trt_output


Please specify unet_xl as the quantized Unet engine to run the quantized solution. The system will load the original engine file by default.

Inference Speedup
-------------------
TRT version  9.3.0
GPU: H100

TRT int8 vs Framework fp16
^^^^^^^^^^^^^^^^^^^^^^^^^^^

+---------------------+------------+-------------+--------------------+------------+---------+------------+
| Pipeline            | Batch Size | Latency (ms)| Pipeline           | Batch Size | Latency | Speedup    |
+=====================+============+=============+====================+============+=========+============+
| Framework fp16 base | 1          | 3056.01     | ModelOpt TRT Int8  | 1          | 1406.68 | 2.172498365|
+---------------------+------------+-------------+--------------------+------------+---------+------------+
| Framework fp16 base | 2          | 4832.24     | ModelOpt TRT Int8  | 2          | 2403.29 | 2.01067703 |
+---------------------+------------+-------------+--------------------+------------+---------+------------+
| Framework fp16 base | 4          | 8433.71     | ModelOpt TRT Int8  | 4          | 4252.6  | 1.983189108|
+---------------------+------------+-------------+--------------------+------------+---------+------------+



TRT int8 vs TRT fp16
^^^^^^^^^^^^^^^^^^^^^^^


+-------------+------------+--------------+---------------+------------+------------+-------------+
| Pipeline    | Batch Size | Latency (ms) | Precision     | Batch Size | Latency    | Speedup     |
+=============+============+==============+===============+============+============+=============+
| fp16 base   | 1          | 1723.97      | ModelOpt Int8 | 1          | 1406.68    | 1.225559473 |
+-------------+------------+--------------+---------------+------------+------------+-------------+
| fp16 base   | 2          | 3004.47      | ModelOpt Int8 | 2          | 2403.29    | 1.250148754 |
+-------------+------------+--------------+---------------+------------+------------+-------------+
| fp16 base   | 4          | 5657.19      | ModelOpt Int8 | 4          | 4252.6     | 1.330289705 |
+-------------+------------+--------------+---------------+------------+------------+-------------+


FP16 inference vs Int8 inference
----------------------------------

.. image:: https://github.com/NVIDIA/NeMo/releases/download/v1.23.0/asset-githubio-home-sdxl_trt_fp16_1.png
   :width: 50%
.. image:: https://github.com/NVIDIA/NeMo/releases/download/v1.23.0/asset-githubio-home-sdxl_trt_int8_1.png
   :width: 50%
Prompt: A photo of a Shiba Inu dog with a backpack riding a bike. It is wearing sunglasses and a beach hat. (FP16 upper vs Int8 lower)




.. image:: https://github.com/NVIDIA/NeMo/releases/download/v1.23.0/asset-githubio-home-sdxl_trt_fp16_2.png
   :width: 50%
.. image:: https://github.com/NVIDIA/NeMo/releases/download/v1.23.0/asset-githubio-home-sdxl_trt_int8_2.png
   :width: 50%
Prompt: A cute corgi lives in a house made out of sushi. (FP16 upper vs Int8 lower)




.. image:: https://github.com/NVIDIA/NeMo/releases/download/v1.23.0/asset-githubio-home-sdxl_trt_fp16_3.png
   :width: 50%
.. image:: https://github.com/NVIDIA/NeMo/releases/download/v1.23.0/asset-githubio-home-sdxl_trt_int8_3.png
   :width: 50%
Prompt: A high contrast portrait of a very happy fuzzy panda dressed as a chef in a high end kitchen making dough. There is a painting of flowers on the wall behind him. (FP16 upper vs Int8 lower)

