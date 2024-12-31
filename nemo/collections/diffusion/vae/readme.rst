============================
Pretraining Variational AutoEncoder
============================

Variational Autoencoder (VAE) is a data compression technique that compresses high-resolution images into a lower-dimensional latent space, capturing essential features while reducing dimensionality. This process allows for efficient storage and processing of image data. VAE has been integral to training Stable Diffusion (SD) models, significantly reducing computational requirements. For instance, SDLX utilizes a VAE that reduces image dimensions by 8x, greatly optimizing the training and inference processes. In this repository, we provide training codes to pretrain the VAE from scratch, enabling users to achieve higher compression ratios in the spatial dimension, such as 16x or 32x.

Installation
============

Please pull the latest NeMo docker to get started, see details about NeMo docker `here <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo>`_.

Validation
========
We also provide a validation code for you to quickly evaluate our pretrained 16x VAE model on a 50K dataset. Once you start the docker, run the following script to start the testing.

.. code-block:: bash

   torchrun --nproc-per-node 8 nemo/collections/diffusion/vae/validate_vae.py --yes data.path=path/to/validation/data log.log_dir=/path/to/checkpoint

Configure the following variables:


1. ``data.path``: Set this to the directory containing your test data (e.g., `.jpg` or `.png` files). The original and VAE-reconstructed images will be logged side by side in Weights & Biases (wandb).

2. ``log.log_dir``: Set this to the directory containing the pretrained checkpoint. You can find our pretrained checkpoint at ``TODO by ethan``

Here are some sample images generated from our pretrained VAE.

``Left``: Original Image, ``Right``: 16x VAE Reconstructed Image

.. list-table::
   :align: center

   * - .. image:: https://github.com/user-attachments/assets/08122f5b-2e65-4d65-87d7-eceae9d158fb
         :width: 1400
         :align: center
     - .. image:: https://github.com/user-attachments/assets/6e805a0d-8783-4d24-a65b-d96a6ba1555d
         :width: 1400
         :align: center
     - .. image:: https://github.com/user-attachments/assets/aab1ef33-35da-444d-90ee-ba3ad58a6b2d
         :width: 1400
         :align: center

Data Preparation
========

1. we expect data to be in the form of WebDataset tar files. If you have a folder of images, you can use `tar` to convert them into WebDataset tar files:

    .. code-block:: bash

        000000.tar
        ├── 1.jpg
        ├── 2.jpg
        000001.tar
        ├── 3.jpg
        ├── 4.jpg

2. next we need to index the webdataset with `energon <https://nvidia.github.io/Megatron-Energon/>`_. navigate to the dataset directory and run the following command:

    .. code-block:: bash

        energon prepare . --num-workers 8 --shuffle-tars

3. then select dataset type `ImageWebdataset` and specify the type `jpg`. Below is an example of the interactive setup:

    .. code-block:: bash
        
        Found 2925 tar files in total. The first and last ones are:
        - 000000.tar
        - 002924.tar
        If you want to exclude some of them, cancel with ctrl+c and specify an exclude filter in the command line.
        Please enter a desired train/val/test split like "0.5, 0.2, 0.3" or "8,1,1": 99,1,0
        Indexing shards  [####################################]  2925/2925
        Sample 0, keys:
        - jpg
        Sample 1, keys:
        - jpg
        Found the following part types in the dataset: jpg
        Do you want to create a dataset.yaml interactively? [Y/n]:
        The following dataset classes are available:
        0. CaptioningWebdataset
        1. CrudeWebdataset
        2. ImageClassificationWebdataset
        3. ImageWebdataset
        4. InterleavedWebdataset
        5. MultiChoiceVQAWebdataset
        6. OCRWebdataset
        7. SimilarityInterleavedWebdataset
        8. TextWebdataset
        9. VQAOCRWebdataset
        10. VQAWebdataset
        11. VidQAWebdataset
        Please enter a number to choose a class: 3
        The dataset you selected uses the following sample type:

        @dataclass
        class ImageSample(Sample):
            """Sample type for an image, e.g. for image reconstruction."""

            #: The input image tensor in the shape (C, H, W)
            image: torch.Tensor

        Do you want to set a simple field_map[Y] (or write your own sample_loader [n])? [Y/n]:

        For each field, please specify the corresponding name in the WebDataset.
        Available types in WebDataset: jpg
        Leave empty for skipping optional field
        You may also access json fields e.g. by setting the field to: json[field][field]
        You may also specify alternative fields e.g. by setting to: jpg,png
        Please enter the field_map for ImageWebdataset:
        Please enter a webdataset field name for 'image' (<class 'torch.Tensor'>):
        That type doesn't exist in the WebDataset. Please try again.
        Please enter a webdataset field name for 'image' (<class 'torch.Tensor'>): jpg
        Done

4. finally, you can use the indexed dataset to train the VAE model. specify `data.path=/path/to/dataset` in the training script `train_vae.py`.

Training
========

We provide a sample training script for launching multi-node training. Simply configure ``data.path`` to point to your prepared dataset to get started.

.. code-block:: bash

   bash nemo/collections/diffusion/vae/train_vae.sh \
   data.path=xxx





