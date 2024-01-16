Datasets
========

ImageNet Data Preparation
-----------------

.. note:: It is the responsibility of each user to check the content of the dataset, review the applicable licenses, and determine if it is suitable for their intended use. Users should review any applicable links associated with the dataset before placing the data on their machine.

Please note that according to the ImageNet terms and conditions, automated scripts for downloading the dataset are not
provided. Instead, one can follow the steps outlined below to download and extract the data.

ImageNet 1k
^^^^^^^^^^^^^^^

1. Create an account on `ImageNet <http://image-net.org/download-images>`_ and navigate to ILSVRC 2012.
   Download "Training images (Task 1 & 2)" and "Validation images (all tasks)" to ``data/imagenet_1k``.
2. Extract the training data:

.. code-block::

  mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
  tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
  find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
  cd ..

3. Extract the validation data and move the images to subfolders:

.. code-block::

  mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
  wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash


ImageNet 21k
^^^^^^^^^^^^^^^

1. Create an account on `ImageNet <http://image-net.org/download-images>`_ and download "ImageNet21k" to
   ``data/imagenet_21k``.
2. Extract the data:

.. code-block::

  tar -xvf winter21_whole.tar.gz && rm -f winter21_whole.tar.gz
  find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done

