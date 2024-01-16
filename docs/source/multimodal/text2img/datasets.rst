Datasets
========

Data pipeline overview
-----------------

.. note:: It is the responsibility of each user to check the content of the dataset, review the applicable licenses, and determine if it is suitable for their intended use. Users should review any applicable links associated with the dataset before placing the data on their machine.

For all text2img multimodal models, we provide a generic pipeline as detailed below to download and prepare the dataset. 
The pipeline is suitable for any multimodal datasets hosted on the HuggingFace data repository
where the data is stored as one or more parquet files. The pipeline processes the dataset into the
WebDataset format, consisting of tar files of equal sizes for efficient training.

The 6 sub-stages are as follows.

    #. download_parquet: Parquet files consisting of text (captions) and image URLs are downloaded from a HuggingFace repository.

    #. download_images: The images are downloaded from their respective URLs and, along with the captions, are packed into tar files following the Webdataset format.

    #. reorganize_tar: (Optional) Due to a variety of reasons (such as unstable network or removal of images), some images may fail to download, resulting in uneven tar files with varying number of examples each. If you are using a training sampler that does not support uneven tar files, you need to re-organize the contents of the tar files so that each one contains an equal number of image-text pairs.

    #. precache_encodings: (Optional) If you are training a model with frozen encoders (e.g. Stable Diffusion), you have the option to precache (precompute) image and/or text encodings (embeddings) in this sub-stage. Precaching these encodings can significantly enhance training throughput.

    #. generate_wdinfo: (Optional) The wdinfo.pkl file, which stores information on dataset shards, is generated.

Depending on your specific circumstance, not all sub-stages need to be run all at once.
For example, for parquet datasets not hosted on HuggingFace or those whose format is not parquet,
sub-stages 2-5 can be used to process locally downloaded datasets.
For webdatasets already downloaded locally, sub-stages 4-5 can be used to precache the encoding to reduce training time.
For models that encode image and text on-the-fly, only sub-stages 1-3 need to be run.

Instruction for configuring each sub-stage is provided as a comment next to each field in
`download_multimodal.yaml <http://TODOURL>`_


Examples of Preparing a Dataset for Training Text2Img Model
-----------------------

Refer to the `Dataset Tutorial <http://TODOURL>`_` for details on how to prepare the training dataset for Training Text2Img models.

