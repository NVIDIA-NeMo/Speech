HFDatasetDataModule
===================

`HFDatasetDataModule` is a PyTorch Lightning `LightningDataModule` that wraps Hugging Face (HF) datasets, enabling seamless integration with NeMo Framework. It allows users to load and preprocess datasets from the Hugging Face hub or custom datasets, supporting multiple data splits.


.. warning::
    This class requires the following packages:
    - `lightning.pytorch`
    - `torch`
    - `datasets`
    - `nemo`

    Ensure they are installed in your environment before using this module.

Class Definition
----------------

.. code-block:: python

    class HFDatasetDataModule(pl.LightningDataModule):
        ...

Initialization
--------------

.. autofunction:: HFDatasetDataModule.__init__

**Parameters:**

- **path_or_dataset** (`str` or `datasets.Dataset` or `datasets.DatasetDict`):
  - Path to the dataset on the Hugging Face Hub or a pre-loaded Hugging Face `Dataset` or `DatasetDict` object.

- **split** (`str` or `list`, optional):
  - Specifies which split(s) of the dataset to load. Can be a single split name (e.g., `"train"`) or a list of split names (e.g., `["train", "validation"]`).

- **collate_fn** (`callable`, optional):
  - A function to collate samples into batches. If `None`, a default collate function is used.

- **num_workers** (`int`, default=2):
  - Number of subprocesses to use for data loading.

- **pin_memory** (`bool`, default=True):
  - If `True`, the data loader will copy Tensors into CUDA pinned memory before returning them.

- **persistent_workers** (`bool`, default=True):
  - If `True`, the data loader will not shutdown the worker processes after a dataset has been consumed once.

- **seq_length** (`int`, default=1024):
  - Sequence length for the data sampler.

- **micro_batch_size** (`int`, default=2):
  - Batch size per micro-batch.

- **global_batch_size** (`int`, default=2):
  - Total batch size across all GPUs.

- **pad_token_id** (`int`, default=0):
  - Token ID used for padding sequences.

- **train_aliases** (`list` of `str`, default=["train", "training"]):
  - Synonyms for the training split.

- **test_aliases** (`list` of `str`, default=["test", "testing"]):
  - Synonyms for the testing split.

- **val_aliases** (`list` of `str`, default=["val", "validation", "valid", "eval"]):
  - Synonyms for the validation split.

- **kwargs**:
  - Additional keyword arguments passed to `datasets.load_dataset`.

Attributes
----------

- **dataset_splits** (`dict`):
  - Dictionary containing the dataset splits (`'train'`, `'val'`, `'test'`).

- **_collate_fn** (`callable`):
  - Function used to collate batches.

- **num_workers** (`int`):
  - Number of workers for data loading.

- **pin_memory** (`bool`):
  - Pin memory flag for data loading.

- **persistent_workers** (`bool`):
  - Persistent workers flag for data loading.

- **seq_length** (`int`):
  - Sequence length for data sampling.

- **micro_batch_size** (`int`):
  - Micro batch size.

- **global_batch_size** (`int`):
  - Global batch size.

- **pad_token_id** (`int`):
  - Padding token ID.

Methods
-------

.. autoclass:: HFDatasetDataModule
    :members:
    :exclude-members: __init__, __repr__, __str__

Detailed Method Descriptions
~~~~~~~~~~~~~~~~~~~~~~~~~~

train_dataloader()
~~~~~~~~~~~~~~~~~

Returns the training `DataLoader`.

**Returns:**

- **DataLoader**:
  - Data loader for the training split.

val_dataloader()
~~~~~~~~~~~~~~~

Returns the validation `DataLoader`.

**Returns:**

- **DataLoader**:
  - Data loader for the validation split.

test_dataloader()
~~~~~~~~~~~~~~~~

Returns the testing `DataLoader`.

**Returns:**

- **DataLoader**:
  - Data loader for the testing split.

map(function=None, split_names=None, **kwargs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applies a function to the specified splits of the dataset.

**Parameters:**

- **function** (`callable`, optional):
  - Function to apply to each example in the dataset.

- **split_names** (`str` or `list`, optional):
  - Specific splits to apply the function to. If `None`, applies to all splits.

- **kwargs**:
  - Additional keyword arguments passed to `Dataset.map`.

**Returns:**

- **None**

Properties
----------

.. py:attribute:: train

    Returns the training split of the dataset.

.. py:attribute:: val

    Returns the validation split of the dataset.

.. py:attribute:: test

    Returns the testing split of the dataset.

Static Methods
--------------

from_dict(dataset_dict, split, **kwargs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creates an `HFDatasetDataModule` from a dictionary.

**Parameters:**

- **dataset_dict** (`dict`):
  - Dictionary representing the dataset.

- **split** (`str` or `list`):
  - Split(s) to load.

- **kwargs**:
  - Additional keyword arguments passed to the constructor.

**Returns:**

- **HFDatasetDataModule**:
  - An instance of `HFDatasetDataModule`.

collate_fn(batch, pad_token_id=0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Default collate function to pad and batch sequences.

**Parameters:**

- **batch** (`list` of `dict`):
  - Batch of samples to collate.

- **pad_token_id** (`int`, default=0):
  - Token ID used for padding.

**Returns:**

- **dict**:
  - Batched and padded tensors.

Usage Examples
--------------

**Loading a Single Split:**

.. code-block:: python

    from nemo_lightning import HFDatasetDataModule

    data_module = HFDatasetDataModule(
        path_or_dataset="rajpurkar/squad",
        split="train",
        micro_batch_size=8,
        global_batch_size=64,
        pad_token_id=0
    )

    train_loader = data_module.train_dataloader()

**Loading Multiple Splits:**

.. code-block:: python

    from nemo_lightning import HFDatasetDataModule

    data_module = HFDatasetDataModule(
        path_or_dataset="rajpurkar/squad",
        split=["train", "validation"],
        micro_batch_size=8,
        global_batch_size=64,
        pad_token_id=0
    )

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

Mapping a Function Over the Dataset
-----------------------------------

.. code-block:: python

    def preprocess(example):
        example['input_ids'] = tokenizer.encode(example['text'])
        return example

    data_module.map(preprocess, split_names=["train", "val"])

