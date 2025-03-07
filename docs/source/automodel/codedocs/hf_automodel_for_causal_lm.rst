HFAutoModelForCausalLM Class
============================

.. currentmodule:: nemo.collections.llm.gpt.model.hf_auto_model_for_causal_lm

Overview
--------
`HFAutoModelForCausalLM` is a PyTorch Lightning module designed to facilitate the integration and training of Hugging Face's causal language models within the NeMo Framework. It leverages functionalities from `lightning.pytorch`, `transformers`, and NeMo's utilities to provide a flexible and efficient setup for language model training and fine-tuning.


Inheritance
-----------
.. inheritance-diagram:: HFAutoModelForCausalLM
    :top-classes: lightning.pytorch.LightningModule
    :parts: 1

Constructor
-----------
.. constructor:: HFAutoModelForCausalLM(*, model_name='gpt2', load_pretrained_weights=True, tokenizer=None, loss_fn=masked_cross_entropy, model_transform=None, model_accelerator=None, trust_remote_code=False, default_dtype=torch.bfloat16, load_in_4bit=False, attn_implementation="sdpa")

    Initializes the `HFAutoModelForCausalLM` module.

    Parameters
    ----------
    model_name : str, optional
        Name or path of the pre-trained model to load (default is `'gpt2'`).
    load_pretrained_weights : bool, optional
        Whether to load pre-trained weights from the specified model (default is `True`).
    tokenizer : `AutoTokenizer`, optional
        Tokenizer instance to use. If `None`, a tokenizer is configured based on `model_name`.
    loss_fn : callable, optional
        Loss function to use for training. Defaults to `masked_cross_entropy`.
    model_transform : callable, optional
        Optional transformation function to apply to the model after loading.
    model_accelerator : callable, optional
        Accelerator function to modify the model for specific hardware accelerators.
    trust_remote_code : bool, optional
        Whether to trust remote code when loading the model (default is `False`).
    default_dtype : torch.dtype, optional
        Default data type for model parameters (default is `torch.bfloat16`).
    load_in_4bit : bool, optional
        Whether to load the model in 4-bit precision for memory efficiency (default is `False`).
    attn_implementation : str, optional
        Attention mechanism implementation to use (default is `"sdpa"`).

Attributes
----------
model_name : str
    Name or path of the pre-trained model.
_tokenizer : `AutoTokenizer` or `None`
    Tokenizer instance used for encoding and decoding text.
model : `AutoModelForCausalLM` or `None`
    The underlying Hugging Face causal language model.
loss_fn : callable
    Loss function used for training.
load_pretrained_weights : bool
    Flag indicating whether to load pre-trained weights.
is_hf_model : bool
    Indicates if the model is a Hugging Face model.
model_transform : callable or `None`
    Optional transformation function applied to the model.
model_accelerator : callable or `None`
    Accelerator function applied to the model.
trust_remote_code : bool
    Flag to trust remote code when loading the model.
default_dtype : torch.dtype
    Default data type for model parameters.
load_in_4bit : bool
    Flag to load the model in 4-bit precision.
attn_implementation : str
    Attention mechanism implementation used.

Properties
----------
.. py:attribute:: tokenizer

    Returns the tokenizer instance. If not already set, it configures a tokenizer based on `model_name`.

Methods
-------
.. automethod:: configure_tokenizer

.. automethod:: configure_model

.. automethod:: forward

.. automethod:: training_step

.. automethod:: validation_step

.. automethod:: save_pretrained

.. automethod:: _remove_extra_batch_keys

configure_tokenizer
~~~~~~~~~~~~~~~~~~
.. staticmethod:: configure_tokenizer(model_name, trust_remote_code=False)

    Configures and returns a tokenizer based on the given `model_name`.

    Parameters
    ----------
    model_name : str
        Name or path of the pre-trained tokenizer to load.
    trust_remote_code : bool, optional
        Whether to trust remote code when loading the tokenizer (default is `False`).

    Returns
    -------
    `AutoTokenizer`
        Configured tokenizer instance.

configure_model
~~~~~~~~~~~~~~
Configures the underlying Hugging Face model. If `load_pretrained_weights` is `True`, it loads the model with pre-trained weights. Otherwise, it initializes the model from scratch using the specified configuration. Additionally, it applies Fully Sharded Data Parallel (FSDP) and tensor parallelism if `device_mesh` is provided and applies any specified model accelerators.

forward
~~~~~~
.. py:method:: forward(batch)

    Performs a forward pass through the model.

    Parameters
    ----------
    batch : dict
        A batch of input data containing necessary inputs for the model.

    Returns
    -------
    `transformers.modeling_outputs.CausalLMOutput`
        The output from the causal language model.

training_step
~~~~~~~~~~~~
.. py:method:: training_step(batch, batch_idx=None)

    Defines the training step. Computes the loss using the specified loss function and logs the training loss.

    Parameters
    ----------
    batch : dict
        A batch of training data.
    batch_idx : int, optional
        Index of the batch (default is `None`).

    Returns
    -------
    torch.Tensor
        Computed loss for the batch.

validation_step
~~~~~~~~~~~~~~
.. py:method:: validation_step(batch, batch_idx)

    Defines the validation step. Computes and logs the validation loss.

    Parameters
    ----------
    batch : dict
        A batch of validation data.
    batch_idx : int
        Index of the batch.

save_pretrained
~~~~~~~~~~~~~~
.. py:method:: save_pretrained(path)

    Saves the model and tokenizer to the specified directory.

    Parameters
    ----------
    path : str
        Directory path where the model and tokenizer will be saved.

_remove_extra_batch_keys
~~~~~~~~~~~~~~~~~~~~~~~
.. py:method:: _remove_extra_batch_keys(batch, reserved_keys=['labels', 'loss_mask'])

    Removes keys from the batch that are not required by the model's forward method, except for reserved keys.

    Parameters
    ----------
    batch : dict
        Dictionary of tensors representing a batch.
    reserved_keys : list of str, optional
        Keys to retain in the batch regardless of the model's forward method requirements (default is `['labels', 'loss_mask']`).

    Returns
    -------
    dict
        Filtered batch containing only the necessary keys.

Utility Functions
-----------------
.. function:: masked_cross_entropy(logits, targets, mask=None)

    Computes the cross-entropy loss, optionally applying a mask to the loss values.

    Parameters
    ----------
    logits : torch.Tensor
        Logits output from the model.
    targets : torch.Tensor
        Ground truth target indices.
    mask : torch.Tensor or None, optional
        Mask to apply to the loss values (default is `None`).

    Returns
    -------
    torch.Tensor
        Computed loss.

Usage Example
-------------
```python
from your_module_name import HFAutoModelForCausalLM

# Initialize the model
model = HFAutoModelForCausalLM(
    model_name='gpt2',
    load_pretrained_weights=True,
    trust_remote_code=False
)

# Configure the model (typically called internally)
model.configure_model()

# Example training step
batch = {
    'input_ids': torch.tensor([[...]]),
    'labels': torch.tensor([[...]]),
    'loss_mask': torch.tensor([[...]])
}
loss = model.training_step(batch)
