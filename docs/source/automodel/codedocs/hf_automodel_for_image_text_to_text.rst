HFAutoModelForImageTextToText
==============================

.. currentmodule:: nemo.collections.vlm.hf.model.hf_auto_model_for_image_text_to_text

A PyTorch Lightning module that wraps Hugging Face's AutoModelForImageTextToText for seamless integration with NeMo Framework. This class facilitates training and evaluation of image-text-to-text models, providing functionalities such as loading pretrained weights, customizing loss functions, and handling processors.

Inheritance
-----------
:class:`HFAutoModelForImageTextToText` inherits from:
- :class:`pytorch_lightning.LightningModule`
- :class:`nemo.lightning.io.IOMixin`
- :class:`nemo.collections.llm.fn.FNMixin`

Initialization
--------------
.. autoclass:: HFAutoModelForImageTextToText
    :members:
    :special-members: __init__

Constructor Parameters
----------------------
**model_name** : `str`, optional
    Name or path of the Hugging Face model to load. Default is `'gpt2'`.
    
**load_pretrained_weights** : `bool`, optional
    Whether to load pretrained weights from the specified model. Default is `True`.
    
**processor** : `transformers.PreTrainedProcessor`, optional
    A Hugging Face processor instance. If not provided, it will be configured based on `model_name`.
    
**loss_fn** : `callable`, optional
    Loss function to use during training. Defaults to `masked_cross_entropy`.
    
**model_transform** : `callable`, optional
    A function to apply transformations to the model after loading.
    
**trust_remote_code** : `bool`, optional
    Whether to trust remote code when loading the model. Default is `False`.
    
**default_dtype** : `torch.dtype`, optional
    Default data type for the model. Default is `torch.bfloat16`.
    
**load_in_4bit** : `bool`, optional
    Whether to load the model in 4-bit precision. Default is `False`.

Attributes
----------
**model_name** : `str`
    Name or path of the Hugging Face model.
    
**_processor** : `transformers.PreTrainedProcessor` or `None`
    Processor instance for preprocessing inputs.
    
**tokenizer** : `transformers.PreTrainedTokenizer` or `None`
    Tokenizer associated with the model.
    
**model** : `transformers.PreTrainedModel` or `None`
    The underlying Hugging Face model.
    
**loss_fn** : `callable`
    The loss function used for training.
    
**load_pretrained_weights** : `bool`
    Flag indicating whether pretrained weights are loaded.
    
**is_hf_model** : `bool`
    Indicates if the model is a Hugging Face model.
    
**model_transform** : `callable` or `None`
    Transformation function applied to the model.
    
**trust_remote_code** : `bool`
    Flag indicating whether to trust remote code.
    
**load_in_4bit** : `bool`
    Flag indicating whether to load the model in 4-bit precision.

Methods
-------
.. autosummary::
    :toctree: _autosummary

    processor
    forward
    training_step
    validation_step
    save_pretrained
    extract_skipped_token_ids
    configure_model
    configure_processor

Detailed Method Descriptions
---------------------------

processor
+++++++++
**Property**

Returns the processor associated with the model. If not already set, it initializes the processor using the `model_name`.

**Getter**
- **Returns**: `transformers.PreTrainedProcessor`

**Setter**
- **Parameters**:
    - **value** : `transformers.PreTrainedProcessor`
        The processor to set.

forward
+++++++
**Signature**: `forward(batch)`

Runs a forward pass through the model with the provided batch.

**Parameters**:
- **batch** : `dict`
    A batch of input data.

**Returns**:
- **outputs** : `transformers.model_outputs.ModelOutput`
    The model's output.

training_step
+++++++++++++
**Signature**: `training_step(batch)`

Executes a single training step.

**Parameters**:
- **batch** : `dict`
    A batch of input data.
    
**Returns**:
- **loss** : `torch.Tensor`
    The computed loss for the batch.

validation_step
+++++++++++++++
**Signature**: `validation_step(batch, batch_idx)`

Executes a single validation step.

**Parameters**:
- **batch** : `dict`
    A batch of input data.
- **batch_idx** : `int`
    Index of the batch.

**Returns**:
- **None**

save_pretrained
+++++++++++++++
**Signature**: `save_pretrained(path)`

Saves the model and processor to the specified path using Hugging Face's `save_pretrained` method.

**Parameters**:
- **path** : `str`
    Directory path where the model and processor will be saved.

**Returns**:
- **None**

extract_skipped_token_ids
+++++++++++++++++++++++++
**Signature**: `extract_skipped_token_ids(tokenizer)`

Identifies and returns token IDs that should be masked in the labels based on predefined special tokens.

**Parameters**:
- **tokenizer** : `transformers.PreTrainedTokenizer`
    The tokenizer to inspect for special tokens.

**Returns**:
- **skipped_token_ids** : `torch.IntTensor`
    Tensor containing the IDs of tokens to skip.

configure_model
+++++++++++++++
**Signature**: `configure_model()`

Initializes the Hugging Face model based on the provided configuration and parameters. Loads pretrained weights if specified.

**Parameters**:
- **None**

**Returns**:
- **None**

configure_processor
+++++++++++++++++++
**Signature**: `configure_processor(model_name, trust_remote_code=False)`

Initializes and returns a Hugging Face `AutoProcessor` based on the model name.

**Parameters**:
- **model_name** : `str`
    Name or path of the Hugging Face model.
- **trust_remote_code** : `bool`, optional
    Whether to trust remote code. Default is `False`.

**Returns**:
- **processor** : `transformers.PreTrainedProcessor`
    The initialized processor.

Utility Functions
-----------------
.. autoclass:: masked_cross_entropy
    :members:
    :undoc-members:

masked_cross_entropy
+++++++++++++++++++
**Signature**: `masked_cross_entropy(logits, targets, mask=None)`

Computes the cross-entropy loss with an optional mask to ignore certain tokens.

**Parameters**:
- **logits** : `torch.Tensor`
    Logits output from the model.
- **targets** : `torch.Tensor`
    Ground truth target tokens.
- **mask** : `torch.Tensor` or `None`, optional
    Mask to apply to the loss. Tokens with mask=0 will be ignored.

**Returns**:
- **loss** : `torch.Tensor`
    The computed loss.

Example Usage
-------------
.. code-block:: python

    from your_module import HFAutoModelForImageTextToText

    # Initialize the model
    model = HFAutoModelForImageTextToText(
        model_name='gpt2',
        load_pretrained_weights=True,
        trust_remote_code=True,
        load_in_4bit=True
    )

    # Configure the model
    model.configure_model()

    # Example training loop using PyTorch Lightning Trainer
    from pytorch_lightning import Trainer

    trainer = Trainer(max_epochs=3)
    trainer.fit(model, train_dataloader, val_dataloader)

    # Save the pretrained model
    model.save_pretrained('/path/to/save')
