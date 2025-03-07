HFAutoModelForSpeechSeq2Seq
============================

.. currentmodule:: nemo.collections.speechlm.models.hf_auto_model_for_speech_seq2seq

A PyTorch Lightning module for speech sequence-to-sequence tasks using Hugging Face transformers and NeMo Framework.


Overview
--------
`HFAutoModelForSpeechSeq2Seq` is a versatile PyTorch Lightning module designed for speech-to-text and other sequence-to-sequence tasks. It leverages Hugging Face's `AutoModelForSpeechSeq2Seq` and integrates seamlessly with NVIDIA NeMo's utilities for enhanced functionality. The class supports loading pretrained models, custom tokenizers, and processors, and provides flexible configuration options for training and inference.

Initialization
--------------
.. constructor:: HFAutoModelForSpeechSeq2Seq(*, model_name='gpt2', load_pretrained_weights=True, tokenizer=None, loss_fn=masked_cross_entropy, model_transform=None, model_accelerator=None, trust_remote_code=False)

    Initializes the `HFAutoModelForSpeechSeq2Seq` module.

    Parameters
    ----------
    model_name : str, optional
        The name or path of the pretrained model to use (default is `'gpt2'`).
    load_pretrained_weights : bool, optional
        Whether to load pretrained weights from the specified `model_name` (default is `True`).
    tokenizer : Optional[AutoTokenizer], optional
        A custom tokenizer instance. If `None`, the default tokenizer for `model_name` is used.
    loss_fn : Callable, optional
        The loss function to use. Defaults to `masked_cross_entropy`.
    model_transform : Optional[Any], optional
        A transformation function or object to apply to the model (default is `None`).
    model_accelerator : Optional[Any], optional
        Accelerator configuration for the model (default is `None`).
    trust_remote_code : bool, optional
        Whether to trust remote code when loading models and tokenizers (default is `False`).

Attributes
----------
model_name : str
    The name or path of the pretrained model.
_tokenizer : Optional[AutoTokenizer]
    The tokenizer instance. Initialized lazily.
_processor : Optional[AutoProcessor]
    The processor instance for handling input features. Initialized lazily.
model : Optional[AutoModelForSpeechSeq2Seq]
    The underlying Hugging Face model.
loss_fn : Callable
    The loss function used for training.
load_pretrained_weights : bool
    Flag indicating whether to load pretrained weights.
is_hf_model : bool
    Flag indicating if the model is a Hugging Face model.
model_transform : Optional[Any]
    Transformation applied to the model.
model_accelerator : Optional[Any]
    Accelerator configuration for the model.
trust_remote_code : bool
    Flag indicating whether to trust remote code for model loading.

Properties
----------
.. py:attribute:: tokenizer

    The tokenizer used for encoding and decoding. Initialized on first access.

.. py:attribute:: processor

    The processor for handling input features. Initialized on first access.

Methods
-------
.. automethod:: configure_tokenizer(model_name)

    Configures and returns the tokenizer for the given `model_name`.

.. automethod:: configure_model(train=True)

    Configures the model for training or evaluation.

.. automethod:: forward(input_features, decoder_input_ids, attention_mask=None)

    Performs a forward pass through the model.

.. automethod:: training_step(batch)

    Defines the training step, computing loss and logging metrics.

.. automethod:: validation_step(batch)

    Defines the validation step, computing loss and logging metrics.

.. automethod:: save_pretrained(path)

    Saves the model, tokenizer, and processor to the specified path.

Function: masked_cross_entropy
------------------------------
.. function:: masked_cross_entropy

   Computes the masked cross-entropy loss.

   Parameters
   ----------
   logits : torch.Tensor
       The predicted logits from the model.
   targets : torch.Tensor
       The target labels.
   mask : Optional[torch.Tensor], optional
       A mask to apply to the loss computation (default is `None`).

   Returns
   -------
   torch.Tensor
       The computed loss.

Usage Example
-------------
.. code-block:: javascript
  :linenos:

  import lightning.pytorch as pl
  from your_module_name import HFAutoModelForSpeechSeq2Seq

  # Initialize the model
  model = HFAutoModelForSpeechSeq2Seq(
    model_name='facebook/wav2vec2-base-960h',
    load_pretrained_weights=True,
    trust_remote_code=True
  )

  # Set up the trainer
  trainer = pl.Trainer(max_epochs=10, gpus=1)

  # Train the model
  trainer.fit(model, train_dataloader, val_dataloader)

  # Save the trained model
  model.save_pretrained('path/to/save')
