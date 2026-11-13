*********
Callbacks
*********

Exponential Moving Average (EMA)
================================

During training, EMA maintains a moving average of the trained parameters.
EMA parameters can produce significantly better results and faster convergence for a variety of different domains and models.

EMA is a simple calculation. EMA Weights are pre-initialized with the model weights at the start of training.

Every training update, the EMA weights are updated based on the new model weights.

.. math::
    ema_w = ema_w * decay + model_w * (1-decay)

Enabling EMA is straightforward. We can pass the additional argument to the experiment manager at runtime.

.. code-block:: bash

    python examples/asr/asr_ctc/speech_to_text_ctc.py \
        model.train_ds.manifest_filepath=/path/to/my/train/manifest.json \
        model.validation_ds.manifest_filepath=/path/to/my/validation/manifest.json \
        trainer.devices=2 \
        trainer.accelerator='gpu' \
        trainer.max_epochs=50 \
        exp_manager.ema.enable=True # pass this additional argument to enable EMA

To change the decay rate, pass the additional argument.

.. code-block:: bash

    python examples/asr/asr_ctc/speech_to_text_ctc.py \
        ...
        exp_manager.ema.enable=True \
        exp_manager.ema.decay=0.999

We also offer other helpful arguments.

.. list-table::
   :header-rows: 1

   * - Argument
     - Description
   * - `exp_manager.ema.validate_original_weights=True`
     - Validate the original weights instead of EMA weights.
   * - `exp_manager.ema.every_n_steps=2`
     - Apply EMA every N steps instead of every step.
   * - `exp_manager.ema.cpu_offload=True`
     - Offload EMA weights to CPU. May introduce significant slow-downs.
