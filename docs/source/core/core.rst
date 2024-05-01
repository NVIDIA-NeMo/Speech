NeMo Models
===========

Basics
------

NeMo models contain everything needed to train and reproduce Conversational AI models:

- neural network architectures 
- datasets/data loaders
- data preprocessing/postprocessing
- data augmentors
- optimizers and schedulers
- tokenizers
- language models

NeMo uses `Hydra <https://hydra.cc/>`_ for configuring both NeMo models and the PyTorch Lightning Trainer.

.. note::
    Every NeMo model has an example configuration file and training script that can be found `here <https://github.com/NVIDIA/NeMo/tree/stable/examples>`__.

The end result of using NeMo, `Pytorch Lightning <https://github.com/PyTorchLightning/pytorch-lightning>`__, and Hydra is that NeMo models all have the same look and feel and are also fully compatible with the PyTorch ecosystem.

Pretrained
----------

NeMo comes with many pretrained models for each of our collections: ASR, NLP, and TTS. Every pretrained NeMo model can be downloaded 
and used with the ``from_pretrained()`` method.

As an example, we can instantiate QuartzNet with the following:

.. code-block:: Python

    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")

To see all available pretrained models for a specific NeMo model, use the ``list_available_models()`` method.

.. code-block:: Python

    nemo_asr.models.EncDecCTCModel.list_available_models()

For detailed information on the available pretrained models, refer to the collections documentation: 

- :doc:`Automatic Speech Recognition (ASR) <../asr/intro>`
- :doc:`Natural Language Processing (NLP) <../nlp/models>`
- :doc:`Text-to-Speech Synthesis (TTS) <../tts/intro>`

Training
--------

NeMo leverages `PyTorch Lightning <https://www.pytorchlightning.ai/>`__ for model training. PyTorch Lightning lets NeMo decouple the
conversational AI code from the PyTorch training code. This means that NeMo users can focus on their domain (ASR, NLP, TTS) and 
build complex AI applications without having to rewrite boiler plate code for PyTorch training.

When using PyTorch Lightning, NeMo users can automatically train with:

- multi-GPU/multi-node
- mixed precision
- model checkpointing
- logging
- early stopping
- and more

The two main aspects of the Lightning API are the `LightningModule <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#>`_ 
and the `Trainer <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_.

PyTorch Lightning ``LightningModule``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every NeMo model is a ``LightningModule`` which is an ``nn.module``. This means that NeMo models are compatible with the PyTorch 
ecosystem and can be plugged into existing PyTorch workflows.

Creating a NeMo model is similar to any other PyTorch workflow. We start by initializing our model architecture, then define the forward pass:

.. code-block:: python

    class TextClassificationModel(NLPModel, Exportable):
        ...
        def __init__(self, cfg: DictConfig, trainer: Trainer = None):
            """Initializes the BERTTextClassifier model."""
            ...
            super().__init__(cfg=cfg, trainer=trainer)

            # instantiate a BERT based encoder
            self.bert_model = get_lm_model(
                config_file=cfg.language_model.config_file,
                config_dict=cfg.language_model.config,
                vocab_file=cfg.tokenizer.vocab_file,
                trainer=trainer,
                cfg=cfg,
            )

            # instantiate the FFN for classification
            self.classifier = SequenceClassifier(
                hidden_size=self.bert_model.config.hidden_size,
                num_classes=cfg.dataset.num_classes,
                num_layers=cfg.classifier_head.num_output_layers,
                activation='relu',
                log_softmax=False,
                dropout=cfg.classifier_head.fc_dropout,
                use_transformer_init=True,
                idx_conditioned_on=0,
            )

.. code-block:: python

        def forward(self, input_ids, token_type_ids, attention_mask):
            """
            No special modification required for Lightning, define it as you normally would
            in the `nn.Module` in vanilla PyTorch.
            """
            hidden_states = self.bert_model(
                input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )
            logits = self.classifier(hidden_states=hidden_states)
            return logits

The ``LightningModule`` organizes PyTorch code so that across all NeMo models we have a similar look and feel.
For example, the training logic can be found in ``training_step``:

.. code-block:: python

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        input_ids, input_type_ids, input_mask, labels = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        train_loss = self.loss(logits=logits, labels=labels)

        lr = self._optimizer.param_groups[0]['lr']

        self.log('train_loss', train_loss)
        self.log('lr', lr, prog_bar=True)

        return {
            'loss': train_loss,
            'lr': lr,
        }

While validation logic can be found in ``validation_step``:

.. code-block:: python

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        if self.testing:
            prefix = 'test'
        else:
            prefix = 'val'

        input_ids, input_type_ids, input_mask, labels = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        val_loss = self.loss(logits=logits, labels=labels)

        preds = torch.argmax(logits, axis=-1)

        tp, fn, fp, _ = self.classification_report(preds, labels)

        return {'val_loss': val_loss, 'tp': tp, 'fn': fn, 'fp': fp}

PyTorch Lightning then handles all of the boiler plate code needed for training. Virtually any aspect of training can be customized 
via PyTorch Lightning `hooks <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#hooks>`_, 
`Plugins <https://pytorch-lightning.readthedocs.io/en/stable/extensions/plugins.html>`_, 
`callbacks <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html>`_, or by overriding `methods <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#methods>`_. 

For more domain-specific information, see:

- :doc:`Automatic Speech Recognition (ASR) <../asr/intro>`
- :doc:`Natural Language Processing (NLP) <../nlp/models>`
- :doc:`Text-to-Speech Synthesis (TTS) <../tts/intro>`

PyTorch Lightning Trainer
~~~~~~~~~~~~~~~~~~~~~~~~~

Since every NeMo model is a ``LightningModule``, we can automatically take advantage of the PyTorch Lightning ``Trainer``. Every NeMo 
`example <https://github.com/NVIDIA/NeMo/tree/v1.0.2/examples>`_ training script uses the ``Trainer`` object to fit the model.

First, instantiate the model and trainer, then call ``.fit``:

.. code-block:: python
    
    # We first instantiate the trainer based on the model configuration.
    # See the model configuration documentation for details.    
    trainer = pl.Trainer(**cfg.trainer)

    # Then pass the model configuration and trainer object into the NeMo model
    model = TextClassificationModel(cfg.model, trainer=trainer)

    # Now we can train with by calling .fit
    trainer.fit(model)

    # Or we can run the test loop on test data by calling
    trainer.test(model=model)

All `trainer flags <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags>`_ can be set from from the NeMo configuration. 
    

Configuration
-------------

Hydra is an open-source Python framework that simplifies configuration for complex applications that must bring together many different 
software libraries. Conversational AI model training is a great example of such an application. To train a conversational AI model, we 
must be able to configure:

- neural network architectures
- training and optimization algorithms 
- data pre/post processing
- data augmentation
- experiment logging/visualization
- model checkpointing   

For an introduction to using Hydra, refer to the `Hydra Tutorials <https://hydra.cc/docs/tutorials/intro>`_.

With Hydra, we can configure everything needed for NeMo with three interfaces:

- Command Line (CLI) 
- Configuration Files (YAML)
- Dataclasses (Python)

YAML
~~~~

NeMo provides YAML configuration files for all of our `example <https://github.com/NVIDIA/NeMo/tree/v1.0.2/examples>`_ training scripts.
YAML files make it easy to experiment with different model and training configurations.

Every NeMo example YAML has the same underlying configuration structure:

- trainer
- exp_manager
- model

Model configuration always contain ``train_ds``, ``validation_ds``, ``test_ds``, and ``optim``.  Model architectures vary across 
domains, therefore, refer to the ASR, NLP, and TTS Collections documentation for more detailed information on Model architecture configuration.

A NeMo configuration file should look similar to the following:

.. code-block:: yaml

    # PyTorch Lightning Trainer configuration
    # any argument of the Trainer object can be set here
    trainer:
        devices: 1 # number of gpus per node
        accelerator: gpu
        num_nodes: 1 # number of nodes
        max_epochs: 10 # how many training epochs to run
        val_check_interval: 1.0 # run validation after every epoch

    # Experiment logging configuration
    exp_manager:
        exp_dir: /path/to/my/nemo/experiments
        name: name_of_my_experiment
        create_tensorboard_logger: True
        create_wandb_logger: True

    # Model configuration
    # model network architecture, train/val/test datasets, data augmentation, and optimization
    model:
        train_ds:
            manifest_filepath: /path/to/my/train/manifest.json
            batch_size: 256
            shuffle: True
        validation_ds:
            manifest_filepath: /path/to/my/validation/manifest.json
            batch_size: 32
            shuffle: False
        test_ds:
            manifest_filepath: /path/to/my/test/manifest.json
            batch_size: 32
            shuffle: False
        optim:
            name: novograd
            lr: .01
            betas: [0.8, 0.5]
            weight_decay: 0.001
        # network architecture can vary greatly depending on the domain
        encoder:
            ...
        decoder:
            ...

More specific details about configuration files for each collection can be found on the following pages:

:ref:`NeMo ASR Configuration Files`
        
CLI
~~~

With NeMo and Hydra, every aspect of model training can be modified from the command-line. This is extremely helpful for running lots 
of experiments on compute clusters or for quickly testing parameters while developing.

All NeMo `examples <https://github.com/NVIDIA/NeMo/tree/v1.0.2/examples>`_ come with instructions on how to
run the training/inference script from the command-line (see `here <https://github.com/NVIDIA/NeMo/blob/4e9da75f021fe23c9f49404cd2e7da4597cb5879/examples/asr/asr_ctc/speech_to_text_ctc.py#L24>`__
for an example).

With Hydra, arguments are set using the ``=`` operator:

.. code-block:: bash

    python examples/asr/asr_ctc/speech_to_text_ctc.py \
        model.train_ds.manifest_filepath=/path/to/my/train/manifest.json \
        model.validation_ds.manifest_filepath=/path/to/my/validation/manifest.json \
        trainer.devices=2 \
        trainer.accelerator='gpu' \
        trainer.max_epochs=50

We can use the ``+`` operator to add arguments from the CLI:

.. code-block:: bash

    python examples/asr/asr_ctc/speech_to_text_ctc.py \
        model.train_ds.manifest_filepath=/path/to/my/train/manifest.json \
        model.validation_ds.manifest_filepath=/path/to/my/validation/manifest.json \
        trainer.devices=2 \
        trainer.accelerator='gpu' \
        trainer.max_epochs=50 \
        +trainer.fast_dev_run=true

We can use the ``~`` operator to remove configurations:

.. code-block:: bash

    python examples/asr/asr_ctc/speech_to_text_ctc.py \
        model.train_ds.manifest_filepath=/path/to/my/train/manifest.json \
        model.validation_ds.manifest_filepath=/path/to/my/validation/manifest.json \
        ~model.test_ds \
        trainer.devices=2 \
        trainer.accelerator='gpu' \
        trainer.max_epochs=50 \
        +trainer.fast_dev_run=true

We can specify configuration files using the ``--config-path`` and ``--config-name`` flags:

.. code-block:: bash

    python examples/asr/asr_ctc/speech_to_text_ctc.py \
        --config-path=conf/quartznet \
        --config-name=quartznet_15x5 \
        model.train_ds.manifest_filepath=/path/to/my/train/manifest.json \
        model.validation_ds.manifest_filepath=/path/to/my/validation/manifest.json \
        ~model.test_ds \
        trainer.devices=2 \
        trainer.accelerator='gpu' \
        trainer.max_epochs=50 \
        +trainer.fast_dev_run=true

Dataclasses
~~~~~~~~~~~

Dataclasses allow NeMo to ship model configurations as part of the NeMo library and also enables pure Python configuration of NeMo models. 
With Hydra, dataclasses can be used to create `structured configs <https://hydra.cc/docs/tutorials/structured_config/intro>`_ for the conversational AI application. 

As an example, refer to the code block below for an *Attenion is All You Need* machine translation model. The model configuration can 
be instantiated and modified like any Python `Dataclass <https://docs.python.org/3/library/dataclasses.html>`_.

.. code-block:: Python

    from nemo.collections.nlp.models.machine_translation.mt_enc_dec_config import AAYNBaseConfig

    cfg = AAYNBaseConfig()

    # modify the number of layers in the encoder
    cfg.encoder.num_layers = 8

    # modify the training batch size
    cfg.train_ds.tokens_in_batch = 8192

.. note:: Configuration with Hydra always has the following precedence CLI > YAML > Dataclass

.. _optimization-label:

Optimization
------------

Optimizers and learning rate schedules are configurable across all NeMo models and have their own namespace. Here is a sample YAML 
configuration for a Novograd optimizer with Cosine Annealing learning rate schedule.

.. code-block:: yaml

    optim:
        name: novograd
        lr: 0.01
    
        # optimizer arguments
        betas: [0.8, 0.25]
        weight_decay: 0.001
    
        # scheduler setup
        sched:
            name: CosineAnnealing
    
            # Optional arguments
            max_steps: -1 # computed at runtime or explicitly set here
            monitor: val_loss
            reduce_on_plateau: false
    
            # scheduler config override
            warmup_steps: 1000
            warmup_ratio: null
            min_lr: 1e-9:

.. note:: `NeMo Examples <https://github.com/NVIDIA/NeMo/tree/v1.0.2/examples>`_ has optimizer and scheduler configurations for every NeMo model.

Optimizers can be configured from the CLI as well:

.. code-block:: bash

    python examples/asr/asr_ctc/speech_to_text_ctc.py \
        --config-path=conf/quartznet \
        --config-name=quartznet_15x5 \
        ...
        # train with the adam optimizer
        model.optim=adam \
        # change the learning rate
        model.optim.lr=.0004 \
        # modify betas 
        model.optim.betas=[.8, .5]

.. _optimizers-label:

Optimizers
~~~~~~~~~~

``name`` corresponds to the lowercase name of the optimizer. To view a list of available optimizers, run:

.. code-block:: Python

    from nemo.core.optim.optimizers import AVAILABLE_OPTIMIZERS

    for name, opt in AVAILABLE_OPTIMIZERS.items():
        print(f'name: {name}, opt: {opt}')

.. code-block:: bash

    name: sgd opt: <class 'torch.optim.sgd.SGD'>
    name: adam opt: <class 'torch.optim.adam.Adam'>
    name: adamw opt: <class 'torch.optim.adamw.AdamW'>
    name: adadelta opt: <class 'torch.optim.adadelta.Adadelta'>
    name: adamax opt: <class 'torch.optim.adamax.Adamax'>
    name: adagrad opt: <class 'torch.optim.adagrad.Adagrad'>
    name: rmsprop opt: <class 'torch.optim.rmsprop.RMSprop'>
    name: rprop opt: <class 'torch.optim.rprop.Rprop'>
    name: novograd opt: <class 'nemo.core.optim.novograd.Novograd'>

Optimizer Params
~~~~~~~~~~~~~~~~

Optimizer params can vary between optimizers but the ``lr`` param is required for all optimizers. To see the available params for an 
optimizer, we can look at its corresponding dataclass.

.. code-block:: python

    from nemo.core.config.optimizers import NovogradParams

    print(NovogradParams())

.. code-block:: bash

    NovogradParams(lr='???', betas=(0.95, 0.98), eps=1e-08, weight_decay=0, grad_averaging=False, amsgrad=False, luc=False, luc_trust=0.001, luc_eps=1e-08)

``'???'`` indicates that the lr argument is required.

Register Optimizer
~~~~~~~~~~~~~~~~~~

To register a new optimizer to be used with NeMo, run:

.. autofunction:: nemo.core.optim.optimizers.register_optimizer

.. _learning-rate-schedulers-label:

Learning Rate Schedulers
~~~~~~~~~~~~~~~~~~~~~~~~

Learning rate schedulers can be optionally configured under the ``optim.sched`` namespace.

``name`` corresponds to the name of the learning rate schedule. To view a list of available schedulers, run: 
    
.. code-block:: Python

    from nemo.core.optim.lr_scheduler import AVAILABLE_SCHEDULERS

    for name, opt in AVAILABLE_SCHEDULERS.items():
        print(f'name: {name}, schedule: {opt}')

.. code-block:: bash

    name: WarmupPolicy, schedule: <class 'nemo.core.optim.lr_scheduler.WarmupPolicy'>
    name: WarmupHoldPolicy, schedule: <class 'nemo.core.optim.lr_scheduler.WarmupHoldPolicy'>
    name: SquareAnnealing, schedule: <class 'nemo.core.optim.lr_scheduler.SquareAnnealing'>
    name: CosineAnnealing, schedule: <class 'nemo.core.optim.lr_scheduler.CosineAnnealing'>
    name: NoamAnnealing, schedule: <class 'nemo.core.optim.lr_scheduler.NoamAnnealing'>
    name: WarmupAnnealing, schedule: <class 'nemo.core.optim.lr_scheduler.WarmupAnnealing'>
    name: InverseSquareRootAnnealing, schedule: <class 'nemo.core.optim.lr_scheduler.InverseSquareRootAnnealing'>
    name: SquareRootAnnealing, schedule: <class 'nemo.core.optim.lr_scheduler.SquareRootAnnealing'>
    name: PolynomialDecayAnnealing, schedule: <class 'nemo.core.optim.lr_scheduler.PolynomialDecayAnnealing'>
    name: PolynomialHoldDecayAnnealing, schedule: <class 'nemo.core.optim.lr_scheduler.PolynomialHoldDecayAnnealing'>
    name: StepLR, schedule: <class 'torch.optim.lr_scheduler.StepLR'>
    name: ExponentialLR, schedule: <class 'torch.optim.lr_scheduler.ExponentialLR'>
    name: ReduceLROnPlateau, schedule: <class 'torch.optim.lr_scheduler.ReduceLROnPlateau'>
    name: CyclicLR, schedule: <class 'torch.optim.lr_scheduler.CyclicLR'>

Scheduler Params
~~~~~~~~~~~~~~~~

To see the available params for a scheduler, we can look at its corresponding dataclass:

.. code-block:: Python

    from nemo.core.config.schedulers import CosineAnnealingParams

    print(CosineAnnealingParams())

.. code-block:: bash

    CosineAnnealingParams(last_epoch=-1, warmup_steps=None, warmup_ratio=None, min_lr=0.0)

Register scheduler
~~~~~~~~~~~~~~~~~~

To register a new scheduler to be used with NeMo, run:

.. autofunction:: nemo.core.optim.lr_scheduler.register_scheduler

Save and Restore
----------------

NeMo models all come with ``.save_to`` and ``.restore_from`` methods.  

Save
~~~~

To save a NeMo model, run:

.. code-block:: Python

    model.save_to('/path/to/model.nemo')

Everything needed to use the trained model is packaged and saved in the ``.nemo`` file. For example, in the NLP domain, ``.nemo`` files 
include the necessary tokenizer models and/or vocabulary files, etc.

.. note:: A ``.nemo`` file is simply an archive like any other ``.tar`` file.

Restore
~~~~~~~

To restore a NeMo model, run:

.. code-block:: Python

    # Here, you should usually use the class of the model, or simply use ModelPT.restore_from() for simplicity.
    model.restore_from('/path/to/model.nemo')

When using the PyTorch Lightning Trainer, a PyTorch Lightning checkpoint is created. These are mainly used within NeMo to auto-resume 
training. Since NeMo models are ``LightningModules``, the PyTorch Lightning method ``load_from_checkpoint`` is available. Note that 
``load_from_checkpoint`` won't necessarily work out-of-the-box for all models as some models require more artifacts than just the 
checkpoint to be restored. For these models, the user will have to override ``load_from_checkpoint`` if they want to use it.

It's highly recommended to use ``restore_from`` to load NeMo models.

Restore with Modified Config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes, there may be a need to modify the model (or it's sub-components) prior to restoring a model. A common case is when
the model's internal config must be updated due to various reasons (such as deprecation, newer versioning, support a new feature).
As long as the model has the same parameters as compared to the original config, the parameters can once again be restored safely.

In NeMo, as part of the .nemo file, the model's internal config will be preserved. This config is used during restoration, and
as shown below we can update this config prior to restoring the model.

.. code-block::

    # When restoring a model, you should generally use the class of the model
    # Obtain the config (as an OmegaConf object)
    config = model_class.restore_from('/path/to/model.nemo', return_config=True)
    # OR
    config = model_class.from_pretrained('name_of_the_model', return_config=True)

    # Modify the config as needed
    config.x.y = z

    # Restore the model from the updated config
    model = model_class.restore_from('/path/to/model.nemo', override_config_path=config)
    # OR
    model = model_class.from_pretrained('name_of_the_model', override_config_path=config)

Register Artifacts
------------------

Conversational AI models can be complicated to restore as more information is needed than just the checkpoint weights in order to use the model.
NeMo models can save additional artifacts in the .nemo file by calling ``.register_artifact``.
When restoring NeMo models using ``.restore_from`` or ``.from_pretrained``, any artifacts that were registered will be available automatically.

As an example, consider an NLP model that requires a trained tokenizer model. 
The tokenizer model file can be automatically added to the .nemo file with the following:

.. code-block:: python

    self.encoder_tokenizer = get_nmt_tokenizer(
        ...
        tokenizer_model=self.register_artifact(config_path='encoder_tokenizer.tokenizer_model',
                                               src='/path/to/tokenizer.model',
                                               verify_src_exists=True),
    )

By default, ``.register_artifact`` will always return a path. If the model is being restored from a .nemo file, 
then that path will be to the artifact in the .nemo file. Otherwise, ``.register_artifact`` will return the local path specified by the user.

``config_path`` is the artifact key. It usually corresponds to a model configuration but does not have to.
The model config that is packaged with the .nemo file will be updated according to the ``config_path`` key.
In the above example, the model config will have 

.. code-block:: YAML

    encoder_tokenizer:
        ...
        tokenizer_model: nemo:4978b28103264263a03439aaa6560e5e_tokenizer.model

``src`` is the path to the artifact and the base-name of the path will be used when packaging the artifact in the .nemo file.
Each artifact will have a hash prepended to the basename of ``src`` in the .nemo file. This is to prevent collisions with basenames 
base-names that are identical (say when there are two or more tokenizers, both called `tokenizer.model`).
The resulting .nemo file will then have the following file:

.. code-block:: bash

    4978b28103264263a03439aaa6560e5e_tokenizer.model

If ``verify_src_exists`` is set to ``False``, then the artifact is optional. This means that ``.register_artifact`` will return ``None`` 
if the ``src`` cannot be found.

Push to Hugging Face Hub
------------------------

NeMo models can be pushed to the `Hugging Face Hub <https://huggingface.co/>`_ with the :meth:`~nemo.core.classes.mixins.hf_io_mixin.HuggingFaceFileIO.push_to_hf_hub` method. This method performs the same actions as ``save_to()`` and then uploads the model to the HuggingFace Hub. It offers an additional ``pack_nemo_file`` argument that allows the user to upload the entire NeMo file or just the ``.nemo`` file. This is useful for large language models that have a massive number of parameters, and a single NeMo file could exceed the max upload size of Hugging Face Hub.


Upload a model to the hub
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    token = "<HF TOKEN>" or None
    pack_nemo_file = True  # False will upload multiple files that comprise the NeMo file onto HF Hub; Generally useful for LLMs

    model.push_to_hf_hub(
       repo_id=repo_id, pack_nemo_file=pack_nemo_file, token=token,
    )

Use a Custom Model Card Template for the Hub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Override the default model card
    template = """ <Your own custom template>
    # {model_name}
    """
    kwargs = {"model_name": "ABC", "repo_id": "nvidia/ABC_XYZ"}
    model_card = model.generate_model_card(template=template, template_kwargs=kwargs, type="hf")

    model.push_to_hf_hub(
        repo_id=repo_id, token=token, model_card=model_card
    )

    # Write your own model card class
    class MyModelCard:
      def __init__(self, model_name):
        self.model_name = model_name

      def __repr__(self):
        template = """This is the {model_name} model""".format(model_name=self.model_name)
        return template

    model.push_to_hf_hub(
        repo_id=repo_id, token=token, model_card=MyModelCard("ABC")
    )


Nested NeMo Models
------------------

In some cases, it may be helpful to use NeMo models inside other NeMo models. For example, we can incorporate language models into ASR models to use in a decoding process to improve accuracy or use hybrid ASR-TTS models to generate audio from the text on the fly to train or finetune the ASR model.

There are 3 ways to instantiate child models inside parent models:

- use subconfig directly
- use the ``.nemo`` checkpoint path to load the child model
- use a pretrained NeMo model

To register a child model, use the ``register_nemo_submodule`` method of the parent model. This method will add the child model to a provided model attribute and, in the serialization process, will handle child artifacts correctly and store the child model config in the parent model config in ``config_field``.

.. code-block:: python

    from nemo.core.classes import ModelPT

    class ChildModel(ModelPT):
        ...  # implement necessary methods

    class ParentModel(ModelPT):
        def __init__(self, cfg, trainer=None):
            super().__init__(cfg=cfg, trainer=trainer)

            # optionally annotate type for IDE autocompletion and type checking
            self.child_model: Optional[ChildModel]
            if cfg.get("child_model") is not None:
                # load directly from config
                # either if config provided initially, or automatically
                # after model restoration
                self.register_nemo_submodule(
                    name="child_model",
                    config_field="child_model",
                    model=ChildModel(self.cfg.child_model, trainer=trainer),
                )
            elif cfg.get('child_model_path') is not None:
                # load from .nemo model checkpoint
                # while saving, config will be automatically assigned/updated
                # in cfg.child_model
                self.register_nemo_submodule(
                    name="child_model",
                    config_field="child_model",
                    model=ChildModel.restore_from(self.cfg.child_model_path, trainer=trainer),
                )
            elif cfg.get('child_model_name') is not None:
                # load from pretrained model
                # while saving, config will be automatically assigned/updated
                # in cfg.child_model
                self.register_nemo_submodule(
                    name="child_model",
                    config_field="child_model",
                    model=ChildModel.from_pretrained(self.cfg.child_model_name, trainer=trainer),
                )
            else:
                self.child_model = None

