Migration guide to use lightning 2.0
============

.. # define a hard line break for html
.. |br| raw:: html

    <br />

.. _dummy_header:

* Replace ``trainer.strategy=null`` with ``trainer.strategy=auto`` as 
  `lightning 2.0 doesn't have None strategy <https://lightning.ai/docs/pytorch/stable/common/trainer.html#:~:text=strategy%20(Union%5Bstr%2C%20Strategy%5D)%20%E2%80%93%20Supports%20different%20training%20strategies%20with%20aliases%20as%20well%20custom%20strategies.%20Default%3A%20%22auto%22.>`_.
..
* Remove ``resume_from_checkpoint`` if being used as a trainer flag and pass the path to 
  `Trainer.fit(ckpt_path="...") method <https://lightning.ai/docs/pytorch/stable/upgrade/from_1_9.html#:~:text=used%20Trainer%E2%80%99s%20flag%20resume_from_checkpoint>`_.
..
* Set ``trainer.strategy = "ddp_find_unused_parameters_true"`` if there are unused parameters in your model as lightning 2.0 has find_unused_parameters as False by default. 
  Reference: `NeMo PR 6433 <https://github.com/NVIDIA/NeMo/pull/6433/files#:~:text=Resolve%20conversation-,cfg.trainer.strategy%20%3D%20%22ddp_find_unused_parameters_true%22,-logging.info>`_. 
  More details about this change: `lightning PR 16611 <https://github.com/Lightning-AI/lightning/pull/16611>`_.
..
* If used Trainer's flag ``replace_sampler_ddp`` replace it with 
  `use_distributed_sampler <https://lightning.ai/docs/pytorch/stable/upgrade/from_1_9.html#:~:text=use%20use_distributed_sampler%3B%20the%20sampler%20gets%20created%20not%20only%20for%20the%20DDP%20strategies>`_.
..
* If using ``CheckpointConnector`` replace it with `_CheckpointConnector <https://github.com/NVIDIA/NeMo/pull/6433/files#diff-fbee9218112b5eb07e4b799b868cbe3ab582336157bde6dc7c881daa63965ff5R20>`_.
..
* To set or get ``ckpt_path`` use ``trainer.ckpt_path`` directly instead of calling protected API via ``trainer._checkpoint_connector._ckpt_path`` 
  or using ``trainer._checkpoint_connector.resume_from_checkpoint_fit_path``.
..
* Change ``import load`` from pytorch_lightning.utilities.cloud_io to ``import _load``.
..
* If used ``from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin`` from replace it with 
  `from pytorch_lightning.plugins.precision import MixedPrecisionPlugin <https://lightning.ai/docs/pytorch/stable/upgrade/from_1_9.html#:~:text=used%20the%20pl.plugins.NativeMixedPrecisionPlugin%20plugin>`_. 
..
* Lightning 2.0 adds ``'16-mixed'``, ``'bf16-mixed'`` as the preicison values for fp16 mixed precision and bf16 mixed precision respectively. 
  For backward compatbility ``16`` or ``'16'`` and ``'bf16'`` also perform mixed precision and is equivalent to ``'16-mixed'`` and ``'bf16-mixed'`` 
  respectively. However, lightning recommends to use ``'16-mixed'`` and ``'bf16-mixed'`` to make it less ambiguous. Due to this, ``MegatronHalfPrecisionPlugin's`` 
  parent class from lightning ``MixedPrecisionPlugin`` class, expects the precision arg to be ``'16-mixed'`` and ``'bf16-mixed'``. As a result it's required to 
  pass ``'16-mixed'`` or ``'bf16-mixed'`` to ``MixedPrecisionPLugin`` whenever the precision passed is any of ``[16, '16', '16-mixed']`` or ``['bf16', 'bf16-mixed']``. 
  This can be taken care as shown here: `NeMo upgrade to lightning 2.0 PR <https://github.com/NVIDIA/NeMo/pull/6433/files#diff-c0fc606b0f7750c3444a51159ce5deaa422a8cc4dd1134c504c4df2fdb683d64R140>`_ 
  and here: `MixedPrecisionPlugin <https://github.com/NVIDIA/NeMo/pull/6433/files#diff-c0fc606b0f7750c3444a51159ce5deaa422a8cc4dd1134c504c4df2fdb683d64R148-R152>`_. Also, ``'32-true'`` 
  is added as a precsion value for pure fp32 along with ``32``, ``'32'`` that existed. This can be taken into account as shown here in the `NeMo upgrade to lightning 2.0 PR <https://github.com/NVIDIA/NeMo/pull/6433/files#diff-e93ccae74f4b67d341676afc9f3c7e2c50f751ec64df84eb3b2a86b62029ef76R269>`_.
..
* Lightning 2.0 renames epoch end hooks from ``training_epoch_end``, ``validation_epoch_end``, ``test_epoch_end`` to ``on_train_epoch_end``, 
  ``on_validation_epoch_end``, ``on_test_epoch_end``. The renamed hooks do not accept the outputs arg but instead outputs needs to be defined 
  as an instance variable of the model class to which the outputs of the step needs to be manually appended. More detailed examples implementing 
  this can be found under migration guide of `lightning's PR 16520 <https://github.com/Lightning-AI/lightning/pull/16520>`_. Example from NeMo 
  can be found `here <https://github.com/NVIDIA/NeMo/pull/6433/files#diff-e93ccae74f4b67d341676afc9f3c7e2c50f751ec64df84eb3b2a86b62029ef76R904-R911>`_.
..
* Lightning 2.0 is not currently supporting multiple dataloders for validation and testing in case of ``dataloader_iter``. The support for this will be added back soon in an 
  upcoming release. If ``dataloader_iter`` is being used and your config passes multiple files to ``validation_ds.file_names`` or ``test_ds.file_names``, please use just one file 
  until this issue is fixed with pytorch lightning.
..
* With lightning 2.0 it's required to set ``limit_val_batches`` and ``num_sanity_val_steps`` to be a multiple of number of microbatches while 
  using ``dataloader_iter`` (applies only to Megatron files that use dataloader_iter) for all pretraining files (not downstream tasks like finetuning). 
  This is being taken care internally in NeMo and does not require anything to be done by the user. However, if you are a developer of NeMo and are 
  building a new model for pretraining that uses ``dataloader_iter`` instead of batch in ``validation_step`` methods please make sure to call 
  ``self._reconfigure_val_batches()`` in ``build_train_valid_test_datasets method`` of your model.
..
* If model is being wrapped with ``LightningDistributedModule`` in ``configure_ddp`` method please replace it with ``_LightningModuleWrapperBase`` 
  as being done here: `NeMo upgrade to lightning 2.0 PR <https://github.com/NVIDIA/NeMo/pull/6433/files#diff-7667eae242a8ef776bff78cd08e79bc81df4896a450f0a781f6ed317a3dfb7ffR136>`_.
..
* If using ``pre_configure_ddp()`` in your DDP, remove it as it's not required anymore. 
  `NeMo upgrade to lightning 2.0 PR <https://github.com/NVIDIA/NeMo/pull/6433/files#diff-7667eae242a8ef776bff78cd08e79bc81df4896a450f0a781f6ed317a3dfb7ffR148-R150>`_.
..
* If any of the tests use CPU as the device, ensure to explicitly pass it in the trainer as 
  ``trainer = pl.Trainer(max_epochs=1, accelerator='cpu')`` since deafult val in PTL >= 2.0 is auto and it picks cuda.
..
* If using ``from pytorch_lightning.loops import TrainingEpochLoop``, replace ``TrainingEpochLoop`` with ``_TrainingEpochLoop``.
..
* If using ``trainer.fit_loop.max_steps``, replace it with ``trainer.fit_loop.epoch_loop.max_steps``.