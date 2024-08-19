NeMo Fundamentals
=================

On this page, we’ll look into how NeMo works, providing you with a solid foundation to effectively use NeMo for you :ref:`specific use case <where_next>`.

NeMo Models
-----------

NVIDIA NeMo is a powerful framework for building and deploying neural network models, including those used in generative AI, speech recognition, and natural language processing. NeMo stands for “Neural Modules,” which are the building blocks of the models created using this platform. NeMo includes all of the following components wrapped into a singular, cohesive unit:

* neural network architecture

* dataset and data loaders

* preprocessing of input data and postprocessing of model outputs

* loss function, optimizer, and schedulers

* any other supporting infrastructure, such as tokenizers, language model configuration, and data augmentation

NeMo models are built on PyTorch, with many of their components being subclasses of ``torch.nn.Module``. Additionally, NeMo models utilize PyTorch Lightning (PTL) for training, which helps reduce the boilerplate code required.

NeMo models are also designed to be easily configurable; often this is done with YAML files. Below we show simplified examples of a NeMo model defined in pseudocode and a config defined in YAML. We highlight the lines where the Python config parameter is read from the YAML file.

.. list-table:: Simplified examples of a model and config.
    :widths: 1 1
    :header-rows: 0

    * - .. code-block:: python
	  :caption: NeMo model definition (Python pseudocode)
	  :linenos:
	  :emphasize-lines: 4, 7, 10, 13, 16, 20

	  class ExampleEncDecModel:
	      # cfg is passed so it only contains "model" section
	      def __init__(self, cfg, trainer):
	          self.tokenizer = init_from_cfg(cfg.tokenizer)


	          self.encoder = init_from_cfg(cfg.encoder)


	          self.decoder = init_from_cfg(cfg.decoder)


	          self.loss = init_from_cfg(cfg.loss)


		  # optimizer configured via parent class


	      def setup_training_data(self, cfg):
	          self.train_dl = init_dl_from_cfg(cfg.train_ds)

	      def forward(self, batch):
	          # forward pass defined,
		  # as is standard for PyTorch models
	          ...

	      def training_step(self, batch):
	          log_probs = self.forward(batch)
	          loss = self.loss(log_probs, labels)
	          return loss


      - .. code-block:: yaml
	  :caption: Experiment config (YAML)
	  :linenos:
	  :emphasize-lines: 4, 7, 10, 13, 16, 20

	  #
	  # configuration of the NeMo model
	  model:
	      tokenizer:
	       ...

	      encoder:
	       ...

	      decoder:
	       ...

	      loss:
	       ...

	      optim:
	       ...


	      train_ds:
	       ...

	  # configuration of the
	  # PyTorch Lightning trainer object
	  trainer:
	      ...


Configuring and Training NeMo Models
------------------------------------

During initialization of the model, the "model" section of the config is passed into the model's constructor (as the variable ``cfg``, see line 3 of the left panel above). The model class will read key parameters from the ``cfg`` variable to configure the model (see highlighted lines in the left panel above).

The other object passed into the model's constructor is a PyTorch Lightning ``trainer`` object, which manages the training process. The trainer handles the standard training `boilerplate <https://lightning.ai/docs/pytorch/stable/common/trainer.html#under-the-hood>`__. For non-standard tasks, PyTorch Lightning (PTL) relies on specific methods defined in our NeMo model. For example, PTL mandates that every model must have a specified ``training_step`` method (left panel above, line 27).

The trainer’s configuration is also specified in the config (right panel above, line 25 onwards). This includes parameters such as ``accelerator``, (number of) ``devices``, ``max_steps``, (numerical) ``precision`` and `more <https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api>`__.


Example Training Script
-----------------------

Below is an example training script for our ``ExampleEncDecModel`` model. We highlight the three most important lines that combine everything we discussed in the previous section:

.. code-block:: python
	:caption: run_example_training.py
	:linenos:
	:emphasize-lines: 10, 11, 12

	import pytorch_lightning as pl
	from nemo.collections.path_to_model_class import ExampleEncDecModel
	from nemo.core.config import hydra_runner

	@hydra_runner(
		config_path="config_file_dir_path",
		config_name="config_file_name"
	)
	def main(cfg):
		trainer = pl.Trainer(**cfg.trainer)
		model = ExampleEncDecModel(cfg.model, trainer)
		trainer.fit(model)

	if __name__ == '__main__':
		main(cfg)


Let's go through the code:

* *Lines 1-3*: import statements (second one is made up for the example).
* *Lines 5-8*: the decorator will look for a config file at ``{config_path}/{config_name}.yaml`` and load its contents into the ``cfg`` object that is passed into the ``main`` function on line 9. This functionality is provided by `Hydra <https://hydra.cc/docs/intro/>`__. Instead of a YAML file, we could also have specified the default config as a dataclass and passed that into the ``@hydra_runner`` decorator.
* *Line 10*: initialize a PTL trainer object using the parameters specified in the ``trainer`` section of the config.
* *Line 11*: initialize a NeMo model, passing in both the parameters in the ``model`` section of the config, and a PTL ``trainer`` object.
* *Line 12*: call ``trainer.fit`` on the model. This one unassuming line will carry out our entire training process. PTL will make sure we iterate over our data and call the ``training_step`` we define for each batch (as well as any other PTL `callbacks <https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html>`__ that may have been defined).



Overriding Configs
------------------

The ``cfg`` object in the script above is a dictionary-like object that contains our configuration parameters. Specifically, it is an `OmegaConf <https://omegaconf.readthedocs.io/en/2.3_branch/usage.html>`__ ``DictConfig`` object. These objects have special features such as dot-notation `access <https://omegaconf.readthedocs.io/en/latest/usage.html#access>`__, `variable interpolation <https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation>`__, and the ability to set `mandatory values <https://omegaconf.readthedocs.io/en/latest/usage.html#mandatory-values>`__.

You can run the script above by running the following:

.. code-block:: bash

	python run_example_training.py

The script will use the default config file specified inside the ``@hydra_runner`` decorator.

To specify a different config file, you can call the script like this:

.. code-block:: diff

	 python run_example_training.py \
	+    --config_path="different_config_file_dir_path" \
	+    --config_name="different_config_file_name"

You can also override, delete, or add elements to the config by calling a script like this:


.. code-block:: diff

	 python run_example_training.py \
	     --config_path="different_config_file_dir_path" \
	     --config_name="different_config_file_name" \
	+    model.optim.lr=0.001 \                                     # overwriting
	+    model.train_ds.manifest_filepath="your_train_data.json" \  # overwriting
	+    ~trainer.max_epochs \                                      # deleting
	+    +trainer.max_steps=1000                                    # adding

Running NeMo Scripts
--------------------

NeMo scripts typically take on the form shown above, where the Python script relies on a config object which has some specified default values that you can choose to override.

The NeMo `examples <https://github.com/NVIDIA/NeMo/tree/main/examples/>`__ directory provides numerous scripts for training and inference of various existing NeMo models. It’s important to note that these scripts include default configurations for model, optimize, and training parameters, which have been fine-tuned by the NeMo team over extensive GPU-hours of experimentation. As a result, we recommend using these default configurations as a starting point for your own experiments


NeMo Inference Scripts
######################

The examples scripts directory also contains many inference scripts such as `transcribe_speech.py <https://github.com/NVIDIA/NeMo/blob/main/examples/asr/transcribe_speech.py>`_. These inference scripts typically differ in structure from training scripts, as they include additional utilities for file I/O (reading and saving files). While inference scripts still use configurations (configs), they don’t require the ``trainer`` and ``model`` sections. Additionally, the default configs for inference scripts are usually specified as dataclasses rather than separate files. You can also modify elements via the command line.

Specifying Training Data
------------------------

NeMo will handle the creation of data loaders for you, as long as you put your data into the expected input format. You may also need to train a tokenizer before starting training. To learn more about data formats, see :doc:`LLM <../nlp/nemo_megatron/gpt/gpt_training>`, :doc:`Multimodal <../multimodal/mllm/datasets>`, :ref:`Speech AI <section-with-manifest-format-explanation>`, and :doc:`Vision models <../vision/datasets>`.


Model Checkpoints
-----------------

Throughout training, the model :doc:`checkpoints <../checkpoints/intro>` will be saved inside ``.nemo`` files. These are archive files containing all the necessary components to restore a usable model. For example:

* model weights (``.ckpt`` files)
* model configuration (``.yaml`` files)
* tokenizer files

The NeMo team also releases pretrained models which you can browse on `NGC <https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC>`_ and `HuggingFace Hub <https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia>`_.


Fine-Tuning
-----------

NeMo allows you to fine-tune models as well as train them from scratch.

You can achieve this by initializing a model with random weights, then replacing some or all of those weights with the pretrained model’s weights. Afterward, continue training as usual, possibly making minor adjustments like reducing the learning rate or freezing specific model parameters.


.. _where_next:

Where To Go Next?
-----------------

Here are some options:

* Explore examples or tutorials: dive into NeMo by exploring our `examples <https://github.com/NVIDIA/NeMo/tree/main/examples>`_ or :doc:`tutorials <./tutorials>`

* Domain-specific documentation:

  * For Large Language Models (LLMs), checkout the :doc:`LLM <../nlp/nemo_megatron/intro>` documentation.
  * For Multimodal tasks, refer to the :doc:`Multimodal <../multimodal/mllm/intro>` documentation.

  * If you’re interested in Automatic Speech Recognition (ASR), explore the :doc:`ASR <../asr/intro>` documentation.
  * For Text-to-Speech (TTS), find details in the :doc:`TTS <../tts/intro>` documentation.
  * For Vision Models, consult the :doc:`Vision Models <../vision/intro>` documentation.

* `NeMo Primer <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/00_NeMo_Primer.ipynb>`__: This tutorial provides a hands-on introduction to NeMo, PyTorch Lightning, and OmegaConf. It covers how to use, modify, save, and restore NeMo models.

* `NeMo Models <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/01_NeMo_Models.ipynb>`__: In this tutorial, you'll learn the fundamentals of creating NeMo models.

* NeMo Core Documentation: Explore the :doc:`NeMo Core <../core/core>` documentation for NeMo, which explains the inner workings of NeMo Framework.

