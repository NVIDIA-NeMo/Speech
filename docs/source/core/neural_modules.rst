Neural Modules
==============

NeMo is built around Neural Modules, conceptual blocks of neural networks that take typed inputs and produce typed outputs. Such 
modules typically represent data layers, encoders, decoders, language models, loss functions, or methods of combining activations.
NeMo makes it easy to combine and re-use these building blocks while providing a level of semantic correctness checking via its neural 
type system.

.. note:: *All Neural Modules inherit from ``torch.nn.Module`` and are therefore compatible with the PyTorch ecosystem.*

There are 3 types on Neural Modules:

    - Regular modules
    - Dataset/IterableDataset
    - Losses

Every Neural Module in NeMo must inherit from `nemo.core.classes.module.NeuralModule` class.

.. autoclass:: nemo.core.classes.module.NeuralModule

Every Neural Modules inherits the ``nemo.core.classes.common.Typing`` interface and needs to define neural types for its inputs and outputs.
This is done by defining two properties: ``input_types`` and ``output_types``. Each property should return an ordered dictionary of 
"port name"->"port neural type" pairs. Here is the example from :class:`~nemo.collections.asr.modules.ConvASREncoder` class:

.. code-block:: python

    @property
    def input_types(self):
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def output_types(self):
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @typecheck()
    def forward(self, audio_signal, length=None):
        ...

The code snippet above means that ``nemo.collections.asr.modules.conv_asr.ConvASREncoder`` expects two arguments:
    * First one, named ``audio_signal`` of shape ``[batch, dimension, time]`` with elements representing spectrogram values.
    * Second one, named ``length`` of shape ``[batch]`` with elements representing lengths of corresponding signals.

It also means that ``.forward(...)`` and ``__call__(...)`` methods each produce two outputs:
    * First one, of shape ``[batch, dimension, time]`` but with elements representing encoded representation (``AcousticEncodedRepresentation`` class).
    * Second one, of shape ``[batch]``, corresponding to their lengths.

.. tip:: It is a good practice to define types and add ``@typecheck()`` decorator to your ``.forward()`` method after your module is ready for use by others.

.. note:: The outputs of ``.forward(...)`` method will always be of type ``torch.Tensor`` or container of tensors and will work with any other Pytorch code. The type information is attached to every output tensor. If tensors without types is passed to your module, it will not fail, however the types will not be checked. Thus, it is recommended to define input/output types for all your modules, starting with data layers and add ``@typecheck()`` decorator to them.

.. note:: To temporarily disable typechecking, you can enclose your code in ```with typecheck.disable_checks():``` statement.


Dynamic Layer Freezing
----------------------

You can selectively freeze any modules inside a Nemo model by specifying a freezing schedule in the config yaml. Freezing stops any gradient updates
to that module, so that its weights are not changed for that step. This can be useful for combatting catastrophic forgetting, for example
when finetuning a large pretrained model on a small dataset.

The default approach is to freeze a module for the first N training steps, but you can also enable freezing for a specific range of steps,
for example, from step 20 - 100, or even activate freezing from some N until the end of training. You can also freeze a module for the entire training run.
Dynamic freezing is specified in training steps, not epochs.

To enable freezing, add the following to your config:

.. code-block:: yaml

  model:
    ...
    freeze_updates:
      enabled: true  # set to false if you want to disable freezing
      
      modules:   # list all of the modules you want to have freezing logic for
        encoder: 200       # module will be frozen for the first 200 training steps
        decoder: [50, -1]  # module will be frozen at step 50 and will remain frozen until training ends
        joint: [10, 100]   # module will be frozen between step 10 and step 100 (step >= 10 and step <= 100)
        transcoder: -1     # module will be frozen for the entire training run
