Exporting NeMo Models
=====================

Exporting Models
----------------

Most of the NeMo models can be exported to ONNX or TorchScript to be deployed for inference in optimized execution environments, such as Riva or Triton Inference Server.  
Export interface is provided by the ``Exportable`` mix-in class. If a model extends ``Exportable``, it can be exported by:

.. code-block:: Python

   from nemo.core.classes import ModelPT, Exportable
   # deriving from Exportable
   class MyExportableModel(ModelPT, Exportable):
   ...

   mymodel = MyExportableModel.from_pretrained(model_name="MyModelName")
   
   # exporting pre-trained model to ONNX file for deployment.	
   mymodel.export('mymodel.onnx', [options])


How to Use Model Export
-----------------------
The following arguments are for ``Exportable.export()``. In most cases, you should only supply the name of the output file and use all defaults:

.. code-block:: Python

    def export(
        self,
        output: str,
        input_example=None,
        output_example=None,
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        onnx_opset_version: int = 13,
        try_script: bool = False,
        set_eval: bool = True,
        check_trace: bool = False,
        use_dynamic_axes: bool = True,
        dynamic_axes=None,
        check_tolerance=0.01,
    ):

The ``output``, ``input_example``, ``output_example``, ``verbose``, ``export_params``, ``do_constant_folding``, ``keep_initializers_as_inputs``, ``onnx_opset_version``, ``set_eval`` options have the same semantics as in Pytorch ``onnx.export()`` and ``jit.trace()`` functions and are passed through. For more information about Pytorch's``onnx.export()``, refer to the `torch.onnx functions documentation
<https://pytorch.org/docs/stable/onnx.html#functions>`_.

The file extension of the ``output`` parameter determines export format: ``.onnx->ONNX``, ``.pt`` or ``.ts`` -> ``TorchScript``. If ``input_example`` is None, ``Exportable.input_example()`` is called.

**TorchScript-specific**: If ``try_script`` is ``True``, ``export()`` tries ``jit.script()`` before ``jit.trace()``.
The ``check_trace`` arg is passed through to ``jit.trace()``.
**ONNX-specific**: If ``use_dynamic_axes`` is True, ``onnx.export()`` is called with dynamic axes. If ``dynamic_axes`` is ``None``, they are inferred from the model's ``input_types`` definition (batch dimension is dynamic, and so is duration etc).

If ``check_trace`` is ``True``, the resulting ONNX also runs on ``input_example`` and the results compared to ``output_example`` using the ``check_tolerance`` argument. Note the higher tolerance default.


How to Make Model Exportable
----------------------------

If you are simply using NeMo models, the previous example is all you need to know.
If you write your own models, this section highlights the things you need to be aware of after extending ``Exportable``.

Exportable Hooks and Overrides
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You should not normally need to override ``Exportable`` default methods. However, ``Exportable.export()`` relies on the assumptions that certain methods are available in your class.

.. code-block:: Python

    @property
    def input_example(self) # => Tuple(input, [(input, ...], [Dict])
         """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples. 
	 """

This function should return a tuple of (normally) Tensors - one per each of model inputs (args to ``forward()``). The last element may be a ``Dict`` to specify non-positional arguments by name, as per Torch ``export()`` convention. For more information, refer to the `Using dictionaries to handle Named Arguments as model inputs
<https://pytorch.org/docs/stable/onnx.html#using-dictionaries-to-handle-named-arguments-as-model-inputs>`_.

.. Note: ``Dict`` currently does not work with Torchscript ``trace()``.

.. code-block:: Python

    @property
    def input_types(self):
    @property
    def output_types(self):
    
Those are needed for inferring in/out names and dynamic axes. If your model derives from ``ModulePT``, those are already there. Another common scenario is that your model contains one or more modules that processes input and generates output. Then, you should override ``Exportable`` methods ``input_module()`` and ``output_module()`` to point to them, like in this example:

.. code-block:: Python

    @property
    def input_module(self):
        return self.fastpitch

    @property
    def output_module(self):
        return self.fastpitch

Your model should also have an export-friendly ``forward()`` method - that can mean different things for ONNX ant TorchScript. For ONNX, you can't have forced named parameters without default, like ``forward(self, *, text)``. For TorchScript, you should avoid ``None`` and use ``Optional`` instead. The criteria are highly volatile and may change with every PyTorch version, so it's a trial-and-error process. There is also the general issue that in many cases, ``forward()`` for inference can be simplified and even use less inputs/outputs. To address this, ``Exportable`` looks for ``forward_for_export()`` method in your model and uses that instead of ``forward()`` to export:

.. code-block:: Python

    # Uses forced named args, many default parameters. 
    def forward(
        self,
        *,
        text,
        durs=None,
        pitch=None,
        speaker=0,
        pace=1.0,
        spec=None,
        attn_prior=None,
        mel_lens=None,
        input_lens=None,
    ):
        # Passes through all self.fastpitch outputs
        return self.fastpitch(
            text=text,
            durs=durs,
            pitch=pitch,
            speaker=speaker,
            pace=pace,
            spec=spec,
            attn_prior=attn_prior,
            mel_lens=mel_lens,
            input_lens=input_lens,
        )


    # Uses less inputs, no '*', returns less outputs:
    def forward_for_export(self, text):
        (
            spect,
            durs_predicted,
            log_durs_predicted,
            pitch_predicted,
            attn_soft,
            attn_logprob,
            attn_hard,
            attn_hard_dur,
            pitch,
        ) = self.fastpitch(text=text)
        return spect, durs_predicted, log_durs_predicted, pitch_predicted

To stay consistent with input_types()/output_types(), there are also those hooks in ``Exportable`` that let you exclude particular inputs/outputs from the export process:

.. code-block:: Python

    @property
    def disabled_deployment_input_names(self):
        """Implement this method to return a set of input names disabled for export"""
        return set(["durs", "pitch", "speaker", "pace", "spec", "attn_prior", "mel_lens", "input_lens"])

    @property
    def disabled_deployment_output_names(self):


Another common requirement for models that are being exported is to run certain net modifications for inference efficiency before exporting - like disabling masks in some convolutions or removing batch normalizations. A better style is to make those happen on ``ModelPT.eval()`` (and reversed on ``.train()``), but it's not always feasible so the following hook is provided in ``Exportable`` to run those:

.. code-block:: Python

    def _prepare_for_export(self, **kwargs):
        """
        Override this method to prepare module for export. This is in-place operation.
        Base version does common necessary module replacements (Apex etc)
        """
    # do graph modifications specific for this model
        replace_1D_2D = kwargs.get('replace_1D_2D', False)
        replace_for_export(self, replace_1D_2D)
    # call base method for common set of modifications
	Exportable._prepare_for_export(self, **kwargs)


Exportable Model Code
~~~~~~~~~~~~~~~~~~~~~

Most importantly, the actual Torch code in your model should be ONNX or TorchScript - compatible (ideally, both).
#. Ensure the code is written in Torch - avoid bare `Numpy or Python operands <https://pytorch.org/docs/stable/onnx.html#write-pytorch-model-in-torch-way>`_.
#. Create your model ``Exportable`` and add an export unit test, to catch any operation/construct not supported in ONNX/TorchScript, immediately.

For more information, refer to the PyTorch documentation:
       - `List of supported operators <https://pytorch.org/docs/stable/onnx.html#supported-operators>`_
       - `Tracing vs. scripting <https://pytorch.org/docs/stable/onnx.html#tracing-vs-scripting>`_ 
       - `AlexNet example <https://pytorch.org/docs/stable/onnx.html#example-end-to-end-alexnet-from-pytorch-to-onnx>`_

