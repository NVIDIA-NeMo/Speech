Community Model Converter Development Guide
===========================================

Guideline Steps for Checkpoint Conversion
-----------------------------------------

1. **Understand Both Frameworks**: Familiarize yourself with the architectures and naming conventions of both HuggingFace and NeMo models.

2. **Load Community Checkpoint**: For example, use HuggingFace’s ``AutoModel`` to load the pre-trained model.

3. **Inspect Model and Config**: Understand the layer names, parameter shapes, and essential configs.

4. **Adjust NeMo Model Configuration**: Modify the NeMo model configuration to match the HuggingFace model’s specifications.

5. **Initialize NeMo Model**: Create an instance of the corresponding NeMo model.

6. **Create Key Mapping**: Define a function to map HuggingFace layer names to NeMo layer names. Adjust for any structural differences.

7. **Rename and Reshape Parameters**: Implement a function to rename keys in the HuggingFace state dictionary and reshape tensors if necessary. For example, QKV weights usually need some special handling from HF to NeMo.

8. **Load Converted Weights into NeMo Model**: Apply the transformed state dictionary to the NeMo model.

9. **Save NeMo Checkpoint**: Save the updated NeMo model as a new checkpoint.

10. **Verification**: Verify the performance of the NeMo model to ensure successful conversion.

11. **Add Docstrings and Comments**: Please kindly comment the expected shapes in the parameter reshaping part.

12. **Add Jenkins Tests**: Please use `Llama Huggingface to NeMo converter test <https://github.com/NVIDIA/NeMo/blob/main/Jenkinsfile#L418>`_  as an example for development.

Script Placement and Naming Conventions
---------------------------------------

- **Script Location**: Place scripts in the ``NeMo/scripts/checkpoint_converters`` directory.

- **Script Naming**: Name your script following the format ``convert_{model}_{source}_to_{target}.py``, such as ``convert_llama_hf_to_nemo.py``.

- **Unified Arguments (APIs)**: User only needs to define input and output files. Configs should be automatically updated.

  - ``--input_name_or_path``: Specify the name or path of the model. Give one example default value.

  - ``--output_path``: Set the path for saving the output .nemo file. This argument is required.

  - ``--hparams_file``: Define the path for the configuration file needed for restoration. Set default path to an existing and working yaml file e.g. ``f"{os.path.dirname(__file__)}/../../examples/nlp/language_modeling/conf/megatron_bert_config.yaml"``. A regular user should not change it, but for advanced/internal users, this can be modified.

  - ``--precision``: Choose the precision for saved checkpoint weights. Options: "bf16", "16", "32". Default: "32".

Code Template
-------------

Below template tries to address the 11 steps in the guideline part. Please also use `Gemma Huggingface to NeMo converter <https://github.com/NVIDIA/NeMo/tree/main/scripts/checkpoint_converters/convert_gemma_hf_to_nemo.py>`__  as an full example for development.

.. code-block:: python

    import os
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer, AutoModel
    from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel
    from nemo.utils import logging
    from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder

    # Add additional imports and custom functions as required

    def create_rename_keys(num_hidden_layers):
        # Your implementation of create_rename_keys function
        ...

    def adjust_tensor_shapes(model, nemo_state_dict):
        # Your implementation of adjust_tensor_shapes function
        ...

    def adjust_nemo_config(model_config, ref_config):
        # Your implementation of adjust_nemo_config function
        ...

    def rename_model_keys(model_state_dict, rename_keys):
        """
        Rename keys in the model's state dictionary based on the provided mappings.

        Parameters:
        model_state_dict (dict): The state dictionary of the model.
        rename_keys (list): A list of tuples with the mapping (old_key, new_key).

        Returns:
        dict: A new state dictionary with updated key names.
        """

        # Create a new state dictionary with updated key names
        new_state_dict = {}

        # Track keys from the original state dict to ensure all are processed
        remaining_keys = set(model_state_dict.keys())

        # Iterate over the rename mappings
        for old_key, new_key in rename_keys:
            if old_key in model_state_dict:
                # Rename the key and remove it from the tracking set
                new_state_dict[new_key] = model_state_dict[old_key]
                remaining_keys.remove(old_key)
            else:
                print(f"Warning: Key '{old_key}' not found in the model state dictionary.")

        # Check if any keys were not converted from old to new
        for old_key in remaining_keys:
            print(f"Warning: Key '{old_key}' was not converted.")

    def get_args():
        # Arg names subject to change, feel free to suggest.
        parser = ArgumentParser()
        parser.add_argument("--input_name_or_path", type=str, default="intfloat/e5-large-unsupervised")
        parser.add_argument(
            "--hparams_file",
            type=str,
            default=f"{os.path.dirname(__file__)}/../../examples/nlp/language_modeling/conf/megatron_bert_config.yaml",
            required=False,
            help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
        )
        parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output .nemo file.")
        parser.add_argument(
            "--precision", type=str, default="32", choices=["bf16", "32"], help="Precision for checkpoint weights saved"
        )

        args = parser.parse_args()
        return args

    def convert(args):
        logging.info(f"Loading checkpoint from HF: `{args.name_or_path}`")
        hf_model = AutoModel.from_pretrained(args.name_or_path)

        nemo_config = OmegaConf.load(args.hparams_file)
        nemo_config.model = adjust_nemo_config(nemo_config.model, hf_model.config.to_dict())

        nemo_config.trainer["precision"] = args.precision
        trainer = MegatronTrainerBuilder(nemo_config).create_trainer()
        model = MegatronBertModel(nemo_config.model, trainer)

        old_state_dict = hf_model.state_dict()
        rename_keys = create_rename_keys(nemo_config.model.num_layers)
        new_state_dict = rename_model_keys(model_state_dict=old_state_dict, rename_keys=rename_keys)
        nemo_state_dict = adjust_tensor_shapes(model, new_state_dict)
        model.load_state_dict(nemo_state_dict, strict=True)

        # Additional verification and processing steps
        ...

        model.save_to(args.save_path)
        logging.info(f'NeMo model saved to: {args.save_path}')

    if __name__ == '__main__':
        args = get_args()
        convert(args)



*Notes:* This template abstracts some functions (create_rename_keys, adjust_tensor_shapes, adjust_nemo_config) which are crucial for the conversion process. These functions need to be adapted based on specific model architectures and requirements. Ensure that the NeMo model’s configuration is properly aligned with the HuggingFace model’s configuration. It is important to thoroughly test the converted model to validate the conversion process.


Development Tips
----------------

A Simple Guide for Model Mapping and Conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Mapping between community model and NeMo model**:

   - Match the configurations between the community model and the NeMo model.
   - Create two text files, ``state_src.txt`` and ``state_tgt.txt``, containing the state dict weights and their shapes for easier reference and debugging.

   Example code to generate ``state_src.txt``:

   .. code-block:: python

       file_path = "state_src.txt"
       state = model.state_dict()
       with open(file_path, 'w') as file:
           for k, v in state.items():
               file.write(f"{k} {v.shape}\n")

   - Utilize language models (LMs) to assist in completing the key mapping through the ``create_rename_keys`` function. Here's an example prompt for Gemma:

     .. code-block:: text

        Map the following key names and tensor shapes from Model A to their equivalents in Model B. Here is an example mapping: Model A's 'model.layer.weight' corresponds to Model B's 'module.block.weight'.
        ============================================================
        embedder.weight torch.Size([256128, 2048])
        ...
        ============================================================

   Based on the results, update the following code accordingly:

   .. code-block:: python

       def create_rename_keys(num_hidden_layers):
           rename_keys = []
           for i in range(num_hidden_layers):
               # encoder layers: output projection, 2 feedforward neural networks, and 2 layernorms
               # @chatgpt to fill in layer-dependent keys above

           # @chatgpt fill in non-layer-dependent keys above
           rename_keys.extend(
               [
                   # ...
               ]
           )

           return rename_keys

   **Note**: Also list all the keys not included in the conversion above.

2. **Common issues when converting: results not matching between Community model and NeMo model**:

   a. Megatron Core uses a special QKV layout, which needs careful handling and reshaping from community models, especially when GQA or MQA is used. Refer to the `Gemma Huggingface to NeMo converter <https://github.com/NVIDIA/NeMo/tree/main/scripts/checkpoint_converters/convert_gemma_hf_to_nemo.py#L144>`__ for guidance.

   b. GLU Variants weights could also be a common source of error. In Megatron Core, the regular feedforward projection weights and gated forward weights are fused together, requiring careful attention to the order of these two. Refer to the `Gemma Huggingface to NeMo converter <https://github.com/NVIDIA/NeMo/tree/main/scripts/checkpoint_converters/convert_gemma_hf_to_nemo.py#L135>`_ for more details.

3. The ``create_hf_model`` function can be used to create a model programmatically. For reproducibility, see the example provided at `GitHub <https://github.com/NVIDIA/NeMo/blob/main/tests/setup/models/create_hf_model.py>`_. This function creates a randomly initialized HuggingFace model for testing purposes. The model can be specified by name or path for creating its config and tokenizer using HuggingFace transformers AutoConfig and AutoTokenizer functions.

Example usage:

.. code-block:: python

    create_hf_model(
        model_name_or_path="/home/TestData/nlp/meta-llama/Llama-2-7b-hf",
        output_dir=os.path.join(args.save_dir, "megatron_llama/llama-ci-hf"),
        config_updates={
            "hidden_size": 256,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 4
        },
        overwrite=args.overwrite,
    )

