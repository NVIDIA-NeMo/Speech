Community Checkpoint Converter
==============================

We provide easy-to-use tools that enable users to convert community checkpoints into the NeMo format. These tools facilitate various operations, including resuming training, Sparse Fine-Tuning (SFT), Parameter-Efficient Fine-Tuning (PEFT), and deployment. For detailed instructions and guidelines, please refer to our documentation.

We offer comprehensive guides to assist both end users and developers:

- **User Guide**: Detailed steps on how to convert community model checkpoints for further training or deployment within NeMo. For more information, please see our :doc:`user_guide`.

- **Developer Guide**: Instructions for developers on how to implement converters for community model checkpoints, allowing for broader compatibility and integration within the NeMo ecosystem. For development details, refer to our :doc:`dev_guide`.

- **Megatron-LM Checkpoint Conversion**: NVIDIA NeMo and NVIDIA Megatron-LM share several foundational technologies. You can convert your GPT-style model checkpoints trained with Megatron-LM into the NeMo Framework using our scripts, see our :doc:`convert_mlm`.

Access the user and developer guides directly through the links below:

.. toctree::
   :maxdepth: 1
   :caption: Conversion Guides

   user_guide
   dev_guide
   convert_mlm
