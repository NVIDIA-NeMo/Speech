# Huggingface Model Import and Forward Pass Tutorial

This tutorial demonstrates two Python scripts for importing a supported model from Huggingface and running a forward pass on the imported model. We will use Starcoder 2 3B for the purposes of this tutorial because it is relatively small and a forward pass can fit on a single GPU (The tutorial was developed on a RTX 5880 with 48GB memory).

The first script is for importing the model's Huggingface checkpoint to NeMo format, and the second script performs a forward pass on the imported model.

For this tutorial, we assume that all steps are being run inside a [NeMo docker container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).

## 1. Importing Model Checkpoint ([import_ckpt.py](./import_ckpt.py))

This script imports the Starcoder2 3B model checkpoint from HuggingFace to NeMo format. Proceed with the following steps:

1. `cd /opt/NeMo/examples/llm/import_and_forward`
1. Run the import_ckpt.py script to import the model checkpoint using `python import_ckpt.py`

This script imports the checkpoint from Huggingface to the `/workspace/starcoder2_3b_nemo2` directory using the `import_ckpt` function.

## 2. Performing a Forward Pass ([forward_step.py](./forward_step.py))

This script performs a forward pass using the imported model. To achieve this, execute the following steps:

1. `cd /opt/NeMo/examples/llm/import_and_forward`
1. Run the forward_step.py script using `python forward_step.py`.

This script initializes the previously imported model and uses it to compute the output using randomly generated input.