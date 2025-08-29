# 🧠 TopIPL: Iterative Pseudo-Labeling for ASR

TopIPL is an **iterative pseudo-labeling algorithm** for training speech recognition models using both labeled and unlabeled data. It integrates seamlessly into the NeMo ASR pipeline and enables **self-training** across epochs with minimal manual intervention.

## 🚀 Key Features

- ⚙️ Supports **semi-supervised ASR training** with dynamic iterative pseudo-label refinement.
- 🧪 Designed for large-scale training using both labeled and unlabeled speech data.
- 🔁 Automatically writes pseudo-labels and updates training configs between iterations.

## 📦 Required Components

TopIPL relies on the following components:

- **[`SDPNeMoRunIPLProcessor`]**  
  Commands for running IPL are generated and submitted using SDP processors and NeMo-Run.  
  See instructions for usage [here](https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/sdp/processors/ipl/README.md).

- **Training Callback: `IPLEpochStopperCallback`**  
  Add this to your training config under `exp_manager` to **stop training at the end of each epoch**, enabling pseudo-label update:

```yaml
exp_manager:
  create_ipl_epoch_stopper_callback: True
  ipl_epoch_stopper_callback_params:
    stop_every_n_epochs: n # Stop training after every n epochs (default: 1)

