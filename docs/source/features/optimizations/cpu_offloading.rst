CPU Offloading
==============

Overview
--------

CPU Offloading in NeMo is a feature that reduces the peak memory usage of the GPU by offloading activations and inactive weights to CPU storage. NeMo supports offloading at the transformer layer level, allowing users to specify the number of transformer layers in their language model that require CPU offloading. During the forward pass, NeMo offloads activations at the optimal time and reloads them as needed during the backward pass.

Features
--------
- Supports training models with long sequence lengths by managing activation memory efficiently.
- Enables high batch sizes per GPU by offloading activation memory.
- Overlaps computation with data transfers (Host2Device and Device2Host) during offloading and reloading.

Usage
-----
- Set cpu_offloading to True to enable CPU offloading.
- Set cpu_offloading_num_layers to a value between 0 and the total number of layers in the model minus one.
- Set cpu_offloading_activations and cpu_offloading_weights based on your needs to offload activations only, weights only, or both.
