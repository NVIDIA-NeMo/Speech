

Supported PEFT methods
----------------------

NeMo supports the following PFET tuning methods

1. **Adapters (Canonical)**: `Parameter-Efficient Transfer Learning for
   NLP <http://arxiv.org/abs/1902.00751>`__

   -  Adapters (Houlsby setup) is one of the first PEFT methods applied
      to NLP. Adapter tuning is more efficient than full fine-tuning
      because the base model weights are frozen, while only a small
      number of adapter module weights are updated. In this method, two
      linear layers with a bottleneck and a non-linear activation are
      inserted into each transformer layer via a residual connection. In
      each case, the output linear layer is initialized to 0 to ensure
      that an untrained adapter does not affect the normal forward pass
      of the transformer layer.
   -  In NeMo, you can customize the adapter bottleneck dimension,
      adapter dropout amount, as well as the type and position of
      normalization layer.

2. **LoRA**: `LoRA: Low-Rank Adaptation of Large Language
   Models <http://arxiv.org/abs/2106.09685>`__

   -  LoRA makes fine-tuning efficient by representing weight updates
      with two low rank decomposition matrices. The original model
      weights remain frozen, while the low rank decomposition matrices
      are updated to adapt to the new data, so the number of trainable
      parameters is kept low. In contrast with adapters, the original
      model weights and adapted weights can be combined during
      inference, avoiding any architectural change or additional latency
      in the model at inference time.
   -  In NeMo, you can customize the adapter bottleneck dimension and
      the target modules to apply LoRA. LoRA can be applied to any linear
      layer. In a transformer model, this includes 1) Q, K, V attention
      projections, 2) attention output layer, and 3) either or both of
      the two transformer MLP layers. For QKV, NeMo's attention
      implementation fuses QKV into a single projection, so our LoRA
      implementation learns a single Low-Rank projection for QKV
      combined.

3. **IA3**: `Few-Shot Parameter-Efficient Fine-Tuning is Better and
   Cheaper than In-Context Learning <http://arxiv.org/abs/2205.05638>`__

   -  IA3 makes fine-tuning efficient by rescaling activations with
      learned vectors. The rescaling layers are injected in the
      attention (for key and value) and feedforward modules in the base
      model. Similar to other PEFT methods, only the rescaling vectors
      are updated during fine-tuning to adapt to the new data so the
      number of updated parameters is low. However, since rescaling
      vectors are much smaller than low rank matrices (LoRA) and
      bottleneck layers (Adapters), IA3 cuts down the number of
      trainable parameters further by an order of magnitude. The
      learning rescaling vectors can also be merged with the base
      weights, leading to no architectural change and no additional
      latency at inference time.
   -  There is no hyperparameter to tune for the IA3 adapter.

4. **P-Tuning**: `GPT Understands,
   Too <https://arxiv.org/abs/2103.10385>`__

   -  P-tuning is an example of the prompt learning family of methods,
      in which trainable virtual tokens are inserted into the model
      input prompt to induce it to perform a task. Virtual tokens (also
      called "continuous" or "soft" tokens) are embeddings that have no
      concrete mapping to strings or characters within the model’s
      vocabulary. They are simply 1D vectors that match the
      dimensionality of real tokens which make up the model's
      vocabulary.
   -  In p-tuning, an intermediate MLP model is used to generate
      virtual token embeddings. We refer to this intermediate model as
      our ``prompt_encoder``. The prompt encoder parameters are randomly
      initialized at the start of p-tuning. All base model parameters
      are frozen, and only the prompt encoder weights are updated at
      each training step.
   -  In Nemo, you can customize the number of virtual tokens, as well
      as the embedding and MLP bottleneck dimensions.
