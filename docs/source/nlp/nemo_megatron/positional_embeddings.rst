Positional embeddings
---------------------

Positional embeddings are used to give the model information about the position of each element in a sequence.  Megatron LLM supports the following positional embedding types:

GPT
^^^

.. list-table:: *Supported positional embeddings in GPT models*
   :widths: 10 30 60
   :header-rows: 1

   * - Parameter value
     - How to use
     - Description

   * - **learned_absolute**
     - .. code::
          
          model.position_embedding_type='learned_absolute'
     - Absolute Position Encodings :cite:`pos-emb-vaswani2023attention` are position embeddings used in Transformer-based models, added to input embeddings in the encoder and decoder sections. These encodings match the dimension of embeddings and are created using sine and cosine functions of various frequencies. Each dimension in the encoding corresponds to a sinusoid with wavelengths forming a geometric progression.

   * - **rope**
     - .. code::

          model.position_embedding_type='rope'
          model.rotary_percentage=1.0
     - Rotary Position Embedding (RoPE) :cite:`pos-emb-su2022roformer` incorporates positional information by utilizing a rotation matrix to encode the absolute positions of tokens while maintaining relative positional relationships in self-attention formulations by leveraging the geometric properties of vectors and complex numbers, applying a rotation based on a preset non-zero constant and the relative positions of the tokens to the word embeddings.
   
   * - **alibi**
     - .. code::

          model.position_embedding_type='alibi'
     - Attention with Linear Biases (ALiBi) :cite:`pos-emb-press2022train` modifies the way attention scores are computed in the attention sublayer of the network. ALiBi introduces a static, non-learned bias after the query-key dot product during the computation of attention scores. This bias is added in the form of a head-specific slope that is determined before training, creating a geometric sequence of slopes for the different heads in the model. The method has an inductive bias towards recency, penalizing attention scores between distant query-key pairs with the penalty increasing as the distance grows, and it leverages different rates of penalty increase across different heads based on the slope magnitude.

   * - **kerple**
     - .. code::

          model.position_embedding_type='kerple'
     - Kernelized Relative Positional Embedding for Length Extrapolation (KERPLE) :cite:`pos-emb-chi2022kerple` generalizes relative positional embeddings (RPE) by kernelizing positional differences using conditionally positive definite (CPD) kernels known for generalizing distance metrics. They transform CPD kernels into positive definite (PD) kernels by adding a constant offset, which is absorbed during softmax normalization in the self-attention mechanism of transformers. This approach allows for a variety of RPEs that facilitate length extrapolation in a principled manner.

   * - **xpos**
     - .. code::

          model.position_embedding_type='xpos'
     - Extrapolatable Position Embedding (xPos) :cite:`pos-emb-sun2022lengthextrapolatable`

   * - **sandwich**
     - .. code::

          model.position_embedding_type='sandwich'
     - Sandwich :cite:`pos-emb-chi2023dissecting`

T5
^^

.. list-table:: *Supported positional embeddings in T5 models*
   :widths: 10 30 60
   :header-rows: 1

   * - Parameter value
     - How to use
     - Description

   * - **learned_absolute**
     - .. code::

          model.encoder.position_embedding_type='learned_absolute'
          model.decoder.position_embedding_type='learned_absolute'
     - Absolute Position Encodings :cite:`pos-emb-vaswani2023attention` are position embeddings used in Transformer-based models, added to input embeddings in the encoder and decoder sections. These encodings match the dimension of embeddings and are created using sine and cosine functions of various frequencies. Each dimension in the encoding corresponds to a sinusoid with wavelengths forming a geometric progression.

   * - **relative**
     - .. code::

          model.encoder.position_embedding_type='relative'
          model.decoder.position_embedding_type='relative'
     - Relative Position Representations :cite:`pos-emb-shaw2018selfattention`

   * - **alibi**
     - .. code::

          model.encoder.position_embedding_type='alibi'
          model.decoder.position_embedding_type='alibi'
     - Attention with Linear Biases (ALiBi) :cite:`pos-emb-press2022train` modifies the way attention scores are computed in the attention sublayer of the network. ALiBi introduces a static, non-learned bias after the query-key dot product during the computation of attention scores. This bias is added in the form of a head-specific slope that is determined before training, creating a geometric sequence of slopes for the different heads in the model. The method has an inductive bias towards recency, penalizing attention scores between distant query-key pairs with the penalty increasing as the distance grows, and it leverages different rates of penalty increase across different heads based on the slope magnitude.

   * - **kerple**
     - .. code::

          model.encoder.position_embedding_type='kerple'
          model.decoder.position_embedding_type='kerple'
     - Kernelized Relative Positional Embedding for Length Extrapolation (KERPLE) :cite:`pos-emb-chi2022kerple` generalizes relative positional embeddings (RPE) by kernelizing positional differences using conditionally positive definite (CPD) kernels known for generalizing distance metrics. They transform CPD kernels into positive definite (PD) kernels by adding a constant offset, which is absorbed during softmax normalization in the self-attention mechanism of transformers. This approach allows for a variety of RPEs that facilitate length extrapolation in a principled manner.

Positional interpolation
------------------------
Position Interpolation (PI) :cite:`pos-emb-chen2023extending` is a method introduced to extend the context window sizes of Rotary Position Embedding (RoPE)-based pretrained large language models (LLMs). The central principle of PI is to reduce the position indices so that they align with the initial context window size through interpolation.

Positional Interpolation is supported in Megatron GPT SFT models. Set RoPE Interpolation factor for sequence length :code:`seq_len_interpolation_factor` to enable it.

.. code::

   model.position_embedding_type='rope'
   model.rotary_percentage=1.0
   model.seq_len_interpolation_factor: 2

References
----------

.. bibliography:: ../nlp_all.bib
    :style: plain
    :labelprefix: pos-emb
    :keyprefix: pos-emb-