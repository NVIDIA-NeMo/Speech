Parameter-Efficient Fine-Tuning (PEFT)
======================================

PEFT is a popular technique used to efficiently finetune large language
models for use in various downstream tasks. When finetuning with PEFT,
the base model weights are frozen, and a few trainable adapter modules
are injected into the model, resulting in a very small number (<< 1%) of
trainble weights. With carefully chosen adapter modules and injection
points, PEFT achieves comparable performance to full finetuning at a
fraction of the computational and storage costs.

NeMo supports four PEFT methods which can be used with various
transformer-based models.

==================== ===== ===== ========= ==
\                    GPT 3 NvGPT LLaMa 1/2 T5
==================== ===== ===== ========= ==
Adapters (Canonical) ✅    ✅    ✅        ✅
LoRA                 ✅    ✅    ✅        ✅
IA3                  ✅    ✅    ✅        ✅
P-Tuning             ✅    ✅    ✅        ✅
==================== ===== ===== ========= ==

Learn more about PEFT in NeMo with the :ref:`peftquickstart` which provides an overview on how PEFT works
in NeMo. Read about the supported PEFT methods
`here <supported_methods.html>`__. For a practical example, take a look at
the `Step-by-step Guide <https://github.com/NVIDIA/NeMo/blob/main/tutorials/nlp/lora.ipynb>`__.

The API guide can be found `here <../../api.html#adapter-mixin-class>`__

.. toctree::
   :maxdepth: 1

   quick_start
   supported_methods