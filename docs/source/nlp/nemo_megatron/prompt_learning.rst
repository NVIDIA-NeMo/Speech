.. _promptlearning:

Prompt Learning
---------------

Within NeMo we refer to **p-tuning** and **prompt tuning** methods collectively as prompt learning. Both methods are parameter efficient alternatives to fine-tuning pretrained language models. Our NeMo implementation makes it possible to use one pretrained GPT model on many downstream tasks without needing to tune the model's full set of parameters. It also allows for adding new tasks to your model without overwriting or disrupting previous tasks for which the model has already been p-tuned/prompt-tuned. Because the original model parameters are frozen and never altered by either method, p-tuning/prompt-tuning also avoids catastrophic forgetting issues often encountered when fine-tuning models. 

Instead of selecting discrete text prompts in a manual or automated fashion, prompt tuning and p-tuning utilize virtual prompt embeddings that can be optimized via gradient descent. The only difference between prompt tuning and p-tuning within NeMo-Megatron is the architecture used to tune the soft prompt tokens during training.

- Our prompt tuning implementation is based off Lester et. al’s EMNLP 2021 paper "`The Power of Scale for Parameter-Efficient Prompt Tuning <https://arxiv.org/abs/2104.08691>`_"
- Our p-tuning implementation is based off Liu et al's paper "`GPT Understands, Too <https://arxiv.org/abs/2103.10385>`_"

Our continuous learning capability for combined p-tuning and prompt tuning with GPT style models is a NeMo specific extension of the author's original work.

Please also checkout our `prompt learning tutorial notebook. <https://github.com/NVIDIA/NeMo/tree/stable/tutorials/nlp/Multitask_Prompt_and_PTuning.ipynb>`_


Terminology
^^^^^^^^^^^
We will be using the terms ``continuous``, ``soft``, and ``virtual`` token interchangeably to refer to embeddings inserted into the model prompt that have no concrete mapping to strings or characters within the model’s vocabulary. These virtual token embeddings exist in contrast to the ``discrete``, ``hard``, or ``real`` tokens that do make up the model’s vocabulary. Virtual tokens are purely 1D vectors with dimensionality equal to that of each real token embedding, matching the ``hidden_size`` hyperparameter. In training and inference, continuous token embeddings are inserted among discrete token embeddings according to a template you provide in the model's config. We will demonstrate how to do this below.

When referring to p-tuning and prompt tuning together, we will be using the phrase prompt learning for simplicity.

Prompt Tuning
^^^^^^^^^^^^^

In prompt-tuning a pretrained GPT model, soft prompt embeddings are initialized as a 2D matrix of size ``total_virtual_tokens X hidden_size``. Each task the model is prompt-tuned to perform has its own 2D embedding matrix associated with it. Tasks do not share any parameters during training or inference. All GPT model parameters are frozen and only the embedding parameters for each task are updated during training.

In prompt tuning you can specify how the embeddings are initialized for each task. You can either

- Initialize embedding parameters according to some random distribution
- Initialize embedding parameters from existing vocabulary embeddings (recommended)

If you choose to initialize virtual token embeddings from existing embedding weights, you can provide the string of words you want to use for initialization in the model's config. This string will be tokenized and tiled or truncated to match the specified number of virtual tokens you would like to use (``total_virtual_tokens``). Vocab embeddings are copied and used to initialize the soft prompt embedding matrix for each task. The vocab embeddings themselves are not updated or changed during prompt tuning.

P-Tuning
^^^^^^^^

In p-tuning, an LSTM model is used to predict virtual token embeddings. We refer to this LSTM model as our ``prompt_encoder``. LSTM parameters are randomly initialized at the start of p-tuning. All GPT model parameters are frozen, and only the LSTM weights are updated at each training step. LSTM parameters are shared between all tasks that are p-tuned at the same time, but the LSTM model outputs unique virtual token embeddings for each task. The virtual tokens predicted by the LSTM are inserted among the discrete token input in the exact same manner as with prompt-tuning. You still specify the number of virtual tokens you want to use by setting ``total_virtual_tokens`` and each virtual token embedding is still a 1D vector of size ``hidden_size``.

Using Both Prompt and P-Tuning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A single pretrained GPT model can use both p-tuning and prompt-tuning. While you must decide to use either p-tuning or prompt-tuning for each task you want your model to perform, you can p-tune your model on a set of tasks *A*, then prompt tune your same model on a different set of tasks *B*, then finally run inference on tasks from both *A* and *B* at the same time. During prompt-tuning or p-tuning, tasks tuned at the same time must use the same number of virtual tokens. During inference, tasks using differing amounts of virtual tokens can be run at the same time.

When p-tuning completes, prompt tuned virtual tokens from the p-tuning ``prompt_encoder`` are automatically moved to the ``prompt_table`` where all prompt tuned and p-tuned soft prompts are stored. The LSTM ``prompt_encoder`` is then removed from the model. This allows us to preserve previously p-tuned soft prompts while still maintaining the ability to add new p-tuned or prompt-tuned soft prompts in the future. The ``prompt_table`` uses the ``taskname`` as a key to look up the correct virtual tokens for a specified task. The ``prompt_table``'s hash table data structure also makes it possible for each task to flexibly use a different number of virtual tokens. 

P-tuning usually requires fewer virtual tokens per task to achieve good results but uses a higher number of parameters compared to prompt-tuning. For example, if you prompt tune a 125M parameter GPT model (with hidden size 768) on two tasks, using 100 virtual tokens per task, the total parameters tuned during prompt tuning would equal 153k (~.1% of the pre-trained model size). If you p-tune the same 125M GPT model on 2 tasks, using an LSTM with two layers and 10 tokens per task, you will be tuning 8.3M parameters (~6.6% of the pre-trained model size). The increased number of parameters used during p-tuning is mitigated by our ``prompt_table``. When p-tuned soft prompts are placed in the prompt table, only the parameters for the predicted virtual tokens are saved. This allows us to keep the benefit of tuning a larger number of parameters during training, while also preserving the parameter efficiency of prompt-tuning during inference and storing of the model.

Because p-tuning shares parameters between tasks during training, p-tuning your model on multiple tasks that are similar might allow your model to share insight between tasks. In the same vein, p-tuning on many very different tasks at once might perform worse than prompt tuning, which tunes a distinct set of parameters per task. **Generally we recommend using p-tuning over prompt tuning.**

Users can also optionally tune the model's full parameters in addition to the soft prompt parameters. See ``model.lm_finetune`` in the Prompt Learning Config section for details on how to configure this.

Dataset Preprocessing
^^^^^^^^^^^^^^^^^^^^^

The prompt learning dataset accepts a list of json/dictionary objects or a list of json file names where each json file contains a collection of json objects. Each json object must include the field ``taskname`` which is a string identifier for the task the data example corresponds to. They should also include one or more fields corresponding to different sections of the discrete text prompt. The input data might look like:

.. code::

  [
    {"taskname": "squad", "context": [CONTEXT_PARAGRAPH_TEXT1], "question": [QUESTION_TEXT1], "answer": [ANSWER_TEXT1]},
    {"taskname": "squad", "context": [CONTEXT_PARAGRAPH_TEXT2], "question": [QUESTION_TEXT2], "answer": [ANSWER_TEXT2]},
    {"taskname": "intent_and_slot", "utterance": [UTTERANCE_TEXT1], "label": [INTENT_TEXT1][SLOT_TEXT1]},
    {"taskname": "intent_and_slot", "utterance": [UTTERANCE_TEXT2], "label": [INTENT_TEXT2][SLOT_TEXT2]},
    {"taskname": "sentiment", "sentence": [SENTENCE_TEXT1], "label": [SENTIMENT_LABEL1]},
    {"taskname": "sentiment", "sentence": [SENTENCE_TEXT2], "label": [SENTIMENT_LABEL2]},
  ]
  
These additional fields can be unlimited in number and will be used to help map different parts of the discrete text input to a prompt template that you define. We show how this mapping works and how to construct your prompt template in the Prompt Formatting section. Data examples for each dataset can all be passed to the dataset class in one file, or in separate ``.jsonl`` files in a list.
  
.. _data-example-label:

Prompt Formatting
^^^^^^^^^^^^^^^^^

To customize different prompts for different tasks, we simply need to specify the prompt task template in the config file at ``model.task_templates``. The virtual token markers ``<|VIRTUAL_PROMPT_#|>`` signify where you want virtual tokens to be placed in the template string. ``<|VIRTUAL_PROMPT_0|>``, ``<|VIRTUAL_PROMPT_1|>``, and ``<|VIRTUAL_PROMPT_2|>`` indicate where a number of virtual tokens matching the values given at ``virtual_token_splits[0]``, ``virtual_token_splits[1]`` and ``virtual_token_splits[2]`` will be placed. The other variable fields ``{var}`` refer to the fields in the data json.

For example, given:

- the data json ``{"sentence1": "And he said, Mama, I'm home.", "sentence2": "He didn't say a word."}``
- virtual token splits set to ``virtual_token_splits = [3, 3, 3]``
- a prompt template set to ``prompt_template = "<|VIRTUAL_PROMPT_0|> Hypothesis: [sentence1], <|VIRTUAL_PROMPT_1|> Premise: [sentence2] <|VIRTUAL_PROMPT_2|> Answer:"``

the input will be translated into ``VVV Hypothesis: And he said, Mama, I'm home. VVV Premise: He didn't say a word. VVV Answer:``, where ``VVV`` are three virtual tokens.

**We recommend you first try prompt learning by placing all virtual tokens at the very beginning of your prompt template** like we do with the ``sentiment`` task example below. We've found this gives strong performance. 
.. code::

  config.model.task_templates = [
    {
        "taskname": "sentiment",
        "prompt_template": "<|VIRTUAL_PROMPT_0|> {sentence} sentiment: {label}",
        "total_virtual_tokens": 10,
        "virtual_token_splits": [10],
        "truncate_field": "sentence",
        "answer_only_loss": False,
    },
    {
        "taskname": "intent_and_slot",
        "prompt_template": "<|VIRTUAL_PROMPT_0|> Predict intent and slot <|VIRTUAL_PROMPT_1|> :\n{utterance}{label}",
        "total_virtual_tokens": 10,
        "virtual_token_splits": [7, 3],
        "truncate_field": None,
        "answer_only_loss": True,
        "answer_field": "label"
    }
  ]

.. _prompt-formatting-label:

``model.task_templates`` Config Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. list-table:: 
    :widths: 15 15 25
    :header-rows: 1
    
    * - **Parameter**
      - **Data type**
      - **Description**
    * - **taskname**
      - string
      - Short string denoting the task, used to lookup task specific virtual tokens from the ``prompt_table``. Refers to the same ``taskname`` in the dataset json objects.
    * - **prompt_template**
      - string
      - a string showing the model where to place virtual tokens and how to map dataset json fields to where they belong in the model prompt
    * - **total_virtual_tokens**
      - int
      - specifies the total number of virtual tokens that will be inserted into the model prompt
    * - **virtual_token_splits**
      - list of ints
      - specifies the number of virtual tokens that belong at each ``<|VIRTUAL_PROMPT_#|>`` marker. ``virtual_token_splits`` values should add up to ``total_virtual_tokens``. The number of ``virtual_token_splits`` should match the number of ``<|VIRTUAL_PROMPT_#|>`` markers.
    * - **answer_only_loss**
      - bool
      - Whether to limit loss calculation to only the answer portion of the prompt during tuning. Strongly recommended for long prompts. 
    * - **answer_field**
      - string
      - The field in the data json corresponding to the answer. The loss will only be calculated on this portion of the prompt if ``answer_only_loss`` is ``True``. The answer field must be at the end of the prompt template. 
    * - **truncate_field** 
      - string
      - specifies which field in the data json to truncate if the length of the input exceeds the maximum sequence length of the model. If ``truncate_field`` is set to ``None``, examples that are too long are simply dropped from the dataset.

Prompt Learning Specific Config Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. list-table::
   :widths: 15 15 25
   :header-rows: 1
   
   * - **Parameter**
     - **Data type**
     - **Description**
   * - **model.nemo_path**
     - string
     - Path to where you want to save your model after prompt tuning/p-tuning, must end in `.nemo`
   * - **model.virtual_prompt_style**
     - string
     - one of 'prompt-tuning', 'p-tuning', or 'inference'
   * - **model.language_model_path**
     - string
     - Path to the GPT language model .nemo file you want to use for prompt learning, not needed if ``restore_path`` is set
   * - **model.restore_path**
     - string
     - Path to a .nemo file of existing ``MegatronGPTPromptLearningModel`` that has already been prompt tuned or p-tuned on at least one task. P-tuned or prompt tuned in this training session will be added to this model's `prompt_table`. Should be set to ``null`` if none.
   * - **model.new_tasks**
     - list of strings
     - List of new tasknames to be prompt or p-tuned, 
   * - **model.existing_tasks**
     - list of strings
     - List of tasks the model has already been p-tuned/prompt-tuned for, needed when a restore path is given. Should be set to ``[]`` if None. 
   * - **model.task_templates**
     - list
     - See the ``model.task_templates`` Config Parameters Table above
   * - **model.prompt_tuning.new_prompt_init_methods**
     - list of strings
     - List of 'text' or 'random', should correspond to the order of tasks listed in ``model.new_tasks``. Only needed if `virtual_prompt_style='prompt-tuning'`
   * - **model.prompt_tuning.new_prompt_init_text**
     - list of strings
     - The text you want to use for soft prompt initalization if ``model.prompt_tuning.new_prompt_init_methods`` is set to 'text' for a task. Should correspond to the order of tasks listed in ``model.new_tasks``. The text is tokenized and clipped or tiled to match ``total_virtual_tokens`` in ``model.task_templates``. The vocab embeddings associated with each token are copied and use to initialize the soft prompts before tuning.
   * - **model.p_tuning.dropout**
     - float
     - LSTM prompt encoder dropout prob
   * - **model.p_tuning.num_layers**
     - int
     - Num layers in LSTM prompt encoder
   * - **model.tensor_model_parallel_size**
     - int
     - intra-layer model parallelism, must match the ``tensor_model_parallel_size`` of the GPT model given at ``language_model_path``
   * - **model.batch_size**
     - int
     - global batch size 
   * - **model.data.train_ds**
     - list of strings
     - list of ``.json`` or ``.jsonl`` training dataset files with json ojects that have the dataset format described above
   * - **model.data.validation_ds**
     - list of strings
     - list of ``.json`` or ``.jsonl`` validation dataset files with json ojects that have the dataset format described above
   * - **model.data.add_eos**
     - bool
     - Whether to add an EOS token at the end of each training example (recommended). 

An example config file can be found at https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/language_modeling/conf/megatron_gpt_prompt_learning_config.yaml

Setting New Tasks
^^^^^^^^^^^^^^^^^

After you p-tune or prompt-tune your model, you can always go back and p-tune or prompt-tune your model on more tasks without over writing the virtual prompts who've trained already. You can also use a different number of ``total_virtual_tokens`` between each training session as long as tasks ptuned or prompt tuned at the same time have the same number of ``total_virtual_tokens``. For this reason, when you ptune on a new task, you need to tell your model which of your tasks are new and which ones already exist (and thus you don't want to tune them). You do this by setting the ``new_tasks`` and ``existing_tasks`` values in the config file.

Example Multi-Task Prompt Tuning Config and Command
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
First define a config called ``multitask-prompt-learning.yaml`` demonstrated below. **In the** ``exp_manager`` **portion of the config,** ``save_nemo_on_train_end`` **should be set to** ``False`` **to avoid unnecessarily saving the incorrect model weights.** This is already done in the example `megatron_gpt_prompt_learning_config.yaml config <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/language_modeling/conf/megatron_gpt_prompt_learning_config.yaml>`_ that you should use as your starting point. The correct prompt learning model will be saved at the ``model.nemo_path`` you set.

.. code::
  
  name: multitask_prompt_tuning
  trainer: ...
  exp_manager: ...
  model:
    seed: 1234
    nemo_path: ${name}.nemo 
    virtual_prompt_style: "prompt-tuning" 
    encoder_seq_length: 2048 
    tensor_model_parallel_size: 1 
    pipeline_model_parallel_size: 1 
    global_batch_size: 16
    micro_batch_size: 4

    restore_path: null 
    language_model_path: models/megatron_125M_gpt.nemo
    existing_tasks: []
    new_tasks: ["sentiment", "intent_and_slot"] 

    task_templates: 
    - taskname: "sentiment" 
      prompt_template: "<|VIRTUAL_PROMPT_0|> {sentence} sentiment: {label}" 
      total_virtual_tokens: 100 
      virtual_token_splits: [100] 
      truncate_field: null
      answer_only_loss: False

    - taskname: "intent_and_slot"
      prompt_template: "<|VIRTUAL_PROMPT_0|> Predict intent and slot <|VIRTUAL_PROMPT_1|> :\n{utterance}{label}" 
      total_virtual_tokens: 100 
      virtual_token_splits: [80, 20]
      truncate_field: null
      answer_only_loss: True
      answer_field: "label"

    prompt_tuning: 
      new_prompt_init_methods: ["text", "text"] 
      new_prompt_init_text: ["financial sentiment analysis postive neutral negative", "intent and slot classification virtual assistant task bot please"] 

    data:
      train_ds: ["data/financial_phrase_bank_train.jsonl", "data/assistent_train.jsonl"]
      validation_ds: ["data/financial_phrase_bank_val.jsonl", "data/assistent_val.jsonl"]
      add_eos: True
      shuffle: True
      num_workers: 1
      pin_memory: True

    optim: ...

(See https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/language_modeling/conf/megatron_gpt_prompt_learning_config.yaml for what should go in the ``trainer``, ``exp_manager``, and ``optim`` sections.)

Then run the command

.. code::
  
  python megatron_gpt_prompt_learning.py --config-name=multitask-prompt-learning.yaml
         

Example Multi-Task P-Tuning Config and Command After Prompt-Tuning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Update ``multitask-prompt-learning.yaml`` from the example above with p-tuning parameters for the new task. Be sure to update ``model.existing_tasks`` with the tasknames from previous prompt learning runs and to use the ``.nemo`` file saved at the end of your last prompt learning session. Values different from the config above have stars commented next to them. 

In this example, the SQuAD task includes the question context as part of the prompt. Because the context is long, we recommend setting ``answer_only_loss`` to ``True`` for this task, and any task where a significant portion of the prompt is not a part of the answer. ``answer_only_loss`` tells the model to only calculate the cross-entropy loss on the answer portion of the training example. Though we recommend placing all virtual tokens at the beginning of the prompt, we place them throughout the prompt in this example to demonstrate how to do so.

.. code::

  name: multitask_p_tuning # ***
  trainer: ...
  exp_manager: ...
  model:
    seed: 1234
    nemo_path: ${name}.nemo 
    virtual_prompt_style: "p-tuning" # ***
    encoder_seq_length: 2048 
    tensor_model_parallel_size: 1 
    pipeline_model_parallel_size: 1 
    global_batch_size: 16
    micro_batch_size: 4

    restore_path: multitask_prompt_tuning.nemo # ***
    language_model_path: models/megatron_125M_gpt.nemo
    existing_tasks: ["sentiment", "intent_and_slot"] # ***
    new_tasks: ["squad"] 

    task_templates: 
    - taskname: "sentiment" 
      prompt_template: "<|VIRTUAL_PROMPT_0|> {sentence} sentiment: {label}" 
      total_virtual_tokens: 100 
      virtual_token_splits: [100] 
      truncate_field: null
      answer_only_loss: False

    - taskname: "intent_and_slot"
      prompt_template: "<|VIRTUAL_PROMPT_0|> Predict intent and slot <|VIRTUAL_PROMPT_1|> :\n{utterance}{label}" 
      total_virtual_tokens: 100 
      virtual_token_splits: [80, 20]
      truncate_field: null
      answer_only_loss: True
      answer_field: "label"

    - taskname: "squad" # ***
      prompt_template: "<|VIRTUAL_PROMPT_0|> Answer the question from the context {question} {context} Answer: {answer}" # *** 
      total_virtual_tokens: 9 # ***
      virtual_token_splits: [9] # ***
      truncate_field: context # ***
      answer_only_loss: True # ***
      answer_field: "answer" # ***

    p_tuning: # ***
        dropout: 0.0 # ***
        num_layers: 2 # ***
        
    data:
      train_ds: ["data/squad_train.jsonl"] # ***
      validation_ds: ["data/squad_val.jsonl"] # ***
      add_eos: True
      shuffle: True
      num_workers: 1
      pin_memory: True

    optim: ...

Then run the command again:

.. code::
  
  python megatron_gpt_prompt_learning.py --config-name=multitask-prompt-learning.yaml


Example Multi-Task Inference 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The inference file can contain a mix of prompts from all the tasks the model has been prompt tuned on. 

.. code::

    python megatron_gpt_prompt_learning_eval.py \
            virtual_prompt_model_file=PATH_TO_NEMO_PROMPT_LEARNING_MODEL_FILE \
            gpt_model_file=PATH_TO_FROZEN_GPT_MODEL_FILE \
            inference.greedy=True \
            inference.add_BOS=False \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            tensor_model_parallel_size=1 \
            pipeline_model_parallel_size=1 \
            prompts=[prompt1,prompt2]
            
``virtual_prompt_model_file`` should be a path to a .nemo file saved after p-tuning/prompt tuning and ``model_file`` is still the path to the gpt model's .nemo file.   

prompts in this case should be a list of .json or .jsonl files containing json objects similar to the ones used during prompt learning. They should have keys that match the fields specified in the prompt template. Fields can be dropped from the prompt dict and their corresponding section of the prompt template will be automatically removed. 

For example, say the prompt template during p-tuning/prompt-tuning looked like:

.. code::

  '<|VIRTUAL_PROMPT_0|> Context: {context} Question: {question} Answer: {answer}'
  
but you don't want to include the answer field during inference. Just don't include the answer field in the prompt dict like below:

.. code::

  {"taskname": "squad", "context": "some paragraph", "question": "question related to paragraph"}
  {"taskname": "squad", "context": "another paragraph", "question": "a different question related to paragraph"}

        
And the dataset class will automatically format your input to have the form:

.. code::

  [
      '<|VIRTUAL_PROMPT_0|> Context: some paragraph Question: question related to paragraph Answer: ',
      '<|VIRTUAL_PROMPT_0|> Context: another paragraph Question: a different question related to paragraph Answer: '
  ]
        
Generally prompt learning inference is just like running inference with a GPT model. The only difference is you need to add ``virtual_prompt_model_file=PATH_TO_NEMO_PROMPT_LEARNING_MODEL_FILE`` to your command if you're using a p-tuned/prompt-tuned model. 

Example prompt learning script: `NeMo/examples/nlp/language_modeling/megatron_gpt_prompt_learning.py.py <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/language_modeling/megatron_gpt_prompt_learning.py>`__.

Example prompt tuned inference script: `NeMo/examples/nlp/language_modeling/megatron_gpt_eval.py <https://github.com/NVIDIA/NeMo/tree/stable/examples/nlp/language_modeling/megatron_gpt_eval.py>`__.
