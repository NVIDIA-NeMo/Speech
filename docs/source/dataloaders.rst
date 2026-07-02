.. _lhotse-dataloading:

==================
Lhotse Dataloading
==================

NeMo supports using `Lhotse`_, a speech data handling library, as a dataloading option. The key features of Lhotse used in NeMo are:

* Dynamic batch sizes
    Lhotse samples mini-batches to satisfy the constraint of total speech duration in a mini-batch (``batch_duration``),
    rather than a specific number of examples (i.e., batch size).
* Dynamic bucketing
    Instead of statically pre-bucketing the data, Lhotse allocates training examples to buckets dynamically.
    This allows more rapid experimentation with bucketing settings (number of buckets, specific placement of bucket duration bins)
    to minimize the amount of padding and accelerate training.
* Quadratic duration penalty
    Adding a quadratic penalty to an utterance's duration allows to sample mini-batches so that the
    GPU utilization is more consistent across big batches of short utterances and small batches of long utterances when using
    models with quadratic time/memory complexity (such as transformer).
* Dynamic weighted data source multiplexing
    An approach to combining diverse data sources (e.g. multiple domains, languages, tasks)
    where each data source is treated as a separate stream with its own sampling probability. The resulting data stream is a
    multiplexer that samples from each sub-stream. This approach ensures that the distribution of different sources is approximately
    constant in time (i.e., stationary); in fact, each mini-batch will have roughly the same ratio of data coming from each source.
    Since the multiplexing is done dynamically, it is very easy to tune the sampling weights.

.. caution:: As of now, Lhotse is mainly supported in most ASR model configurations. We aim to gradually extend this support to other speech tasks.

.. _Lhotse: https://github.com/lhotse-speech/lhotse
.. _Lhotse Cuts documentation: https://lhotse.readthedocs.io/en/latest/cuts.html
.. |tutorial_shar| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/lhotse-speech/lhotse/blob/master/examples/04-lhotse-shar.ipynb

Architecture overview
---------------------

The Lhotse dataloader is a pipeline of small components. Each YAML option you
set lands in exactly one of them, so it pays to know which is which::

    input_cfg entry  ──►  parser_fn  ──►  Adapter (IteratorNode)
                          (registered                 │
                           via @data_type_parser)     ▼
                                            CutSet (lazy iterator graph)
                                                      │
                              SamplingConstraint  ──► CutSampler
                                                      │
                                                      ▼
                                          IterableDatasetWrapper
                                                      │
                                                      ▼
                                            user-defined Dataset
                                                      │
                                                      ▼
                                                 DataLoader
                                                 (or StatefulDataLoader)

Components, top to bottom:

* **input_cfg entry** — one YAML dict identified by ``type:`` (e.g.
  ``type: nemo_tarred``). Listed below in :ref:`lhotse-format-reference`.
* **parser_fn** — registered with the ``@data_type_parser`` decorator in
  ``nemo/collections/common/data/lhotse/cutset.py``. Reads the entry and
  returns ``(CutSet, is_tarred)``. Users can add their own (see
  :ref:`lhotse-extension-hooks`).
* **Adapter** — a class that knows how to iterate one specific on-disk
  format (e.g. ``LazyNeMoTarredIterator``, ``LazyParquetIterator``,
  ``NeMoMultimodalConversationJsonlAdapter``). All recent adapters are
  Lhotse :class:`~lhotse.lazy.IteratorNode` subclasses and support
  ``indexed=True`` for O(1) random access — see
  :ref:`indexed-resumable-dataloading`.
* **CutSet** — Lhotse's lazy manifest wrapper. Composing multiple sources
  produces a graph of iterator nodes (mux, mix, map, filter, …) underneath.
* **SamplingConstraint** — defines what "length" means for batch packing:
  :class:`~lhotse.dataset.sampling.base.TimeConstraint` (audio duration,
  default), :class:`~lhotse.dataset.sampling.base.TokenConstraint` (token
  count, multimodal), ``MultimodalSamplingConstraint`` /
  ``FixedBucketBatchSizeConstraint2D`` (NeMo extensions; see
  :ref:`lhotse-sampling-constraints`).
* **CutSampler** — :class:`~lhotse.dataset.sampling.DynamicCutSampler` or
  :class:`~lhotse.dataset.sampling.DynamicBucketingSampler`, picked
  automatically based on ``use_bucketing``.
* **IterableDatasetWrapper** — Lhotse helper that turns the sampler-produced
  ``CutSet`` mini-batches into a stream the PyTorch ``DataLoader`` can
  consume.
* **Dataset class** — supplied by the model code; converts a ``CutSet``
  mini-batch into a ``dict[str, Tensor]``. The same dataset class can serve
  multiple model architectures because all batching is upstream.

.. _lhotse-format-reference:

Supported input formats
-----------------------

Every entry in ``input_cfg`` is identified by ``type:``. The table below is
the canonical list of every type the dataloader understands today, what it
returns, and the on-disk shape it expects.

.. list-table::
   :header-rows: 1
   :widths: 18 32 14 8 8 10 10

   * - ``type:``
     - Purpose
     - Yields
     - Audio
     - Tarred
     - Indexable
     - Adapter / parser
   * - ``nemo``
     - NeMo non-tarred JSON manifest (per-file audio)
     - ``Cut``
     - yes
     - no
     - yes
     - ``LazyNeMoIterator``
   * - ``nemo_tarred``
     - NeMo tarred manifest + audio tar shards
     - ``Cut``
     - yes
     - yes
     - yes
     - ``LazyNeMoTarredIterator``
   * - ``lhotse``
     - Plain Lhotse cuts JSONL
     - ``Cut``
     - yes
     - no
     - yes
     - lhotse ``LazyJsonlIterator`` / ``LazyIndexedManifestIterator``
   * - ``lhotse_shar``
     - Lhotse Shar (sharded archive directory)
     - ``Cut``
     - yes
     - yes
     - yes
     - lhotse ``LazySharIterator``
   * - ``parquet``
     - Parquet file with audio bytes column
     - ``Cut``
     - yes
     - no
     - yes (row groups)
     - ``LazyParquetIterator``
   * - ``txt``
     - One example per line, raw text
     - ``TextExample``
     - no
     - n/a
     - no
     - ``LhotseTextAdapter``
   * - ``txt_jsonl``
     - One JSON object per line; configurable text field
     - ``TextExample``
     - no
     - n/a
     - yes
     - ``LhotseTextJsonlAdapter``
   * - ``txt_pair``
     - Source + target text files for translation
     - ``SourceTargetTextExample``
     - no
     - n/a
     - no
     - ``LhotseTextPairAdapter``
   * - ``multimodal_conversation``
     - Multi-turn chat with mixed text/audio turns (JSONL)
     - ``NeMoMultimodalConversation``
     - optional
     - optional
     - yes
     - ``NeMoMultimodalConversationJsonlAdapter``
   * - ``share_gpt``
     - ShareGPT-format JSONL → conversation
     - ``NeMoMultimodalConversation``
     - optional
     - optional
     - yes
     - ``NeMoMultimodalConversationShareGPTJsonlAdapter``
   * - ``share_gpt_webdataset``
     - ShareGPT in WebDataset tar shards
     - ``NeMoMultimodalConversation``
     - optional
     - yes
     - yes
     - ``NeMoMultimodalConversationShareGPTWebdatasetAdapter``
   * - ``lhotse_as_conversation``
     - Read ASR data and emit it as ASR conversation
     - ``NeMoMultimodalConversation``
     - yes
     - inherits
     - inherits
     - transform on ``read_cutset_from_config``
   * - ``sqa_as_conversation``
     - Spoken-QA → 3-turn conversation (question / audio / answer)
     - ``NeMoMultimodalConversation``
     - yes
     - inherits
     - inherits
     - transform
   * - ``s2s_as_conversation``
     - Duplex S2S → conversation
     - ``NeMoMultimodalConversation``
     - yes
     - inherits
     - inherits
     - transform
   * - ``s2s_duplex_overlap_as_s2s_duplex``
     - Overlapping agent/user segments → unified S2S timeline
     - ``Cut``
     - yes
     - inherits
     - inherits
     - transform
   * - ``s2s_duplex_reverse_role``
     - Swap user and agent in a duplex cut
     - ``Cut``
     - yes
     - inherits
     - inherits
     - transform
   * - ``lhotse_magpietts_data_as_continuation``
     - MagpieTTS dataset → S2S duplex continuation
     - ``Cut``
     - yes
     - inherits
     - inherits
     - transform
   * - ``nemo_tarred_to_duplex``
     - Single-supervision NeMo → duplex (user speech + agent silence)
     - ``Cut``
     - yes
     - yes
     - inherits
     - transform
   * - ``multi_speaker_simulator``
     - Synthetic multi-speaker mixtures from a manifest
     - ``Cut``
     - yes
     - n/a
     - no
     - ``MultiSpeakerMixtureGenerator``
   * - ``group``
     - Wrap a list of entries with a shared ``weight`` and ``tags``
     - (nested)
     - n/a
     - n/a
     - n/a
     - n/a

Notes:

* "Inherits" means the type is a transform that wraps another underlying
  source via ``read_cutset_from_config(config)``. Such entries accept the
  underlying source's keys (e.g. ``cuts_path`` and ``manifest_filepath``)
  *in addition to* their own.
* Tarred NeMo manifests support a ``_skipme`` key to omit specific manifest
  rows without repacking tars (set to ``True``, ``1``, or a reason string).
* Lhotse Shar is documented in the upstream tutorial: |tutorial_shar|.

Conversation / multimodal types — when to use which
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Six types yield ``NeMoMultimodalConversation`` from very different sources.
Pick by the shape of your input data:

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Your data
     - ``type:``
     - Notes
   * - JSONL of multi-turn chats with mixed text/audio turns
     - ``multimodal_conversation``
     - Native chat schema; audio turns reference paths or tar members
   * - JSONL in ShareGPT chat schema
     - ``share_gpt``
     - Adds ShareGPT-specific role/value parsing
   * - ShareGPT data packed in WebDataset tar shards
     - ``share_gpt_webdataset``
     - Same parsing as ``share_gpt``, reads tarred shards
   * - ASR data in NeMo or Lhotse format
     - ``lhotse_as_conversation``
     - Builds a 2-turn (instruction+audio / transcript) conversation per cut
   * - Spoken-QA data with ``question`` / ``answer`` fields
     - ``sqa_as_conversation``
     - Builds a 3-turn (question / audio / answer) conversation per cut
   * - Duplex S2S data with user/agent supervisions
     - ``s2s_as_conversation``
     - Maps duplex roles onto chat turns

The last three (``*_as_conversation``) are *transforms*: they delegate to
``read_cutset_from_config(config)`` for the underlying audio source, so the
nested keys like ``manifest_filepath``, ``cuts_path``, or ``shar_path``
belong on the same entry.

Enabling Lhotse via configuration
----------------------------------

.. note:: Using Lhotse with tarred datasets will make the dataloader infinite, ditching the notion of an "epoch". "Epoch" may still be logged in W&B/TensorBoard, but it will correspond to the number of executed training loops between validation loops.

Start with an existing NeMo experiment YAML configuration. Typically, you'll only need to add a few options to enable Lhotse.
These options are::

    # NeMo generic dataloading arguments
    model.train_ds.manifest_filepath=...
    model.train_ds.tarred_audio_filepaths=...   # for tarred datasets only
    model.train_ds.num_workers=4
    model.train_ds.min_duration=0.3             # optional
    model.train_ds.max_duration=30.0            # optional
    model.train_ds.shuffle=true                 # optional

    # Lhotse dataloading related arguments
    ++model.train_ds.use_lhotse=True
    ++model.train_ds.batch_duration=1100
    ++model.train_ds.quadratic_duration=30
    ++model.train_ds.num_buckets=30
    ++model.train_ds.num_cuts_for_bins_estimate=10000
    ++model.train_ds.bucket_buffer_size=10000
    ++model.train_ds.shuffle_buffer_size=10000

    # PyTorch Lightning related arguments
    ++trainer.use_distributed_sampler=false
    ++trainer.limit_train_batches=1000
    trainer.val_check_interval=1000
    trainer.max_steps=300000

.. note:: The default values above are a reasonable starting point for a hybrid RNN-T + CTC ASR model on a 32GB GPU with a data distribution dominated by 15s long utterances.

Let's briefly go over each of the Lhotse dataloading arguments:

* ``use_lhotse`` enables Lhotse dataloading
* ``batch_duration`` is the total max duration of utterances in a mini-batch and controls the batch size; the more shorter utterances, the bigger the batch size, and vice versa.
* ``quadratic_duration`` adds a quadratically growing penalty for long utterances; useful in bucketing and transformer type of models. The value set here means utterances this long will count as if with a doubled duration.
* ``num_buckets`` is the number of buckets in the bucketing sampler. Bigger value means less padding but also less randomization.
* ``num_cuts_for_bins_estimate`` is the number of utterance we will sample before the start of the training to estimate the duration bins for buckets. Larger number results in a more accurate estimatation but also a bigger lag before starting the training.
* ``bucket_buffer_size`` is the number of utterances (data and metadata) we will hold in memory to be distributed between buckets. With bigger ``batch_duration``, this number may need to be increased for dynamic bucketing sampler to work properly (typically it will emit a warning if this is too low).
* ``shuffle_buffer_size`` is an extra number of utterances we will hold in memory to perform approximate shuffling (via reservoir-like sampling). Bigger number means more memory usage but also better randomness.

The PyTorch Lightning ``trainer`` related arguments:

* ``use_distributed_sampler=false`` is required because Lhotse has its own handling of distributed sampling.
* ``val_check_interval``/``limit_train_batches``
    These are required for dataloaders with tarred/Shar datasets
    because Lhotse makes the dataloader infinite, so we'd never go past epoch 0. This approach guarantees
    we will never hang the training because the dataloader in some node has less mini-batches than the others
    in some epochs. The value provided here will be the effective length of each "pseudo-epoch" after which we'll
    trigger the validation loop.
* ``max_steps`` is the total number of steps we expect to be training for. It is required for the same reason as ``limit_train_batches``; since we'd never go past epoch 0, the training would have never finished.

Some other Lhotse related arguments we support:

* ``cuts_path`` can be provided to read data from a Lhotse CutSet manifest instead of a NeMo manifest.
    Specifying this option will result in ``manifest_filepaths`` and ``tarred_audio_filepaths`` being ignored.
* ``shar_path``
    Can be provided to read data from a Lhotse Shar manifest instead of a NeMo manifest.
    Specifying this option will result in ``manifest_filepaths`` and ``tarred_audio_filepaths`` being ignored.
    This argument can be a string (single Shar directory), a list of strings (Shar directories),
    or a list of 2-item lists, where the first item is a Shar directory path, and the other is a sampling weight.
    The user can also provide a dict mapping Lhotse Shar fields to a list of shard paths with data for that field.
    For details about Lhotse Shar format, see: |tutorial_shar|
* ``bucket_duration_bins``
    Duration bins are a list of float values (seconds) that when provided, will skip the initial bucket bin estimation
    and save some time. It has to have a length of ``num_buckets - 1``. An optimal value can be obtained by running CLI:
    ``lhotse cut estimate-bucket-bins -b $num_buckets my-cuts.jsonl.gz``
* ``use_bucketing`` is a boolean which indicates if we want to enable/disable dynamic bucketing. By defalt it's enabled.
* ``text_field`` is the name of the key in the JSON (NeMo) manifest from which we should be reading text (default="text").
* ``lang_field`` is the name of the key in the JSON (NeMo) manifest from which we should be reading language tag (default="lang"). This is useful when working e.g. with ``AggregateTokenizer``.
* ``batch_size``
    Limits the number of examples in a mini-batch to this number, when combined with ``batch_duration``.
    When ``batch_duration`` is not set, it acts as a static batch size.
* ``seed`` sets a random seed for the shuffle buffer.

* ``indexed`` (default ``False``) opts the dataloader into Lhotse's indexed-manifest
  path, giving every adapter O(1) random access and graph-token-based exact restore.
  Requires ``.idx`` sidecars next to every JSONL/tar file. See
  :ref:`indexed-resumable-dataloading` below.

* ``use_stateful_dataloader`` (default ``False``) swaps PyTorch's
  ``DataLoader`` for ``torchdata.stateful_dataloader.StatefulDataLoader`` so
  that per-worker iterator state is captured in checkpoints and restored
  exactly on resume. Pair with ``indexed: true`` for full O(1) restore.

The full and always up-to-date list of supported options can be found in ``LhotseDataLoadingConfig`` class.

.. _asr-dataset-config-format:

Extended multi-dataset configuration format
--------------------------------------------

Combining a large number of datasets and defining weights for them can be tricky.
We offer an extended configuration format that allows you to explicitly define datasets,
dataset groups, and their weights either inline in the experiment configuration,
or as a path to a separate YAML file.

In addition to the features above, this format introduces a special ``tags`` dict-like field.
The keys and values in ``tags`` are automatically attached to every sampled example, which
is very useful when combining multiple datasets with different properties.
The dataset class which converts these examples to tensors can partition the mini-batch and apply
different processing to each group.
For example, you may want to construct different prompts for the model using metadata in ``tags``.

How ``tags`` is applied
^^^^^^^^^^^^^^^^^^^^^^^

Every key/value pair in ``tags`` becomes an attribute on every cut produced
by that entry. The dataloader walks the cuts via ``cuts.map(...)`` and runs::

    for key, val in tags.items():
        setattr(cut, key, val)

So in your dataset class you read them back as ordinary attributes::

    def __getitem__(self, cuts):
        for cut in cuts:
            lang   = cut.lang
            task   = cut.task
            ctx    = cut.context
            ...

Tags set on a ``group`` apply to every nested entry; tags set on an inner
entry override the outer ones for that source. Conflicts with built-in cut
fields (``id``, ``duration``, ``supervisions``, …) silently overwrite the
built-in — pick tag names that don't collide.

.. note:: When fine-tuning a model that was trained with ``input_cfg`` option, typically you'd only need
    to override the following options: ``input_cfg=null`` and ``manifest_filepath=path/to/manifest.json``.

Example 1. Combine two datasets with equal weights and attach custom metadata in ``tags`` to each cut:

.. code-block:: yaml

    input_cfg:
      - type: nemo_tarred
        manifest_filepath: /path/to/manifest__OP_0..512_CL_.json
        tarred_audio_filepath: /path/to/tarred_audio/audio__OP_0..512_CL_.tar
        weight: 0.4
        tags:
          lang: en
          pnc: no
      - type: nemo_tarred
        manifest_filepath: /path/to/other/manifest__OP_0..512_CL_.json
        tarred_audio_filepath: /path/to/other/tarred_audio/audio__OP_0..512_CL_.tar
        weight: 0.6
        tags:
          lang: pl
          pnc: yes

Example 2. Combine multiple (4) datasets, corresponding to different tasks (ASR, AST).
Each task gets its own group and its own weight.
Then within each task, each dataset get its own within-group weight as well.
The final weight is the product of outer and inner weight:

.. code-block:: yaml

    input_cfg:
      - type: group
        weight: 0.7
        tags:
          task: asr
        input_cfg:
          - type: nemo_tarred
            manifest_filepath: /path/to/asr1/manifest__OP_0..512_CL_.json
            tarred_audio_filepath: /path/to/tarred_audio/asr1/audio__OP_0..512_CL_.tar
            weight: 0.6
            tags:
              source_lang: en
              target_lang: en
          - type: nemo_tarred
            manifest_filepath: /path/to/asr2/manifest__OP_0..512_CL_.json
            tarred_audio_filepath: /path/to/asr2/tarred_audio/audio__OP_0..512_CL_.tar
            weight: 0.4
            tags:
              source_lang: pl
              target_lang: pl
      - type: group
        weight: 0.3
        tags:
          task: ast
        input_cfg:
          - type: nemo_tarred
            manifest_filepath: /path/to/ast1/manifest__OP_0..512_CL_.json
            tarred_audio_filepath: /path/to/ast1/tarred_audio/audio__OP_0..512_CL_.tar
            weight: 0.2
            tags:
              source_lang: en
              target_lang: pl
          - type: nemo_tarred
            manifest_filepath: /path/to/ast2/manifest__OP_0..512_CL_.json
            tarred_audio_filepath: /path/to/ast2/tarred_audio/audio__OP_0..512_CL_.tar
            weight: 0.8
            tags:
              source_lang: pl
              target_lang: en

Configuring multimodal dataloading
-----------------------------------

Our configuration format supports specifying data sources from other modalities than just audio.
At this time, this support is extended to audio and text modalities. We provide the following parser types:

**Raw text files.** Simple text files where each line is an individual text example. This can represent standard language modeling data.
This parser is registered under ``type: txt``.

Data format examples::

    # file: document_0.txt
    This is a language modeling example.
    Wall Street is expecting major news tomorrow.

    # file: document_1.txt
    Invisible bats have stormed the city.
    What an incredible event!

Dataloading configuration example::

    input_cfg:
      - type: txt
        paths: /path/to/document_{0..1}.txt
        language: en  # optional

Python object example::

    from nemo.collections.common.data.lhotse.text_adapters import TextExample

    example = TextExample(
        text="This is a language modeling example.",
        language="en",  # optional
    )

Python dataloader instantiation example::

    from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config

    dl = get_lhotse_dataloader_from_config({
            "input_cfg": [
                {"type": "txt", "paths": "/path/to/document_{0..1}.txt", "language": "en"},
            ],
            "use_multimodal_dataloading": True,
            "batch_size": 4,
        },
        global_rank=0,
        world_size=1,
        dataset=MyDatasetClass(),  # converts CutSet -> dict[str, Tensor]
        tokenizer=my_tokenizer,
    )

**Raw text file pairs.** Pairs of raw text files with corresponding lines. This can represent machine translation data.
This parser is registered under ``type: txt_pair``.

Data format examples::

    # file: document_en_0.txt
    This is a machine translation example.
    Wall Street is expecting major news tomorrow.

    # file: document_pl_0.txt
    To jest przykład tłumaczenia maszynowego.
    Wall Street spodziewa się jutro ważnych wiadomości.

Dataloading configuration example::

    input_cfg:
      - type: txt_pair
        source_path: /path/to/document_en_{0..N}.txt
        target_path: /path/to/document_pl_{0..N}.txt
        source_language: en  # optional
        target_language: pl  # optional

Python object example::

    from nemo.collections.common.data.lhotse.text_adapters import SourceTargetTextExample

    example = SourceTargetTextExample(
        source=TextExample(
            text="This is a language modeling example.",
            language="en",  # optional
        ),
        target=TextExample(
            text="To jest przykład tłumaczenia maszynowego.",
            language="pl",  # optional
        ),
    )

Python dataloader instantiation example::

    from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config

    dl = get_lhotse_dataloader_from_config({
            "input_cfg": [
                {
                    "type": "txt_pair",
                    "source_path": "/path/to/document_en_{0..N}.txt",
                    "target_path": "/path/to/document_pl_{0..N}.txt",
                    "source_language": "en"
                    "target_language": "en"
                },
            ],
            "use_multimodal_dataloading": True,
            "prompt_format": "t5nmt",
            "batch_size": 4,
        },
        global_rank=0,
        world_size=1,
        dataset=MyDatasetClass(),  # converts CutSet -> dict[str, Tensor]
        tokenizer=my_tokenizer,
    )

**NeMo multimodal conversations.** A JSON-Lines (JSONL) file that defines multi-turn conversations with mixed text and audio turns.
This parser is registered under ``type: multimodal_conversation``.

Data format examples::

    # file: chat_0.jsonl
    {"id": "conv-0", "conversations": [{"from": "user", "value": "speak to me", "type": "text"}, {"from": "assistant": "value": "/path/to/audio.wav", "duration": 17.1, "type": "audio"}]}
    {"id": "conv-1", "conversations": [{"from": "user", "value": "speak to me", "type": "text"}, {"from": "assistant": "value": "/path/to/audio.wav", "duration": 5, "offset": 17.1, "type": "audio"}]}

Dataloading configuration example::

    token_equivalent_duration: 0.08
    input_cfg:
      - type: multimodal_conversation
        manifest_filepath: /path/to/chat_{0..N}.jsonl
        audio_locator_tag: [audio]

Python object example::

    from lhotse import Recording
    from nemo.collections.common.data.lhotse.text_adapters import MultimodalConversation, TextTurn, AudioTurn

    conversation = NeMoMultimodalConversation(
        id="conv-0",
        turns=[
            TextTurn(value="speak to me", role="user"),
            AudioTurn(cut=Recording.from_file("/path/to/audio.wav").to_cut(), role="assistant", audio_locator_tag="[audio]"),
        ],
        token_equivalent_duration=0.08,  # this value will be auto-inserted by the dataloader
    )

Python dataloader instantiation example::

    from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config

    dl = get_lhotse_dataloader_from_config({
            "input_cfg": [
                {
                    "type": "multimodal_conversation",
                    "manifest_filepath": "/path/to/chat_{0..N}.jsonl",
                    "audio_locator_tag": "[audio]",
                },
            ],
            "use_multimodal_dataloading": True,
            "token_equivalent_duration": 0.08,
            "prompt_format": "llama2",
            "batch_size": 4,
        },
        global_rank=0,
        world_size=1,
        dataset=MyDatasetClass(),  # converts CutSet -> dict[str, Tensor]
        tokenizer=my_tokenizer,
    )

**Indexed mode for text/multimodal sources.** All of the parsers above
(``txt_jsonl``, ``nemo_sft_jsonl``, ``multimodal_conversation``, ``share_gpt``,
``share_gpt_webdataset``) accept ``indexed: true`` and integrate with
``StatefulDataLoader``-based exact resume. ``txt`` and ``txt_pair`` are
intentionally streaming-only. See :ref:`indexed-resumable-dataloading`.

**Dataloading and bucketing of text and multimodal data.** When dataloading text or multimodal data, pay attention to the following config options (we provide example values for convenience):

* ``use_multimodal_sampling: true`` tells Lhotse to switch from measuring audio duration to measuring token counts; required for text.

* ``prompt_format: "prompt-name"`` will apply a specified PromptFormatter during data sampling to accurately reflect its token counts.

* ``measure_total_length: true`` customizes length measurement for decoder-only and encoder-decoder models. Decoder-only models consume a linear sequence of context + answer, so we should measure the total length (``true``). On the other hand, encoder-decoder models deal with two different sequence lengths: input (context) sequence length for the encoder, and output (answer) sequence length for the decoder. For such models set this to ``false``.

* ``min_tokens: 1``/``max_tokens: 4096`` filters examples based on their token count (after applying the prompt format).

* ``min_tpt: 0.1``/``max_tpt: 10`` filter examples based on their output-token-per-input-token-ratio. For example, a ``max_tpt: 10`` means we'll filter every example that has more than 10 output tokens per 1 input token. Very useful for removing sequence length outliers that lead to OOM. Use ``estimate_token_bins.py`` to view token count distributions for calbirating this value.

* (multimodal-only) ``token_equivalent_duration: 0.08`` is used to be able to measure audio examples in the number of "tokens". For example, if we're using fbank with 0.01s frame shift and an acoustic model that has a subsampling factor of 0.08, then a reasonable setting for this could be 0.08 (which means every subsampled frame counts as one token). Calibrate this value to fit your needs.

**Text/multimodal bucketing and OOMptimizer.** Analogous to bucketing for audio data, we provide two scripts to support efficient bucketing:

* ``scripts/speech_llm/estimate_token_bins.py`` which estimates 1D or 2D buckets based on the input config, tokenizer, and prompt format. It also estimates input/output token count distribution and suggested ``max_tpt`` (token-per-token) filtering values.

* (experimental) ``scripts/speech_llm/oomptimizer.py`` which works with SALM/BESTOW GPT/T5 models and estimates the optimal ``bucket_batch_size`` for a given model config and bucket bins value. Given the complexity of Speech LLM some configurations may not be supported yet at the time of writing (e.g., model parallelism).

To enable bucketing, set ``batch_size: null`` and use the following options:

* ``use_bucketing: true``

* ``bucket_duration_bins`` - the output of ``estimate_token_bins.py``. If ``null``, it will be estimated at the start of training at the cost of some run time (not recommended).

* (oomptimizer-only) ``bucket_batch_size`` - the output of OOMptimizer.

* (non-oomptimizer-only) ``batch_tokens`` is the maximum number of tokens we want to find inside a mini-batch. Similarly to ``batch_duration``, this number does consider padding tokens too, therefore enabling bucketing is recommended to maximize the ratio of real vs padding tokens. Note that it's just a heuristic for determining the optimal batch sizes for different buckets, and may be less efficient than using OOMptimizer.

* (non-oomptimizer-only) ``quadratic_factor`` is a quadratic penalty to equalize the GPU memory usage between buckets of short and long sequence lengths for models with quadratic memory usage. It is only a heuristic and may not be as efficient as using OOMptimizer.

**Joint dataloading of text/audio/multimodal data.** The key strength of this approach is that we can easily combine audio datasets and text datasets,
and benefit from every other technique we described in this doc, such as: dynamic data mixing, data weighting, dynamic bucketing, and so on.

Single-config vs. ``multi_config: true``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default the dataloader builds **one** ``CutSet`` and **one** sampler from
the top-level config. Setting ``multi_config: true`` switches to a
**multi-modality** layout where each named sub-block (typically ``audio:``
and ``text:``) is parsed as its own dataloader config, with its own
sampling/bucketing options, and the per-modality samplers are fused at the
batch level.

When ``multi_config: true`` is set:

* Top-level keys (``num_workers``, ``shuffle``, ``seed``, ``sample_rate``,
  …) apply globally and are inherited by every sub-block.
* Per-modality overrides — including the ``input_cfg`` itself — go inside
  the named sub-block (``audio: ...`` / ``text: ...``).
* The per-modality samplers are combined into one stream by
  ``sampler_fusion``.

This approach is described in the `EMMeTT`_ paper. There's also a notebook tutorial called Multimodal Lhotse Dataloading. We construct a separate sampler (with its own batching settings) for each modality,
and specify how the samplers should be fused together via the option ``sampler_fusion``:

* ``sampler_fusion: "round_robin"`` will iterate single sampler per step, taking turns. For example: step 0 - audio batch, step 1 - text batch, step 2 - audio batch, etc.

* ``sampler_fusion: "randomized_round_robin"`` is similar, but at each chooses a sampler randomly using ``sampler_weights: [w0, w1]`` (weights can be unnormalized).

* ``sampler_fusion: "zip"`` will draw a mini-batch from each sampler at every step, and merge them into a single ``CutSet``. This approach combines well with multimodal gradient accumulation (run forward+backward for one modality, then the other, then the update step).

.. _EMMeTT: https://arxiv.org/abs/2409.13523

Example. Combine an ASR (audio-text) dataset with an MT (text-only) dataset so that mini-batches have some examples from both datasets:

.. code-block:: yaml

    model:
      ...
      train_ds:
        multi_config: True,
        sampler_fusion: zip
        shuffle: true
        num_workers: 4

        audio:
          prompt_format: t5nmt
          use_bucketing: true
          min_duration: 0.5
          max_duration: 30.0
          max_tps: 12.0
          bucket_duration_bins: [[3.16, 10], [3.16, 22], [5.18, 15], ...]
          bucket_batch_size: [1024, 768, 832, ...]
          input_cfg:
            - type: nemo_tarred
              manifest_filepath: /path/to/manifest__OP_0..512_CL_.json
              tarred_audio_filepath: /path/to/tarred_audio/audio__OP_0..512_CL_.tar
              weight: 0.5
              tags:
                context: "Translate the following to English"

        text:
          prompt_format: t5nmt
          use_multimodal_sampling: true
          min_tokens: 1
          max_tokens: 256
          min_tpt: 0.333
          max_tpt: 3.0
          measure_total_length: false
          use_bucketing: true
          bucket_duration_bins: [[10, 4], [10, 26], [15, 10], ...]
          bucket_batch_size: [512, 128, 192, ...]
          input_cfg:
            - type: txt_pair
              source_path: /path/to/en__OP_0..512_CL_.txt
              target_path: /path/to/pl__OP_0..512_CL_.txt
              source_language: en
              target_language: pl
              weight: 0.5
              tags:
                question: "Translate the following to Polish"

.. caution:: We strongly recommend to use multiple shards for text files as well so that different nodes and dataloading workers are able to randomize the order of text iteration. Otherwise, multi-GPU training has a high risk of duplication of text examples.

.. _lhotse-sampling-constraints:

Sampling constraints
--------------------

A :class:`~lhotse.dataset.sampling.base.SamplingConstraint` decides what
"length" means when the sampler packs a mini-batch. NeMo uses four:

* :class:`~lhotse.dataset.sampling.base.TimeConstraint` — default.
  Length = audio duration in seconds. Enforces ``max_duration`` /
  ``batch_duration`` / ``quadratic_duration``.
* :class:`~lhotse.dataset.sampling.base.TokenConstraint` — activated by
  ``use_multimodal_sampling: true`` for text-only flows. Length = token
  count after applying the tokenizer (and optionally the prompt format).
  Enforces ``max_tokens`` / ``batch_tokens`` / ``quadratic_factor``.
* ``MultimodalSamplingConstraint`` — Lhotse-style mixed-modality
  packing. Activated by setting both ``use_multimodal_sampling: true``
  and a ``token_equivalent_duration`` so audio cuts are measured in
  equivalent-token units alongside text. Enforces all of the above plus
  ``min_tpt``/``max_tpt`` (token-per-token ratio filtering).
* ``FixedBucketBatchSizeConstraint2D`` — activated automatically when
  ``bucket_duration_bins`` is given as a list of ``[duration, tokens]``
  pairs **and** ``bucket_batch_size`` is set. Each bucket gets its own
  fixed batch size; this is the layout produced by
  ``estimate_duration_bins_2d.py`` and the OOMptimizer.

You usually don't pick a constraint by name — it's inferred from the
combination of YAML options. The names matter when you read NeMo's source,
extend the system with a custom constraint, or interpret error messages.

.. _indexed-resumable-dataloading:

Resumable / indexed dataloading
-------------------------------

Setting ``indexed: true`` (per-source or top-level) plus
``use_stateful_dataloader: true`` (top-level) opts NeMo's Lhotse dataloader
into Lhotse's indexed iterator graph and torchdata's
``StatefulDataLoader``. The combination gives you:

* O(1) checkpoint/restore of the *whole* dataloading pipeline — sampler RNG,
  bucketer state, multiplexer choice RNG, per-source iterator cursors, and
  per-worker prefetch queues — without any replay from the start of the epoch.
* Random access (``__getitem__``) over every supported adapter.

When set at the top level, ``indexed: true`` is propagated by
``read_dataset_config`` through the ``propagate_attrs`` cascade, so a single
top-level flag covers every nested ``input_cfg`` group. You can still override
it per-source if needed.

Per-adapter support
^^^^^^^^^^^^^^^^^^^

The following ``input_cfg`` types accept ``indexed: true`` today and require an
``.idx`` sidecar next to each data file:

* ``nemo`` / ``nemo_tarred`` — JSONL manifest gets ``manifest.json.idx``;
  every audio tar in ``tarred_audio_filepaths`` gets ``shard.tar.idx``.
* ``lhotse`` (plain) — ``cuts.jsonl`` gets ``cuts.jsonl.idx``.
* ``lhotse_shar`` — every uncompressed ``cuts.<NNNNNN>.jsonl`` and field tar
  inside the Shar dir.
* ``parquet`` — no sidecar required, but the file must expose row-group
  statistics (the default for files written by pyarrow / pandas).
* ``txt_jsonl`` — every file in ``paths``.
* ``multimodal_conversation`` and ``share_gpt`` — JSONL manifest plus optional
  audio tars in ``tarred_audio_filepaths``.
* ``share_gpt_webdataset`` — every ``shard-*.tar`` inside ``data_dir``.

``txt`` and ``txt_pair`` remain streaming-only (no random-access support).

Two caveats to be aware of:

* ``indexed: true`` is incompatible with ``extra_fields`` and ``slice_length``
  on ``nemo``/``nemo_tarred``: those features mutate or expand cuts in a way
  that has no stable index. Pre-process the manifest offline if you need them
  in an indexed pipeline.
* Only **uncompressed** files can be indexed (no ``.jsonl.gz``,
  ``.tar.gz``, etc.) and only files on a backend that supports indexed reads
  (local FS, S3-compatible object stores, AIStore).

Building ``.idx`` sidecars
^^^^^^^^^^^^^^^^^^^^^^^^^^

Two equivalent ways:

1. Lhotse's CLI per file::

       lhotse index jsonl path/to/cuts.jsonl
       lhotse index tar  path/to/shard.tar
       lhotse index shar path/to/shar_dir/

2. NeMo's batch helper that takes a config and indexes everything it
   references in one shot::

       python scripts/dataloading/build_indexes.py path/to/input_cfg.yaml

   The script walks ``input_cfg`` (including nested ``group`` entries and
   per-entry YAML references), dispatches the right tar layout for each
   adapter (NeMo one-member-per-sample vs. WebDataset/Shar pair format), and
   skips files that already have an up-to-date ``.idx``. Use ``--force`` to
   rebuild, ``--workers N`` for parallelism, ``--dry-run`` to preview.

   Pass ``--indexes-root /path/to/mirror`` to write the sidecars to a
   separate directory tree that mirrors the data files' layout instead of
   placing them next to the data — see :ref:`lhotse-indexes-root` below.

.. _lhotse-indexes-root:

Storing ``.idx`` sidecars in a separate directory (``indexes_root``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, every ``.idx`` lives next to its data file
(``cuts.jsonl`` ↔ ``cuts.jsonl.idx``). If your data sits on shared, slow,
or read-only storage (NFS, S3, AIStore), you may want to keep the indexes
on a fast local disk instead. Set ``indexes_root`` at the top of the
dataloader config:

.. code-block:: yaml

    data:
      train_ds:
        indexed: true
        use_stateful_dataloader: true
        indexes_root: /scratch/idx     # mirror lives here
        input_cfg:
          - type: nemo_tarred
            manifest_filepath: /shared/data/asr/manifest__OP_0..127_CL_.jsonl
            tarred_audio_filepaths: ais://bucket/asr/audio__OP_0..127_CL_.tar

Index lookups for each data file ``D`` resolve to
``<indexes_root>/<D-with-scheme-stripped>.idx``. Examples::

    /shared/data/asr/manifest_0.jsonl    -> /scratch/idx/shared/data/asr/manifest_0.jsonl.idx
    ais://bucket/asr/audio_0.tar        -> /scratch/idx/bucket/asr/audio_0.tar.idx

The setting cascades through ``read_dataset_config`` to every nested
``input_cfg`` entry, so a single top-level value covers the whole pipeline.
You can override it per-source on any entry that needs a different mirror.

Two ways to populate the mirror:

1. **Build the indexes there to begin with**::

       python scripts/dataloading/build_indexes.py \
           --indexes-root /scratch/idx path/to/input_cfg.yaml

   The script reads each data file in place, computes the offsets, and
   writes the ``.idx`` directly to the mirrored target.

2. **Prefetch existing remote indexes** when sidecars already live next to
   the data on shared/object storage and you just want a local copy::

       python scripts/dataloading/prefetch_indexes.py \
           --indexes-root /scratch/idx path/to/input_cfg.yaml

   ``prefetch_indexes.py`` walks the same ``input_cfg``, locates every
   sidecar at its natural location (via lhotse's ``open_best``, so
   ``ais://`` / ``s3://`` / ``http://`` are all supported as sources),
   and copies it into the local mirror. Use ``--source-indexes-root``
   when the source sidecars themselves live under another mirror.

Both scripts accept ``--force``, ``--workers N``, and ``--dry-run``.

End-to-end YAML example
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    model:
      train_ds:
        # Top-level switches enable indexed restore for every source below.
        indexed: true
        use_stateful_dataloader: true
        force_finite: true
        force_map_dataset: true

        sample_rate: 16000
        num_workers: 4
        seed: 42
        shard_seed: randomized

        # Bucketing and the rest of the dataloader knobs work exactly as before.
        use_bucketing: true
        num_buckets: 30
        batch_duration: 1100
        quadratic_duration: 30

        input_cfg:
          - type: nemo_tarred
            manifest_filepath: /data/asr/manifest__OP_0..127_CL_.jsonl
            tarred_audio_filepaths: /data/asr/audio__OP_0..127_CL_.tar
            weight: 0.7
          - type: lhotse
            cuts_path: /data/extra/cuts.jsonl
            weight: 0.3

Resume contract
^^^^^^^^^^^^^^^

When ``use_stateful_dataloader: true`` is set, Lightning's checkpoint will
contain the full lhotse iterator graph state under the dataloader key. On
resume:

* iterator positions advance to where they were at save time (no replay from
  position 0);
* ``set_epoch`` is a no-op while restored state is pending, so the resumed run
  continues the same epoch instead of starting a new one;
* ``num_workers`` and ``world_size`` must match between save and restore (a
  hard requirement of ``StatefulDataLoader``).

Non-indexed pipelines fall back to Lhotse's ``_fast_forward()`` replay (O(N)
in batches consumed before the checkpoint) and require ``num_workers`` only to
be consistent for replay-based restore — not exact restore.

For the iterator graph contract itself, see Lhotse's
`indexed manifests guide <https://lhotse.readthedocs.io/en/latest/indexed-manifests.html>`_.

Pre-computing bucket duration bins
------------------------------------

We recommend to pre-compute the bucket duration bins in order to accelerate the start of the training -- otherwise, the dynamic bucketing sampler will have to spend some time estimating them before the training starts.
The following script may be used:

.. code-block:: bash

    $ python scripts/speech_recognition/estimate_duration_bins.py -b 30 manifest.json

    # The script's output:
    Use the following options in your config:
            num_buckets=30
            bucket_duration_bins=[1.78,2.34,2.69,...
    <other diagnostic information about the dataset>

For multi-dataset setups, one may provide a dataset config directly:

.. code-block:: bash

    $ python scripts/speech_recognition/estimate_duration_bins.py -b 30 input_cfg.yaml

    # The script's output:
    Use the following options in your config:
            num_buckets=30
            bucket_duration_bins=[1.91,3.02,3.56,...
    <other diagnostic information about the dataset>

It's also possible to manually specify the list of data manifests (optionally together with weights):

.. code-block:: bash

    $ python scripts/speech_recognition/estimate_duration_bins.py -b 30 [[manifest.json,0.7],[other.json,0.3]]

    # The script's output:
    Use the following options in your config:
            num_buckets=30
            bucket_duration_bins=[1.91,3.02,3.56,...
    <other diagnostic information about the dataset>

2D bucketing
-------------

To achieve maximum training efficiency for some classes of models it is necessary to stratify the sampling
both on the input sequence lengths and the output sequence lengths.
One such example are attention encoder-decoder models, where the overall GPU memory usage can be factorized
into two main components: input-sequence-length bound (encoder activations) and output-sequence-length bound
(decoder activations).
Classical bucketing techniques only stratify on the input sequence length (e.g. duration in speech),
which leverages encoder effectively but leads to excessive padding on on decoder's side.

To amend this we support a 2D bucketing technique which estimates the buckets in two stages.
The first stage is identical to 1D bucketing, i.e. we determine the input-sequence bucket bins so that
every bin holds roughly an equal duration of audio.
In the second stage, we use a tokenizer and optionally a prompt formatter (for prompted models) to
estimate the total number of tokens in each duration bin, and sub-divide it into several sub-buckets,
where each sub-bucket again holds roughly an equal number of tokens.

To run 2D bucketing with 30 buckets sub-divided into 5 sub-buckets each (150 buckets total), use the following script:

.. code-block:: bash

    $ python scripts/speech_recognition/estimate_duration_bins_2d.py \
        --tokenizer path/to/tokenizer.model \
        --buckets 30 \
        --sub-buckets 5 \
        input_cfg.yaml

    # The script's output:
    Use the following options in your config:
            use_bucketing=1
            num_buckets=30
            bucket_duration_bins=[[1.91,10],[1.91,17],[1.91,25],...
    The max_tps setting below is optional, use it if your data has low quality long transcript outliers:
            max_tps=[13.2,13.2,11.8,11.8,...]

Note that the output in ``bucket_duration_bins`` is a nested list, where every bin specifies
the maximum duration and the maximum number of tokens that go into the bucket.
Passing this option to Lhotse dataloader will automatically enable 2D bucketing.

Note the presence of ``max_tps`` (token-per-second) option.
It is optional to include it in the dataloader configuration: if you do, we will apply an extra filter
that discards examples which have more tokens per second than the threshold value.
The threshold is determined for each bucket separately based on data distribution, and can be controlled
with the option ``--token_outlier_threshold``.
This filtering is useful primarily for noisy datasets to discard low quality examples / outliers.

We also support aggregate tokenizers for 2D bucketing estimation:

.. code-block:: bash

    $ python scripts/speech_recognition/estimate_duration_bins_2d.py \
        --tokenizer path/to/en/tokenizer.model path/to/pl/tokenizer1.model \
        --langs en pl \
        --buckets 30 \
        --sub-buckets 5 \
        input_cfg.yaml

To estimate 2D buckets for a prompted model such as Canary-1B, provide prompt format name and an example prompt.
For Canary-1B, we'll also provide the special tokens tokenizer. Example:

.. code-block:: bash

    $ python scripts/speech_recognition/estimate_duration_bins_2d.py \
        --prompt-format canary \
        --prompt "[{'role':'user','slots':{'source_lang':'en','target_lang':'de','task':'ast','pnc':'yes'}}]" \
        --tokenizer path/to/spl_tokens/tokenizer.model path/to/en/tokenizer.model path/to/de/tokenizer1.model \
        --langs spl_tokens en de \
        --buckets 30 \
        --sub-buckets 5 \
        input_cfg.yaml

Pushing GPU utilization to the limits with bucketing and OOMptimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default approach of specifying a ``batch_duration``, ``bucket_duration_bins`` and ``quadratic_duration``
is quite flexible, but is not maximally efficient. We observed that in practice it often leads to under-utilization
of GPU memory and compute for most buckets (especially those with shorter durations).
While it is impossible to estimate GPU memory usage up-front, we can determine it empirically with a bit of search.

OOMptimizer is an approach that given a NeMo model, optimizer, and a list of buckets (1D or 2D)
estimates the maximum possible batch size to use for each bucket.
It performs a binary search over batch sizes that succeed or lead to CUDA OOM until convergence.
We find that the resulting bucketing batch size profiles enable full GPU utilization in training,
while it only takes a couple of minutes to complete the search.

In order to run OOMptimizer, you only need the bucketing bins (from previous sections) and a model configuration:

.. code-block:: bash

    $ python scripts/speech_recognition/oomptimizer.py \
        --config-path fast-conformer_aed.yaml \
        --module-name nemo.collections.asr.models.EncDecMultiTaskModel \
        --buckets '[[3.975,30],[3.975,48],[4.97,37],...]'

    # The script's output:
    <output logs from the search>
    The final profile is:
            bucket_duration_bins=[[3.975,30],[3.975,48],...]
            bucket_batch_size=[352,308,280,...]
            max_tps=12.0
            max_duration=40.0

Use the resulting options in your training configuration (typically under namespace ``model.train_ds``) to apply the profile.

It's also possible to run OOMptimizer using a pretrained model's name and bucket bins corresponding
to your fine-tuning data:

.. code-block:: bash

    $ python scripts/speech_recognition/oomptimizer.py \
        --pretrained-name nvidia/canary-1b \
        --buckets '[2.0,3.1,5.6,6.6,...]'

Note that your training script can perform some additional actions using GPU RAM that cannot be anticipated by the OOMptimizer.
By default, we let the script use up to 90% of GPU's RAM for this estimation to account for that.
In the unlikely case you run into an OutOfMemoryError during training, you can try re-estimating the profile with the option ``--memory-fraction 0.75`` (or another value) that will further cap OOMptimizer's available GPU RAM.

Seeds and randomness
---------------------

In Lhotse dataloading configuration we have two parameters controlling randomness: ``seed`` and ``shard_seed``.
Both of them can be either set to a fixed number, or one of two string options ``"randomized"`` and ``"trng"``.
Their roles are:

* ``seed`` is the base random seed, and is one of several factors used to initialize various RNGs participating in dataloading.

* ``shard_seed`` controls the shard randomization strategy in distributed data parallel setups when using sharded tarred datasets.

Below are the typical examples of configuration with an explanation of the expected outcome.

Case 1 (default): ``seed=<int>`` and ``shard_seed="trng"``:

* The ``trng`` setting discards ``seed`` and causes the actual random seed to be drawn using OS's true RNG. Each node/GPU/dataloading worker draws its own unique random seed when it first needs it.

* Each node/GPU/dataloading worker yields data in a different order (no mini-batch duplication).

* On each training script run, the order of dataloader examples are **different**.

* Since the random seed is unpredictable, the exact dataloading order is not replicable.

Case 2: ``seed=<int>`` and ``shard_seed="randomized"``:

* The ``randomized`` setting uses ``seed`` along with DDP ``rank`` and dataloading ``worker_id`` to set a unique but deterministic random seed in each dataloading process across all GPUs.

* Each node/GPU/dataloading worker yields data in a different order (no mini-batch duplication).

* On each training script run, the order of dataloader examples are **identical** as long as ``seed`` is the same.

* This setup guarantees 100% dataloading reproducibility.

* Resuming training without changing of the ``seed`` value will cause the model to train on data it has already seen. For large data setups, not managing the ``seed`` may cause the model to never be trained on a majority of data. This is why this mode is not the default.

* If you're combining DDP with model parallelism techniques (Tensor Parallel, Pipeline Parallel, etc.) you need to use ``shard_seed="randomized"``. Using ``"trng"`` will cause different model parallel ranks to desynchronize and cause a deadlock.

* Generally the seed can be managed by the user by providing a different value each time the training script is launched. For example, for most models the option to override would be ``model.train_ds.seed=<value>``. If you're launching multiple tasks queued one after another on a grid system, you can generate a different random seed for each task, e.g. on most Unix systems ``RSEED=$(od -An -N4 -tu4 < /dev/urandom | tr -d ' ')`` would generate a random uint32 number that can be provided as the seed.

Other, more exotic configurations:

* With ``shard_seed=<int>``, all dataloading workers will yield the same results. This is only useful for unit testing and maybe debugging.

* With ``seed="trng"``, the base random seed itself will be drawn using a TRNG. It will be different on each GPU training process. This setting is not recommended.

* With ``seed="randomized"``, the base random seed is set to Python's global RNG seed. It might be different on each GPU training process. This setting is not recommended.

CP/TP-safe batches with ``BroadcastingDataLoader``
---------------------------------------------------

Context-parallel (CP) and tensor-parallel (TP) training require all ranks
within the same ``(cp, tp)`` sub-mesh of a DP slot to process the **same**
global batch each step — CP shards the sequence dimension and TP shards
the feature dimension, so a divergent global batch breaks the per-rank
shape contract that CP/TP collectives assume.

Independent Lhotse loaders on each rank with ``shard_seed="randomized"``
guarantee that *seeded* shard cursors line up, but they don't protect
against background-thread non-determinism (``concurrent_bucketing``,
worker scheduling jitter, etc.). The empirical signature is per-rank
``cu_seqlens`` divergence at a fraction of training steps, which then
deadlocks NCCL collectives with mismatched shapes.

The :class:`~nemo.collections.common.data.lhotse.broadcasting.BroadcastingDataLoader`
fixes this at the data layer: construct the real Lhotse loader on a
single DP-source rank (``cp_rank == 0`` and ``tp_rank == 0``) and let the
wrapper broadcast each batch to the other ranks in the ``(cp, tp)``
sub-mesh over NCCL. Iteration ends in lockstep via a continue/stop
broadcast — no length needs to be known up-front.

.. code-block:: python

    from torch.distributed.device_mesh import init_device_mesh

    from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
    from nemo.collections.common.data.lhotse.broadcasting import (
        BroadcastingDataLoader,
        is_dp_source_rank,
    )

    mesh = init_device_mesh("cuda", (dp, cp, tp), mesh_dim_names=("dp", "cp", "tp"))

    if is_dp_source_rank(mesh):
        source = get_lhotse_dataloader_from_config(
            config=cfg.train_ds,
            global_rank=dp_rank,
            world_size=dp_size,
            dataset=dataset,
            tokenizer=tokenizer,
        )
    else:
        source = None

    return BroadcastingDataLoader(source=source, device_mesh=mesh)

The wrapper delegates ``state_dict`` / ``load_state_dict`` to the source
loader on the source rank (no-ops on non-source ranks), so checkpoint and
resume keep working transparently with regular ``DataLoader``,
``torchdata.StatefulDataLoader``, or any other source object that
implements those methods.

The wrapper is a no-op when ``device_mesh`` is ``None`` or every named
axis present in the mesh has size 1, so the same call site works for
single-GPU, DDP-only, and CP/TP runs without a separate code path.

Train vs. validation / test configs
-----------------------------------

The training and validation/test sections of a NeMo recipe use the same
underlying dataloader builder but have a different shape and a different
default behavior.

**Training (``train_ds``).** A single config that produces one infinite
``CutSet``. The dataloader is wrapped to never run out of data, so
``trainer.max_steps`` (and ``limit_train_batches`` for tarred sources)
controls the run length:

.. code-block:: yaml

    model:
      train_ds:
        sample_rate: 16000
        num_workers: 4
        shuffle: true
        use_bucketing: true
        num_buckets: 30
        batch_duration: 1100
        input_cfg:
          - type: nemo_tarred
            manifest_filepath: /data/asr/manifest__OP_0..127_CL_.json
            tarred_audio_filepaths: /data/asr/audio__OP_0..127_CL_.tar

**Validation / test (``validation_ds`` / ``test_ds``).** A *named* dict of
configs — one per evaluation set — that produces finite iteration:

.. code-block:: yaml

    model:
      validation_ds:
        sample_rate: 16000
        batch_size: 16
        # Per-set entries; keys become the metric prefixes in logging.
        datasets:
          dev_clean:
            cuts_path: /data/dev-clean/cuts.jsonl
          dev_other:
            cuts_path: /data/dev-other/cuts.jsonl

The most common eval-side overrides:

* ``shuffle: false`` — deterministic order.
* ``force_finite: true`` — break out of the infinite-mux that's safe for
  training but would loop forever in eval.
* ``use_bucketing: false`` — bucketing trades padding for randomness; on a
  small eval set the savings are negligible and a fixed batch size makes
  results easier to interpret.
* ``num_workers: 0`` (or a small number) — eval is short, the worker
  startup cost matters more.

When the model code expects a single eval set, use the plain ``cuts_path`` /
``manifest_filepath`` form at the same level as ``train_ds`` instead of the
``datasets:`` dict.

Preparing your data
-------------------

Three minimal recipes covering the main on-disk formats.

**NeMo manifest** — one JSON object per line, fields read by ``LazyNeMoIterator``::

    {"audio_filepath": "/data/utt_0001.wav", "duration": 3.42, "text": "hello world", "lang": "en"}
    {"audio_filepath": "/data/utt_0002.wav", "duration": 5.10, "text": "another example", "lang": "en"}

For tarred NeMo manifests, see
``scripts/speech_recognition/convert_to_tarred_audio_dataset.py`` in the NeMo
repo.

**Lhotse cuts JSONL** — build a ``CutSet`` from raw recordings + supervisions:

.. code-block:: python

    from lhotse import CutSet, Recording, SupervisionSegment

    cuts = []
    for path, transcript in pairs:
        rec = Recording.from_file(path)
        sup = SupervisionSegment(
            id=rec.id, recording_id=rec.id,
            start=0.0, duration=rec.duration,
            text=transcript, language="en",
        )
        cut = rec.to_cut()
        cut.supervisions = [sup]
        cuts.append(cut)

    CutSet.from_cuts(cuts).to_file("cuts.jsonl")  # uncompressed!

For Lhotse Shar (sharded archive), see the upstream tutorial: |tutorial_shar|.

**Parquet** — write a ``pyarrow`` table with the column names the
``LazyParquetIterator`` reads (``audio``, ``text``, ``duration``,
optional ``lang``):

.. code-block:: python

    import pyarrow as pa, pyarrow.parquet as pq

    table = pa.table({
        "audio":    [open(p, "rb").read() for p in paths],
        "text":     transcripts,
        "duration": durations,
        "lang":     ["en"] * len(paths),
    })
    pq.write_table(table, "shard_000.parquet")  # row-group stats kept by default

Once your manifests are written, build the indexed sidecars in one shot::

    python scripts/dataloading/build_indexes.py path/to/input_cfg.yaml

See :ref:`indexed-resumable-dataloading` for the resumable side.

.. _lhotse-storage-backends:

Storage backends: local, object store, AIStore
----------------------------------------------

Every input path the dataloader reads goes through Lhotse's ``open_best``,
which routes file paths and URIs to the right backend automatically:

* **Local files** — paths like ``/data/...`` work out of the box, no
  configuration needed.
* **Generic object stores via ``smart_open``** — ``s3://``, ``gs://``,
  ``http://``, ``https://`` URIs work after ``pip install smart_open``.
  Authentication uses the underlying SDK's defaults (e.g. AWS env vars).
* **AIStore** — ``ais://bucket/key`` URIs work after ``pip install aistore``
  and ``export AIS_ENDPOINT=http://...``. Optional tuning env vars
  ``AIS_CONNECT_TIMEOUT`` and ``AIS_READ_TIMEOUT`` are honored by the SDK.

The same routing applies to ``.idx`` sidecars: they are read and written
next to the data file, so the backend must accept writes at that location
or the indexes need to be pre-built locally and uploaded.

AIStore GetBatch (separate optimization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For tarred multimodal-conversation manifests, NeMo also supports AIStore's
batched object-fetch API (``GetBatch``) via ``USE_AIS_GET_BATCH=true``,
which issues one batched fetch per minibatch instead of per-cut tar reads.
This is independent of using AIStore as a generic backend — see
:doc:`speechlm2/datasets` for the speech-LM-specific details, including
how it composes with ``indexed: true``.

.. _lhotse-extension-hooks:

Registering a custom format
---------------------------

Adding a new ``type:`` to the ``input_cfg`` registry is one decorator and
one function:

.. code-block:: python

    from nemo.collections.common.data.lhotse.cutset import data_type_parser
    from lhotse import CutSet

    @data_type_parser("my_format")
    def read_my_format(config) -> tuple[CutSet, bool]:
        cuts = CutSet(MyAdapter(path=config.path, ...))
        is_tarred = True  # True ⇒ IterableDataset path; False ⇒ map-style
        return cuts, is_tarred

The parser must accept arbitrary keys: ``read_dataset_config`` cascades
options like ``indexed``, ``shard_seed``, ``metadata_only``,
``force_finite``, ``audio_locator_tag`` from the top of the YAML down into
every entry via ``propagate_attrs``. Missing keys should fall back to
sensible defaults via ``config.get(...)``.

To make ``MyAdapter`` participate in the indexed/resumable path
(:ref:`indexed-resumable-dataloading`), implement Lhotse's
:class:`~lhotse.lazy.IteratorNode` contract — see
`indexed manifests guide <https://lhotse.readthedocs.io/en/latest/indexed-manifests.html>`_
for the requirements.

Common pitfalls
---------------

The most common foot-guns when standing up a NeMo Lhotse recipe:

1. **Forgetting** ``trainer.use_distributed_sampler=false``. NeMo's Lhotse
   integration handles distributed sampling itself; leaving Lightning's
   default on causes silent batch duplication across DP ranks.

2. **No** ``max_steps`` **with tarred / Shar data.** Tarred sources are
   infinite by design, so without ``trainer.max_steps`` (and
   ``limit_train_batches`` for the periodic validation cadence) training
   never completes the first "epoch". Always set both.

3. **Compressed inputs cannot be indexed.** ``.jsonl.gz`` and ``.tar.gz``
   work for streaming, but ``indexed: true`` requires uncompressed,
   seekable files. Re-extract or re-write before building ``.idx``.

4. **Mismatched** ``num_workers`` / ``world_size`` **on resume.** Exact
   per-worker resume with ``StatefulDataLoader`` requires both to match
   between save and restore. Replay-based restore with the regular
   ``DataLoader`` is more lenient.

5. ``indexed: true`` **is incompatible with** ``extra_fields`` **and**
   ``slice_length`` on ``nemo`` / ``nemo_tarred``. Both expand or rewrite
   cuts in a way that has no stable index. Pre-process the manifest
   offline if you need them in an indexed pipeline.

6. ``shard_seed: "trng"`` **deadlocks under TP/PP.** Tensor- and pipeline-
   parallel ranks must see the same shard order, but ``"trng"`` draws an
   independent seed per worker. Use ``shard_seed: "randomized"`` whenever
   you have model parallelism on top of DDP.

7. **Missing** ``force_finite: true`` **on validation.** Validation configs
   that reuse training infrastructure inherit the infinite-mux behavior;
   without ``force_finite: true`` the validation loop never terminates.

.. _lhotse-config-reference:

``LhotseDataLoadingConfig`` field reference
-------------------------------------------

The complete option schema lives in ``LhotseDataLoadingConfig``
(``nemo/collections/common/data/lhotse/dataloader.py``). It carries ~80
fields; the categorization below mirrors the source order and groups
options by what they control.

**Inputs.** ``input_cfg``, ``manifest_filepath``,
``tarred_audio_filepaths``, ``cuts_path``, ``shar_path``,
``skip_missing_manifest_entries``.

**Sampling — basic.** ``batch_size``, ``batch_duration``,
``quadratic_duration``, ``min_duration``, ``max_duration``, ``min_tps``,
``max_tps``.

**Sampling — bucketing.** ``use_bucketing``, ``num_buckets``,
``bucket_duration_bins``, ``bucket_batch_size``, ``bucket_buffer_size``,
``num_cuts_for_bins_estimate``, ``concurrent_bucketing``.

**Sampling — multimodal.** ``use_multimodal_sampling``, ``prompt_format``,
``pretokenize``, ``audio_locator_tag``, ``token_equivalent_duration``,
``batch_tokens``, ``quadratic_factor``, ``min_tokens``, ``max_tokens``,
``min_tpt``, ``max_tpt``, ``measure_total_length``.

**Sampling — fusion (multi-config).** ``multi_config``, ``sampler_fusion``,
``sampler_weights``.

**Indexed / resumable.** ``indexed``, ``use_stateful_dataloader``,
``indexes_root``. See :ref:`indexed-resumable-dataloading` and
:ref:`lhotse-indexes-root`.

**Mixing & weighting.** ``reweight_temperature``, ``max_open_streams``.

**I/O & distributed.** ``num_workers``, ``pin_memory``, ``shard_seed``,
``seed``, ``shuffle``, ``shuffle_buffer_size``, ``drop_last``,
``force_finite``, ``force_map_dataset``, ``force_iterable_dataset``,
``metadata_only``, ``cuda_expandable_segments``.

**On-the-fly augmentation.**

* Speed/RIR — ``perturb_speed``, ``rir_enabled``, ``rir_path``, ``rir_prob``.
* Noise — ``noise_path``, ``noise_snr``, ``noise_mix_prob``.
* Lowpass — ``lowpass_enabled``, ``lowpass_frequencies_interval``,
  ``lowpass_prob``.
* Compression — ``compression_enabled``, ``compression_prob``,
  ``compression_level_interval``, ``compression_codecs``,
  ``compression_codec_weights``, ``compression_enable_for_custom_fields``.
* Clipping — ``clipping_enabled``, ``clipping_gain_db``,
  ``clipping_normalize``, ``clipping_oversampling``, ``clipping_prob``,
  ``clipping_prob_hard``.
* Concatenation — ``concatenate_samples``, ``concatenate_gap_seconds``,
  ``concatenate_duration_factor``, ``concatenate_merge_supervisions``,
  ``db_norm``.

**Cut transforms.** ``truncate_duration``, ``truncate_offset_type``,
``cut_into_windows_duration``, ``cut_into_windows_hop``,
``pad_min_duration``, ``pad_direction``, ``cut_text_into_windows_tokens``,
``keep_excessive_supervisions``.

**Field-name overrides.** ``text_field``, ``lang_field``,
``channel_selector``, ``sample_rate``.

**Filtering.** ``max_cer``, ``min_context_speaker_similarity``, ``keep``.

For exact types and defaults, see the dataclass definition in the source
file — it is the single source of truth.

See also
--------

* :doc:`speechlm2/datasets` — speech-LM-specific data classes, AIStore
  GetBatch with indexed mode, and the SpeechLM ``DataModule`` resume
  contract.
* :doc:`asr/datasets` — ASR-specific data preparation conventions.
* :doc:`audio/datasets` — audio (codec, enhancement) data flows.
* `Lhotse PyTorch Datasets <https://lhotse.readthedocs.io/en/latest/datasets.html>`_
  — upstream sampler API, ``StatefulDataLoader`` integration, custom RNG
  state in batch transforms.
* `Lhotse indexed manifests <https://lhotse.readthedocs.io/en/latest/indexed-manifests.html>`_
  — the iterator-graph contract that makes O(1) restore work.
