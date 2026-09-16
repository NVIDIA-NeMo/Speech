Datasets
========

The speechlm2 collection supports datasets that contain both audio and text data for training models that can understand speech and generate appropriate responses.
This section describes the dataset format, preparation, and usage with the speechlm2 models.

.. seealso::

   :doc:`/dataloaders` is the canonical reference for the underlying Lhotse
   dataloader: ``input_cfg`` shape, supported formats, sampling/bucketing
   options, indexed manifests + resumable dataloading, and
   ``LhotseDataLoadingConfig`` field schema. The page below covers what's
   speech-LM-specific on top of that — datamodule resume contract,
   AIStore GetBatch, conversation type semantics in the SALM/duplex
   recipes.

Dataset Format
--------------

Duplex S2S models use the Lhotse framework for audio data management. The primary datasets used are:

1. **DuplexS2SDataset**: For general duplex speech-to-speech models
2. **SALMDataset**: Specifically for the Speech-Augmented Language Model (SALM), which processes speech+text and outputs text.
3. **DuplexSTTDataset**: For the DuplexSTTModel, which processes conversational audio and generates text responses.
4. **DuplexEARTTSDataset**: Dataset for Duplex EARTTS model, extending DuplexS2SDataset with additional output fields for TTS, including audio prompting. It optionally prepends an audio prompt (speaker reference) to target_audio, which is used to initialize speaker conditioning in the EARTTS model. The dataset provides audio_prompt, audio_prompt_lens, non_prompt_mask, aligned_attention_mask, and aligned_position_ids, and supports custom speaker reference audio through the context_audio field, while preserving full compatibility with the original data format.

DuplexS2S Dataset Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A typical dataset for speechlm2 models consists of:

1. **Audio files**: Contains source audio (input speech) and possibly target audio (output speech)
2. **Text transcriptions**: Associated text for both input and output speech
3. **Role identifiers**: To distinguish between speakers (e.g., "user" vs "agent")

The dataset organization is built around the concept of conversation turns, with each turn containing audio and text from either a user or an agent/assistant.

The datasets are primarily managed using Lhotse's CutSet format, which provides efficient handling of audio data and annotations. A typical Lhotse manifest includes:

- Audio recording information (path, duration, sample rate)
- Supervision information (transcripts, speaker roles, timing)
- Optional additional annotations

Example of a Lhotse cut:

.. code-block:: python

    {
        "id": "conversation_1",
        "start": 0,
        "duration": 10.7,
        "channel": 0,
        "supervisions": [
            {
                "id": "conversation_1_turn_0",
                "text": "Can you help me with this problem?",
                "start": 0,
                "duration": 5.2,
                "speaker": "user"
            },
            {
                "id": "conversation_1_turn_1",
                "text": "I can help you with that.",
                "start": 5.2,
                "duration": 3.1,
                "speaker": "assistant"
            }
        ],
        "recording": {
            "id": "conversation_1_user",
            "path": "/path/to/audio/conversation_1_user.wav",
            "sampling_rate": 16000,
            "num_samples": 171200,
            "duration": 10.7
        },
        "custom": {
            "target_audio": {
                "id": "conversation_1_assistant",
                "path": "/path/to/audio/conversation_1_assistant.wav",
                "sampling_rate": 22050,
                "num_samples": 235935,
                "duration": 10.7
            }
        }
    }

The DuplexS2SDataset performs several key operations when processing data:

1. **Turn Identification**: Each cut contains a list of `supervisions` with objects of type `lhotse.SupervisionSegment` that represent conversation turns with corresponding text and speaker information.

2. **Speaker Role Separation**: The text of each supervision is tokenized and identified as the model's output (when `supervision.speaker` is in `output_roles`, e.g., "agent" or "Assistant") or the model's input (when in `input_roles`, e.g., "user" or "User").

3. **Token Sequence Generation**:
   - `target_tokens` and `source_tokens` arrays are created with a length equal to `lhotse.utils.compute_num_frames(cut.duration, frame_length, cut.sampling_rate)`
   - The `frame_length` parameter (typically 80ms) determines the temporal resolution of token assignments
   - Each token is assigned to a position based on its corresponding audio segment's timing

4. **Token Offset Calculation**:
   - The starting position for each turn's tokens is determined using `lhotse.utils.compute_num_frames(supervision.start, frame_length)`
   - This ensures tokens are aligned with their corresponding audio segments

5. **Length Validation**:
   - If token sequences are too long compared to the audio duration, warnings are emitted
   - Tokens that extend beyond the audio length are truncated

This process ensures that the model can correctly align audio input with corresponding text, and learn to generate appropriate responses based on the conversation context.

DuplexS2SDataset
****************

This dataset class is designed for models that handle both speech understanding and speech generation. It processes audio inputs and prepares them for the model along with corresponding text.

.. code-block:: python

    from nemo.collections.speechlm2.data import DuplexS2SDataset
    
    dataset = DuplexS2SDataset(
        tokenizer=model.tokenizer,                   # Text tokenizer
        frame_length=0.08,                          # Frame length in seconds
        source_sample_rate=16000,                   # Input audio sample rate
        target_sample_rate=22050,                   # Output audio sample rate
        input_roles=["user", "User"],               # Roles considered as input
        output_roles=["agent", "Assistant"]         # Roles considered as output
    )

SALMDataset Structure
^^^^^^^^^^^^^^^^^^^^^

Data used for SALM can be either regular speech-to-text data (in any NeMo or Lhotse format), or a dataset of multi-turn conversions.
For the most part, please refer to :doc:`the ASR datasets documentation <../asr/datasets>` for details on data formats and multimodal dataloading.

When using speech-to-text data, you'll need read it with a special ``lhotse_as_conversation`` data reader
that creates a two-turn, query+response, multi-modal conversation data types out of regular Lhotse cuts.
This approach makes SALM training more flexible, allowing straightforward combination of single-turn and multi-turn data.

Each audio turn is represented as a single token, defined in ``audio_locator_tag`` property, and automatically added to the model's tokenizer inside model code.
This token is replaced during the training/generation pass with its corresponding audio segment representation.

Example YAML configuration using existing ASR datasets with ``lhotse_as_conversation``:

.. code-block:: yaml

    data:
      train_ds:
        prompt_format: "llama3"  # Choose based on your model
        token_equivalent_duration: 0.08
        input_cfg:
          # Example 1: Using standard ASR Lhotse manifests (JSONL)
          - type: lhotse_as_conversation
            cuts_path: /path/to/librispeech_train_clean_100.jsonl.gz
            audio_locator_tag: "<|audioplaceholder|>"
            tags:
              context: "Transcribe the following audio:"
              # Optional system prompt can be uncommented
              # system_prompt: "You are a helpful assistant that transcribes audio accurately."
          
          # Example 2: Using tarred NeMo manifests
          - type: lhotse_as_conversation
            manifest_filepath: /path/to/tedlium_train_manifest.jsonl.gz
            tarred_audio_filepaths: /path/to/tedlium_shards/shard-{000000..000009}.tar
            audio_locator_tag: "<|audioplaceholder|>"
            tags:
              context: "Write down what is said in this recording:"
              
          # Example 3: Using Lhotse SHAR format
          - type: lhotse_as_conversation
            shar_path: /path/to/fisher_shar/
            audio_locator_tag: "<|audioplaceholder|>"
            tags:
              context: "Listen to this clip and write a transcript:"
    
      # ... other settings

Alternatively, one can provide an existing YAML file with their dataset composition and wrap 
it in a ``lhotse_as_conversation`` reader as follows:

.. code-block:: yaml

    data:
      train_ds:
        input_cfg:
          - type: lhotse_as_conversation
            input_cfg: /path/to/dataset_config.yaml
            audio_locator_tag: "<|audioplaceholder|>"
            tags:
              context: "Transcribe the following audio:"
              # Optional system prompt can be uncommented
              # system_prompt: "You are a helpful assistant that transcribes audio accurately."


The ``lhotse_as_conversation`` reader automatically creates a two-turn conversation from each ASR example:
1. Optionally, if ``system_prompt`` tag is provided, it's added as a special system turn for LLM models that support system prompts.
2. A user turn containing the audio and a text context (from the ``context`` tag)
3. An assistant turn containing the transcription (from the cut's supervision text)

If a ``context`` tag is provided in the configuration, it's added as a text turn before the audio.

SALMDataset
***********

This dataset class is specialized for the SALM model, which focuses on understanding speech input and generating text output.

.. code-block:: python

    from nemo.collections.speechlm2.data import SALMDataset

    dataset = SALMDataset(
        tokenizer=model.tokenizer,                   # Text tokenizer
    )


Indexed ShareGPT audio formats
------------------------------

The ``share_gpt`` and ``share_gpt_webdataset`` readers support the indexed,
checkpointable conversation formats below.  The YAML blocks are complete
``input_cfg`` leaves: they can be stored as individual YAML files and passed to
the index tools.  Source JSONL and tar files are immutable; ``.idx``,
``.wds-v2.idx``, ``.sgroute``, and ``.idxpack`` files are derived artifacts.
For stateful training, put ``index_pack`` on each outer dataset entry and keep
``indexed: true`` and ``use_stateful_dataloader: true`` on the training dataset.

System prompt policy
^^^^^^^^^^^^^^^^^^^^

ShareGPT turns whose ``from`` field is ``system`` remain system turns. The
``share_gpt``, ``share_gpt_webdataset``, and ``multimodal_conversation`` readers
also accept a configured prompt through ``tags``:

.. code-block:: yaml

    tags:
      system_prompt: "You are a helpful assistant."
      override_system_prompt: true

Without ``override_system_prompt``, a system prompt already present in the data
is preserved and the configured prompt is inserted only when one is absent.
When the override is enabled, the configured prompt replaces all system turns
from the data.

Audio fields and placeholder binding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The accepted top-level audio fields have deterministic precedence:
``sound``, then ``speech``, then ``ori_sound``.  The first non-empty field wins;
lower-precedence fields are not merged.  Each field accepts either one string
or a flat list of strings.  Audio paths bind in JSON list order:

* one path may be reused by any number of user/human placeholders;
* several paths with one placeholder expand into several audio turns;
* several paths with several placeholders bind one path per placeholder and
  must have equal cardinality.

Assistant placeholders remain literal text.  The default placeholders are
``<sound>`` and ``<speech>``; list them explicitly when source data uses both.
These minimal rows make the precedence rule concrete:

.. code-block:: json

    {
      "id": "alias-precedence",
      "sound": "chosen-by-sound.wav",
      "speech": "ignored-speech.wav",
      "ori_sound": "ignored-original.wav",
      "conversations": [
        {"from": "human", "value": "<sound>"},
        {"from": "gpt", "value": "Done."}
      ]
    }

.. code-block:: json

    {
      "id": "alias-speech",
      "speech": "chosen-by-speech.wav",
      "conversations": [
        {"from": "human", "value": "<speech>"},
        {"from": "gpt", "value": "Done."}
      ]
    }

.. code-block:: json

    {
      "id": "alias-ori-sound",
      "ori_sound": "chosen-by-ori.wav",
      "conversations": [
        {"from": "human", "value": "<sound>"},
        {"from": "gpt", "value": "Done."}
      ]
    }

Loose JSONL with direct audio
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Relative local paths use ``audio_root`` when configured, otherwise the
manifest directory.  This row uses a scalar relative ``sound``:

.. code-block:: json

    {
      "id": "loose-relative-scalar",
      "sound": "audio/request.wav",
      "conversations": [
        {"from": "human", "value": "Please inspect <sound>."},
        {"from": "gpt", "value": "The request is clear."}
      ]
    }

.. code-block:: yaml

    # indexed-sharegpt-example: loose-relative-scalar
    - type: share_gpt
      manifest_filepath: /data/sharegpt/relative.jsonl
      audio_root: /data/sharegpt
      audio_locator_tag: <|audio|>
      audio_placeholders: [<sound>, <speech>]
      shuffle: true
      shard_seed: 17
      indexed: true
      indexes_root: /index-mirror/site-a
      index_pack: /index-packs/site-a/loose-relative.idxpack
      index_pack_max_open_files: 32
      force_finite: false

``s3://`` and ``ais://`` are direct object paths.  A flat list may mix them;
the following two placeholders bind left then right:

.. code-block:: json

    {
      "id": "loose-remote-list",
      "sound": [
        "s3://example-audio/clips/left.flac",
        "ais://example-audio/clips/right.flac"
      ],
      "conversations": [
        {"from": "human", "value": "Compare <sound> with <speech>."},
        {"from": "gpt", "value": "They differ."}
      ]
    }

.. code-block:: yaml

    # indexed-sharegpt-example: loose-remote-list
    - type: share_gpt
      manifest_filepath: /data/sharegpt/remote.jsonl
      audio_locator_tag: <|audio|>
      audio_placeholders: [<sound>, <speech>]
      shuffle: true
      shard_seed: 17
      indexed: true
      indexes_root: /index-mirror/site-b
      index_pack: /index-packs/site-b/loose-remote.idxpack
      index_pack_max_open_files: 32
      force_finite: false

Absolute POSIX paths are read unchanged when they exist locally.  To preserve
an original manifest across clusters, ``audio_path_prefix_map`` rewrites an
absolute source prefix before opening audio.  Mapping is component-aware,
selects the unique longest prefix, and requires every absolute source path to
match when a map is configured.  Destinations may be local paths or
``s3://``/``ais://`` roots; the source JSONL is not rewritten.

.. code-block:: json

    {
      "id": "loose-absolute-prefix-map",
      "sound": "/lustre/source/audio/request.wav",
      "conversations": [
        {"from": "human", "value": "Inspect <sound>."},
        {"from": "gpt", "value": "Done."}
      ]
    }

.. code-block:: yaml

    # indexed-sharegpt-example: loose-absolute-prefix-map
    - type: share_gpt
      manifest_filepath: /data/sharegpt/absolute.jsonl
      audio_path_prefix_map:
        /source/site-a/: s3://example-audio/source-root/
      audio_locator_tag: <|audio|>
      audio_placeholders: [<sound>, <speech>]
      shuffle: true
      shard_seed: 17
      indexed: true
      indexes_root: /index-mirror/site-b
      index_pack: /index-packs/site-b/loose-absolute.idxpack
      index_pack_max_open_files: 32
      force_finite: false

Audited exact-row exclusions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An immutable source may have a separately approved set of unusable rows.  For
indexed ShareGPT JSONL only, ``excluded_manifest_lines`` removes those rows
from the logical sample domain before parsing or audio I/O.  Lines are
one-based, positive, unique integers.  The list must be sorted and accompanied
by both digests:

.. code-block:: yaml

    - type: share_gpt
      manifest_filepath: /data/sharegpt/frozen.jsonl
      audio_root: /data/sharegpt
      audio_locator_tag: <|audio|>
      audio_placeholders: [<sound>, <speech>]
      shuffle: true
      shard_seed: 17
      indexed: true
      index_pack: /index-packs/site-a/frozen.idxpack
      excluded_manifest_lines: [12, 48, 93]
      excluded_manifest_lines_sha256: <sha256-of-compact-json-list-plus-newline>
      approved_exclusion_audit_sha256: <sha256-of-immutable-approval-artifact>
      force_finite: false

Compute the line-set digest over the exact UTF-8 bytes ``[12,48,93]\\n``.
The approval digest must be a lowercase 64-character SHA-256.  Construction
fails on an invalid line, duplicate, out-of-range line, digest mismatch,
missing approval digest, or use without ``indexed: true``.

Logical graph tokens are mapped to the remaining physical rows before
partitioning output is decoded.  ``len(reader)`` therefore reports source
rows minus approved exclusions; ``__getitem__`` and iterator graph origins
use the logical index, while collection-mode routing uses the corresponding
physical manifest row.  Checkpoint state records both exclusion digests and
refuses a resume when the set or approval changes.  Datasets without
exclusions retain their legacy state shape.

This mechanism does not enable ``skip_missing_manifest_entries`` and does
not change ``fault_tolerant_audio_loading``. Keep the former disabled and set
the latter to ``false`` when the contract is to skip only an audited exact set
and fail on every unrelated manifest or audio error.

Aligned JSONL and tar shards
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The legacy paired mode aligns expanded manifest and tar shard lists one to
one.  Each JSONL row names a regular member in its corresponding tar; ordering
inside the tar is not the lookup key.

.. code-block:: json

    {
      "id": "aligned-jsonl-tar",
      "sound": "audio/000.wav",
      "conversations": [
        {"from": "human", "value": "Transcribe <sound>."},
        {"from": "gpt", "value": "hello"}
      ]
    }

.. code-block:: text

    manifests/part-000.jsonl  <->  audio/part-000.tar
                                      audio/000.wav
    manifests/part-001.jsonl  <->  audio/part-001.tar
                                      audio/001.wav

.. code-block:: yaml

    # indexed-sharegpt-example: aligned-jsonl-tar
    - type: share_gpt
      manifest_filepath: /data/aligned/manifests/part-{000..127}.jsonl
      tarred_audio_filepaths: /data/aligned/audio/part-{000..127}.tar
      tar_lookup_mode: paired
      audio_locator_tag: <|audio|>
      audio_placeholders: [<sound>, <speech>]
      shuffle: true
      shard_seed: 17
      indexed: true
      indexes_root: /index-mirror/site-a
      force_finite: false

Paired mode requires a JSONL ``.idx`` and native tar ``.idx`` for every shard.
It is a supported loose-index compatibility path, but paired ShareGPT audio
tars are not supported by ``convert_indexes_to_idxpack.py``.  Do not add an
``index_pack`` to this leaf; use collection mode for new pack-based datasets.

JSONL with an unordered tar collection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Collection mode is for one or more manifest shards whose audio members are
distributed across an ordered tar collection without positional shard pairing.
The route builder resolves each requested name once and writes an immutable,
row-aligned ``.sgroute``.  Runtime uses the row's ``(tar shard, member index)``
records and never builds a per-worker global filename table.

Rows with no configured audio placeholder in a ``human``/``user`` turn are
treated as text-only and receive an empty route, even if they retain a stale
``sound``/``speech``/``ori_sound`` field.  Placeholder-like text in assistant
turns is literal and does not request audio.  A user placeholder still requires
a valid audio field and the usual path/placeholder cardinality checks.

.. code-block:: json

    {
      "id": "unordered-tar-collection",
      "sound": ["clips/left.flac", "clips/right.flac"],
      "conversations": [
        {"from": "human", "value": "First <sound>, then <speech>."},
        {"from": "gpt", "value": "Counted both clips."}
      ]
    }

.. code-block:: text

    multi-clip.jsonl
    audio_00.tar
      unrelated.flac
      clips/right.flac
    audio_01.tar
      clips/left.flac

.. code-block:: yaml

    # indexed-sharegpt-example: unordered-tar-collection
    - type: share_gpt
      manifest_filepath: /data/multi-clip/multi-clip.jsonl
      tarred_audio_filepaths: /data/multi-clip/audio_{00..63}.tar
      tar_lookup_mode: collection
      tar_routing_filepath: multi-clip.sgroute
      audio_locator_tag: <|audio|>
      audio_placeholders: [<sound>, <speech>]
      shuffle: true
      shard_seed: 17
      indexed: true
      indexes_root: /index-mirror/site-a
      index_pack: /index-packs/site-a/multi-clip.idxpack
      index_pack_max_open_files: 32
      force_finite: false

``tar_routing_filepath`` is canonical; ``tar_routing_index`` is a compatibility
alias.  Defining both with different values is an error.  In packed mode, a
relative route is published and resolved beside ``index_pack``; in loose mode,
it resolves below ``indexes_root``.  A relative route without either root is an
error.  The pack must contain ``role=manifest, kind=jsonl`` and an
offset-bearing ``role=tar_collection, kind=nemo_tar``.  Paths-only native tar
catalogs cannot serve collection routes.

The current route format is ``SGROUTE2``.  Its builder records both a complete
manifest-content digest and a compact source-identity digest.  One-time build
and artifact validation recompute the complete content digest; training-worker
startup compares only local stat identity or a stable remote version, ETag, or
checksum.  Runtime therefore performs no full-manifest payload read while
still rejecting a changed source before the first sample is yielded.  A remote
backend that exposes no stable object identity is rejected.

Conventional and variable-member WebDataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``share_gpt_webdataset`` groups consecutive regular tar members by POSIX parent
plus the filename prefix before the first dot.  Conventional one-audio samples
are valid WDS v2 samples and pack normally:

.. code-block:: json

    {
      "id": "conventional-wds-v2",
      "sound": "sample-000.wav",
      "conversations": [
        {"from": "human", "value": "Transcribe <sound>."},
        {"from": "gpt", "value": "hello"}
      ]
    }

.. code-block:: text

    shard-000.tar
      sample-000.json
      sample-000.wav
      sample-001.wav
      sample-001.json

.. code-block:: yaml

    # indexed-sharegpt-example: conventional-wds-v2
    - type: share_gpt_webdataset
      data_dir: /data/wds/conventional
      wds_sample_index_version: 2
      audio_locator_tag: <|audio|>
      audio_placeholders: [<sound>, <speech>]
      shuffle: true
      shard_seed: 17
      indexed: true
      indexes_root: /index-mirror/site-a
      index_pack: /index-packs/site-a/wds-conventional.idxpack
      index_pack_max_open_files: 32
      force_finite: false

Legacy conventional rows may retain an original source path in ``sound`` (or
an audio alias) even when packing renames the sole tar payload to
``<sample-key>.<codec>``.  When a sample has exactly one audio payload, that
one-to-one binding is deterministic and remains supported.  Variable-member
samples still resolve strictly by exact name or one unambiguous basename.

Variable-member samples may use the same key for ``1.json``, ``1.flac``, and
``1.1.flac``.  The JSON path list determines placeholder order; physical audio
member order does not.  Resolution first matches the exact member name and
then an unambiguous basename.

.. code-block:: json

    {
      "id": "variable-wds-v2",
      "speech": ["nested/1.flac", "nested/1.1.flac"],
      "conversations": [
        {"from": "human", "value": "Compare <speech> and <sound>."},
        {"from": "gpt", "value": "The clips differ."}
      ]
    }

.. code-block:: text

    variable-shard-000.tar
      nested/1.1.flac
      nested/1.json
      nested/1.flac
      nested/2.json
      nested/2.flac

.. code-block:: yaml

    # indexed-sharegpt-example: variable-wds-v2
    - type: share_gpt_webdataset
      data_dir: /data/wds/variable-members
      wds_sample_index_version: 2
      audio_locator_tag: <|audio|>
      audio_placeholders: [<sound>, <speech>]
      shuffle: true
      shard_seed: 17
      indexed: true
      indexes_root: /index-mirror/site-a
      index_pack: /index-packs/site-a/wds-variable-members.idxpack
      index_pack_max_open_files: 32
      force_finite: false

Every WDS v2 tar has ``<tar>.wds-v2.idx`` plus canonical
``<tar>.wds-v2.idx.meta.json`` metadata.  Its pack collection is
``role=wds_tar, kind=wds_tar_v2`` and retains sample offsets.  The pack catalog
defines the runtime shard order, so production workers perform no recursive WDS scan.
If ``wids-meta.json`` is absent, only the one-time build/conversion CPU process
may discover tars below ``data_dir``.

Text-only conversations
^^^^^^^^^^^^^^^^^^^^^^^

Loose ``share_gpt`` JSONL may contain text-only rows with no audio field and no
audio placeholder.  They can coexist with audio rows and use the same manifest
index/pack.  WDS v2 intentionally requires at least one non-JSON payload per
sample, and tar-paired or collection routing should not be used to represent a
text-only source.

.. code-block:: json

    {
      "id": "text-only",
      "conversations": [
        {"from": "human", "value": "What is two plus two?"},
        {"from": "gpt", "value": "Four."}
      ]
    }

.. code-block:: yaml

    # indexed-sharegpt-example: text-only
    - type: share_gpt
      manifest_filepath: /data/sharegpt/text-only.jsonl
      audio_locator_tag: <|audio|>
      audio_placeholders: [<sound>, <speech>]
      shuffle: true
      shard_seed: 17
      indexed: true
      indexes_root: /index-mirror/site-a
      index_pack: /index-packs/site-a/text-only.idxpack
      index_pack_max_open_files: 32
      force_finite: false

Build, convert, and validate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run these commands from the NeMo repository root.  Build loose sidecars first;
omitting ``--kind`` builds every kind discovered in the leaf.

Direct or text-only JSONL needs the manifest ``.idx`` before conversion:

.. code-block:: bash

    python scripts/dataloading/build_indexes.py \
      --indexes-root /index-mirror/site-a --kind jsonl loose-sharegpt.yaml
    python scripts/dataloading/convert_indexes_to_idxpack.py \
      --indexes-root /index-mirror/site-a \
      --output /immutable-packs/site-a/loose-sharegpt.idxpack loose-sharegpt.yaml

The direct JSONL pack contains ``role=manifest, kind=jsonl``.  The same layout
serves mixed audio/text or text-only manifests.

Aligned paired mode needs both loose index kinds and is not converted:

.. code-block:: bash

    python scripts/dataloading/build_indexes.py \
      --indexes-root /index-mirror/site-a --kind jsonl --kind nemo_tar aligned-sharegpt.yaml

Build WDS v2 sidecars, then convert them.  Conversion validates every offset
file and metadata companion against its source before publishing the pack:

.. code-block:: bash

    python scripts/dataloading/build_indexes.py \
      --indexes-root /index-mirror/site-a --kind wds_tar_v2 wds-v2.yaml
    python scripts/dataloading/convert_indexes_to_idxpack.py \
      --indexes-root /index-mirror/site-a \
      --output /immutable-packs/site-a/wds-v2.idxpack wds-v2.yaml

For collection mode, build the JSONL and native-tar offsets before the route.
The command performs that dependency ordering even though the filters appear
on one line:

.. code-block:: bash

    python scripts/dataloading/build_indexes.py --indexes-root /index-mirror/site-a --kind jsonl --kind nemo_tar --kind sharegpt_route collection-sharegpt.yaml
    python scripts/dataloading/convert_indexes_to_idxpack.py \
      --indexes-root /index-mirror/site-a \
      --output /immutable-packs/site-a/multi-clip.idxpack collection-sharegpt.yaml

Do not pass ``--native-tar-paths-only`` for collection mode: routes require the
offset-bearing tar collection.  Add converter ``--overwrite`` only when an
intentional atomic replacement is authorized; normal pack roots are immutable.
The repository submission wrapper uses the corresponding form:

.. code-block:: bash

    python submit_build_indexes.py \
      --cluster site-a --blend collection-sharegpt.yaml \
      --build-packs --pack-with-tar-offsets

The full validator consumes a training recipe, not a leaf-only YAML.  It uses
the production tokenizer, ``SALMDataset``, audio loading, tokenization, and
collation, and fails unless all requested batches materialize.  Output contains
only IDs, counters, and timings, never conversation or audio payloads:

.. code-block:: bash

    torchrun --standalone --nnodes=1 --nproc-per-node=8 \
      scripts/dataloading/validate_dataloader.py \
      --config recipe.yaml --data-blend-dir /data-blends/site-a \
      --output-dir /validation/site-a --phase baseline --run-idx 0 \
      --steps 100 --checkpoint-at -1 --section train_ds \
      --mode full --no-metadata-only

Use the full validator before training; fast mode validates metadata/indexed
iteration but does not prove that audio can be decoded and collated.

Portability and fail-closed behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An ``.idxpack`` stores the configured source strings, ordered shard catalog,
source identities, and collection keys.  It is not cluster-path portable when
local paths or prefix-map destinations change.  Rebuild local tar indexes,
``.sgroute`` files, and packs under a new immutable root for each deployment site.
A pack may be reused only when configured path strings and source identities
are identical, as can occur with shared ``s3://``/``ais://`` object names.
Never rewrite a source JSONL to make an old pack appear portable.

Indexed dataset entries fail closed rather than selecting a sequential reader:

* WDS v2 requires ``indexed: true``; sequential fallback is not allowed.
* Each WDS sample must have contiguous regular members, exactly one JSON member,
  at least one non-JSON payload, unique names, valid object JSON, and one sample
  key.  Duplicate names, non-contiguous sample-key reuse, stale metadata, or a
  corrupt offset digest are errors.
* A requested WDS member must match exactly or by one unambiguous basename.
  A conventional sample with exactly one payload may bind its single declared
  source path to that payload for legacy path-preserving JSON compatibility.
  Otherwise missing members, an ambiguous basename, or two declared paths
  resolving to one member are errors.
* A placeholder without an audio path, a non-string/nested audio list, or
  incompatible path/placeholder cardinality is an error.
* Collection mode requires ``indexed: true``, tar paths, a ``.sgroute``, matching
  route source/row/map digests, and an offset-bearing tar pack collection.
  Missing names, duplicate candidate names, stale routes, paths-only tar
  collections, and unused route records are errors.
* Packed WDS workers use the catalog and packed offsets.  A missing pack or any
  attempt to fall back to recursive shard discovery is an error for production
  indexed configurations.

AIStore GetBatch (multimodal conversations) (experimental)
**********************************************************

`AIStore GetBatch <https://docs.nvidia.com/aistore/get_batch>`_ is a server-side
batched object-fetch API; see the `paper <https://arxiv.org/html/2602.22434v1>`_
for the design and motivation.

For tarred multimodal conversation manifests (``NeMoMultimodalConversationJsonlAdapter``
and ``NeMoMultimodalConversationShareGPTJsonlAdapter``), set the environment variable
``USE_AIS_GET_BATCH=true`` to enable AIStore GetBatch loading:

.. code-block:: bash

    USE_AIS_GET_BATCH=true python examples/speechlm2/salm_train.py \
      --config-name=salm_automodel ...

When enabled:

* The adapters skip opening tar files and instead build URL-backed cuts whose
  ``AudioSource`` points at the per-shard audio location (the JSONL ``value``
  field is trusted to match the tar layout).
* ``SALMDataset`` constructs its loader as
  ``AudioSamples(use_batch_loader=True, fault_tolerant=True, mono_downmix=True)``,
  which issues a single batched fetch per minibatch instead of per-cut reads.
* ``collate_conversation_audio_fault_tolerant`` delegates loading and collation
  to ``AudioSamples`` and drops every conversation whose cuts didn't survive
  the fetch — preserving the legacy fault-tolerant semantics.

Leave the env var unset to keep the original tar-iterating loader.

Combining with ``indexed: true``
""""""""""""""""""""""""""""""""

``USE_AIS_GET_BATCH=true`` coexists with ``indexed: true`` on
``LazyNeMoTarredIterator`` (and on the multimodal-conversation adapters).
Indexed mode keeps the JSONL-driven O(1) global indexing and graph-token
checkpointing, while AIStore GetBatch handles the actual audio fetch:

* The audio-tar ``.idx`` sidecar is **not** required when GetBatch is enabled
  — the iterator skips opening tar files entirely and emits URL-backed cuts
  whose ``AudioSource`` points at ``{tar_path}/{audio_filename}``
  (``type="url"`` for ``ais://...`` paths, ``type="file"`` otherwise).
* Manifest JSONLs still need their ``.idx`` sidecars; they drive the indexed
  iterator graph and the ``state_dict`` / ``load_state_dict`` round-trip.
* Audio bytes are fetched lazily by ``AudioSamples(use_batch_loader=True)`` at
  collation time, which issues one batched GetBatch request per minibatch.

Use this combination when shards live on AIStore and you want both the
network efficiency of GetBatch and the exact-resume guarantees of the
indexed/stateful pipeline.

DuplexSTTDataset
****************

This dataset class is specialized for the DuplexSTTModel, which processes duplex conversational audio and generates text responses. Unlike DuplexS2SDataset which outputs speech, this dataset prepares data for speech-to-text conversion in duplex conversations.

.. code-block:: python

    from nemo.collections.speechlm2.data import DuplexSTTDataset

    dataset = DuplexSTTDataset(
        tokenizer=model.tokenizer,                   # Text tokenizer
        frame_length=0.08,                           # Frame length for audio processing
        source_sample_rate=16000,                    # Audio sample rate
        input_roles=["User"],                        # Roles to use as input
        output_roles=["Assistant"],                  # Roles to generate text for
    )

DataModule
----------

The DataModule class in the speechlm2 collection manages dataset loading, preparation, and batching for PyTorch Lightning training:

.. code-block:: python

    from nemo.collections.speechlm2.data import DataModule
    
    datamodule = DataModule(
        cfg_data,                  # Configuration dictionary for data
        tokenizer=model.tokenizer, # Text tokenizer
        dataset=dataset            # Instance of DuplexS2SDataset, DuplexSTTDataset, or SALMDataset
    )

The DataModule takes care of:

1. Setting up proper data parallel ranks for dataloaders
2. Instantiating the dataloaders with configuration from YAML
3. Managing multiple datasets for validation/testing
4. Persisting the train dataloader's iterator state across checkpoints
   (when ``use_stateful_dataloader: true``)

Checkpointed / resumable training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The DataModule caches the train dataloader on first ``train_dataloader()``
call and exposes ``state_dict()`` / ``load_state_dict()`` that delegate to the
cached dataloader when it supports them. Lightning's trainer wires those into
every checkpoint automatically, so an experiment configured with::

    data:
      train_ds:
        indexed: true
        use_stateful_dataloader: true
        ...

resumes O(1) — sampler RNG, bucketer state, multiplexer choice RNG,
per-source iterator cursors, and per-worker prefetch queues are all restored
exactly without replay.

With a regular ``DataLoader`` (``use_stateful_dataloader`` unset or
``False``) ``state_dict``/``load_state_dict`` become no-ops and resume falls
back to Lhotse's ``_fast_forward()`` replay path.

Two constraints to keep in mind across save/restore:

* ``num_workers`` and ``world_size`` must match between save and restore
  (a hard requirement of ``StatefulDataLoader``).
* All data files must be **uncompressed** and accompanied by ``.idx``
  sidecars. Build them in one shot with ``scripts/dataloading/build_indexes.py``
  (see :ref:`indexed-resumable-dataloading` in the main Lhotse dataloading
  guide).

Bucketing for Efficient Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The DataModule supports bucketing for more efficient training. Bucketing groups samples of similar lengths together, which reduces padding and improves training efficiency. The key bucketing parameters are:

1. **batch_duration**: Target cumulative duration (in seconds) of samples in a batch
2. **bucket_duration_bins**: List of duration thresholds for bucketing
3. **use_bucketing**: Flag to enable/disable bucketing
4. **num_buckets**: Number of buckets to create
5. **bucket_buffer_size**: Number of samples to load into memory for bucket assignment

Example bucketing configuration:

.. code-block:: yaml

    train_ds:
      # ... other settings
      batch_duration: 100  # Target 100 seconds per batch
      bucket_duration_bins: [8.94766, 10.1551, 11.64118, 19.30376, 42.85]  # Duration thresholds
      use_bucketing: true  # Enable bucketing
      num_buckets: 5  # Create 5 buckets
      bucket_buffer_size: 5000  # Buffer size for bucket assignment

When bucketing is enabled:

1. Samples are grouped into buckets based on their duration
2. Each batch contains samples from the same bucket
3. The actual batch size can vary to maintain a consistent total duration
4. The target batch_duration ensures efficient GPU memory usage

Bucketing helps to:
- Reduce padding and increase effective batch size
- Improve training efficiency and convergence
- Manage memory usage with variable-length inputs

Data Configuration
------------------

A typical data configuration in YAML includes:

.. code-block:: yaml

    data:

      train_ds:
        sample_rate: ${data.target_sample_rate}
        input_cfg:
          - type: lhotse_shar
            shar_path: /path/to/train_data
        seed: 42
        shard_seed: "randomized"
        num_workers: 4
        # Optional bucketing settings
        batch_duration: 100
        bucket_duration_bins: [8.94766, 10.1551, 11.64118, 19.30376, 42.85]
        use_bucketing: true
        num_buckets: 5
        bucket_buffer_size: 5000
        # batch_size: 4  # alternative to bucketing
    
      validation_ds:
        datasets:
          val_set_name_0:
            shar_path: /path/to/validation_data_0
          val_set_name_1:
            shar_path: /path/to/validation_data_1
        sample_rate: ${data.target_sample_rate}
        batch_size: 4
        seed: 42
        shard_seed: "randomized"

Note that the actual dataset paths and blend are defined by the YAML config, not Python code. This makes it easy to change the dataset composition without modifying the code.
To learn more about the YAML data config, see :ref:`the Extended multi-dataset configuration format <asr-dataset-config-format>` section in the ASR documentation.

Preparing S2S Datasets
----------------------

Creating Lhotse Manifests
^^^^^^^^^^^^^^^^^^^^^^^^^

To prepare your own dataset, you'll need to create Lhotse manifests from your audio files and transcripts:

.. code-block:: python

    from lhotse import CutSet, Recording, SupervisionSegment
    
    # Create a recording for user and assistant
    recording_user = Recording(
        id="conversation_1_user",
        path="/path/to/audio/conversation_1_user.wav",
        sampling_rate=16000,
        num_samples=171200,
        duration=10.7
    )
    recording_assistant = Recording(
        id="conversation_1_assistant",
        path="/path/to/audio/conversation_1_assistant.wav",
        sampling_rate=22050,
        num_samples=235935,
        duration=10.7
    )
    
    # Create supervision for this recording
    supervisions = [
        SupervisionSegment(
            id="conversation_1_turn_0",
            recording_id="conversation_1",
            start=0,
            duration=5.2,
            text="Can you help me with this problem?",
            speaker="user"
        ),
        SupervisionSegment(
            id="conversation_1_turn_1",
            recording_id="conversation_1",
            start=5.5,
            duration=3.1,
            text="I can help you with that.",
            speaker="assistant"
        ),
    ]
    
    # Create a CutSet
    # The assistant's response is located in target_audio field which makes it easy to replace
    # when using multiple models or speakers for synthetic data generation.
    cut = recording.to_cut()
    cut.supervisions = supervisions
    cut.target_audio = recording_assistant
    cutset = CutSet([cut])
    
    # Save to disk
    cutset.to_file("path/to/manifest.jsonl.gz")

Converting to SHAR Format
^^^^^^^^^^^^^^^^^^^^^^^^^

For efficient training, it's recommended to convert your Lhotse manifests to SHAR (SHarded ARchive) format:

.. code-block:: python

    from lhotse import CutSet
    from lhotse.shar import SharWriter
    
    cutset = CutSet.from_file("path/to/manifest.jsonl.gz")
    cutset.to_shar("path/to/train_shar", fields={"recording": "flac", "target_audio": "flac"}, shard_size=100)
    

Training with Prepared Datasets
-------------------------------

Once your datasets are prepared, you can use them to train a model:

.. code-block:: python

    # Load configuration
    config_path = "path/to/config.yaml"
    cfg = OmegaConf.load(config_path)
    
    # The training data paths are available in the config file:
    # cfg.data.train_ds.input_cfg[0].shar_path = "path/to/train_shar"
    
    # Create dataset and datamodule
    dataset = DuplexS2SDataset(
        tokenizer=model.tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        target_sample_rate=cfg.data.target_sample_rate,
        input_roles=cfg.data.input_roles,
        output_roles=cfg.data.output_roles,
    )
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)
    
    # Train the model
    trainer.fit(model, datamodule)

Example S2S Datasets
--------------------

While there are no publicly available datasets specifically formatted for Duplex S2S models yet, you can adapt conversation datasets with audio recordings such as:

1. Fisher Corpus
2. Switchboard Corpus
3. CallHome
4. Synthetic conversation datasets generated using TTS

You would need to format these datasets as Lhotse manifests with appropriate speaker role annotations to use them with the speechlm2 S2S models.
