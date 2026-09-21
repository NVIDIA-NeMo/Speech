Evaluating multi-speaker (SOT) models
=====================================

Multi-speaker models emit serialized-output-transcription (SOT) text, where a speaker tag such as
``<spk:0>`` precedes each speaker's run. Scoring such a transcript needs cpWER — concatenated
minimum-permutation word error rate — which is permutation-invariant over speakers, so it scores
*what was said* and *who said it* together.

This page covers how to produce a prediction manifest, how to score one, and the settings that
change the number.

.. seealso::

   :doc:`/speechlm2/models` for the models themselves, and :doc:`/speechlm2/datasets` for the
   manifest format they consume.

Two stages, on purpose
----------------------

Evaluation is split into inference and scoring:

.. code-block:: bash

   # 1. inference -- writes a manifest, and scores it inline by default
   python examples/speechlm2/streaming_stt_generate.py \
       pretrained_name=<hf dir> inputs=data/test.json \
       output_manifest=eval/run.jsonl

   # 2. scoring -- re-score that manifest with different settings, no GPU
   python examples/speechlm2/streaming_stt_score.py \
       manifest=eval/run.jsonl cpwer_normalizer=chime8

The split exists because changing *how* scoring works should not mean re-running inference: scoring
is cheap and CPU-only, inference is neither.

Re-running inference *is* reproducible. ``streaming_stt_generate.py`` sets no seed by default, but
two runs of one checkpoint on one machine at identical settings produce byte-identical manifests.
What changes a hypothesis is changing a **setting**. Two settings matter more than they look:

``batch_size``
   The chunked decoder pads every stream's response to the length of the slowest stream in its
   batch, so batch composition reaches the KV cache and moves the text.

``seed``
   Counter-intuitively, setting a seed **changes** results rather than pinning them. ``seed``
   enables ``torch.use_deterministic_algorithms``, and the fused Triton depthwise-striding
   subsampler is gated on that being *off* -- so a seeded run silently takes the PyTorch
   subsampling path while an unseeded run takes the fused one. The two round differently (see the
   warning below), and on a 4-utterance sample two of four transcripts differed. Use ``seed`` to
   make a run reproducible across *machines*, not to reproduce an unseeded run on this one.

Compare runs only at equal settings, and prefer re-scoring an archived manifest, which is exact by
construction.

.. warning::

   Encoder arithmetic is not stable across versions. The fused subsampler carries
   convolution → ReLU → depthwise in fp32 registers and rounds to bf16 **once** at the end, where
   the unfused path materialises each layer and so rounds the intermediate before the depthwise
   reads it. The fused result is the more accurate of the two, but different, and in bf16
   (8-bit mantissa) the difference propagates through every encoder layer. Measured on a
   2912-session corpus when the fused kernel landed: 973 of 2912 hypotheses changed, WER improved
   0.14 pp, and cpWER micro moved +0.03 pp. Numbers recorded before such a change are not directly
   comparable to numbers recorded after it.

.. note::

   Before the fix in ``cache_feature_bufferer.py`` that put its preprocessor in eval mode, this was
   not true: dither was applied during inference and the same audio decoded differently every run.
   Manifests produced before that fix are not reproducible, and numbers recorded from them cannot
   be regenerated.

Scoring inline and scoring offline produce identical numbers — both call the same code, and a test
pins that.

The manifest
------------

``streaming_stt_generate.py`` writes one JSON object per line.

.. list-table::
   :header-rows: 1
   :widths: 24 10 66

   * - Key
     - Type
     - Meaning
   * - ``id``
     - str
     - Lhotse cut id, including an ``-<offset>-<duration>`` suffix. Not the audio stem.
   * - ``duration``
     - float
     - The **padded** duration when ``pad_extra_duration`` is set. Rows are ordered by ascending
       duration, not manifest order.
   * - ``text`` / ``pred_text``
     - str
     - Reference and hypothesis, **normalized and tag-stripped**. Convenient to read; not
       re-scorable (see below).
   * - ``text_raw`` / ``pred_text_raw``
     - str
     - Reference and hypothesis **verbatim**, speaker tags intact. These are what scoring reads.
   * - ``wer`` / ``ins`` / ``del`` / ``sub``
     - float
     - Speaker-agnostic **rates**.
   * - ``wer_errors`` / ``wer_ref_words`` / ``wer_ins`` / ``wer_del`` / ``wer_sub``
     - int
     - The same thing as **counts**. A bare name is a rate, a prefixed name is a count.
   * - ``cpwer``
     - float
     - Per-session cpWER as a **fraction**, or ``null``.
   * - ``cpwer_errors`` / ``cpwer_ref_words`` / ``cpwer_ins`` / ``cpwer_del`` / ``cpwer_sub``
     - int
     - Counts. **Absent** on a row that was not scored — see `Two ways a row has no rate`_.
   * - ``custom``
     - dict
     - The input manifest row, nested whole. Per-subset bucketing reads its key from here.
   * - ``_run``
     - dict
     - What produced the manifest: ``build``, ``head_sha``, ``placement``, ``seg_mode``,
       ``inference_normalizer``, ``oracle_spk_targets`` and the rest.

.. warning::

   ``text`` and ``pred_text`` **cannot** be re-scored. Every Whisper-style normalizer begins by
   deleting ``<...>`` spans, so the speaker tags are gone by the time they are written, and
   re-normalizing an already-normalized string is not the same as normalizing the original —
   ``chime8("it cost one thousand five hundred dollars")`` is seven words, while
   ``chime8(whisper(...))`` is ``"it cost 1500 dollars"``, four. That is a 43% swing in the
   denominator. The scorer refuses a manifest that has only these fields rather than producing a
   wrong number.

   Manifests written before ``text_raw`` existed can be recovered by joining their source manifest::

       python examples/speechlm2/streaming_stt_score.py manifest=eval/old.jsonl \
           reference_manifest=data/test.json hypothesis_field=pred_text_annotated

Scoring axes
------------

Different scorers disagree about what counts as a speaker tag, whether malformed residue is a word,
and what to do with text before the first tag. Each disagreement is an independent axis, so another
scorer's number can be reproduced without adopting its whole pipeline. **Every default reproduces
this repository's historical behaviour**, pinned against a snapshot.

.. list-table::
   :header-rows: 1
   :widths: 32 14 54

   * - Axis
     - Default
     - Effect
   * - ``cpwer_normalizer``
     - inherit
     - ``whisper`` | ``hf`` | ``chime8`` | ``none``. ``None`` inherits ``use_normalizer``.
   * - ``cpwer_tag_syntax_ref`` / ``_hyp``
     - ``spk``
     - ``spk`` (``<spk:N>``) | ``bracket`` (``[sN]``) | ``spk+bracket`` | ``canonical``
       (adds ``speaker N:``). Per role, because a reference and a hypothesis need not be parsed
       the same way.
   * - ``cpwer_tag_case_sensitive``
     - ``True``
     - ``False`` treats ``<SPK:0>`` as a tag rather than as two words.
   * - ``cpwer_untagged_speaker_ref`` / ``_hyp``
     - ``0``
     - Bucket for words before the first tag. ``null`` discards them — and, with no tag at all,
       yields no streams, so the row abstains.
   * - ``cpwer_keep_empty_streams``
     - ``True``
     - ``False`` drops a stream whose **raw** text is empty, before normalization.
   * - ``cpwer_drop_tag_residue``
     - ``True``
     - ``False`` scores malformed residue such as an unclosed ``<spk:0`` as text.
   * - ``cpwer_speaker_order``
     - ``index``
     - ``first_seen`` orders streams by first appearance. Changes only the ins/del/sub split,
       never the error total or the reference word count.
   * - ``cpwer_ceiling_source``
     - ``strip_tags``
     - How the no-tag ceiling's pseudo-hypothesis is built. Affects the ceiling only.

``cpwer_placement`` (``prefix`` | ``suffix``) and ``cpwer_max_speakers`` are settings rather than
axes: they describe the data, and no reference scorer has an equivalent.

The normalizer is usually the one that matters. On a 2912-session conversational corpus, switching
from ``whisper`` to ``chime8`` moved cpWER by 0.50 pp, while flipping all six parser-shaped axes at
once moved it by 0.0000 pp — that corpus contains no bracket tags, no ``speaker N:`` spellings and
no untagged references, so those axes were inert on it. They are insurance against model-output
drift, not a fix.

Reading the report
------------------

.. code-block:: text

   cpwer_axes: cpwer_ceiling_source='strip_tags',cpwer_drop_tag_residue=True,...
   reference-comparable: no (2 of 9 comparable axes match)
       differs: cpwer_normalizer='whisper' -> reference uses 'chime8'
   WER: 24.95% [normalizer=whisper]
   cpWER (micro): 30.24%
   cpWER (macro): 31.04%
   cpWER (subset-macro): 34.31%
   cpWER no-tag ceiling: 43.81% (a word-perfect but unattributed hypothesis) [source=strip_tags]
      1spk reference speakers: 19.83%
      4spk reference speakers: 45.73%
     under20s-ami-ihm-test            25.88%  n=365
     sessions 2912  admitted 2912  abstained 0  zero-ref-words 0  empty-hyp 2  untagged-hyp 82

Three different averages appear, and they answer different questions:

``cpWER (micro)``
   Total errors over total reference words. The headline: long sessions count more.

``cpWER (macro)``
   Unweighted mean over sessions. A short session counts as much as a long one.

``cpWER (subset-macro)``
   Unweighted mean over subsets. Prevents one large corpus from dominating.

``cpWER no-tag ceiling``
   What a word-perfect but completely unattributed hypothesis would score. Computed from the
   reference alone, so it is model-independent and comparable across runs — useful as a fixed
   anchor. It is **not** 100%: the solver still credits the flat stream's best-matching speaker.

The ``cpwer_axes`` stamp and the ``reference-comparable`` verdict record which settings produced the
number, so a scored manifest can be checked later without the original command line. The verdict
ranges over nine axes; ``cpwer_ceiling_source`` is excluded because the reference scorer has no
ceiling and so cannot have an opinion about it.

Scale is split by artifact, never by flag. Per-row manifest fields are **fractions**; the console,
the log and ``metrics.json`` are **percent**, and ``metrics.json`` carries an explicit ``"scale"``
key plus unrounded ``*_fraction`` copies so a re-pool never has to undo rounding.

Two ways a row has no rate
--------------------------

Both leave ``cpwer`` without a number, for different reasons, and they are distinguished by **key
presence** rather than by a sentinel:

*Abstained* — the reference parsed to zero speaker streams, so there was nothing to align against.
The row is not scored: ``cpwer`` is ``null`` and the five count keys are **absent**. Counted in
``cpwer_abstained``.

*Zero reference words* — the reference parsed into streams that normalize to nothing, for instance
a turn consisting only of filler. The row **is** scored: ``cpwer`` is ``0.0`` and all five counts
are present. Its errors pool into the micro numerator against a zero denominator contribution, so a
corpus of such rows can exceed 100%, and a literal ``0.0`` joins the macro. Counted in
``cpwer_zero_ref_words``.

All counters are printed even at zero: a counter that appears only when non-zero cannot be told
apart from one that was never computed.

Long-form audio
---------------

cpWER cannot be scored over segmented inference. ``<spk:N>`` is arrival-ordered *within* each decode
window, so ``<spk:0>`` in one segment is generally a different person than in the next, and no
session-global permutation can undo a per-segment relabeling. Setting ``max_segment_duration``
together with inline cpWER raises; the scorer refuses the resulting manifest for the same reason.

Segmented inference can still write a manifest — set ``score_inline=false`` — and its WER is
unaffected.
