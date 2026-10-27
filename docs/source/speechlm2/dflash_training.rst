DFlash and DFlash2 draft training
=================================

Use ``examples/speechlm2/dflash_train.py`` with ``conf/dflash.yaml`` to train a
compact DFlash or DFlash2 draft against a frozen, audio-conditioned
``SALMAutomodel`` target. The YAML inherits SALM's target, data, and trainer
defaults from ``salm_automodel.yaml`` and adds the ``dflash`` settings. Choose
``dflash.variant`` (``dflash`` or ``dflash2``) and provide a reserved
``dflash.mask_token_id``. The entry point trains and exports draft-only weights while
preserving SALM's audio-placeholder expansion. The shipped DFlash2 settings mirror Automodel's
recipe, including its two-tap grouped dynamic convolution, top-16 rank-256 path selector, and
separately normalized backbone and selector losses. ``num_anchors`` limits sampled blocks per row;
packed THD batches use one row per rank. For non-packed BSHD batches, up to
``batch_size * num_anchors`` blocks are allocated per rank; reduce ``num_anchors`` when using larger batches.
Packed-document boundaries are enforced by document-local anchor selection, attention masks, and position IDs.
Both variants use dense draft logits (DFlash2 needs them for candidate selection), with
``use_fused_linear_ce=false``. This integration supports BSHD and packed THD batches with
``tp_size=pp_size=cp_size=1`` and does not directly load the published packed NVFP4 inference
checkpoint into BF16 training modules.
Target features are captured with SALM's original padding; before the draft forward,
leading padding is moved to the end of each row so it cannot enter a draft context.

``trainer.precision`` controls both target and draft dtypes during training. For BF16
training, either the shipped ``bf16-flash`` or ``bf16-automodel`` trainer precision
is supported. Target inference backend compatibility is validated before constructing the draft.
The frozen target runs in the configured parameter dtype without TE FP8 autocast.
Packed decoder inputs are supported; packed audio-encoder execution
(``model.packed_encoder_sequences`` / ``model.packed_encoder_cp``) is not enabled by this entry point.

``dflash.lr`` defaults to 6e-4 with AdamW and no scheduler. To reproduce an adaptation
schedule, supply ``dflash.optimizer`` and ``dflash.lr_scheduler`` explicitly; for example:

.. code-block:: yaml

   dflash:
     optimizer:
       _target_: torch.optim.AdamW
       lr: 1.5e-4
       betas: [0.9, 0.999]
       eps: 1e-8
       weight_decay: 0.0
     lr_scheduler:
       _target_: nemo.core.optim.lr_scheduler.CosineAnnealing
       warmup_steps: 500
       min_lr: 1e-6
       max_steps: 8000

For a frozen target initialized from a Hugging Face directory after FSDP sharding,
``model.init_from_checkpoint_strict=true`` (default) requires every model parameter
in that checkpoint. Set it to ``false`` only for intentional partial initialization.

To warm-start a draft, set ``dflash.init_from_pretrained`` to a local HuggingFace
draft export matching the configured architecture. Loading rejects missing,
unexpected, or mismatched weights. This initializes draft weights only; optimizer,
scheduler, and step counters start fresh. The frozen target is still selected
independently with the model's checkpoint settings.

The shipped recipe defaults to ``dflash.label_source=target_argmax``, supervising
the draft with the frozen backbone's greedy next-token predictions under the
original, teacher-forced context. Set ``dflash.label_source=ground_truth`` to use
reference tokens instead. These are hard labels, not soft-logit distillation or generated
rollouts. The input tokens and audio-conditioned hidden features remain the
original context. Labels use the final-normalized state at the preceding token;
packed document boundaries and padding never supply a predecessor. Only
supervised positions are projected, in chunks bounded by
``dflash.target_argmax_chunk_size`` (default 128), so full-sequence vocabulary
logits are not retained. Draft features still use pre-final-normalization layer
outputs. Validation uses the same configured label source.

Offline model-generated labels are also supported through the dataset-label path.
Put the generated transcripts in the dataset's supervised assistant responses and
set ``dflash.label_source=ground_truth``. Here ``ground_truth`` means the supplied
dataset tokens; they may be human references or model predictions. For ordinary
``lhotse_as_conversation`` ASR data, the response comes from
``cut.supervisions[0].text`` (NeMo ASR manifest ``text`` before conversion).
Preserve audio alignment and prompt context, and write a separate manifest so the
original references remain available for evaluation. For example, with the data
configuration pointing to the generated-label manifest:

.. code-block:: yaml

   dflash:
     variant: dflash2
     label_source: ground_truth

The target still computes hidden features online, now under the generated response
context. This uses the same training path as reference labels. Selecting
``target_argmax`` instead recomputes labels under that context and overrides the
supplied response tokens as supervision. See
``.claude/skills/train-dflash/SKILL.md`` for the generated-label preparation workflow.

For projection-only adaptation of a pretrained DFlash2 draft, set
``dflash.projection_only=true`` with ``dflash.init_from_pretrained``. Only the
input feature projection ``fc.weight`` is trainable; the target, draft decoder,
norms, convolutions, and candidate selector remain frozen. Gradients still flow
through the frozen draft into the projection, including with activation
checkpointing. The optimizer contains only trainable parameters, while draft
checkpoints and Hugging Face exports retain all draft weights.

For example, to adapt a compatible block8 DFlash2 export to block16:

.. code-block:: yaml

   dflash:
     variant: dflash2
     init_from_pretrained: /path/to/block8-draft
     block_size: 16
     label_source: target_argmax
     target_argmax_chunk_size: 128
     projection_only: true
     use_fused_linear_ce: false

Use this as an override to ``dflash.yaml``, preserving the
warm-start architecture and target-layer taps. The configured block size is
used for training and saved in the exported draft config; strict loading still
requires every weight shape to match. Set ``projection_only=false`` to train
the full draft. A weights-only warm start can change block size; resuming a
training checkpoint instead requires the same optimizer parameter selection.

Launching training
------------------

Supply the same target and data overrides used for SALM training. For example,
with a compatible BF16 target checkpoint and pretrained draft:

.. code-block:: bash

   torchrun --nproc_per_node=8 examples/speechlm2/dflash_train.py \
     --config-name=dflash \
     model.init_from_checkpoint=/path/to/target-checkpoint \
     dflash.init_from_pretrained=/path/to/compatible-draft \
     dflash.block_size=16 \
     dflash.label_source=target_argmax

Use ``+run_validate_only=true`` for validation without training. Ordinary SALM
training continues to use ``salm_train.py`` and ``salm_automodel.yaml``.
