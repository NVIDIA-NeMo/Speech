#!/usr/bin/env python
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=C0116
# pylint: disable=C0301
import importlib
import math
import sys
from numbers import Number

import click
import pytorch_lightning as pl
import torch
from lhotse import compute_num_samples
from omegaconf import OmegaConf

from nemo.collections.asr.models.asr_model import ASRModel
from nemo.collections.common.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, MaskType, NeuralType
from nemo.utils import logging


class ProfilingBatchGenerator:
    """
    ProfilingBatchGenerator is used to generate artificial mini-batches for model training
    and tracking the progress of batch size optimization.

    The high-level usage API is the following::

        >>> gen = ProfilingBatchGenerator(schema)
        ... finished = False
        ... while not finished:
        ...     batch = gen(input_seq_len, output_seq_len)
        ...     try:
        ...         training_step(model, batch)
        ...         oom = False
        ...     except torch.cuda.OutOfMemoryError:
        ...         oom = True
        ...     finished = gen.advance(oom)
        ... solution = gen.max_batch_size  # The solution of the search problem.
        ... gen.reset()  # Can re-use for other sequence lengths now.

    The search terminates once the difference between max working batch size and min OOM batch size
    divided by the latter is smaller than ``rel_gap_thresh`` that difference amounts to a single element.
    For example, a max working batch size is 96 and min OOM batch size is 100 indicates a gap of 0.04,
    which would terminate the search with threshold of 0.05.

    In order to generate mini-batches compatible with a given model, the generator:

    * accepts a ``schema`` argument in its constructor, and

    * accepts input/output sequence lengths in each call to generate a mini-batch.

    ``schema`` has the following structure::


        >>> {
        ...     "cls": tuple | MyBatchType,
        ...     "inputs": [
        ...         {
        ...             "type": NeuralType(...) | Literal["dummy"],
        ...             "seq_length": Literal["input", "output"],
        ...             "vocab_size": int,  # optional, required only for LabelsType
        ...             "name": str,  # optional, indicates kwarg
        ...         },
        ...         ...,
        ...     ]
        ... }

    ``cls`` indicates how we should construct the mini-batch. Typically you can just use ``tuple`` for most
    batch schemas. However, if the model expects a specific, e.g., dataclass, you can tell ``ProfilingBatchGenerator``
    to use it. The mini-batch object will be constructed using the items in ``inputs``.

    Each element of ``inputs`` specifies a NeMo NeuralType which needs to have a defined ``elements_type``.
    The supported types are ``AudioSignal``, ``LengthsType`` and ``LabelsType``.
    If "type" is not a NeuralType, we interpret that as a placeholder tensor that's not relevant but expected
    by the model/batch constructor. In addition, ``"seq_length"`` key is used to determine whether we should apply
    input or output sequence length to a given tensor.

    Optional keys:

    * ``vocab_size`` is required for ``LabelsType`` so that we can generate proper label values.

    * ``name`` is required if objects of ``cls`` have to be constructed using keyword arguments.

    A simple schema example for a model using audio/lengths tensor pair (unsupervised/self-supervised)::

        >>> {
        ...     "cls": tuple,
        ...     "inputs": [
        ...         {"type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
        ...         {"type": NeuralType(("B"), LengthsType()), "seq_length": "input"},
        ...     ]
        ... }

    """

    def __init__(
        self,
        schema: dict,
        start_batch_size: int = 32,
        rel_gap_thresh: float = 0.05,
        device: str = "cuda",
    ):
        self.schema = schema
        self.start_batch_size = start_batch_size
        self.rel_gap_thresh = rel_gap_thresh
        self.device = device
        self.reset()

    def __call__(self, input_seq_length: int, output_seq_length: int):
        B = self._current
        select_seq_length = {"input": input_seq_length, "output": output_seq_length}
        batch = []
        names = []
        for item in self.schema["inputs"]:
            nt = item["type"]
            if isinstance(nt, str) and nt == "constant":
                if isinstance(val := item["value"], str) and val == "batch":
                    tnsr = torch.tensor([B], dtype=torch.long, device=self.device)
                else:
                    tnsr = torch.tensor([val], dtype=torch.long, device=self.device)
            elif not isinstance(nt, NeuralType):  # placeholder
                tnsr = torch.tensor([])
            elif isinstance(nt.elements_type, AudioSignal):
                seq_length = select_seq_length[item["seq_length"]]
                tnsr = torch.randn(B, seq_length, dtype=torch.float32, device=self.device)
            elif isinstance(nt.elements_type, LengthsType):
                seq_length = select_seq_length[item["seq_length"]]
                tnsr = torch.ones(B, dtype=torch.long, device=self.device) * seq_length
            elif isinstance(nt.elements_type, MaskType):
                seq_length = select_seq_length[item["seq_length"]]
                tnsr = torch.ones(B, seq_length, device=self.device)
            elif isinstance(nt.elements_type, LabelsType):
                seq_length = select_seq_length[item["seq_length"]]
                tnsr = torch.randint(0, item["vocab_size"], size=(B, seq_length), device=self.device)
            else:
                raise RuntimeError("Unexpected item in oomptimizer schema: {item}")
            batch.append(tnsr)
            names.append(item.get("name"))
        args = [elem for name, elem in zip(names, batch) if name is None]
        kwargs = {name: elem for name, elem in zip(names, batch) if name is not None}
        if not kwargs and self.schema["cls"] == tuple:
            return tuple(args)
        return self.schema["cls"](*args, **kwargs)

    @property
    def max_batch_size(self) -> int | None:
        """
        Return the solution of the batch size search problem.
        It will keep returning None until the search is done.
        """
        if (
            self._max_ok is not None
            and self._min_err is not None
            and (self.current_rel_gap <= self.rel_gap_thresh or self._min_err - self._max_ok <= 1)
        ):
            return self._max_ok
        return None

    @property
    def current_rel_gap(self) -> float | None:
        """
        Return the current gap between the largest batch that works and the smallest batch that triggers OOM.
        The gap is defined as the batch size difference divided by the larger element.
        E.g., if the best found batch size is 95 and the smallest that triggers OOM is 100, the gap is 0.05.
        """
        if self._min_err is None or self._max_ok is None:
            return None
        return (self._min_err - self._max_ok) / self._min_err

    def reset(self):
        """Reset the generator to prepare it for a new search."""
        self._current = self.start_batch_size
        self._max_ok = None  # max batch size that works
        self._min_err = None  # min batch size that doesn't work

    def advance(self, oom: bool) -> bool:
        """
        Adjusts the current batch size based on the outcome.
        Returns a bool indicating whether the calibration is complete.
        """
        if self.max_batch_size is not None:
            return True

        if oom:
            # Training step failed with OOM.
            # Update the minimum known batch size that causes an error.
            self._min_err = min(float("inf") if self._min_err is None else self._min_err, self._current)
            # Training step failed on OOM
            if self._max_ok is None:
                # We haven't found a batch size that works yet, keep going 2x down.
                self._current = round(self._current / 2)
            else:
                # Try the middle-point between the known extremes.
                self._current = round((self._max_ok + self._min_err) / 2)
        else:
            # Training step successful.
            # Update the maximum known batch size that works.
            self._max_ok = max(-1 if self._max_ok is None else self._max_ok, self._current)
            if self._min_err is None:
                # We haven't found a batch size that causes an error yet, keep going 2x higher
                self._current *= 2
            else:
                # Try the middle-point between the known extremes.
                self._current = round((self._max_ok + self._min_err) / 2)

        if self._current == 0:
            raise RuntimeError(
                "We diverged and arrived batch_size=0. Perhaps the input is too large for this model and hardware."
            )

        return False


class FloatList(click.Option):
    """Support passing bucket duration bins as [1.1,2.5,5.6,...]"""

    name = "list[float]"

    def type_cast_value(self, ctx, value):
        if isinstance(value, list) and all(isinstance(v, float) for v in value):
            return value
        try:
            import ast

            ans = ast.literal_eval(value)
            if isinstance(ans[0], list):
                ans = [tuple(item) for item in ans]
            return ans
        except ValueError:
            raise click.BadParameter(value)


@click.command(context_settings={'show_default': True})
@click.option(
    "-n",
    "--pretrained-name",
    type=str,
    default=None,
    help="Name of a pretrained model to use, e.g. 'nvidia/canary-1b'.",
)
@click.option(
    "-m",
    "--module-name",
    type=str,
    default=None,
    help="Full path to NeMo's module corresponding to CONFIG_PATH, e.g. 'nemo.collections.asr.models.EncDecMultiTaskModel'.",
)
@click.option(
    "-c", "--config-path", type=str, default=None, help="Path to the training configuration file for MODULE_NAME."
)
@click.option(
    "--schema",
    type=str,
    default="audio",
    help="Which schema to use (typically used for choosing the modality, i.e., 'audio' / 'text'",
)
@click.option(
    "-b",
    "--buckets",
    cls=FloatList,
    default=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
    help="List of upper-bound bucket bins (i.e. first bucket is [0.0 - item0), second bucket is [item0 - item1), etc.). "
    "We also support a nested list for 2D bucketing, e.g. [[2.0, 10],[2.0,20],[4.5,15],[4.5,30],...], "
    "where each item is a pair of (max_input_seq_len, max_output_seq_len) for a given bucket.",
)
@click.option(
    "-t",
    "--threshold",
    type=float,
    default=0.05,
    help="Search stopping criterion in range [0, 1], lower is more precise. Interpret as the uncerainty gap, i.e. (min_oom_batch_size - max_ok_batch_size) / min_oom_batch_size.",
)
@click.option("-s", "--start-batch-size", type=int, default=32, help="Initial batch size to start the search from.")
@click.option(
    "-r",
    "--ratio",
    type=int,
    default=12,  # conservative estimate towards longer transcripts
    help="The output_sequence_length to input_sequence_length ratio for the purpose of determing the maximum output sequence lengths. "
    "The interpretation depends on input and output modalities. Examples: for audio->text it's tokens per second. "
    "For text->audio it's seconds per token. For audio->audio it's output seconds per input second. "
    "For text->text it's output tokens per input token. "
    "In general larger ratio means longer output sequences and increased memory consumption. "
    "The default value is set adequately for automatic speech recognition. "
    "This argument is ignored when 2D buckets are provided to --buckets option. "
    "For GPT-style models, use --ratio=1 ",
)
@click.option(
    "-f",
    "--memory-fraction",
    type=float,
    default=0.9,
    help="Limits the use of CUDA memory for this process to MEMORY_FRACTION of the total device memory. "
    "By default we force 5% memory to be unused to account for non-training-loop related CUDA memory usage"
    "in actual training scripts.",
)
@click.option(
    "-d",
    "--device",
    default="cuda:0",
    help="Device string to be passed to torch.device; due to MEMORY_FRACTION option, "
    "it must specify the device index (e.g. cuda:0). "
    "You can also leave the default index and select a specific GPU using env var CUDA_VISIBLE_DEVICES=<idx>",
)
@click.option(
    "-y",
    "--dtype",
    default="bfloat16",
    help="Float precision to use for computation (used together with autocast).",
)
@click.option(
    "--ddp/--no-ddp",
    type=bool,
    default=True,
    help="Whether we should simulate DDP GPU RAM usage. Stores an extra copy of the model in GPU memory. Enabled by default.",
)
def oomptimizer(
    pretrained_name: str | None,
    module_name: str | None,
    config_path: str | None,
    schema: str,
    buckets: list[float],
    threshold: float,
    start_batch_size: int,
    ratio: int,
    memory_fraction: float,
    device: str,
    dtype: str,
    ddp: bool,
):
    """
    OOMptimizer finds the optimal batch sizes for training your model with bucketing dataloading.
    It performs a search over batch sizes until it converges by measuring the GPU memory usage for
    a model's training step and optimizer update.

    \b
    There are two main usage patterns: for using a pretrained model or an untrained model configuration.
    The latter is more flexible but requires the user to provide two separate arguments. Examples:
    * python oomptimizer.py --pretrained-name nvidia/canary-1b
    * python oomptimizer.py --module-name nemo.collections.asr.models.EncDecMultiTaskModel \
        --config-path examples/asr/conf/speech_multitask/fast-conformer_aed.yaml

    Dynamic bucketing is notoriously difficult to tune as you risk running into CUDA OOM many steps into the training.
    In order to simplify finding the optimal settings, OOMptimizer scans each bucket to find the maximum possible
    batch size that doesn't trigger a CUDA OOM.

    \b
    The suggested workflow is the following:
    1) Run scripts/speech_recognition/estimate_duration_bins.py to get the duration distribution of your data.
        (consider running estimate_duration_bins_2d.py for models with a strong dependency on output sequence length
        such as attention-encoder-decoder models).
    2) Run OOMptimizer to find the optimal batch sizes for your specific model, optimizer, and GPU.
    3) Use these optimal settings in your actual training script and enjoy optimal GPU utilization OOM-free.

    In the unlikely event that OOMptimizer bucket batch sizes are still leading to OOMs,
    please try a lower setting of the MEMORY_FRACTION option, e.g. 0.75 (75% of GPU memory).
    This may be required in very complex setups where there are additional GPU RAM loads that can't be anticipated
    through the combination of training_step and optimizer update.
    """
    if all(opt is None for opt in (pretrained_name, module_name, config_path)):
        click.secho(
            "You need to provide either PRETRAINED_NAME or the pair of MODULE_NAME and CONFIG_PATH.", fg="yellow"
        )
        sys.exit(1)
    logging.setLevel(logging.CRITICAL)
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device)

    model_clones = []
    for _ in range(2 if ddp else 1):
        if pretrained_name is not None:
            assert (
                config_path is None and module_name is None
            ), "--pretrained-name cannot be used together with --module-name/--config-path"
            click.echo(f"Intializing ASR model from pretrained checkpoint {pretrained_name}.")
            trainer = pl.Trainer(barebones=True)
            trainer.log_every_n_steps = 1000000
            model = ASRModel.from_pretrained(pretrained_name, trainer=trainer).to(device)
        else:
            assert config_path is not None, "--module-name requires --config-path to be specified as well."
            assert module_name is not None, "--config-path requires --module-name to be specified as well."
            cfg = OmegaConf.load(config_path)
            trainer = MegatronLMPPTrainerBuilder(cfg).create_trainer()
            trainer.log_every_n_steps = 1000000
            namespace, name = module_name.rsplit('.', maxsplit=1)
            model_cls = getattr(importlib.import_module(namespace), name)
            model = model_cls.restore_from_pretrained_models(cfg, trainer=trainer).to(device)
            model.log = lambda *args, **kwargs: None
        model_clones.append(model)
    model = model_clones[-1]
    model.init_consumed_samples = 0
    model._compute_consumed_samples_after_training_step = lambda *args, **kwargs: 1

    from megatron.core.parallel_state import initialize_model_parallel
    from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_model_parallel_for_nemo

    initialize_model_parallel_for_nemo(
        world_size=1, global_rank=0, local_rank=0, micro_batch_size=16, global_batch_size=16
    )
    torch.distributed.init_process_group("nccl", world_size=1, rank=0)
    initialize_model_parallel()

    if not hasattr(model, "oomptimizer_schema"):
        click.secho(
            f"We read model of type {type(model)} which doesn't seem to support OOMptimizer "
            f"(we could not find the property .oomptimizer_schema).",
            fg="red",
        )
        sys.exit(1)

    schema = model.oomptimizer_schema(schema)

    click.echo("Setting up the optimizers.")
    optimizer = model.configure_optimizers()
    if isinstance(optimizer, tuple):
        optimizer = optimizer[0][0]

    # warmup - preallocate model/optimizer memory for all modality modules
    for sch_ in ("text", "audio"):
        gen_ = ProfilingBatchGenerator(model.oomptimizer_schema(sch_), start_batch_size=1)
        with torch.autocast("cuda", getattr(torch, dtype)):
            if sch_ == "audio":
                batch_ = gen_(17519, 13)
            else:
                batch_ = gen_(9, 7)
            optimizer.zero_grad()
            out = model.training_step(iter([batch_]))
            optimizer.step()

    is_2d_bucketing = all(
        isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(v, Number) for v in item)
        for item in buckets
    )
    # Determine modality for input and output.
    modalities = [
        (
            "text"
            if any(
                isinstance(item["type"].elements_type, LabelsType) and item["seq_length"] == direction
                for item in schema["inputs"]
                if not isinstance(item["type"], str)
            )
            else "audio"
        )
        for direction in ("input", "output")
    ]

    def get_max_seq_lens(buckets):

        def _determine_lens_for_bucket(bin):
            if is_2d_bucketing:
                input_len, output_len = bin
            else:
                input_len = bin
                output_len = math.ceil(ratio * input_len)
            sampling_rate = getattr(
                model, "sample_rate", 16000
            )  # TODO: may need to extend schema for broader model coverage
            match modalities:
                case "audio", "audio":
                    return (
                        compute_num_samples(input_len, sampling_rate=sampling_rate),
                        compute_num_samples(output_len, sampling_rate=sampling_rate),
                    )
                case "audio", "text":
                    return (compute_num_samples(input_len, sampling_rate=sampling_rate), output_len)
                case "text", "audio":
                    return (
                        input_len,
                        compute_num_samples(output_len, sampling_rate=sampling_rate),
                    )
                case "text", "text":
                    return input_len, output_len
                case _:
                    raise RuntimeError(f"Unexpected modality combination: {_}")

        return [_determine_lens_for_bucket(bin) for bin in buckets]

    click.echo("Starting profiling.")
    max_seq_lens = get_max_seq_lens(buckets)
    gen = ProfilingBatchGenerator(schema=schema, start_batch_size=start_batch_size, rel_gap_thresh=threshold)
    profile = {}

    # Iterate buckets from the largest to the smallest sequences. This usually ends up creating
    # a tiny bit smaller batches, likely due to worse memory fragmentation.
    with torch.autocast("cuda", getattr(torch, dtype)):
        for bucket, (seq_len_in, seq_len_out) in reversed(list(zip(buckets, max_seq_lens))):
            click.echo(f"The current sequence lengths are: input={seq_len_in} output={seq_len_out}.")
            gen.reset()
            batch_idx = 0

            def step():
                click.echo(
                    f"\t[BEGIN step] [CUDA RAM CURRENT: {torch.cuda.memory_allocated() / (1024 * 1024):.1f}MB] [CUDA RAM MAX: {torch.cuda.max_memory_allocated() / (1024*1024):.1f}MB]"
                )
                batch = gen(seq_len_in, seq_len_out)
                oom = False
                try:
                    click.echo(
                        f"\tCurrent settings | batch_size={gen._current} | gap: {gen.current_rel_gap}... ", nl=False
                    )
                    optimizer.zero_grad()
                    # In SpeechLLM training_step performs both forward and backward; no need for manual backward
                    out = model.training_step(iter([batch]))
                    optimizer.step()
                except torch.cuda.OutOfMemoryError as e:
                    click.secho(f"OOM!", fg="yellow")
                    oom = True
                except RuntimeError as e:
                    if "cuFFT error: CUFFT_INTERNAL_ERROR" not in str(e):
                        raise
                    click.secho(f"OOM!", fg="yellow")
                    oom = True
                else:
                    click.secho(f"OK!", fg="green")
                finally:
                    click.echo(
                        f"\t[END step] [CUDA RAM CURRENT: {torch.cuda.memory_allocated() / (1024 * 1024):.1f}MB] [CUDA RAM MAX: {torch.cuda.max_memory_allocated() / (1024*1024):.1f}MB]"
                    )
                    del batch
                    # Note: We could call empty_cache() to free up some more memory on the GPU,
                    #       but we have found out empirically that this causes a mismatched condition
                    #       between OOMptimizer and the actual training. During training, there is some
                    #       degree of memory fragmentation and it's better to simulate that in OOMptimizer.
                    # torch.cuda.memory.empty_cache()
                    torch.cuda.reset_max_memory_allocated()
                return oom

            oom = step()
            while not (finished := gen.advance(oom)):
                click.echo("\t" + "=" * 80)
                oom = step()

            click.secho(
                f"=> Optimal setting for bucket={bucket} (input={seq_len_in} output={seq_len_out}) is max_batch_size={gen.max_batch_size}",
                fg="green",
            )
            profile[(bucket, seq_len_in, seq_len_out)] = gen.max_batch_size
            gen.start_batch_size = gen.max_batch_size * 2

    # Reverse the profile to be ascendingly sorted again.
    profile = dict(reversed(list(profile.items())))

    click.echo("The 1st stage profile is:")
    for (bucket, seq_len_in, seq_len_out), bs in profile.items():
        click.echo(f"Bucket={bucket} (input={seq_len_in} output={seq_len_out}) => max_batch_size={bs}")

    if is_2d_bucketing:
        # 2D bucketing doesn't support bucket merging.
        final_profile = [["[" + ",".join(map(str, b)) + "]", bs] for (b, _, __), bs in profile.items()]
        max_input_len, max_output_len = buckets[-1]
        ratio = max_output_len / max_input_len
    else:
        click.echo("Bucket merging stage...")
        final_profile = []
        for idx, ((bucket, seq_len_in, seq_len_out), bs) in enumerate(profile.items()):
            if idx == 0:
                final_profile.append([bucket, bs])
                continue
            if bs == final_profile[-1][1]:
                click.echo(f"Merging bucket {idx} with bucket {idx-1} due to identical batch sizes.")
                final_profile[-1][0] = bucket
                continue
            final_profile.append([bucket, bs])
        max_input_len = final_profile[-1][0]

    click.secho(f"The profile was created with the following settings:")
    click.secho(f"* using {memory_fraction:.1%} of available GPU RAM.")
    click.secho(f"* {'' if ddp else 'not '}simulating DDP memory overhead.")
    click.secho(f"* using AMP with dtype={dtype}.")
    click.secho("The final profile is:", bold=True)
    click.secho("\tbucket_duration_bins=[" + ",".join(str(seqlen) for seqlen, bs in final_profile) + "]", bold=True)
    click.secho("\tbucket_batch_size=[" + ",".join(str(bs) for seqlen, bs in final_profile) + "]", bold=True)
    click.secho("\t(The following flags are suitable for ASR/speech-to-text models):")
    click.secho(f"\tmax_tps={ratio}", bold=True)
    click.secho(f"\tmax_duration={max_input_len}", bold=True)


if __name__ == "__main__":
    oomptimizer()
