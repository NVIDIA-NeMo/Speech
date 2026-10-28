# Data, prompts and rollout signal

## The training row

Use Parquet with one recording/instruction per row. A minimal WER row is:

```json
{
  "id": "train-0001",
  "messages": [{"role": "user", "content": "Transcribe the audio."}],
  "audio_filepath": "/shared/audio/train-0001.wav",
  "instruction": "Transcribe the audio.",
  "expected_answer": "the reference transcript",
  "language": "en",
  "task": "ordinary_asr",
  "reward_metric": "wer",
  "responses_create_params": {"max_output_tokens": 512}
}
```

Read `$RL/nemo_rl/data/asr.py` and `$RL/nemo_rl/data/datasets/response_datasets/response_dataset.py` at the selected revision.
The `messages` column preserves the extra fields through `ResponseDataset`; omitting it can discard audio/metadata.
The ASR processor renders **`instruction + "\n<|audio|>"`**, with no system message or reference in the prompt.
Keep `messages` and `instruction` consistent. Reference text belongs only in reward metadata.

Audio must be nonempty, readable by SoundFile and 16 kHz; the processor averages channels. It reads the entire file:
**an `offset`/`duration` field does not crop it**. Materialize intended segments first, preserving parent-recording ID,
offset and content hash in the selection ledger. Reject mismatched transcripts, missing audio and accidental overlaps.
Do not silently truncate long recordings or discard them merely to improve throughput.

`responses_create_params.max_output_tokens` is a positive integer or null. Null inherits the global output ceiling.
The effective allowance is bounded by the row cap, global cap and remaining expanded context. Check finished outputs
and model tokenization when choosing a cap; reference length is a guide, not permission to cut every answer at that length.

Keep duration, source, accent/language, speaker/recording ID, audio identity, prompt ID and selection reason in a sidecar.
These fields support deduplication and balanced selection even when the processor does not forward them to the reward.
Shuffle the consumer deterministically: the reference recipe has `data.shuffle=false`.

## Reward contract

For word-based ASR, reuse `$RL/nemo_rl/environments/asr_reward.py:asr_reward` for mining and training. It returns
**1 − fractional WER**, without clipping negative rewards. Its normalizer preserves Indic combining marks. It is not
automatically the benchmark normalizer. Save both training reward and evaluation error when they differ.

An empty reference uses the count of hypothesis words as its error. Include nonspeech only if it is a target task,
and balance it with speech on which the model sometimes omits words or returns empty output. A mostly-empty-output
dataset can teach a bad shortcut. A completed hallucination is a legitimate poor alternative, not a row to erase.

Mandarin CER in the public RL scorer calls `nemo_skills.evaluation.evaluator.audio.evaluate_cer` with
`whitespace_mode="han_adjacent"`. This optional scoring dependency is distinct from the Skills launcher/PR.
Use a public Skills release/main revision that provides that API, or implement the customer's CER contract as a tested
reward adapter. Test the behavior, not just importability: a function accepting `**kwargs` may ignore a new option.
For example, the intended rule treats `今 天 use AI` and `今天use AI` alike while preserving English word boundaries.
Do not install an unmerged Skills feature branch or silently score Mandarin with whitespace WER.
The built-in Mandarin path requires `task="ordinary_asr"`, `reward_metric="cer"` and
`cer_whitespace_mode="han_adjacent"`, and a nonempty normalized reference.

MER, strict labels and other task rewards need the adaptation described in [Tasks and presets](tasks-and-presets.md).
Changing the `reward_metric` string alone does not implement a new reward.

## Efficient decoding

Start with roughly 100–200 training recordings, stratified by relevant task/language/domain, expanding for class
coverage when needed. The greedy baseline is separate from eight stochastic samples. Match the selected training
temperature, top-p/top-k, tokenizer, stop tokens, text/audio order and audio preprocessing. Disable speculation for
the initial comparison. Do not copy a different model's hard-coded EOS ID.

`examples/speechlm2/finetuning/vllm_task_eval.py` is a **greedy** evaluator, not a multi-rollout miner. For mining,
write a small job-local driver using the same Speech vLLM plugin and the request format below. This is a construction
pattern, not an additional installed CLI. Configure `llm` and `tokenizer` for the full seed in the matched environment;
set `stop_ids`, `top_p` and `top_k` from the resolved RL generation config (vLLM uses `-1` for unrestricted top-k,
where the RL config may use null). `rows` are a bounded shard of the schema above.

```python
import soundfile as sf
from vllm import SamplingParams

requests, sampling = [], []
for row in rows:
    audio, sr = sf.read(row["audio_filepath"], dtype="float32", always_2d=True)
    if sr != 16000 or not len(audio):
        raise ValueError(row["audio_filepath"])
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["instruction"] + "\n<|audio|>"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    requests.append({"prompt": prompt, "multi_modal_data": {"audio": (audio.mean(axis=1), sr)}})
    sampling.append(SamplingParams(
        n=8, temperature=0.8, top_p=top_p, top_k=top_k,
        max_tokens=min(row["responses_create_params"]["max_output_tokens"] or global_output_cap, global_output_cap),
        stop_token_ids=stop_ids, seed=mining_seed,
    ))
outputs = llm.generate(requests, sampling)
```

Write each completed shard before scoring. For every response save row/audio identity, prompt, raw text, token count,
finish/stop reason, reward and decoding/scorer versions. Keep raw text even if evaluation strips formatting. Emit one
greedy response separately with `n=1, temperature=0`. Persist failures and length-capped outputs separately from valid
completed alternatives; do not count a transport failure as a wrong answer or claim cap-only contrast as useful signal.
Keep capped samples for diagnosis and record the training recipe's actual masking policy.

Cache identity includes seed **revision/weights**, tokenizer, audio+segment, prompt, sampling and cap settings. Reward
changes permit rescoring raw text; they do not justify calling incompatible decodes matched rollouts. Decode only the
missing rows/sample counts. Reuse one loaded model for many shards, stream bounded batches, and keep audio/checkpoints
on shared storage. Parallelize independent shards when resources allow; release idle engines after their queue drains.

## Selection and accounting

For every row compute usable sample count, reward range/standard deviation, best/median reward, greedy reward and
the fraction of samples better than greedy. For binary tasks retain observed correct/incorrect counts. Start with
nonconstant groups containing useful answers; use a task-meaningful spread floor if tiny scoring noise dominates.
There is no universal 0.25/0.50 error-gap rule. Keep challenging groups with relative improvement even if none is perfect.

Within each stratum, sample with a fixed RNG seed; cap near-duplicate recordings/speakers and balance error types and
protected capabilities. Audit a few high-spread examples against audio to exclude reference errors. Expand mining
where useful contrast per GPU-hour is high **and** coverage is needed. Relax a selection threshold before spending
64 samples on every candidate, but record the change and its effect on accepted quality/yield.

Offline selection chooses recordings and prompts, **not saved target responses for replay**. GSPO generates fresh
responses on-policy. Publish the selected Parquet, its hash, membership and source/signal counts; retain the raw mining
results. Early small releases can start training while further train-only mining proceeds, but freeze each run's input.

All-equal groups are rejected dynamically during training. Track actual retained exposure by stratum and unique audio;
offline balance and observed eight-sample signal do not guarantee balanced gradients or continued signal after learning.
