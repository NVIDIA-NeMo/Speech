"""
Extract named entities from paper (title, authors, abstract) using
Qwen3-30B-A3B-Instruct via vLLM.

Each paper produces THREE separate prompts (title, authors, abstract).
Samples N times per prompt (default 16) and keeps entities appearing
in at least half the samples (default 8) for robustness.

Uses vLLM structured outputs (JSON schema) to force valid output.

Usage:
    python ner_llm.py data/abstract_results.json -o data/ner_llm_results.json
    python ner_llm.py data/abstract_results.json -n 16 -k 8 -o data/ner_llm_results.json
"""
import os
import re
import json
import argparse
import logging
from collections import Counter
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

logging.getLogger().setLevel(logging.ERROR)

# NER_MODEL = "/data/user_data/siqiouya/ckpts/pretrained/llm/Qwen3-30B-A3B-Instruct-2507-FP8/"
NER_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

# JSON schema to force the model to output {"entities": ["str", ...]}
NER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "List of named entities extracted from the text"
        }
    },
    "required": ["entities"],
    "additionalProperties": False
}

TITLE_TEMPLATE = """\
You are a named entity extraction system. Given a paper's title, \
extract all named entities. Include: model/system names, method/algorithm names, \
dataset names, task names, and any other proper nouns or technical terms that \
a speech recognition system might struggle with.

Return a JSON object with a single key "entities" containing an array of strings, one per entity. No duplicates.

[TITLE]
{title}
"""

AUTHORS_TEMPLATE = """\
You are a named entity extraction system. Given a paper's author list, \
extract all named entities. Include: author names, organization/affiliation names, \
and location names.

Return a JSON object with a single key "entities" containing an array of strings, one per entity. No duplicates.

[AUTHORS]
{authors}
"""

ABSTRACT_TEMPLATE = """\
You are a named entity extraction system. Given a paper's abstract, \
extract all named entities. Include: model/system names, method/algorithm names, \
dataset names, task names, organization names, location names, \
and any other proper nouns or technical terms that a speech recognition system might struggle with.

Return a JSON object with a single key "entities" containing an array of strings, one per entity. No duplicates.

[ABSTRACT]
{abstract}
"""


def _clean_text(text):
    """Remove URLs and emails from text."""
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\{[^}]*\}\S*@\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    return text.strip()


def _parse_paper_sections(raw):
    """Split raw extracted text into title, authors, and abstract body."""
    lines = raw.strip().split('\n', 1)
    title = lines[0].strip()
    rest = lines[1] if len(lines) > 1 else ''

    abstract_match = re.split(r'\n\s*Abstract\s*\n', rest, maxsplit=1)
    if len(abstract_match) == 2:
        authors = abstract_match[0].strip()
        abstract = abstract_match[1].strip()
    else:
        authors = ''
        abstract = rest.strip()

    return title, authors, abstract


def build_prompts(results, tokenizer):
    """Build chat-formatted prompts from abstract results.

    Returns a list of dicts, one per paper, each containing:
        - 'title_prompt', 'authors_prompt', 'abstract_prompt' (str or None)
    """
    paper_prompts = []
    for data in results:
        raw = data.get("abstract", "").strip()
        entry = {"title_prompt": None, "authors_prompt": None, "abstract_prompt": None}

        if not raw:
            paper_prompts.append(entry)
            continue

        title, authors, abstract = _parse_paper_sections(raw)
        title = _clean_text(title)
        authors = _clean_text(authors)
        abstract = _clean_text(abstract)

        print(f"title: {title}")
        print(f"authors: {authors}")
        print(f"abstract: {abstract}")

        # Title prompt
        if title:
            messages = [{"role": "user", "content": TITLE_TEMPLATE.format(title=title)}]
            entry["title_prompt"] = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

        # Authors prompt
        if authors:
            messages = [{"role": "user", "content": AUTHORS_TEMPLATE.format(authors=authors)}]
            entry["authors_prompt"] = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

        # Abstract prompt
        if abstract:
            messages = [{"role": "user", "content": ABSTRACT_TEMPLATE.format(abstract=abstract)}]
            entry["abstract_prompt"] = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

        paper_prompts.append(entry)
    return paper_prompts


def parse_entity_list(response):
    """Parse entities from JSON-schema-constrained model response."""
    text = response.strip()
    try:
        parsed = json.loads(text)
        entities = parsed.get("entities", [])
        if isinstance(entities, list):
            seen = set()
            unique = []
            for e in entities:
                s = str(e).strip()
                if s and s not in seen:
                    seen.add(s)
                    unique.append(s)
            return unique
    except json.JSONDecodeError:
        pass
    return []


def majority_vote(all_responses, min_count):
    """Apply majority-vote filtering across multiple sampled responses."""
    entity_counter = Counter()
    for response in all_responses:
        sample_entities = parse_entity_list(response)
        for e in set(sample_entities):
            entity_counter[e] += 1
    return [e for e, c in entity_counter.most_common() if c >= min_count]


def main():
    parser = argparse.ArgumentParser(
        description="Extract named entities from paper abstracts using Qwen3 via vLLM"
    )
    parser.add_argument(
        "input",
        help="Path to abstract_results.json (from extract_abstract.py)",
    )
    parser.add_argument(
        "-o", "--output",
        default="data/ner_llm_results.json",
        help="Output JSON file path (default: data/ner_llm_results.json)",
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int, default=16,
        help="Number of samples per prompt (default: 16)",
    )
    parser.add_argument(
        "-k", "--min-count",
        type=int, default=8,
        help="Minimum number of samples an entity must appear in to be kept (default: 8)",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"Loaded {len(results)} papers from {args.input}")
    print(f"Loading {NER_MODEL} ...")

    llm = LLM(
        model=NER_MODEL,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        max_model_len=4096,
    )
    tokenizer = llm.get_tokenizer()

    structured_outputs = StructuredOutputsParams(json=NER_JSON_SCHEMA)
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=2048,
        n=args.num_samples,
        structured_outputs=structured_outputs,
    )

    # Build per-section prompts
    paper_prompts = build_prompts(results, tokenizer)

    # Flatten all valid prompts into a single batch for efficient generation.
    # Each entry: (paper_index, section_key, prompt_text)
    SECTIONS = ["title_prompt", "authors_prompt", "abstract_prompt"]
    batch = []
    for i, pp in enumerate(paper_prompts):
        for section in SECTIONS:
            prompt = pp[section]
            if prompt is not None:
                batch.append((i, section, prompt))

    all_prompts = [item[2] for item in batch]
    print(f"Generating NER for {len(results)} papers × 3 sections "
          f"({len(all_prompts)} total prompts, {args.num_samples} samples each, "
          f"keeping entities with >= {args.min_count} hits) ...")

    outputs = llm.generate(all_prompts, sampling_params=sampling_params, use_tqdm=True)

    # Map outputs back to (paper_index, section)
    # output_map[paper_index][section_key] = list of response texts
    output_map = {}
    for (paper_idx, section_key, _), out in zip(batch, outputs):
        if paper_idx not in output_map:
            output_map[paper_idx] = {}
        output_map[paper_idx][section_key] = [o.text for o in out.outputs]

    # Build final results: majority-vote per section, then merge
    SECTION_NAMES = {
        "title_prompt": "title",
        "authors_prompt": "authors",
        "abstract_prompt": "abstract",
    }
    final = []
    for i, data in enumerate(results):
        sections = output_map.get(i, {})
        per_section = {}
        all_entities = []

        for section_key, section_name in SECTION_NAMES.items():
            responses = sections.get(section_key, [])
            entities = majority_vote(responses, args.min_count)
            per_section[section_name] = entities
            all_entities.extend(entities)

        # Deduplicate merged entities while preserving order
        seen = set()
        merged = []
        for e in all_entities:
            if e not in seen:
                seen.add(e)
                merged.append(e)

        final.append({
            "path": data["path"],
            "entity_count": len(merged),
            "entities": merged,
            "entities_by_section": per_section,
        })

    # Save
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to: {args.output}")

    total = sum(d["entity_count"] for d in final)
    print(f"\nTotal entities extracted: {total} "
          f"(n={args.num_samples}, min_count={args.min_count})")
    for d in final:
        by_sec = d["entities_by_section"]
        print(f"  {os.path.basename(d['path'])}: {d['entity_count']} entities "
              f"(title={len(by_sec['title'])}, "
              f"authors={len(by_sec['authors'])}, "
              f"abstract={len(by_sec['abstract'])})")


if __name__ == "__main__":
    main()