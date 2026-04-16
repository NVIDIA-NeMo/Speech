"""
Extract full structured text from academic PDFs.

Extracts title, authors, abstract, body sections, and references
as separate fields for downstream entity extraction.

Usage:
    python extract_fulltext.py pdf_paths.txt -o fulltext_results.json
"""
import os
import re
import json
import argparse

import pymupdf
import pymupdf4llm
from tqdm import tqdm


def _clean_text(text):
    """Remove URLs, emails, and picture placeholders from text."""
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\{[^}]*\}\S*@\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    # Remove pymupdf picture placeholders
    text = re.sub(r'==>.*?<==', '', text)
    return text.strip()


def _find_section_boundaries(text):
    """Find numbered section headings and their positions.

    Returns list of (position, heading_text, level) tuples sorted by position.
    Level 1 = top-level section (e.g., "1 Introduction"), level 2 = subsection, etc.
    """
    boundaries = []
    for m in re.finditer(r'^(\d{1,2}(?:\.\d{1,2}){0,3})\s+([A-Z].{2,80})$', text, re.MULTILINE):
        num = m.group(1)
        heading = m.group(2).strip()
        # Skip lines that look like picture placeholders or table content
        if '==>' in heading or '<==' in heading or heading.startswith('|'):
            continue
        level = num.count('.') + 1
        boundaries.append((m.start(), f"{num} {heading}", level))
    return boundaries


def _find_named_section(text, name):
    """Find position of a named section (Abstract, References, etc.)."""
    pattern = rf'^\s*{name}\s*$'
    m = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
    return m.start() if m else -1


def extract_fulltext_from_pdf(pdf_path):
    """Extract full structured text from a PDF.

    Returns dict with keys:
        - title: str
        - authors: str
        - abstract: str
        - body_sections: list of {"heading": str, "text": str}
        - references: str
        - full_body_text: str (all body text concatenated)
        - total_words: int
    """
    try:
        doc = pymupdf.open(pdf_path)
        text = pymupdf4llm.to_text(doc, use_ocr=False, force_text=False, header=False, footer=False)
    except Exception as e:
        print(f"Error opening {pdf_path}: {e}")
        return None

    # --- Find key boundaries ---
    # Abstract position
    abstract_pos = _find_named_section(text, 'Abstract')

    # Introduction (first numbered section) - marks end of abstract
    intro_match = re.search(r'^\d+\s+Introduction', text, re.MULTILINE)
    intro_pos = intro_match.start() if intro_match else -1

    # References section
    ref_pos = _find_named_section(text, 'References')

    # Appendix / Limitations / Ethics / Acknowledgments (after references)
    appendix_pos = -1
    for name in ['Appendix', 'Limitations', 'Ethics', 'Acknowledgments', 'Acknowledgements']:
        pos = _find_named_section(text, name)
        if pos > ref_pos and (appendix_pos == -1 or pos < appendix_pos):
            appendix_pos = pos
    # Also check for lettered appendix sections like "A Details"
    for m in re.finditer(r'^[A-Z]\s+\w', text, re.MULTILINE):
        if m.start() > (ref_pos if ref_pos != -1 else len(text) * 0.8):
            if appendix_pos == -1 or m.start() < appendix_pos:
                appendix_pos = m.start()
            break

    # --- Extract title and authors ---
    if abstract_pos != -1:
        preamble = text[:abstract_pos].strip()
    elif intro_pos != -1:
        preamble = text[:intro_pos].strip()
    else:
        preamble = text[:500].strip()

    lines = preamble.split('\n', 1)
    title = lines[0].strip()
    authors = lines[1].strip() if len(lines) > 1 else ''
    authors = _clean_text(authors)

    # --- Extract abstract ---
    if abstract_pos != -1 and intro_pos != -1:
        abstract_text = text[abstract_pos:intro_pos]
        # Remove the "Abstract" heading itself
        abstract_text = re.sub(r'^\s*Abstract\s*\n', '', abstract_text, flags=re.IGNORECASE)
        abstract = _clean_text(abstract_text.strip())
    elif abstract_pos != -1:
        # No intro found, take next 2000 chars after Abstract heading
        abstract_text = text[abstract_pos:abstract_pos + 2000]
        abstract_text = re.sub(r'^\s*Abstract\s*\n', '', abstract_text, flags=re.IGNORECASE)
        abstract = _clean_text(abstract_text.strip())
    else:
        abstract = ''

    # --- Extract body sections ---
    body_start = intro_pos if intro_pos != -1 else (abstract_pos + len(abstract) if abstract_pos != -1 else 0)
    body_end = ref_pos if ref_pos != -1 else len(text)

    body_text = text[body_start:body_end]
    # Clean picture placeholders before section parsing
    body_text = re.sub(r'==>.*?<==', '', body_text)
    # Clean page numbers on their own lines
    body_text = re.sub(r'^\d{4,}\s*$', '', body_text, flags=re.MULTILINE)
    body_sections = []

    # Find section boundaries within the body
    boundaries = _find_section_boundaries(body_text)

    if boundaries:
        for idx, (pos, heading, level) in enumerate(boundaries):
            if idx + 1 < len(boundaries):
                next_pos = boundaries[idx + 1][0]
            else:
                next_pos = len(body_text)

            section_text = body_text[pos:next_pos]
            # Remove the heading line from the section text
            section_text = section_text[len(heading):].strip()
            section_text = _clean_text(section_text)

            if section_text:
                body_sections.append({
                    "heading": heading,
                    "text": section_text,
                })
    else:
        # No sections found, use the whole body
        cleaned = _clean_text(body_text)
        if cleaned:
            body_sections.append({
                "heading": "body",
                "text": cleaned,
            })

    full_body_text = _clean_text(body_text)

    # --- Extract references ---
    if ref_pos != -1:
        ref_end = appendix_pos if appendix_pos != -1 else len(text)
        ref_text = text[ref_pos:ref_end]
        # Remove the "References" heading
        ref_text = re.sub(r'^\s*References\s*\n', '', ref_text, flags=re.IGNORECASE)
        references = _clean_text(ref_text.strip())
    else:
        references = ''

    total_words = len(full_body_text.split())

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "body_sections": body_sections,
        "full_body_text": full_body_text,
        "references": references,
        "total_words": total_words,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract full structured text from PDFs")
    parser.add_argument(
        "pdf_paths_file",
        help="Path to file containing PDF paths (one per line)",
    )
    parser.add_argument(
        "-o", "--output",
        default="fulltext_results.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    with open(args.pdf_paths_file, 'r') as f:
        pdf_paths = [line.strip() for line in f if line.strip()]

    print(f"Found {len(pdf_paths)} PDFs to process")

    results = []
    for pdf_path in tqdm(pdf_paths, desc="Extracting full text"):
        print(f"\nProcessing: {os.path.basename(pdf_path)}")
        entry = extract_fulltext_from_pdf(pdf_path)
        if entry is None:
            entry = {
                "title": "", "authors": "", "abstract": "",
                "body_sections": [], "full_body_text": "",
                "references": "", "total_words": 0,
            }
        entry["path"] = pdf_path
        results.append(entry)

        print(f"  title: {entry['title'][:80]}...")
        print(f"  authors: {entry['authors'][:80]}...")
        print(f"  abstract: {len(entry['abstract'])} chars")
        print(f"  body sections: {len(entry['body_sections'])}")
        print(f"  body text: {entry['total_words']} words")
        print(f"  references: {len(entry['references'])} chars")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to: {args.output}")
    print(f"  Total papers: {len(results)}")
    print(f"  Avg words/paper: {sum(r['total_words'] for r in results) / max(len(results), 1):.0f}")


if __name__ == "__main__":
    main()
