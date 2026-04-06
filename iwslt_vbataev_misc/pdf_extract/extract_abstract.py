import os
import json
import argparse
import pymupdf.layout
import pymupdf4llm
from tqdm import tqdm

def extract_abstract_from_pdf(pdf_path):
    """Extract named entities from a PDF file."""
    try:
        # Open and extract text from PDF
        doc = pymupdf.open(pdf_path)
        text = pymupdf4llm.to_text(doc, use_ocr=False, force_text=False, header=False, footer=False)
        
        # Remove section after introduction section
        pos_intro = text.find('\n1 Introduction')
        assert pos_intro != -1
        text_abstract = text[:pos_intro]
        return text_abstract
    
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return ''


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Extract abstract from PDFs"
    )
    parser.add_argument(
        "pdf_paths_file",
        help="Path to file containing PDF paths (one per line)"
    )
    parser.add_argument(
        "-o", "--output",
        default="data/abstract_results.json",
        help="Output JSON file path (default: data/ner_results.json)"
    )
    args = parser.parse_args()
    
    # Read PDF paths from file
    with open(args.pdf_paths_file, 'r') as f:
        pdf_paths = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(pdf_paths)} PDFs to process")
    
    # Process each PDF
    results = []
    for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
        pdf_name = os.path.basename(pdf_path)
        print(f"\nProcessing: {pdf_name}")
        abstract = extract_abstract_from_pdf(pdf_path)
        results.append({
            "path": pdf_path,
            "abstract": abstract,
        })
    
    # Save results to JSON
    output_file = args.output
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_file}")

if __name__ == "__main__":
    main()
