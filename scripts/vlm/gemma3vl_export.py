"""Export Gemma3VL NeMo checkpoints to Hugging Face format."""

import argparse
from huggingface_hub import hf_hub_download
import importlib
import os
from pathlib import Path
import sys
from nemo.collections import llm


def main():
  parser = argparse.ArgumentParser(
      description=(
          "Export NeMo vision language model checkpoint to Hugging Face format."
      )
  )
  parser.add_argument(
      "--nemo_ckpt_path",
      type=str,
      required=True,
      default=None,
      help="Path to the NeMo checkpoint directory.",
  )
  parser.add_argument(
      "--output_hf_path",
      type=str,
      required=True,
      default=None,
      help="Path to save the converted Hugging Face checkpoint.",
  )
  parser.add_argument(
      "--model_name",
      type=str,
      required=False,
      default=None,
      help="Name of the model on Hugging Face.",
  )

  args = parser.parse_args()

  llm.export_ckpt(
      path=Path(args.nemo_ckpt_path),
      target="hf",
      output_path=Path(args.output_hf_path),
      overwrite=True,
  )
  if args.model_name:
    # Copy necessary files if exist from HuggingFace for Gemma3VL model export.
    copy_file_list = [
        "preprocessor_config.json",
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    for file_name in copy_file_list:
      try:
        downloaded_path = hf_hub_download(
            repo_id=args.model_name,
            filename=file_name,
            local_dir=args.output_hf_path,
        )
        print(f"Downloaded {downloaded_path} during export gamma3vl models.")
      except:
        print(f"Ignore {file_name} during export gamma3vl models.")


if __name__ == "__main__":
  main()
