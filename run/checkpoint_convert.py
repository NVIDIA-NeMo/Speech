import argparse

from nemo.collections import llm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-checkpoint-path", type=str, required=True)
    parser.add_argument("--nemo-checkpoint-path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    llm.import_ckpt(
        model=llm.DeepSeekModel(llm.DeepSeekV3Config()),
        source=args.hf_checkpoint_path,
        output_path=args.nemo_checkpoint_path,
    )
