from pathlib import Path
from typing import TYPE_CHECKING, Optional

from nemo.lightning.base import NEMO_DATASETS_CACHE

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers import TokenizerSpec
    from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset


def get_dataset_root(name: str) -> Path:
    output = Path(NEMO_DATASETS_CACHE) / name
    output.mkdir(parents=True, exist_ok=True)

    return output


def create_sft_dataset(
    path: Path,
    tokenizer: "TokenizerSpec",
    seq_length: int = 2048,
    add_bos: bool = False,
    add_eos: bool = True,
    add_sep: bool = False,
    seed: int = 1234,
    label_key: str = 'output',
    answer_only_loss: bool = True,
    truncation_field: str = 'input',
    pad_to_max_length: bool = False,
    index_mapping_dir: Optional[str] = None,
    prompt_template: str = '{input} {output}',
    truncation_method: str = 'right',
    memmap_workers: int = 2,
    hf_dataset: bool = False,
    **kwargs,
) -> "GPTSFTDataset":
    from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset

    return GPTSFTDataset(
        file_path=str(path),
        tokenizer=tokenizer,
        max_seq_length=seq_length,
        memmap_workers=memmap_workers,
        hf_dataset=hf_dataset,
        add_bos=add_bos,
        add_eos=add_eos,
        add_sep=add_sep,
        seed=seed,
        label_key=label_key,
        answer_only_loss=answer_only_loss,
        truncation_field=truncation_field,
        pad_to_max_length=pad_to_max_length,
        index_mapping_dir=index_mapping_dir,
        prompt_template=prompt_template,
        truncation_method=truncation_method,
        **kwargs,
    )
