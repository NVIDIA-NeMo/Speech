from pathlib import Path
from typing import TYPE_CHECKING, Optional

from nemo.lightning.base import NEMO_DATASETS_CACHE

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers import TokenizerSpec
    from nemo.collections.nlp.data.language_modeling.megatron.t5_sft_dataset import T5SFTDataset


def get_dataset_root(name: str) -> Path:
    output = Path(NEMO_DATASETS_CACHE) / name
    output.mkdir(parents=True, exist_ok=True)

    return output


def create_sft_dataset(
    path: Path,
    tokenizer: "TokenizerSpec",
    seq_length: int = 512,
    seq_length_dec: int = 128,
    add_bos: bool = True,
    add_eos: bool = True,
    replace_bos_with_pad: bool = False,
    seed: int = 1234,
    index_mapping_dir: Optional[str] = None,
    memmap_workers: int = 2,
    hf_dataset: bool = False,
    **kwargs,
) -> "T5SFTDataset":
    from nemo.collections.nlp.data.language_modeling.megatron.t5_sft_dataset import T5SFTDataset

    return T5SFTDataset(
        file_path=str(path),
        src_tokenizer=tokenizer,
        tgt_tokenizer=tokenizer,
        max_src_seq_length=seq_length,
        max_tgt_seq_length=seq_length_dec,
        memmap_workers=memmap_workers,
        hf_dataset=hf_dataset,
        add_bos_to_input=add_bos,
        add_eos_to_input=add_eos,
        replace_bos_with_pad=replace_bos_with_pad,
        index_mapping_dir=index_mapping_dir,
    )
