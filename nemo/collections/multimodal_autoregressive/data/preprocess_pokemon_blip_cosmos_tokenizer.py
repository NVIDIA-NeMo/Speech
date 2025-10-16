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

from argparse import ArgumentParser

import numpy as np
import torch
from datasets import load_dataset
from einops import rearrange
from tqdm import tqdm

from nemo.collections.common.video_tokenizers.cosmos_tokenizer import CausalVideoTokenizer
from nemo.collections.common.video_tokenizers.utils import numpy2tensor, pad_video_batch
from nemo.collections.multimodal_autoregressive.tokenizer.cosmos_multimodal_tokenizer import CosmosMultiModalTokenizer

try:
    from nemo.collections.nlp.data.language_modeling.megatron import indexed_dataset

    HAVE_NLP = True
except (ImportError, ModuleNotFoundError):
    HAVE_NLP = False

"""
You can run this script as follows

python3 nemo/collections/multimodal_autoregressive/data/preprocess_pokemon_blip_cosmos_tokenizer.py

NOTE : Make sure you install tiktoken==0.6.0
"""


def to_imgstr(image_tokens_flattened):
    """Convert the image tokens to string

    Given image tokens e.g [1,5,32] as input, this produces the appropriate string tokens
    e.g., "<|visual token 000001|><|visual token 000005|><|visual token 000032|>"

    Args:
        image_tokens : The image tokens as an integer list
        tokenizer: EMU3 Tokenizer

    Returns:
        str: The image token converted to string
    """
    image_tokens_flattened = image_tokens_flattened.cpu().numpy().tolist()
    visual_tokens = [
        '<|visual token {token_id:0>6d}|>'.format(token_id=token_id) for token_id in image_tokens_flattened
    ]
    visual_tokens_str = "".join(visual_tokens)
    return visual_tokens_str


def main(args):
    """Main function"""

    dataset = load_dataset(args.dataset)

    text_tokenizer = CosmosMultiModalTokenizer.from_pretrained(args.multimodal_tokenizer_path)
    image_tokenizer = CausalVideoTokenizer.from_pretrained(
        tokenizer_type=args.image_encoder, load_encoder=True, load_decoder=False, load_full_model=False
    )

    builders = {}
    key = 'text'
    builders[key] = indexed_dataset.make_builder(
        f'{args.output_prefix}.bin',
        impl='mmap',
        chunk_size=64,
        pad_id=text_tokenizer.pad_token if getattr(text_tokenizer, "pad_token", None) is not None else 0,
        retrieval_db=None,
        vocab_size=text_tokenizer.vocab_size,
        stride=64,
    )

    dataset = dataset['train']

    for data in tqdm(dataset):
        image, caption = data['image'], data['text']
        image = image.resize((512, 512))
        image_numpy_array = np.array(image)
        image_numpy_array = rearrange(image_numpy_array, 'h w (t c) -> t h w c', t=1)
        batch_image_array = image_numpy_array[np.newaxis, ...]
        padded_input_image_batch, crop_region = pad_video_batch(batch_image_array)
        input_tensor = numpy2tensor(
            padded_input_image_batch, dtype=image_tokenizer._dtype, device=image_tokenizer._device
        )
        output_indices, output_latent_vectors = image_tokenizer.encode(input_tensor)
        output_indices_flattened = output_indices.reshape(-1)

        imgstr = to_imgstr(output_indices_flattened)
        image_prompt = text_tokenizer.boi_token + text_tokenizer.img_token + imgstr + text_tokenizer.eoi_token

        prompt = (
            f'{text_tokenizer.bos_token}You are a helpful assistant. '
            'Draw a picture for the caption given by the user. '
            f'USER: {caption}. ASSISTANT: {image_prompt}{text_tokenizer.eos_token}'
        )

        int_tokens = text_tokenizer(prompt).input_ids
        builders[key].add_item(torch.IntTensor(int_tokens))
        builders[key].end_document()

    builders[key].finalize(
        f'{args.output_prefix}.idx',
    )
    print(f' Output .bin and .idx files saved to {args.output_prefix}')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        "--output_prefix",
        required=True,
        type=str,
        help="The directory along with the output file name to write "
        "the .idx and .bin files (e.g /path/to/output/sample)",
    )
    parser.add_argument(
        "--image_encoder",
        type=str,
        help="Discrete image encoder. Options are (Cosmos-Tokenizer-DV8x16x16/Cosmos-Tokenizer-DV4x8x8)",
        default='Cosmos-Tokenizer-DV8x16x16',
    )
    parser.add_argument(
        "--dataset", type=str, help="The hugging face dataset", default='reach-vb/pokemon-blip-captions'
    )
    parser.add_argument(
        "--multimodal_tokenizer_path",
        required=True,
        type=str,
        help="The path to the multimodal tokenizer. (nemo/collections/multimodal_autoregressive/tokenizer)",
    )
    args = parser.parse_args()

    with torch.no_grad():
        main(args)
