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

import math
import os
import pickle
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

try:
    from nemo.collections.nlp.data.language_modeling.megatron import indexed_dataset

    HAVE_NLP = True
except (ImportError, ModuleNotFoundError):
    HAVE_NLP = False

"""
You can run this script as follows

torchrun --nproc-per-node 8 preprocess_coyo.py \
    --input_image_dir /path/to/images \
    --input_captions_dir /path/to/captions \
    --output_dir /path/to/output/prefix \

NOTE : Make sure you install tiktoken==0.6.0
NOTE : Make sure the images and captions have the same filename (Images should be .jpg and Captions .pkl)
"""

EMU_HUB = "BAAI/Emu3-Gen"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"


def smart_resize(image, factor: int = 8, min_pixels: int = 512 * 512, max_pixels: int = 1024 * 1024):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    height, width = image.size
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 5:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 5, got {max(height, width) / min(height, width)}"
        )

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    image = image.resize((h_bar, w_bar))
    return image


def to_imgstr(image_tokens, tokenizer):
    """Convert the image tokens to string

    Given image tokens e.g [1,5,32] as input, this produces the appropriate string tokens
    e.g., "<|visual token 000001|><|visual token 000005|><|visual token 000032|>"

    Args:
        image_tokens : The image tokens as an integer list
        tokenizer: EMU3 Tokenizer

    Returns:
        str: The image token converted to string
    """
    image_tokens = image_tokens.cpu().numpy().tolist()
    image_token_str = [
        ['<|visual token {token_id:0>6d}|>'.format(token_id=token_id) for token_id in token_row]
        for token_row in image_tokens
    ]
    image_row_str = ["".join(token_row) for token_row in image_token_str]
    imgstr = tokenizer.eol_token.join(image_row_str)
    return imgstr


def main(args):
    """Main Function"""

    gpu_rank = torch.cuda.current_device()
    world_size = torch.cuda.device_count()

    tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map="cuda", trust_remote_code=True).eval()

    # prepare input
    text = "Please describe the image"

    builders = {}
    key = 'text'
    builders[key] = indexed_dataset.make_builder(
        f'{args.output_prefix}.bin',
        impl=args.dataset_impl,
        chunk_size=args.chunk_size,
        pad_id=tokenizer.pad_id if getattr(tokenizer, "pad_id", None) is not None else 0,
        retrieval_db=None,
        vocab_size=tokenizer.vocab_size,
        stride=args.chunk_stride_size,
    )

    filepaths_final = glob(f'{args.input_image_dir}/*.jpg')

    pbar = tqdm(filepaths_final)
    total_images_to_process = len(filepaths_final)
    total_images_to_process_per_gpu = total_images_to_process // torch.cuda.device_count()
    if total_images_to_process_per_gpu > 30000:
        print(
            'WARNING : Found more than 30k images to process per GPU. '
            'This job might take more than 3 hours to process as tested on H100 gpus'
        )
    print(
        f'Total images to process : {total_images_to_process_per_gpu}. '
        'Each GPU will get {total_images_to_process_per_gpu} files'
    )

    for idx, filepath in enumerate(pbar):
        pbar.update(1)
        if idx % world_size != gpu_rank:
            continue
        try:
            image = Image.open(filepath)
            caption_filename = filepath.split('/')[-1].replace('.jpg', '.pkl')
            caption_path = Path(args.input_captions_dir).joinpath(caption_filename)
            if not os.path.isfile(caption_path):
                print(f'WARNING : Caption file does not exist {caption_path}. So skipping')
                continue
            if image.mode == 'L':
                print(f'WARNING : Image {filepath} is gray scale. So skipping')
                continue
            image = smart_resize(image)
            image_tensor = torchvision.transforms.functional.pil_to_tensor(image).unsqueeze(0)
            image_tokens = image_tokenizer.encode(image_tensor.to(image_tokenizer.device, image_tokenizer.dtype))
            bs, h, w = image_tokens.shape

            imgstr = to_imgstr(image_tokens[0], tokenizer=tokenizer)
            image_prompt = (
                tokenizer.boi_token
                + f'{h}*{w}'
                + tokenizer.img_token
                + imgstr
                + tokenizer.eol_token
                + tokenizer.eof_token
                + tokenizer.eoi_token
            )

            caption = ""
            with open(caption_path, 'rb') as f:
                caption_data = pickle.load(f)
                caption = caption_data['captions']['llava']

            prompt = (
                f'{tokenizer.bos_token}You are a helpful assistant. '
                f'USER: {image_prompt}{text}. ASSISTANT: {caption}{tokenizer.eos_token}'
            )
            int_tokens = tokenizer(prompt).input_ids
            builders[key].add_item(torch.IntTensor(int_tokens))
            builders[key].end_document()
        except Exception as e:
            print(f'Error in handling {filepath}. Exception {e} raised. Continuing to next file')
            continue

    builders[key].finalize(
        f'{args.output_prefix}.idx',
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_image_dir", required=True, type=str, help="The directory which contains images.")
    parser.add_argument(
        "--input_captions_dir",
        required=True,
        type=str,
        help="The directory which contains captions (as .pkl file with same names as the image names).",
    )
    parser.add_argument(
        "--output_prefix",
        required=True,
        type=str,
        help="The directory along with the output file name to "
        "write the .idx and .bin files (e.g /path/to/output/sample)",
    )
    parser.add_argument('--dataset_impl', type=str, default='mmap', choices=['lazy', 'cached', 'mmap', 'retmmap'])
    parser.add_argument('--chunk_size', type=int, default=64, help='chunk size used for retrieval')
    parser.add_argument(
        '--resize_image', type=bool, default=True, help='Resizes the image to be between mix_pixels and max_pixels'
    )
    parser.add_argument(
        '--spatial_factor',
        type=int,
        default=8,
        help='The spatial downsample factor the image will be downsampled/upsampled'
        'to fit between min_pixels and max_pixels if resize_image is set to True',
    )
    parser.add_argument(
        '--min_pixels',
        type=int,
        default=512 * 512,
        help='The minimum number of pixels in the image. '
        'Picture will be upsampled if smaller and resize_image is set to True',
    )
    parser.add_argument(
        '--max_pixels',
        type=int,
        default=1024 * 1024,
        help='The maximum number of pixels in the image. '
        'Picture will be downsampled if smaller and resize_image is set to False',
    )
    parser.add_argument(
        '--chunk_stride_size', type=int, default=64, help='the stride size for neighbor chunks used for retrieval'
    )

    args = parser.parse_args()

    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)

    with torch.no_grad():
        main(args)
