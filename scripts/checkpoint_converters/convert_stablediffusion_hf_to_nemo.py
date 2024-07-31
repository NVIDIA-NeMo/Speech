# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

r"""
Conversion script to convert HuggingFace StableDiffusion checkpoints into nemo checkpoint.
  Example to run this conversion script:
    python convert_hf_starcoder2_to_nemo.py \
     --input_name_or_path <path_to_sc2_checkpoints_folder> \
     --output_path <path_to_output_nemo_file> --model <unet|vae>
"""

import os
from argparse import ArgumentParser

import numpy as np
import safetensors
import torch
import torch.nn

from nemo.utils import logging


def filter_keys(rule, dict):
    keys = list(dict.keys())
    nd = {k: dict[k] for k in keys if rule(k)}
    return nd


def map_keys(rule, dict):
    new = {rule(k): v for k, v in dict.items()}
    return new


def split_name(name, dots=0):
    l = name.split(".")
    return ".".join(l[: dots + 1]), ".".join(l[dots + 1 :])


def is_prefix(shortstr, longstr):
    # is the first string a prefix of the second one
    return longstr == shortstr or longstr.startswith(shortstr + ".")


def numdots(str):
    return str.count(".")


class SegTree:
    def __init__(self):
        self.nodes = dict()
        self.val = None
        self.final_val = 0
        self.convert_name = None

    def __len__(self):
        return len(self.nodes)

    def is_leaf(self):
        return len(self.nodes) == 0

    def add(self, name, val=0):
        prefix, subname = split_name(name)
        if subname == '':
            self.nodes[name] = SegTree()
            self.nodes[name].val = val
            return
        if self.nodes.get(prefix) is None:
            self.nodes[prefix] = SegTree()
        self.nodes[prefix].add(subname, val)

    def change(self, name, val):
        self.add(name, val)

    def __getitem__(self, name: str):
        if hasattr(self, name):
            return getattr(self, name)
        val = self.nodes.get(name)
        if val is None:
            # straight lookup failed, do a prefix lookup
            keys = list(self.nodes.keys())
            p_flag = [is_prefix(k, name) for k in keys]
            if not any(p_flag):
                return None
            # either more than 1 match (error) or exactly 1 (success)
            if np.sum(p_flag) > 1:
                logging.warning(f"warning: multiple matches of key {name} with {keys}")
            else:
                i = np.where(p_flag)[0][0]
                n = numdots(keys[i])
                prefix, substr = split_name(name, n)
                return self.nodes[prefix][substr]
        return val


def model_to_tree(model):
    keys = list(model.keys())
    tree = SegTree()
    for k in keys:
        tree.add(k, "leaf")
    return tree


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to Huggingface UNet checkpoints",
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument("--precision", type=str, default="32", help="Model precision")
    parser.add_argument("--model", type=str, default="unet", required=True, choices=['unet', 'vae'])
    parser.add_argument("--debug", action='store_true', help="Useful for debugging purposes.")

    args = parser.parse_args()
    return args


def load_hf_ckpt(in_dir, args):
    ckpt = {}
    assert os.path.isdir(in_dir), "Currently supports only directories with a safetensor file in it."
    with safetensors.safe_open(in_dir + "/diffusion_pytorch_model.safetensors", framework="pt") as f:
        for k in f.keys():
            ckpt[k] = f.get_tensor(k)
    return args, ckpt


def dup_convert_name_recursive(tree: SegTree, convert_name=None):
    '''inside this tree, convert all nodes recursively
    optionally, convert the name of the root as given by name (if not None)
    '''
    if tree is None:
        return
    if convert_name is not None:
        tree.convert_name = convert_name
    # recursively copy the name into convert_name
    for k, v in tree.nodes.items():
        dup_convert_name_recursive(v, k)


def sanity_check(hf_tree, hf_unet, nemo_unet):
    # check if i'm introducing new keys
    for hfk, nk in hf_to_nemo_mapping(hf_tree).items():
        if nk not in nemo_unet.keys():
            logging.info(nk)
        if hfk not in hf_unet.keys():
            logging.info(hfk)


def convert_input_keys(hf_tree: SegTree):
    '''map the input blocks of huggingface model'''
    # map `conv_in` to first input block
    dup_convert_name_recursive(hf_tree['conv_in'], 'input_blocks.0.0')

    # start counting blocks from now on
    nemo_inp_blk = 1
    down_blocks = hf_tree['down_blocks']
    down_blocks_keys = sorted(list(down_blocks.nodes.keys()), key=int)
    for downblockid in down_blocks_keys:
        block = down_blocks[str(downblockid)]
        # compute number of resnets, attentions, downsamplers in this block
        resnets = block.nodes.get('resnets', SegTree())
        attentions = block.nodes.get('attentions', SegTree())
        downsamplers = block.nodes.get('downsamplers', SegTree())

        if len(attentions) == 0:  # no attentions, this is a DownBlock2d
            for resid in sorted(list(resnets.nodes.keys()), key=int):
                resid = str(resid)
                resnets[resid].convert_name = f"input_blocks.{nemo_inp_blk}.0"
                map_resnet_block(resnets[resid])
                nemo_inp_blk += 1
        elif len(attentions) == len(resnets):
            # there are attention blocks here -- each resnet+attention becomes a block
            for resid in sorted(list(resnets.nodes.keys()), key=int):
                resid = str(resid)
                resnets[resid].convert_name = f"input_blocks.{nemo_inp_blk}.0"
                map_resnet_block(resnets[resid])
                attentions[resid].convert_name = f"input_blocks.{nemo_inp_blk}.1"
                map_attention_block(attentions[resid])
                nemo_inp_blk += 1
        else:
            logging.warning("number of attention blocks is not the same as resnets - whats going on?")
        # if there is a downsampler, then also append it
        if len(downsamplers) > 0:
            for k in downsamplers.nodes.keys():
                downsamplers[k].convert_name = f"input_blocks.{nemo_inp_blk}.{k}"
                dup_convert_name_recursive(downsamplers[k]['conv'], 'op')
            nemo_inp_blk += 1


def clean_convert_names(tree):
    tree.convert_name = None
    for k, v in tree.nodes.items():
        clean_convert_names(v)


def map_attention_block(att_tree: SegTree):
    '''this HF tree can either be an AttentionBlock or a DualAttention block
    currently assumed AttentionBlock
    '''

    # TODO(@rohitrango): Add check for dual attention block, but this works for both SD and SDXL
    def check_att_type(tree):
        return "att_block"

    if check_att_type(att_tree) == 'att_block':
        dup_convert_name_recursive(att_tree['norm'], 'norm')
        dup_convert_name_recursive(att_tree['proj_in'], 'proj_in')
        dup_convert_name_recursive(att_tree['proj_out'], 'proj_out')
        tblockids = list(att_tree['transformer_blocks'].nodes.keys())
        for t in tblockids:
            tblock = att_tree[f'transformer_blocks.{t}']
            tblock.convert_name = f"transformer_blocks.{t}"
            dup_convert_name_recursive(tblock['attn1'], 'attn1')
            dup_convert_name_recursive(tblock['attn2'], 'attn2')
            dup_convert_name_recursive(tblock['norm1'], 'attn1.norm')
            dup_convert_name_recursive(tblock['norm2'], 'attn2.norm')
            dup_convert_name_recursive(tblock['norm3'], 'ff.net.0')
            # map ff
            tblock['ff'].convert_name = "ff"
            tblock['ff.net'].convert_name = 'net'
            dup_convert_name_recursive(tblock['ff.net.0'], '1')
            dup_convert_name_recursive(tblock['ff.net.2'], '3')
    else:
        logging.warning("failed to identify type of attention block here.")


def map_resnet_block(resnet_tree: SegTree):
    '''this HF tree is supposed to have all the keys for a resnet'''
    dup_convert_name_recursive(resnet_tree.nodes.get('time_emb_proj'), 'emb_layers.1')
    dup_convert_name_recursive(resnet_tree['norm1'], 'in_layers.0')
    dup_convert_name_recursive(resnet_tree['conv1'], 'in_layers.1')
    dup_convert_name_recursive(resnet_tree['norm2'], 'out_layers.0')
    dup_convert_name_recursive(resnet_tree['conv2'], 'out_layers.2')
    dup_convert_name_recursive(resnet_tree.nodes.get('conv_shortcut'), 'skip_connection')


def hf_to_nemo_mapping(tree: SegTree):
    mapping = {}
    for nodename, subtree in tree.nodes.items():
        convert_name = subtree.convert_name
        convert_name = (convert_name + ".") if convert_name is not None else ""
        if subtree.is_leaf() and subtree.convert_name is not None:
            mapping[nodename] = subtree.convert_name
        else:
            submapping = hf_to_nemo_mapping(subtree)
            for k, v in submapping.items():
                mapping[nodename + "." + k] = convert_name + v
    return mapping


def convert_cond_keys(tree: SegTree):
    # map all conditioning keys
    if tree.nodes.get("add_embedding"):
        logging.info("Add embedding found...")
        tree['add_embedding'].convert_name = 'label_emb.0'
        dup_convert_name_recursive(tree['add_embedding.linear_1'], '0')
        dup_convert_name_recursive(tree['add_embedding.linear_2'], '2')
    if tree.nodes.get("time_embedding"):
        logging.info("Time embedding found...")
        tree['time_embedding'].convert_name = 'time_embed'
        dup_convert_name_recursive(tree['time_embedding.linear_1'], '0')
        dup_convert_name_recursive(tree['time_embedding.linear_2'], '2')


def convert_middle_keys(tree: SegTree):
    '''middle block is fixed (resnet -> attention -> resnet)'''
    mid = tree['mid_block']
    resnets = mid['resnets']
    attns = mid['attentions']
    mid.convert_name = 'middle_block'
    resnets['0'].convert_name = '0'
    resnets['1'].convert_name = '2'
    attns['0'].convert_name = '1'
    map_resnet_block(resnets['0'])
    map_resnet_block(resnets['1'])
    map_attention_block(attns['0'])


def convert_output_keys(hf_tree: SegTree):
    '''output keys is similar to input keys'''
    nemo_inp_blk = 0
    up_blocks = hf_tree['up_blocks']
    up_blocks_keys = sorted(list(up_blocks.nodes.keys()), key=int)

    for downblockid in up_blocks_keys:
        block = up_blocks[str(downblockid)]
        # compute number of resnets, attentions, downsamplers in this block
        resnets = block.nodes.get('resnets', SegTree())
        attentions = block.nodes.get('attentions', SegTree())
        upsamplers = block.nodes.get('upsamplers', SegTree())

        if len(attentions) == 0:  # no attentions, this is a UpBlock2D
            for resid in sorted(list(resnets.nodes.keys()), key=int):
                resid = str(resid)
                resnets[resid].convert_name = f"output_blocks.{nemo_inp_blk}.0"
                map_resnet_block(resnets[resid])
                nemo_inp_blk += 1

        elif len(attentions) == len(resnets):
            # there are attention blocks here -- each resnet+attention becomes a block
            for resid in sorted(list(resnets.nodes.keys()), key=int):
                resid = str(resid)
                resnets[resid].convert_name = f"output_blocks.{nemo_inp_blk}.0"
                map_resnet_block(resnets[resid])
                attentions[resid].convert_name = f"output_blocks.{nemo_inp_blk}.1"
                map_attention_block(attentions[resid])
                nemo_inp_blk += 1
        else:
            logging.warning("number of attention blocks is not the same as resnets - whats going on?")

        # if there is a upsampler, then also append it
        if len(upsamplers) > 0:
            nemo_inp_blk -= 1
            upsamplenum = (
                1 if len(attentions) == 0 else 2
            )  # if there are attention modules, upsample is module2, else it is module 1 (to stay consistent with SD)
            upsamplers['0'].convert_name = f"output_blocks.{nemo_inp_blk}.{upsamplenum}"
            dup_convert_name_recursive(upsamplers['0.conv'], 'conv')
            nemo_inp_blk += 1


def convert_finalout_keys(hf_tree: SegTree):
    dup_convert_name_recursive(hf_tree['conv_norm_out'], "out.0")
    dup_convert_name_recursive(hf_tree['conv_out'], "out.1")


def convert_encoder(hf_tree: SegTree):
    encoder = hf_tree['encoder']
    encoder.convert_name = 'encoder'
    dup_convert_name_recursive(encoder['conv_in'], 'conv_in')
    dup_convert_name_recursive(encoder['conv_out'], 'conv_out')
    dup_convert_name_recursive(encoder['conv_norm_out'], 'norm_out')

    # each block contains resnets and downsamplers
    # there are also optional attention blocks in the down module, but I havent encountered them yet
    encoder['down_blocks'].convert_name = 'down'
    for downid, downblock in encoder['down_blocks'].nodes.items():
        downblock.convert_name = downid
        downsamplers = downblock.nodes.get('downsamplers', SegTree())
        dup_convert_name_recursive(downblock['resnets'], 'block')
        # check for conv_shortcuts here
        for resid, resnet in downblock['resnets'].nodes.items():
            if resnet.nodes.get('conv_shortcut') is not None:
                resnet.nodes['conv_shortcut'].convert_name = 'nin_shortcut'
        if len(downsamplers) > 0:
            dup_convert_name_recursive(downsamplers['0'], 'downsample')

    # map the `mid_block` ( NeMo's mid layer is hardcoded in terms of number of modules)
    encoder['mid_block'].convert_name = 'mid'
    dup_convert_name_recursive(encoder[f'mid_block.resnets.0'], 'block_1')
    dup_convert_name_recursive(encoder[f'mid_block.resnets.1'], 'block_2')

    # attention part
    att = encoder['mid_block.attentions.0']
    att.convert_name = 'attn_1'
    dup_convert_name_recursive(att['group_norm'], 'norm')
    dup_convert_name_recursive(att['to_k'], 'k')
    dup_convert_name_recursive(att['to_q'], 'q')
    dup_convert_name_recursive(att['to_v'], 'v')
    dup_convert_name_recursive(att['to_out.0'], 'proj_out')


def convert_decoder(hf_tree: SegTree):
    decoder = hf_tree['decoder']
    decoder.convert_name = 'decoder'
    dup_convert_name_recursive(decoder['conv_in'], 'conv_in')
    dup_convert_name_recursive(decoder['conv_out'], 'conv_out')
    dup_convert_name_recursive(decoder['conv_norm_out'], 'norm_out')
    # each block contains resnets and downsamplers
    # map the `mid_block` ( NeMo's mid layer is hardcoded in terms of number of modules)
    decoder['mid_block'].convert_name = 'mid'
    dup_convert_name_recursive(decoder[f'mid_block.resnets.0'], 'block_1')
    dup_convert_name_recursive(decoder[f'mid_block.resnets.1'], 'block_2')
    # attention blocks
    att = decoder['mid_block.attentions.0']
    att.convert_name = 'attn_1'
    dup_convert_name_recursive(att['group_norm'], 'norm')
    dup_convert_name_recursive(att['to_k'], 'k')
    dup_convert_name_recursive(att['to_q'], 'q')
    dup_convert_name_recursive(att['to_v'], 'v')
    dup_convert_name_recursive(att['to_out.0'], 'proj_out')

    # up blocks contain resnets and upsamplers
    decoder['up_blocks'].convert_name = 'up'
    num_up_blocks = len(decoder['up_blocks'])
    for upid, upblock in decoder['up_blocks'].nodes.items():
        upblock.convert_name = str(num_up_blocks - 1 - int(upid))
        upsamplers = upblock.nodes.get('upsamplers', SegTree())
        dup_convert_name_recursive(upblock['resnets'], 'block')
        # check for conv_shortcuts here
        for resid, resnet in upblock['resnets'].nodes.items():
            if resnet.nodes.get('conv_shortcut') is not None:
                resnet.nodes['conv_shortcut'].convert_name = 'nin_shortcut'
        if len(upsamplers) > 0:
            dup_convert_name_recursive(upsamplers['0'], 'upsample')


def convert(args):
    logging.info(f"loading checkpoint {args.input_name_or_path}")
    _, hf_ckpt = load_hf_ckpt(args.input_name_or_path, args)
    hf_tree = model_to_tree(hf_ckpt)

    if args.model == 'unet':
        logging.info("converting unet...")
        convert_input_keys(hf_tree)
        convert_cond_keys(hf_tree)
        convert_middle_keys(hf_tree)
        convert_output_keys(hf_tree)
        convert_finalout_keys(hf_tree)
        # get mapping

    elif args.model == 'vae':
        logging.info("converting vae...")
        dup_convert_name_recursive(hf_tree['quant_conv'], 'quant_conv')
        dup_convert_name_recursive(hf_tree['post_quant_conv'], 'post_quant_conv')
        convert_encoder(hf_tree)
        convert_decoder(hf_tree)

    else:
        logging.error("incorrect model specification.")
        return

    # check mapping
    mapping = hf_to_nemo_mapping(hf_tree)
    if len(mapping) != len(hf_ckpt.keys()):
        logging.warning("not all keys are matched properly.")
    nemo_ckpt = {}

    for hf_key, nemo_key in mapping.items():
        nemo_ckpt[nemo_key] = hf_ckpt[hf_key]
    # save this
    torch.save(nemo_ckpt, args.output_path)
    logging.info(f"Saved nemo file to {args.output_path}")


if __name__ == '__main__':
    args = get_args()
    convert(args)
