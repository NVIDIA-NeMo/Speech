# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


import json
import os

import numpy as np
import tensorrt as trt
import tensorrt_llm
import tensorrt_llm.profiler as profiler
import torch
from PIL import Image
from tensorrt_llm import logger
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.runtime import ModelRunner, Session, TensorInfo
from torchvision import transforms
from transformers import CLIPImageProcessor


def trt_dtype_to_torch(dtype):
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.bfloat16:
        return torch.bfloat16
    else:
        raise TypeError("%s is not supported" % dtype)


class MultimodalModelRunner:

    def __init__(self, visual_engine_dir, llm_engine_dir):
        self.runtime_rank = tensorrt_llm.mpi_rank()
        device_id = self.runtime_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        self.device = "cuda:%d" % (device_id)

        self.stream = torch.cuda.Stream(torch.cuda.current_device())
        torch.cuda.set_stream(self.stream)

        # parse model type from visual engine config
        with open(os.path.join(visual_engine_dir, "config.json"), "r") as f:
            config = json.load(f)
        self.model_type = config['builder_config']['model_type']
        self.vision_precision = config['builder_config']['precision']

        self.num_frames = config['builder_config'].get('num_frames', None)
        self.image_size = config['builder_config'].get('image_size', None)

        self.profiling_iterations = 20

        self.init_image_encoder(visual_engine_dir)
        self.init_tokenizer(llm_engine_dir)
        self.init_llm(llm_engine_dir)

    def init_tokenizer(self, llm_engine_dir):
        if os.path.exists(os.path.join(llm_engine_dir, 'huggingface_tokenizer')):
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(llm_engine_dir, 'huggingface_tokenizer'))
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            from sentencepiece import SentencePieceProcessor

            sp = SentencePieceProcessor(os.path.join(llm_engine_dir, 'tokenizer.model'))

            class return_obj:

                def __init__(self, input_ids):
                    self.input_ids = input_ids

                def __getitem__(self, name):
                    if name in "input_ids":
                        return self.input_ids
                    else:
                        raise AttributeError(f"'return_obj' has no item '{name}'")

            # sentencepiece does not follow the same interface as HF
            class HFTokenizerInterface:

                def encode(self, x, return_tensors=None, **kwargs):
                    out = sp.encode(x)
                    if return_tensors == "pt":
                        out = torch.tensor(out)
                    return return_obj(out)

                def __call__(self, x, return_tensors=None, **kwargs):
                    return self.encode(x, return_tensors, **kwargs)

                def decode(self, x, **kwargs):
                    return sp.decode(x.tolist())

                def batch_decode(self, x, **kwargs):
                    return self.decode(x, **kwargs)

            self.tokenizer = HFTokenizerInterface()
            self.tokenizer.eos_token_id = sp.eos_id()
            self.tokenizer.bos_token_id = sp.bos_id()
            self.tokenizer.pad_token_id = sp.pad_id()

            self.tokenizer.padding_side = "right"

    def init_image_encoder(self, visual_engine_dir):
        vision_encoder_path = os.path.join(visual_engine_dir, 'visual_encoder.engine')
        logger.info(f'Loading engine from {vision_encoder_path}')
        with open(vision_encoder_path, 'rb') as f:
            engine_buffer = f.read()
        logger.info(f'Creating session from engine {vision_encoder_path}')
        self.visual_encoder_session = Session.from_serialized_engine(engine_buffer)

    def init_llm(self, llm_engine_dir):
        self.model = ModelRunner.from_dir(
            llm_engine_dir, rank=tensorrt_llm.mpi_rank(), debug_mode=False, stream=self.stream
        )
        self.model_config = self.model.session._model_config
        self.runtime_mapping = self.model.session.mapping

    def video_preprocess(self, video_path):
        from decord import VideoReader

        if isinstance(video_path, str):
            vr = VideoReader(video_path)
            num_frames = self.num_frames
            if num_frames == -1:
                frames = [Image.fromarray(frame.asnumpy()[:, :, ::-1]).convert('RGB') for frame in vr]
            else:
                # equally sliced frames into self.num_frames frames
                # if self.num_frames is greater than the number of frames in the video, we will repeat the last frame
                num_frames = min(num_frames, len(vr))
                indices = np.linspace(0, len(vr) - 1, num=num_frames, dtype=int)
                frames = [Image.fromarray(vr[idx].asnumpy()[:, :, ::-1]).convert('RGB') for idx in indices]
                if len(frames) < num_frames:
                    frames += [frames[-1]] * (num_frames - len(frames))
        elif isinstance(video_path, np.ndarray):
            num_frames = self.num_frames
            if num_frames == -1:
                frames = [Image.fromarray(frame[:, :, ::-1]).convert('RGB') for frame in video_path]
            else:
                # equally sliced frames into self.num_frames frames
                # if self.num_frames is greater than the number of frames in the video, we will repeat the last frame
                num_frames = min(num_frames, video_path.shape[0])
                indices = np.linspace(0, video_path.shape[0] - 1, num=num_frames, dtype=int)
                frames = [Image.fromarray(video_path[idx][:, :, ::-1]).convert('RGB') for idx in indices]
                if len(frames) < num_frames:
                    frames += [frames[-1]] * (num_frames - len(frames))
        else:
            frames = self.video_path

        processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.bfloat16)
        frames = processor.preprocess(frames, return_tensors="pt")['pixel_values']
        # make dtype consistent with vision encoder
        media_tensors = frames.to(
            tensorrt_llm._utils.str_dtype_to_torch(self.vision_precision)
        )  # [num_frames, 3, H, W]
        return media_tensors.unsqueeze(0)  # [1, num_frames, 3, H, W]

    def preprocess(self, warmup, pre_prompt, post_prompt, image, attention_mask, batch_size):
        if not warmup:
            profiler.start("Vision")

        visual_features, visual_atts = self.get_visual_features(image, attention_mask)

        if not warmup:
            profiler.stop("Vision")

        pre_input_ids = self.tokenizer(pre_prompt, return_tensors="pt", padding=True).input_ids
        if post_prompt[0] is not None:
            post_input_ids = self.tokenizer(post_prompt, return_tensors="pt", padding=True).input_ids
            if self.model_type == 'video-neva':
                length = pre_input_ids.shape[1] + post_input_ids.shape[1] + visual_atts.shape[2] * visual_atts.shape[1]
            else:
                length = pre_input_ids.shape[1] + post_input_ids.shape[1] + visual_atts.shape[1]
        else:
            post_input_ids = None
            length = pre_input_ids.shape[1] + visual_atts.shape[1]

        input_lengths = torch.IntTensor([length] * batch_size).to(torch.int32)

        input_ids, ptuning_args = self.setup_fake_prompts(
            visual_features, pre_input_ids, post_input_ids, input_lengths
        )

        return input_ids, input_lengths, ptuning_args, visual_features

    def generate(
        self,
        pre_prompt,
        post_prompt,
        image,
        decoder_input_ids,
        max_new_tokens,
        attention_mask,
        warmup,
        batch_size,
        top_k,
        top_p,
        temperature,
        repetition_penalty,
        num_beams,
    ):
        if not warmup:
            profiler.start("Generate")

        input_ids, input_lengths, ptuning_args, visual_features = self.preprocess(
            warmup, pre_prompt, post_prompt, image, attention_mask, batch_size
        )

        if warmup:
            return None

        profiler.start("LLM")
        end_id = self.tokenizer.eos_token_id

        ptuning_args[0] = torch.stack([ptuning_args[0]])
        output_ids = self.model.generate(
            input_ids,
            sampling_config=None,
            prompt_table=ptuning_args[0],
            max_new_tokens=max_new_tokens,
            end_id=end_id,
            pad_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.all_special_ids[0]
            ),
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
            output_sequence_lengths=False,
            return_dict=False,
        )

        profiler.stop("LLM")

        if tensorrt_llm.mpi_rank() == 0:
            # Extract a list of tensors of shape beam_width x output_ids.
            output_beams_list = [
                self.tokenizer.batch_decode(
                    output_ids[batch_idx, :, input_lengths[batch_idx] :], skip_special_tokens=True
                )
                for batch_idx in range(batch_size)
            ]

            stripped_text = [
                [output_beams_list[batch_idx][beam_idx].strip() for beam_idx in range(num_beams)]
                for batch_idx in range(batch_size)
            ]
            profiler.stop("Generate")
            return stripped_text
        else:
            profiler.stop("Generate")
            return None

    def get_visual_features(self, image, attention_mask):
        visual_features = {'input': image.to(tensorrt_llm._utils.str_dtype_to_torch(self.vision_precision))}
        if attention_mask is not None:
            visual_features['attention_mask'] = attention_mask
        tensor_info = [TensorInfo('input', str_dtype_to_trt(self.vision_precision), image.shape)]
        if attention_mask is not None:
            tensor_info.append(TensorInfo('attention_mask', trt.DataType.INT32, attention_mask.shape))

        visual_output_info = self.visual_encoder_session.infer_shapes(tensor_info)

        visual_outputs = {
            t.name: torch.empty(tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device=image.device)
            for t in visual_output_info
        }

        ok = self.visual_encoder_session.run(visual_features, visual_outputs, self.stream.cuda_stream)
        assert ok, "Runtime execution failed for vision encoder session"
        self.stream.synchronize()

        image_embeds = visual_outputs['output']
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        return image_embeds, image_atts

    def setup_fake_prompts(self, visual_features, pre_input_ids, post_input_ids, input_lengths):
        # Assemble fake prompts which points to image embedding actually
        if hasattr(self, 'num_frames') and (visual_features.shape[1] == self.num_frames):
            visual_features = visual_features.view(visual_features.shape[0], -1, visual_features.shape[-1])

        fake_prompt_id = torch.arange(
            self.model_config.vocab_size,
            self.model_config.vocab_size + visual_features.shape[0] * visual_features.shape[1],
        )
        fake_prompt_id = fake_prompt_id.reshape(visual_features.shape[0], visual_features.shape[1])

        if post_input_ids is not None:
            input_ids = [pre_input_ids, fake_prompt_id, post_input_ids]
        else:
            input_ids = [fake_prompt_id, pre_input_ids]
        input_ids = torch.cat(input_ids, dim=1).contiguous().to(torch.int32)

        ptuning_args = self.ptuning_setup(visual_features, input_ids, input_lengths)

        return input_ids, ptuning_args

    def ptuning_setup(self, prompt_table, input_ids, input_lengths):
        hidden_size = self.model_config.hidden_size * self.runtime_mapping.tp_size
        if prompt_table is not None:
            task_vocab_size = torch.tensor(
                [prompt_table.shape[1]],
                dtype=torch.int32,
            ).cuda()
            prompt_table = prompt_table.view((prompt_table.shape[0] * prompt_table.shape[1], prompt_table.shape[2]))

            assert prompt_table.shape[1] == hidden_size, "Prompt table dimensions do not match hidden size"

            prompt_table = prompt_table.cuda().to(
                dtype=tensorrt_llm._utils.str_dtype_to_torch(self.model_config.dtype)
            )
        else:
            prompt_table = torch.empty([1, hidden_size]).cuda()
            task_vocab_size = torch.zeros([1]).cuda()

        if self.model_config.remove_input_padding:
            tasks = torch.zeros([torch.sum(input_lengths)], dtype=torch.int32).cuda()
        else:
            tasks = torch.zeros(input_ids.shape, dtype=torch.int32).cuda()

        return [prompt_table, tasks, task_vocab_size]

    def setup_inputs(self, input_text, raw_image, batch_size):
        attention_mask = None

        if self.model_type == "neva":
            image_size = self.image_size
            dtype = torch.float32
            transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            image = transform(raw_image).to(dtype).unsqueeze(0)

            if input_text is None:
                input_text = "Hi! What is in this image?"

            pre_prompt = "<extra_id_0>System\n\n<extra_id_1>User\n"
            post_prompt = f"\n{input_text}\n<extra_id_1>Assistant\n"
        elif self.model_type == "video-neva":
            image = self.video_preprocess(raw_image)  # shape (1, num_frames, 3, H, W)

            if input_text is None:
                input_text = "Hi! What is in this video?"

            # SteerLM prompt template
            pre_prompt = """<extra_id_0>System\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n<extra_id_1>User"""
            post_prompt = (
                f"\n{input_text}\n<extra_id_1>Assistant\n<extra_id_2>quality:4,toxicity:0,humor:0,creativity:0,helpfulness:4,correctness:4,coherence:4,complexity:4,verbosity:4\n"
                ""
            )
        else:
            raise RuntimeError(f"Invalid model type {self.model_type}")

        # Repeat inputs to match batch size
        pre_prompt = [pre_prompt] * batch_size
        post_prompt = [post_prompt] * batch_size
        if image.dim() == 5:
            image = image.expand(batch_size, -1, -1, -1, -1).contiguous()
        else:
            image = image.expand(batch_size, -1, -1, -1).contiguous()
        image = image.to(self.device)

        # Generate decoder_input_ids for enc-dec models
        # Custom prompts can be added as:
        # decoder_input_ids = model.tokenizer(decoder_prompt).input_ids
        decoder_input_ids = None

        return input_text, pre_prompt, post_prompt, image, decoder_input_ids, attention_mask

    def run(
        self,
        input_text,
        input_image,
        max_new_tokens,
        batch_size,
        top_k,
        top_p,
        temperature,
        repetition_penalty,
        num_beams,
        run_profiling=False,
        check_accuracy=False,
    ):
        input_text, pre_prompt, post_prompt, processed_image, decoder_input_ids, attention_mask = self.setup_inputs(
            input_text, input_image, batch_size
        )

        self.generate(
            pre_prompt,
            post_prompt,
            processed_image,
            decoder_input_ids,
            max_new_tokens,
            attention_mask=attention_mask,
            warmup=True,
            batch_size=batch_size,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
        )
        num_iters = self.profiling_iterations if run_profiling else 1
        for _ in range(num_iters):
            output_text = self.generate(
                pre_prompt,
                post_prompt,
                processed_image,
                decoder_input_ids,
                max_new_tokens,
                attention_mask=attention_mask,
                warmup=False,
                batch_size=batch_size,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                num_beams=num_beams,
            )
        if self.runtime_rank == 0:
            self.print_result(input_text, output_text, batch_size, num_beams, run_profiling, check_accuracy)
        return output_text

    def print_result(self, input_text, output_text, batch_size, num_beams, run_profiling, check_accuracy):
        if not run_profiling and not check_accuracy:
            return
        logger.info("---------------------------------------------------------")
        if self.model_type != 'nougat':
            logger.info(f"\n[Q] {input_text}")
        logger.info(f"\n[A] {output_text[0]}")

        if num_beams == 1:
            output_ids = self.tokenizer(output_text[0][0], add_special_tokens=False)['input_ids']
            logger.info(f"Generated {len(output_ids)} tokens")

        if check_accuracy:
            for i in range(batch_size - 1):
                if not (output_text[i] == output_text[i + 1]):
                    logger.info(f"Output {i} and {i + 1} do not match")
                    assert False

                assert 'robot' in output_text[0][0].lower()

        if run_profiling:
            msec_per_batch = lambda name: 1000 * profiler.elapsed_time_in_sec(name) / self.profiling_iterations
            logger.info('Latencies per batch (msec)')
            logger.info('TRT vision encoder: %.1f' % (msec_per_batch('Vision')))
            logger.info('TRTLLM LLM generate: %.1f' % (msec_per_batch('LLM')))
            logger.info('Multimodal generate: %.1f' % (msec_per_batch('Generate')))

        logger.info("---------------------------------------------------------")

    def load_test_media(self, input_media):
        if self.model_type == "video-neva":
            media = input_media
        elif self.model_type == "neva":
            media = Image.open(input_media).convert('RGB')
        else:
            raise RuntimeError(f"Invalid model type {self.model_type}")

        return media
