# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import asyncio

import gradio as gr

from nemo.collections.nlp.modules.common.chat_css import CSS
from nemo.collections.nlp.modules.common.chatbot_component import Chatbot
from nemo.collections.nlp.modules.common.megatron.retrieval_services.util import (
    convert_retrieved_to_md,
    request_data,
    text_generation,
)

__all__ = ['RetroDemoWebApp', 'get_demo']

TURN_TOKEN = '<extra_id_1>'

DEFAULT_SYSTEM = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
SYSTEM_TOKEN = '<extra_id_0>System\n'
# HUMAN_TOKEN = 'Human:'
# ASSITANT_TOKEN = 'Assistant:'


def create_gen_function(port=5555, chat=False):
    if chat:

        def get_generation(
            prompt, preamble, greedy, add_BOS, token_to_gen, min_tokens, temp, top_p, top_k, repetition, end_strings
        ):
            if preamble is not None and preamble != '':
                prompt = SYSTEM_TOKEN + preamble + prompt
            data = {
                "sentences": [prompt],
                "tokens_to_generate": int(token_to_gen),
                "temperature": temp,
                "add_BOS": add_BOS,
                "top_k": top_k,
                "top_p": top_p,
                "greedy": greedy,
                "all_probs": False,
                "repetition_penalty": repetition,
                "min_tokens_to_generate": int(min_tokens),
                "end_strings": [i.strip() for i in end_strings.split(',') if len(i) != 0],
            }
            response = text_generation(data, port=port)
            sentences = response['sentences']
            bot_message = sentences[0]
            bot_message = bot_message[len(prompt) :]
            return bot_message

    else:

        def get_generation(
            prompt, greedy, add_BOS, token_to_gen, min_tokens, temp, top_p, top_k, repetition, end_strings
        ):
            data = {
                "sentences": [prompt],
                "tokens_to_generate": int(token_to_gen),
                "temperature": temp,
                "add_BOS": add_BOS,
                "top_k": top_k,
                "top_p": top_p,
                "greedy": greedy,
                "all_probs": False,
                "repetition_penalty": repetition,
                "min_tokens_to_generate": int(min_tokens),
                "end_strings": [i.strip() for i in end_strings.split(',') if len(i) != 0],
            }
            response = text_generation(data, port=port)
            sentences = response['sentences']
            bot_message = sentences[0]
            bot_message = bot_message[len(prompt) :]
            return bot_message

    return get_generation


def get_demo(share, username, password, server_port=5555, web_port=9889, loop=None):
    asyncio.set_event_loop(loop)
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=2, width=200):
                greedy_flag = gr.Checkbox(label="Greedy")
                add_BOS = gr.Checkbox(label="Add BOS token", value=False)
                token_to_gen = gr.Number(label='Number of Tokens to generate', value=300, type=int)
                min_token_to_gen = gr.Number(label='Min number of Tokens to generate', value=1, type=int)
                temperature = gr.Slider(minimum=0.0, maximum=10.0, value=1.0, label='Temperature', step=0.1)
                top_p = gr.Slider(minimum=0.0, maximum=1.0, step=0.02, value=0.9, label='Top P')
                top_k = gr.Slider(minimum=0, maximum=10000, step=2, value=0, label='Top K')
                repetition_penality = gr.Slider(
                    minimum=1.0, maximum=5.0, step=0.02, value=1.2, label='Repetition penalty'
                )
                end_strings = gr.Textbox(label="End strings (comma separated)", value="<|endoftext|>,", lines=1,)
            with gr.Column(scale=1, min_width=800):
                input_prompt = gr.Textbox(
                    label="Input",
                    value="Ariel was playing basketball. 1 of her shots went in the hoop. 2 of her shots did not go in the hoop. How many shots were there in total?",
                    lines=5,
                )
                output_box = gr.Textbox(value="", label="Output")
                btn = gr.Button(value="Submit")
                btn.click(
                    create_gen_function(server_port, chat=False),
                    inputs=[
                        input_prompt,
                        greedy_flag,
                        add_BOS,
                        token_to_gen,
                        min_token_to_gen,
                        temperature,
                        top_p,
                        top_k,
                        repetition_penality,
                        end_strings,
                    ],
                    outputs=[output_box],
                )
    demo.launch(share=share, server_port=web_port, server_name='0.0.0.0', auth=(username, password))


def get_chatbot_demo(share, username, password, server_port=5555, web_port=9889, loop=None):
    asyncio.set_event_loop(loop)
    with gr.Blocks(css=CSS) as demo:
        # store the mutliple turn conversation
        with gr.Row():
            with gr.Column(scale=2, width=200):
                # store the mutliple turn conversation
                session_state = gr.State(value=[])
                greedy_flag = gr.Checkbox(label="Greedy", value=True)
                add_BOS = gr.Checkbox(label="Add BOS token", value=False)
                token_to_gen = gr.Number(label='Number of Tokens to generate', value=300, type=int)
                min_token_to_gen = gr.Number(label='Min number of Tokens to generate', value=1, type=int)
                temperature = gr.Slider(minimum=0.0, maximum=10.0, value=1.0, label='Temperature', step=0.1)
                top_p = gr.Slider(minimum=0.0, maximum=1.0, step=0.02, value=0.9, label='Top P')
                top_k = gr.Slider(minimum=0, maximum=10000, step=2, value=0, label='Top K')
                repetition_penality = gr.Slider(
                    minimum=1.0, maximum=5.0, step=0.02, value=1.2, label='Repetition penalty'
                )
                end_strings = gr.Textbox(
                    label="End strings (comma separated)", value=f"<|endoftext|>,<extra_id_1>,", lines=1,
                )
                gr.HTML("<hr/>")
                human_name = gr.Textbox(label="Human Name", value="User", line=1,)
                assistant_name = gr.Textbox(label="Assistant Name", value="Assistant", line=1,)
                preamble = gr.Textbox(label="System", value=DEFAULT_SYSTEM, lines=2,)
            with gr.Column(scale=1, min_width=800):
                chatbot = Chatbot(elem_id="chatbot").style(height=800)
                msg = gr.Textbox(label="User", value="", lines=1,)
                clear = gr.Button("Clear")

                def user(user_message, history, session_state):
                    session_state.append(user_message)
                    user_message = user_message.replace('\n', '<br>')
                    return "", history + [[user_message, None]]

                def bot(
                    history,
                    preamble,
                    greedy_flag,
                    add_BOS,
                    token_to_gen,
                    min_token_to_gen,
                    temperature,
                    top_p,
                    top_k,
                    repetition_penality,
                    end_strings,
                    human_name,
                    assistant_name,
                    session_state,
                ):
                    prompt_text = ''
                    names = [human_name, assistant_name]
                    for i, meg in enumerate(session_state):
                        name = names[i % 2]
                        prompt_text += TURN_TOKEN + name + '\n' + meg + '\n'
                    prompt_text += TURN_TOKEN + assistant_name + '\n'
                    bot_message = create_gen_function(server_port, chat=True)(
                        prompt_text,
                        preamble,
                        greedy_flag,
                        add_BOS,
                        token_to_gen,
                        min_token_to_gen,
                        temperature,
                        top_p,
                        top_k,
                        repetition_penality,
                        end_strings,
                    )
                    if bot_message.endswith(TURN_TOKEN):
                        bot_message = bot_message[: -len(TURN_TOKEN)]
                    history[-1][1] = bot_message
                    session_state.append(bot_message.strip())
                    return history

                msg.submit(user, [msg, chatbot, session_state], [msg, chatbot], queue=False).then(
                    bot,
                    [
                        chatbot,
                        preamble,
                        greedy_flag,
                        add_BOS,
                        token_to_gen,
                        min_token_to_gen,
                        temperature,
                        top_p,
                        top_k,
                        repetition_penality,
                        end_strings,
                        human_name,
                        assistant_name,
                        session_state,
                    ],
                    chatbot,
                )

                def clear_fun(session_state):
                    session_state.clear()
                    return None

                clear.click(clear_fun, [session_state], chatbot, queue=False)
        demo.launch(share=share, server_port=web_port, server_name='0.0.0.0', auth=(username, password))


class RetroDemoWebApp:
    def __init__(self, text_service_ip, text_service_port, combo_service_ip, combo_service_port):
        self.text_service_ip = text_service_ip
        self.text_service_port = text_service_port
        self.combo_service_ip = combo_service_ip
        self.combo_service_port = combo_service_port

    def get_retro_generation(
        self,
        prompt,
        greedy,
        add_BOS,
        token_to_gen,
        min_tokens,
        temp,
        top_p,
        top_k,
        repetition,
        neighbors,
        weight,
        end_strings,
    ):
        data = {
            "sentences": [prompt],
            "tokens_to_generate": int(token_to_gen),
            "temperature": temp,
            "add_BOS": add_BOS,
            "top_k": top_k,
            "top_p": top_p,
            "greedy": greedy,
            "all_probs": False,
            "repetition_penalty": repetition,
            "min_tokens_to_generate": int(min_tokens),
            "neighbors": int(neighbors),
            "end_strings": [i.strip() for i in end_strings.split(',') if len(i) != 0],
        }
        self.update_weight(weight)
        output_json = text_generation(data, self.text_service_ip, self.text_service_port)
        sentences = output_json['sentences']
        retrieved = output_json['retrieved']
        return sentences[0], convert_retrieved_to_md(retrieved)

    def update_weight(self, weight):
        data = {"update_weight": [weight, 1.0 - weight]}
        return request_data(data, self.combo_service_ip, self.combo_service_port)

    def add_doc(self, doc, add_eos):
        data = {
            "sentences": [doc],
            "add_eos": add_eos,
        }
        return request_data(data, self.combo_service_ip, self.combo_service_port)

    def reset_index(self):
        data = {"reset": None}
        return request_data(data, self.combo_service_ip, self.combo_service_port)

    def run_demo(self, share, username, password, port):
        with gr.Blocks(css="table, th, td { border: 1px solid blue; table-layout: fixed; width: 100%; }") as demo:
            with gr.Row():
                with gr.Column(scale=2, width=200):
                    greedy_flag = gr.Checkbox(label="Greedy", value=True)
                    add_BOS = gr.Checkbox(label="Add BOS token", value=False)
                    token_to_gen = gr.Number(label='Number of Tokens to generate', value=30, type=int)
                    min_token_to_gen = gr.Number(label='Min number of Tokens to generate', value=1, type=int)
                    temperature = gr.Slider(minimum=0.0, maximum=10.0, value=1.0, label='Temperature', step=0.1)
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, step=0.02, value=0.9, label='Top P')
                    top_k = gr.Slider(minimum=0, maximum=10000, step=2, value=0, label='Top K')
                    repetition_penality = gr.Slider(
                        minimum=1.0, maximum=5.0, step=0.02, value=1.2, label='Repetition penalty'
                    )
                    end_strings = gr.Textbox(label="End strings (comma separated)", value="<|endoftext|>,", lines=1,)
                    k_neighbors = gr.Slider(minimum=0, maximum=50, step=1, value=2, label='Retrieved Documents')
                    weight = gr.Slider(
                        minimum=0.0, maximum=1.0, value=1.0, label='Weight for the Static Retrieval DB', step=0.02
                    )
                    add_retrival_doc = gr.Textbox(label="Add New Retrieval Doc", value="", lines=5,)
                    add_EOS = gr.Checkbox(label="Add EOS token to Retrieval Doc", value=False)
                    with gr.Row():
                        add_btn = gr.Button(value="Add")
                        reset_btn = gr.Button(value="Reset Index")
                    output_status = gr.Label(value='')
                    add_btn.click(self.add_doc, inputs=[add_retrival_doc, add_EOS], outputs=[output_status])
                    reset_btn.click(self.reset_index, inputs=[], outputs=[output_status])

                with gr.Column(scale=1, min_width=800):
                    input_prompt = gr.Textbox(
                        label="Input",
                        value="Ariel was playing basketball. 1 of her shots went in the hoop. 2 of her shots did not go in the hoop. How many shots were there in total?",
                        lines=5,
                    )
                    output_box = gr.Textbox(value="", label="Output")
                    btn = gr.Button(value="Submit")
                    output_retrieval = gr.HTML()
                    btn.click(
                        self.get_retro_generation,
                        inputs=[
                            input_prompt,
                            greedy_flag,
                            add_BOS,
                            token_to_gen,
                            min_token_to_gen,
                            temperature,
                            top_p,
                            top_k,
                            repetition_penality,
                            k_neighbors,
                            weight,
                            end_strings,
                        ],
                        outputs=[output_box, output_retrieval],
                    )
        demo.launch(share=share, server_port=port, server_name='0.0.0.0', auth=(username, password))
