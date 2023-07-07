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

try:
    import gradio as gr

    GRADIO_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    GRADIO_AVAILABLE = False

from nemo.collections.nlp.modules.common.chat_css import CSS
from nemo.collections.nlp.modules.common.megatron.retrieval_services.util import (
    convert_retrieved_to_md,
    request_data,
    text_generation,
)

__all__ = ['RetroDemoWebApp', 'get_demo']

TURN_TOKEN = '<extra_id_1>'

PROMPT_PRESETS = {
    "DIALOGUE": {
        "SYSTEM_TURN_TOKEN": '',
        "USER_TURN_TOKEN": '<extra_id_1>',
        "BOT_TURN_TOKEN": '<extra_id_2>',
        "END_OF_NAME": '',
        "END_OF_TURN": '\n',
    },
    "DIALOGUE2": {
        "SYSTEM_TURN_TOKEN": '<extra_id_0>System\n',
        "USER_TURN_TOKEN": '<extra_id_1>',
        "BOT_TURN_TOKEN": '<extra_id_1>',
        "END_OF_NAME": '\n',
        "END_OF_TURN": '\n',
    },
}


PRESETS = {
    "K1-Greedy": {"temperature": 1.0, "top_p": 0.9, "top_k": 1, "repetition_penalty": 1.0,},
    "K50": {"temperature": 0.75, "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.0,},
    "K50-Creative": {"temperature": 0.85, "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.0,},
    "K50-Precise": {"temperature": 0.1, "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.0,},
    "K50-Original": {"temperature": 0.9, "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.0,},
    "Nucleus9": {"temperature": 0.8, "top_p": 0.9, "top_k": 10000, "repetition_penalty": 1.0,},
    "Custom": {"temperature": 0.75, "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.0,},
}


def check_gradio_import():
    if not GRADIO_AVAILABLE:
        msg = (
            f"could not find the gradio library.\n"
            f"****************************************************************\n"
            f"To install it, please follow the steps below:\n"
            f"pip install gradio==3.34.0\n"
        )
        raise ImportError(msg)


def create_gen_function(port=5555, chat=False):
    def get_generation(prompt, greedy, add_BOS, token_to_gen, min_tokens, temp, top_p, top_k, repetition, end_strings):
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
        if bot_message.find('<extra_id_0') < 0:
            # hack due to the problem that huggingface's tokenizer strips out the <extra_id_x> token
            prompt = prompt.replace('<extra_id_0>', '').replace('<extra_id_1>', '').replace('<extra_id_2>', '')
        bot_message = bot_message[len(prompt) :]
        return bot_message

    return get_generation


def get_demo(share, username, password, server_port=5555, web_port=9889, loop=None):
    check_gradio_import()
    asyncio.set_event_loop(loop)
    with gr.Blocks(css=CSS) as demo:
        with gr.Row():
            with gr.Column(scale=2, width=200):
                # store the mutliple turn conversation
                token_to_gen = gr.Number(label='Number of Tokens to generate', value=300, type=int)
                min_token_to_gen = gr.Number(label='Min number of Tokens to generate', value=1, type=int)
                seed = gr.Number(label='Random seed', value=0, type=int)
                end_strings = gr.Textbox(label="End strings (comma separated)", value="<extra_id_1>,", lines=1,)
                add_BOS = gr.Checkbox(label="Add BOS token", value=False)
                sampling_method = gr.Dropdown(
                    list(PRESETS.keys()), label='Sampling Presets', default='K50', value='K50'
                )
                temperature = gr.Slider(minimum=0.0, maximum=5.0, value=0.75, label='Temperature', step=0.1)
                top_p = gr.Slider(minimum=0.0, maximum=1.0, step=0.02, value=0.95, label='Top P')
                top_k = gr.Slider(minimum=0, maximum=1024, step=2, value=50, label='Top K')

                repetition_penality = gr.Slider(
                    minimum=1.0, maximum=5.0, step=0.02, value=1.0, label='Repetition penalty'
                )

                def set_sampling(x):
                    return list(PRESETS[x].values())

                sampling_method.change(
                    set_sampling, inputs=[sampling_method], outputs=[temperature, top_p, top_k, repetition_penality]
                )

            with gr.Column(scale=1, min_width=900):
                text = gr.Textbox(label="Playground", value="", lines=60, placeholder="Type something here...",)
                submit_btn = gr.Button("Generate")
                clear = gr.Button("Clear")

                def on_submit(
                    prompt_text,
                    token_to_gen,
                    temperature,
                    top_p,
                    top_k,
                    repetition_penality,
                    seed,
                    end_strings,
                    add_BOS,
                    min_token_to_gen,
                ):

                    output = create_gen_function(server_port)(
                        prompt_text,
                        False,
                        add_BOS,
                        token_to_gen,
                        min_token_to_gen,
                        temperature,
                        top_p,
                        top_k,
                        repetition_penality,
                        end_strings,
                    )
                    print(output)
                    print('-------------------')
                    return prompt_text + output

                def clear_fun():
                    return ''

                submit_btn.click(
                    on_submit,
                    [
                        text,
                        token_to_gen,
                        temperature,
                        top_p,
                        top_k,
                        repetition_penality,
                        seed,
                        end_strings,
                        add_BOS,
                        min_token_to_gen,
                    ],
                    [text],
                    queue=False,
                )
                clear.click(clear_fun, None, text, queue=False)
        demo.queue(concurrency_count=16).launch(
            share=share, server_port=web_port, server_name='0.0.0.0', auth=(username, password)
        )


def get_chatbot_demo(
    share, username, password, server_port=5555, web_port=9889, loop=None, value=False, defaults=None
):
    check_gradio_import()
    from nemo.collections.nlp.modules.common.chatbot_component import Chatbot

    asyncio.set_event_loop(loop)
    with gr.Blocks(css=CSS) as demo:
        with gr.Row():
            with gr.Column(scale=2, width=200):
                # store the mutliple turn conversation
                session_state = gr.State(value=[])
                token_to_gen = gr.Number(label='Number of Tokens to generate', value=300, type=int)
                seed = gr.Number(label='Random seed', value=0, type=int)
                prompt_presets = gr.Dropdown(
                    list(PROMPT_PRESETS.keys()), label='Template Presets', default='DIALOGUE2', value='DIALOGUE2'
                )
                sampling_method = gr.Dropdown(
                    list(PRESETS.keys()), label='Sampling Presets', default='K50', value='K50'
                )
                with gr.Accordion("Sampling Parameters", open=False):
                    temperature = gr.Slider(
                        minimum=0.0, maximum=5.0, value=0.75, label='Temperature', step=0.1, interactive=False
                    )
                    top_p = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.02, value=0.95, label='Top P', interactive=False
                    )
                    top_k = gr.Slider(minimum=0, maximum=1024, step=2, value=50, label='Top K', interactive=False)
                    repetition_penality = gr.Slider(
                        minimum=1.0, maximum=5.0, step=0.02, value=1.0, label='Repetition penalty', interactive=False
                    )

                with gr.Accordion("Value Parameters", open=True, visible=value):
                    keys = ['quality', 'toxicity', 'humor', 'creativity', 'violence', 'helpfulness', 'not_appropriate']
                    quality_value = gr.Slider(
                        minimum=0, maximum=9, step=1, value=9, label='Quality', interactive=True, visible=True
                    )
                    toxicity_value = gr.Slider(
                        minimum=0, maximum=9, step=1, value=0, label='Toxicity', interactive=True, visible=True
                    )
                    humor_value = gr.Slider(
                        minimum=0, maximum=9, step=1, value=0, label='Humor', interactive=True, visible=True
                    )
                    creativity_value = gr.Slider(
                        minimum=0, maximum=9, step=1, value=0, label='Creativity', interactive=True, visible=True
                    )
                    violence_value = gr.Slider(
                        minimum=0, maximum=9, step=1, value=0, label='Violence', interactive=True, visible=True
                    )
                    helpfulness_value = gr.Slider(
                        minimum=0, maximum=9, step=1, value=9, label='Helpfulness', interactive=True, visible=True
                    )
                    not_appropriate_value = gr.Slider(
                        minimum=0, maximum=9, step=1, value=0, label='Not Appropriate', interactive=True, visible=True
                    )
                    used_value = gr.CheckboxGroup(keys, value=keys)

                    def change_visibility(x):
                        values = []
                        for key in keys:
                            if key in x:
                                values.append(gr.update(visible=True))
                            else:
                                values.append(gr.update(visible=False))
                        return values

                    used_value.change(
                        change_visibility,
                        inputs=[used_value],
                        outputs=[
                            quality_value,
                            toxicity_value,
                            humor_value,
                            creativity_value,
                            violence_value,
                            helpfulness_value,
                            not_appropriate_value,
                        ],
                    )

                def set_sampling(x):
                    if x == 'Custom':
                        values = [gr.update(value=v, interactive=True) for v in PRESETS[x].values()]
                        return values
                    else:
                        values = [gr.update(value=v, interactive=False) for v in PRESETS[x].values()]
                        return values

                sampling_method.change(
                    set_sampling, inputs=[sampling_method], outputs=[temperature, top_p, top_k, repetition_penality]
                )

                gr.HTML("<hr>")
                human_name = gr.Textbox(label="Human Name", value=defaults['user'], line=1,)
                assistant_name = gr.Textbox(label="Assistant Name", value=defaults['assistant'], line=1,)
                preamble = gr.Textbox(label="System", value=defaults['system'], lines=2,)

                def set_prompt(x):
                    if x == "DIALOGUE":
                        return '', ''
                    return defaults['user'], defaults['assistant']

                prompt_presets.change(set_prompt, inputs=[prompt_presets], outputs=[human_name, assistant_name])

            with gr.Column(scale=1, min_width=900):
                chatbot = Chatbot(elem_id="chatbot").style(height=800)
                msg = gr.Textbox(label="User", value="", lines=1,)
                clear = gr.Button("Clear")

                def user(user_message, history, session_state):
                    session_state.append(user_message)
                    user_message = user_message.replace('\n', '<br>')
                    return "", history + [[user_message, None]]

                def get_value_str(values_array, used_value):
                    if len(used_value) == 0:
                        return ''
                    assert len(values_array) == len(keys)
                    value_str = '<extra_id_2>'
                    elements = []
                    for i, key in enumerate(keys):
                        if key in used_value:
                            elements.append(f'{key}:{values_array[i]}')
                    value_str += ','.join(elements) + '\n'
                    return value_str

                def bot(
                    history,
                    preamble,
                    token_to_gen,
                    temperature,
                    top_p,
                    top_k,
                    repetition_penality,
                    seed,
                    human_name,
                    assistant_name,
                    session_state,
                    prompts_presets,
                    quality_value,
                    toxicity_value,
                    humor_value,
                    creativity_value,
                    violence_value,
                    helpfulness_value,
                    not_appropriate_value,
                    used_value,
                ):

                    values_array = [
                        quality_value,
                        toxicity_value,
                        humor_value,
                        creativity_value,
                        violence_value,
                        helpfulness_value,
                        not_appropriate_value,
                    ]
                    if value:
                        value_str = get_value_str(values_array, used_value)
                    else:
                        value_str = ''

                    prompt_preset = PROMPT_PRESETS[prompts_presets]
                    prompt_text = ''
                    names = [human_name, assistant_name]
                    turn_tokens = [prompt_preset['USER_TURN_TOKEN'], prompt_preset['BOT_TURN_TOKEN']]
                    for i, meg in enumerate(session_state):
                        name = names[i % 2]
                        turn = turn_tokens[i % 2]
                        prompt_text += turn + name + prompt_preset['END_OF_NAME'] + meg + prompt_preset['END_OF_TURN']
                    prompt_text += (
                        prompt_preset['BOT_TURN_TOKEN'] + assistant_name + prompt_preset['END_OF_NAME'] + value_str
                    )
                    prompt_text = prompt_preset['SYSTEM_TURN_TOKEN'] + preamble + prompt_text
                    bot_message = create_gen_function(server_port)(
                        prompt_text,
                        False,
                        False,
                        token_to_gen,
                        1,
                        temperature,
                        top_p,
                        top_k,
                        repetition_penality,
                        '<extra_id_1>',
                    )
                    if bot_message.endswith(TURN_TOKEN):
                        bot_message = bot_message[: -len(TURN_TOKEN)]
                    history[-1][1] = bot_message
                    print(prompt_text)
                    print(bot_message)
                    print('-------------------')
                    session_state.append(value_str + bot_message.strip())
                    return history

                msg.submit(user, [msg, chatbot, session_state], [msg, chatbot], queue=False).then(
                    bot,
                    [
                        chatbot,
                        preamble,
                        token_to_gen,
                        temperature,
                        top_p,
                        top_k,
                        repetition_penality,
                        seed,
                        human_name,
                        assistant_name,
                        session_state,
                        prompt_presets,
                        quality_value,
                        toxicity_value,
                        humor_value,
                        creativity_value,
                        violence_value,
                        helpfulness_value,
                        not_appropriate_value,
                        used_value,
                    ],
                    [chatbot],
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
        check_gradio_import()
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
