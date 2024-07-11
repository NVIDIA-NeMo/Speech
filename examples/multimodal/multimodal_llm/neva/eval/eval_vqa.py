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
#

"""
This script is used for evaluating Video Question Answering task by leveraging LLM API as a judge.
It accepts one JSON file. The JSON file should have the following structure:
[
    {
        "video": "YRvBOLRgZNc_2".mp4",
        "question_id": "v_yVgL8sJQxYo_2_5",
        "question": "What tools are used to apply foundation on the skin between <5s> and <60s>?",
        "ref_answer": "A brush and blender.",
        "duration": 102.002002002002,
        "pred_answer": "A brush",
    },
    {
        "video": "yVgL8sJQxYo_2.mp4",    # not a must-to-have field
        "question": "How long does the action of applying foundation take?",
        "question_id": "v_yVgL8sJQxYo_2_5"
        "ref_answer": "The action takes around 55 seconds (<60s> - <5s>)."
        "duration": 102.002002002002,    # not a must-to-have field
        "pred_answer": "This action takes around 50 seconds.",
    }
  
  ...
]

`video` and `duration` are two optional fields. If not provided, the script will ignore them.

Notice that the time token here is represented as  '<%ss>'.format(time_in_seconds).

For the external LLM API, we use `meta/llama3-70b-instruct"` as an example.
You can go to: https://build.nvidia.com/explore/discover to choose the one that fits your needs.
Notice the API might be a little bit different.

You also need an `API_TOKEN` from here: https://build.nvidia.com/explore/discover#llama3-70b
Click the `Get API Key` and save your key in the environment variable `API_TOKEN`.

USAGE:
API_TOKEN=<YOUR API> python eval_qa.py --input_file <path_to_json_file> --output_dir <path_to_output_dir> --save_mid_result
"""

import argparse
import ast
import json
import os
import re

import requests


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Video Question Answering task.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the prediction file. json list file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--save_mid_result", action="store_true", help="Whether to save the intermediate results.")
    return parser.parse_args()


INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
# MODEL="mistralai/mixtral-8x22b-instruct-v0.1"  # no `system` role
MODEL = "meta/llama3-70b-instruct"


def request_nvidia_api(messages):
    API_TOKEN = os.getenv("API_TOKEN", "")  # ADD NGC API TOKEN HERE
    if not API_TOKEN:
        raise ValueError("Please provide the API_TOKEN in the environment variable.")
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "accept": "text/event-stream",
        "content-type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.5,
        "top_p": 1.0,
        "max_tokens": 2048,
        "seed": 42,
        "stream": True,
    }
    invoke_url = INVOKE_URL
    response = requests.post(invoke_url, headers=headers, json=payload, stream=True)
    output = ""
    for line in response.iter_lines():
        if line == b'data: [DONE]':
            break
        if line:
            res = json.loads(line.decode("utf-8").split("data: ")[1])
            if 'content' in res['choices'][0]['delta']:
                output += res['choices'][0]['delta']['content']
    return output.lstrip().strip()


def convert_time_token(text):
    # use regular expression to convert <12> <56>  to <12s> <56s>
    return re.sub(r'<(\d+)>', r'<\1s>', text)


def get_result(question, answer, pred, key, output_dir, save_mid_result=False):
    messages = [
        {
            "role": "system",
            "content": "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
            "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
            "------"
            "##INSTRUCTIONS: "
            "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
            "- Consider synonyms or paraphrases as valid matches.\n"
            "- Evaluate the correctness of the prediction compared to the answer.",
        },
        {
            "role": "user",
            "content": "Please evaluate the following video-based question-answer pair:\n\n"
            f"Question: {question}\n"
            f"Correct Answer: {answer}\n"
            f"Predicted Answer: {pred}\n\n"
            "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
            "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
            "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}.",
        },
    ]
    try:
        response_message = request_nvidia_api(messages)
        response_dict = ast.literal_eval(response_message)
    except Exception as e:
        print(f"Error processing file {key}: {e}")
        return []
    qa_set = {"question": question, "ref_answer": answer, "pred_answer": pred}
    result_qa_pair = [response_dict, qa_set]
    if save_mid_result:
        with open(f"{output_dir}/{key}.json", "w") as f:
            json.dump(result_qa_pair, f)
    return result_qa_pair


def main():
    args = parse_args()
    input_file = args.input_file
    output_dir = args.output_dir
    save_mid_result = args.save_mid_result
    with open(input_file, "r") as f:
        data = json.load(f)

    tasks = []
    key = 0
    for item in data:
        question = item["question"]
        item["ref_answer"] = convert_time_token(item["ref_answer"])
        tasks.append((question, item["ref_answer"], item["pred_answer"], key, output_dir, save_mid_result))
        key += 1

    # TODO: parallelize the requests
    results = []
    while len(tasks) > 0:
        task = tasks.pop()
        key = task[3]
        cur_result = get_result(*task)
        if cur_result == []:
            tasks.append(task)
            continue
        results.append((key, cur_result))

    score_sum = count = yes_count = no_count = 0
    for key, result in results:
        try:
            count += 1
            score_sum += int(result[0]["score"])

            if "yes" in result[0]["pred"].lower():
                yes_count += 1
            elif "no" in result[0]["pred"].lower():
                no_count += 1
        except Exception as e:
            print(f"Error processing file {key}")

    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)
    result_file = os.path.join(output_dir, "metrics.json")
    metrics = {
        "average_score": average_score,
        "accuracy": accuracy,
        "no_count": no_count,
        "yes_count": yes_count,
        "model": MODEL,
    }
    print("Metrics: ", metrics)
    with open(result_file, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
