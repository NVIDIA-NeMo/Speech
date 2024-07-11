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
This script is used to generate the QA data from the DVC dataset by using the NVIDIA LLM API.
The DVC dataset should have the below structure:
 
 {
    "-4RXOT_UfpM_1": {
    "duration": 34.00066733400067,
    "timestamps": [],
    "sentences": []
  },
  "-4RXOT_UfpM_2": {
    "duration": 119.01901901901903,
    "timestamps": [
      [5, 22],
      [22, 56],
      [90, 114]
    ],
    "sentences": [
      "Apply concealer on the eyelids and blend with sponge",
      "Apply powder on the eyelids with brush",
      "Apply eyeshadow on the crease with brush"
    ]
    ....
}

The generated dataset format is as follows:
[
    {
        "video_id": "-4RXOT_UfpM_1",
        "conversations": [
      {
        "role": "user",
        "content": "Provide a detailed description of the makeup tutorial video."
      },
      {
        "role": "assistant",
        "content": "<0s> <34.00066733400067s> The video does not describe any specific event."
      },
      {
        "role": "user",
        "content": "Is there any event described in the video between <0s> and <34.00066733400067s>?"
      },
      {
        "role": "assistant",
        "content": "No, there is no event described in the video between <0s> and <34.00066733400067s>."
      }
    ],
    "duration": 34.00066733400067,
    },
    {
    "video_id": "-4RXOT_UfpM_2",
    "conversations": [
      {
        "role": "user",
        "content": "Provide a detailed description of the makeup tutorial video."
      },
      {
        "role": "assistant",
        "content": "<5s> <22s> Concealer is applied to the eyelids and blended using a sponge. \n<22s> <56s> Powder is then applied to the eyelids using a brush. \n<90s> <114s> Eyeshadow is applied to the crease with a brush."
      },
      {
        "role": "user",
        "content": "How long does the application of concealer on the eyelids take in the video?"
      },
      {
        "role": "assistant",
        "content": "The application of concealer on the eyelids takes from <5s> to <22s>, which is 17 seconds."
      },
      ....
        ],
    "duration": 119.01901901901903,
    },
....
]

## USAGE:
The default model is llama3-70b-instruct: https://build.nvidia.com/explore/discover#llama3-70b
Go to the url and click the `Get Api Key` in the top right corner to get the API token.

You can also go to https://build.nvidia.com/explore/discover to explore different models and get the API token.
Notice the invoke_url for different model is different. Please check the url on the related model page.

export API_TOKEN=<API_TOKEN_YoU_GOT_FROM_THE_WEBSITE>
python generate_qa_data.py --input_json <path_to_input_json>
                        --output_dir <path_to_output_dir>   # the middle response and the final train.json would be be saved here
                        --generate_one_sample   # to generate one test sample  or just remove it to generate the whole dataset


Please refer to the `convert_instruction_dataset.py` script for converting QA dataset to the format required by the model training script.
"""

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import List, Union

import requests
from tqdm import tqdm

MODEL = "meta/llama3-70b-instruct"
INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Conversation:
    video_id: str
    conversations: List[Message]
    duration: Union[str, None] = None
    source: str = "video"


def request_nvidia_api(messages, temperature):
    API_TOKEN = os.getenv("API_TOKEN", "")  # ADD NGC API TOKEN HERE

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "accept": "text/event-stream",
        "content-type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
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


PROMPT_QA = """Based on the provided video dense captions, create a set of questions and answers that adhere to the following criteria:

- Construct 1 to 5 questions that are direct and seek clear, definitive answers related to the temporal aspects of events within the video. 
- If the dense caption is not provided, only ask if there is a dense caption or not.
- If the dense caption is only 1 sentence, only generate 1 to 2 simple questions.
- Focus solely on the events explicitly mentioned within the given timeframe of the video. Avoid questions that require assumptions or details not provided in the description, such as actions occurring before or after the specified timestamps.
- Only 2 questions may need info from two or more sentences in the dense caption to be answered.
- Each question should directly relate to the visible actions or details mentioned in the provided context, ensuring that answers can be definitively based on the given description.
- Do not create questions about events or actions that are implied but not described, such as the teammate passing the ball before the moment captured in the provided context.
- Formulate questions that explore the content within the given timestamps, emphasizing the sequence of actions, or any other detail explicitly mentioned.
- Your questions should explore details beyond the given sentences but remain within the scope of the video's visual content. Avoid inquiries about events not covered in the descriptions or requiring additional context beyond the video clip presented.
- Incorporate the event timestamps directly into both the questions and answers to maintain consistency and clarity.
- The conversation should reflect your role as a visual AI assistant, focusing on providing insightful and precise answers based on the video's content without inferring or assuming information not evident in the video.
- Structure your responses in JSON format, with each entry consisting of a 'question' and 'answer' pair. Ensure your explanations reveal your analytical process, focusing on how you interpret the video content to address the queries raised.
- Some question may be a follow-up question to the previous question.

Example JSON response structure:
[
    {
        "question": "Generated question",
        "answer": "Answer to the question."
    },
    ...
]

**Always** use timestamp placeholders '<TIMEs>' to denote specific times and durations within your questions and answers, ensuring clarity in temporal references."
Do not include anything besides json in the response.
"""

PROMPT_REWRITE_DENSE_CAPTION = """Based on the provided video dense captions, rewrite it and generate a prompt to adhere to the following criteria:

- Keep the keywords and details from the original dense caption and prompt.
- Ensure the rewritten dense caption and prompt are concise and clear, maintaining the original meaning and context.
- The rewritten dense caption should always denote specific times and durations within the video, ensuring clarity in temporal references.
- The rewritten prompt should be diverse and engaging, encouraging the model/assistant to provide concise and clear dense captions based on the video's content.
- Ensure that the prompt is a single sentence.
- If the prompt requires a specific video type or event, ensure that it is mentioned in the prompt.
- If the prompt require to dense caption formally, dense captions should always begin with the start and end timestamp '<START TIMEs> <END TIMEs>', then the DESCRIPTION, and include \\n to separate the sentences.
- If the prompt doesn't require to dense caption formally, you can have begin and end timestamps in the middle of the dense caption (like from/between <TIMEs> to/and <TIMEs>). You can use either <space> or \\n to separate the sentences in the dense caption.
- You can include domain specific keywords, like sport, medical, warehouse, etc., to specify the type of video.
- If the dense caption specify specific events, ensure that they are mentioned in the prompt.
- The rewritten dense caption and prompt should be structured in JSON format, with the 'prompt' and 'caption' keys.

Example JSON response structure:
{
    "prompt": "Rewrited prompt",
    "caption": "Rewrited dense caption."
}

Note: **Always** use timestamp placeholders '<TIMEs>' to denote specific times and durations in the rewritten dense caption, ensuring clarity in temporal references.

Example Prompts:
- Provide a detailed description of the given basketball video.
- Write a informative and formal summary of the sport video.
- Write a informative summary of the medical video.
- Give me a detailed description of the provided video.

Do not include anything besides json in the response.
"""


def load_data(data_path):
    result = []

    # Open the cleaned JSON file in read mode
    with open(data_path, 'r') as f:
        data_dict = json.load(f)
    for key, value in data_dict.items():
        print(value)
        if len(value["sentences"]) == 0:
            dense_caption = f"<0s> <{value['duration']}s> No event is described in the video."
        else:
            dense_caption = []
            for sentence, timestamp in zip(value["sentences"], value["timestamps"]):
                dense_caption.append(f"<{timestamp[0]}s> <{timestamp[1]}s> {sentence}")
            dense_caption = "\n".join(dense_caption)

        dense_caption = "Below is a dense caption of the given video:\n" + dense_caption
        result.append((key, dense_caption, value.get("duration", None)))

    return result


def generate_question_answers(dense_caption):
    answer = request_nvidia_api(
        [{"role": "system", "content": PROMPT_QA}, {"role": "user", "content": dense_caption}], 0.6
    )

    try:
        return json.loads(answer)
    except Exception as e:
        print(f"Failed to parse the response: {answer}")
        return None


def rewrite_dense_caption(dense_caption):
    answer = request_nvidia_api(
        [{"role": "system", "content": PROMPT_REWRITE_DENSE_CAPTION}, {"role": "user", "content": dense_caption}], 0.6
    )

    try:
        return json.loads(answer)
    except Exception as e:
        print(f"Failed to parse the response: {answer}")
        return None


def generate_one(video_id, dense_caption, duration):
    question_answers = generate_question_answers(dense_caption)
    if question_answers is None:
        return None

    rewritten_dense_caption = rewrite_dense_caption(dense_caption)
    if rewritten_dense_caption is None:
        return None

    # Random insert the dense caption into the question_answers
    idx = random.randint(0, len(question_answers))
    conversations = []
    for i in range(len(question_answers)):
        if i == idx:
            conversations.append(Message("user", rewritten_dense_caption["prompt"]))
            conversations.append(Message("assistant", rewritten_dense_caption["caption"]))
        if "question" in question_answers[i] and "answer" in question_answers[i]:
            conversations.append(Message("user", question_answers[i]["question"]))
            conversations.append(Message("assistant", question_answers[i]["answer"]))
        else:
            print(f"Invalid question answer: {question_answers[i]}")
    return Conversation(video_id=video_id, duration=duration, conversations=conversations)


def wrapped_generate(param):
    video_id, dense_caption, duration = param
    results = []
    failed = 0

    while len(results) < 3 and failed < 3:
        data = generate_one(video_id, dense_caption, duration)
        if data is None:
            failed += 1
            time.sleep(5)
            continue
        results.append(asdict(data))

    return results, video_id


def batch_generate(data, save_dir):

    # Assuming `save_dir` is a string
    save_dir = Path(save_dir)
    temp_dir = save_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Skip the existing files
    existing_files = set(file.stem for file in temp_dir.glob("*.json") if len(json.loads(file.read_text())) > 0)
    data = [
        (video_id, dense_caption, duration)
        for video_id, dense_caption, duration in data
        if video_id not in existing_files
    ]

    with Pool(50) as executor:
        for results, video_id in tqdm(executor.imap_unordered(wrapped_generate, data), total=len(data)):
            if len(results) == 0:
                print(f"Failed to generate for {video_id}")

            with open(temp_dir / f"{video_id}.json", "w") as f:
                json.dump(results, f, indent=2)

    # Combine the results
    result = []
    for file in tqdm(temp_dir.glob("*.json")):
        with open(file, "r") as f:
            result.extend(json.load(f))

    with open(save_dir / "train.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Generated {len(result)} conversations")


def main(data_path, save_dir, generate_one_sample):
    data = load_data(data_path=data_path)
    if generate_one_sample:
        print("Generating QA dataset for 1 data sample...")
        print(generate_question_answers(data[0][1]))
        print(rewrite_dense_caption(data[0][1]))
        print(generate_one(data[0][0], data[0][1], None))
    else:
        print("Generating QA dataset...")
        batch_generate(data, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch generate QA data from DVC dataset.")
    parser.add_argument("--input_json", type=str, help="Path to the JSON file containing the dataset.")
    parser.add_argument("--output_dir", type=str, help="Directory to save the generated data.")
    parser.add_argument("--generate_one_sample", action="store_true", help="Test QA generation for 1 data sample.")

    args = parser.parse_args()
    main(args.input_json, args.output_dir, args.generate_one_sample)
