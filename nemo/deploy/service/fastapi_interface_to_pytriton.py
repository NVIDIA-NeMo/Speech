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
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings

from nemo.deploy.nlp import NemoQueryLLMPyTorch
from nemo.utils import logging


class TritonSettings(BaseSettings):
    """
    TritonSettings class that gets the values of TRITON_HTTP_ADDRESS and TRITON_PORT.
    """

    _triton_service_port: int
    _triton_service_ip: str

    def __init__(self):
        super(TritonSettings, self).__init__()
        try:
            self._triton_service_port = int(os.environ.get('TRITON_PORT', 8000))
            self._triton_service_ip = os.environ.get('TRITON_HTTP_ADDRESS', '0.0.0.0')
        except Exception as error:
            logging.error("An exception occurred trying to retrieve set args in TritonSettings class. Error:", error)
            return

    @property
    def triton_service_port(self):
        """
        Returns the port number for the Triton service.
        """
        return self._triton_service_port

    @property
    def triton_service_ip(self):
        """
        Returns the IP address for the Triton service.
        """
        return self._triton_service_ip


app = FastAPI()
triton_settings = TritonSettings()


class CompletionRequest(BaseModel):
    """
    Represents a request for text completion.

    Attributes:
        model (str): The name of the model to use for completion.
        prompt (str): The input text to generate a response from.
        messages (list[dict]): A list of message dictionaries for chat completion.
        max_tokens (int): The maximum number of tokens to generate in the response.
        temperature (float): Sampling temperature for randomness in generation.
        top_p (float): Cumulative probability for nucleus sampling.
        top_k (int): Number of highest-probability tokens to consider for sampling.
        logprobs (int): Number of log probabilities to include in the response, if applicable.
    """

    model: str
    prompt: str = 'hello'
    messages: list[dict] = [{}]
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.0
    top_k: int = 0
    logprobs: int = None

    @model_validator(mode='after')
    def set_greedy_params(self):
        """Validate parameters for greedy decoding."""
        if self.temperature == 0 and self.top_p == 0:
            logging.warning("Both temperature and top_p are 0. Setting top_k to 1 to ensure greedy sampling.")
            self.top_k = 1
        return self


@app.get("/v1/health")
def health_check():
    """
    Health check endpoint to verify that the API is running.

    Returns:
        dict: A dictionary indicating the status of the application.
    """
    return {"status": "ok"}


@app.get("/v1/triton_health")
async def check_triton_health():
    """
    This method exposes endpoint "/triton_health" which can be used to verify if Triton server is accessible while
    running the REST or FastAPI application.
    Verify by running: curl http://service_http_address:service_port/v1/triton_health and the returned status should
    inform if the server is accessible.
    """
    triton_url = (
        f"http://{triton_settings.triton_service_ip}:{str(triton_settings.triton_service_port)}/v2/health/ready"
    )
    logging.info(f"Attempting to connect to Triton server at: {triton_url}")
    try:
        response = requests.get(triton_url, timeout=5)
        if response.status_code == 200:
            return {"status": "Triton server is reachable and ready"}
        else:
            raise HTTPException(status_code=503, detail="Triton server is not ready")
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Cannot reach Triton server: {str(e)}")


def convert_numpy(obj):
    """
    Convert NumPy arrays in output to lists
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    else:
        return obj


def _helper_fun(url, model, prompts, temperature, top_k, top_p, compute_logprob, max_length, apply_chat_template):
    """
    run_in_executor doesn't allow to pass kwargs, so we have this helper function to pass args as a list
    """
    nq = NemoQueryLLMPyTorch(url=url, model_name=model)
    output = nq.query_llm(
        prompts=prompts,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        compute_logprob=compute_logprob,
        max_length=max_length,
        apply_chat_template=apply_chat_template,
        init_timeout=300,
    )
    return output


async def query_llm_async(
    *, url, model, prompts, temperature, top_k, top_p, compute_logprob, max_length, apply_chat_template
):
    """
    Sends requests to `NemoQueryLLMPyTorch.query_llm` in a non-blocking way, allowing the server to process
    concurrent requests. This way enables batching of requests in the underlying Triton server.
    """
    import asyncio
    import concurrent

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool,
            _helper_fun,
            url,
            model,
            prompts,
            temperature,
            top_k,
            top_p,
            compute_logprob,
            max_length,
            apply_chat_template,
        )
    return result


@app.post("/v1/completions/")
async def completions_v1(request: CompletionRequest):
    """
    Defines the completions endpoint and queries the model deployed on PyTriton server.
    """
    url = f"http://{triton_settings.triton_service_ip}:{triton_settings.triton_service_port}"
    logging.info(f"Request: {request}")
    prompts = request.prompt
    if not isinstance(request.prompt, list):
        prompts = [request.prompt]

    output = await query_llm_async(
        url=url,
        model=request.model,
        prompts=prompts,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        compute_logprob=True if request.logprobs == 1 else False,
        max_length=request.max_tokens,
        apply_chat_template=False,
    )

    output_serializable = convert_numpy(output)
    output_serializable["choices"][0]["text"] = output_serializable["choices"][0]["text"][0][0]
    if request.logprobs == 1:
        output_serializable["choices"][0]["logprobs"]["token_logprobs"] = output_serializable["choices"][0][
            "logprobs"
        ]["token_logprobs"][0]
        output_serializable["choices"][0]["logprobs"]["top_logprobs"] = output_serializable["choices"][0]["logprobs"][
            "top_logprobs"
        ][0]
    logging.info(f"Output: {output_serializable}")
    return output_serializable


def dict_to_str(messages):
    """
    Serializes dict to str
    """
    return json.dumps(messages)


@app.post("/v1/chat/completions/")
async def chat_completions_v1(request: CompletionRequest):
    """
    Defines the chat completions endpoint and queries the model deployed on PyTriton server.
    """
    url = f"http://{triton_settings.triton_service_ip}:{triton_settings.triton_service_port}"
    logging.info(f"Request: {request}")
    prompts = request.messages
    if not isinstance(request.messages, list):
        prompts = [request.messages]
    # Serialize the dictionary to a JSON string represnetation to be able to convert to numpy array
    # (str_list2numpy) and back to list (str_ndarray2list) as required by PyTriton. Using the dictionaries directly
    # with these methods is not possible as they expect string type.
    json_prompts = [dict_to_str(prompts)]
    output = await query_llm_async(
        url=url,
        model=request.model,
        prompts=json_prompts,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        compute_logprob=True if request.logprobs == 1 else False,
        max_length=request.max_tokens,
        apply_chat_template=True,
    )
    # Add 'role' as 'assistant' key to the output dict
    output["choices"][0]["message"] = {"role": "assistant", "content": output["choices"][0]["text"]}
    output["object"] = "chat.completion"

    del output["choices"][0]["text"]

    output_serializable = convert_numpy(output)
    output_serializable["choices"][0]["message"]["content"] = output_serializable["choices"][0]["message"]["content"][
        0
    ][0]
    logging.info(f"Output: {output_serializable}")
    return output_serializable
