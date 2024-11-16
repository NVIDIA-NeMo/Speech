# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from nemo.deploy.nlp import NemoQueryLLM
from nemo.utils import logging


class TritonSettings(BaseSettings):
    _triton_service_port: int
    _triton_service_ip: str
    _triton_request_timeout: str

    def __init__(self):
        super(TritonSettings, self).__init__()
        try:
            self._triton_service_port = int(os.environ.get('TRITON_PORT', 8080))
            self._triton_service_ip = os.environ.get('TRITON_HTTP_ADDRESS', '0.0.0.0')
            self._triton_request_timeout = int(os.environ.get('TRITON_REQUEST_TIMEOUT', 60))
            self._openai_format_response = os.environ.get('OPENAI_FORMAT_RESPONSE', 'False').lower() == 'true'
            self._output_generation_logits = os.environ.get('OUTPUT_GENERATION_LOGITS', 'False').lower() == 'true'
        except Exception as error:
            logging.error("An exception occurred trying to retrieve set args in TritonSettings class. Error:", error)
            return

    @property
    def triton_service_port(self):
        return self._triton_service_port

    @property
    def triton_service_ip(self):
        return self._triton_service_ip

    @property
    def triton_request_timeout(self):
        return self._triton_request_timeout

    @property
    def openai_format_response(self):
        """
        Retuns the response from Triton server in OpenAI compatible format if set to True.
        """
        return self._openai_format_response

    @property
    def output_generation_logits(self):
        """
        Retuns the generation logits along with text in Triton server output if set to True.
        """
        return self._output_generation_logits


app = FastAPI()
triton_settings = TritonSettings()


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.0
    top_k: int = 1
    stream: bool = False
    stop: str | None = None
    frequency_penalty: float = 1.0


@app.get("/v1/health")
def health_check():
    return {"status": "ok"}


@app.get("/v1/triton_health")
async def check_triton_health():
    """
    This method exposes endpoint "/triton_health" which can be used to verify if Triton server is accessible while running the REST or FastAPI application.
    Verify by running: curl http://service_http_address:service_port/v1/triton_health and the returned status should inform if the server is accessible.
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


@app.post("/v1/completions/")
def completions_v1(request: CompletionRequest):
    try:
        url = triton_settings.triton_service_ip + ":" + str(triton_settings.triton_service_port)
        nq = NemoQueryLLM(url=url, model_name=request.model)
        output = nq.query_llm(
            prompts=[request.prompt],
            max_output_len=request.max_tokens,
            # when these below params are passed as None
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
            init_timeout=triton_settings.triton_request_timeout,
            openai_format_response=triton_settings.openai_format_response,
            output_generation_logits=triton_settings.output_generation_logits,
        )
        if triton_settings.openai_format_response:
            return output
        else:
            return {
                "output": output[0][0],
            }
    except Exception as error:
        logging.error("An exception occurred with the post request to /v1/completions/ endpoint:", error)
        return {"error": "An exception occurred"}
