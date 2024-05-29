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


use_pytriton = True
try:
    from pytriton.model_config import ModelConfig
    from pytriton.triton import Triton, TritonConfig
except Exception:
    use_pytriton = False

from nemo.deploy.deploy_base import DeployBase


class DeployPyTriton(DeployBase):
    """
    Deploys any models to Triton Inference Server that implements ITritonDeployable interface in nemo.deploy.

    Example:
        from nemo.deploy import DeployPyTriton, NemoQueryLLM
        from nemo.export import TensorRTLLM

        trt_llm_exporter = TensorRTLLM(model_dir="/path/for/model/files")
        trt_llm_exporter.export(
            nemo_checkpoint_path="/path/for/nemo/checkpoint",
            model_type="llama",
            n_gpus=1,
        )

        nm = DeployPyTriton(model=trt_llm_exporter, triton_model_name="model_name", port=8000)
        nm.deploy()
        nm.run()
        nq = NemoQueryLLM(url="localhost", model_name="model_name")

        prompts = ["hello, testing GPT inference", "another GPT inference test?"]
        output = nq.query_llm(prompts=prompts, max_output_len=100)
        print("prompts: ", prompts)
        print("")
        print("output: ", output)
        print("")

        prompts = ["Give me some info about Paris", "Do you think Londan is a good city to visit?", "What do you think about Rome?"]
        output = nq.query_llm(prompts=prompts, max_output_len=250)
        print("prompts: ", prompts)
        print("")
        print("output: ", output)
        print("")

    """

    def __init__(
        self,
        triton_model_name: str,
        triton_model_version: int = 1,
        checkpoint_path: str = None,
        model=None,
        max_batch_size: int = 128,
        port: int = 8000,
        address="0.0.0.0",
        allow_grpc=True,
        allow_http=True,
        streaming=False,
        pytriton_log_verbose=0,
    ):
        """
        A nemo checkpoint or model is expected for serving on Triton Inference Server.

        Args:
            triton_model_name (str): Name for the service
            triton_model_version(int): Version for the service
            checkpoint_path (str): path of the nemo file
            model (ITritonDeployable): A model that implements the ITritonDeployable from nemo.deploy import ITritonDeployable
            max_batch_size (int): max batch size
            port (int) : port for the Triton server
            address (str): http address for Triton server to bind.
        """

        super().__init__(
            triton_model_name=triton_model_name,
            triton_model_version=triton_model_version,
            checkpoint_path=checkpoint_path,
            model=model,
            max_batch_size=max_batch_size,
            port=port,
            address=address,
            allow_grpc=allow_grpc,
            allow_http=allow_http,
            streaming=streaming,
            pytriton_log_verbose=pytriton_log_verbose,
        )

    def deploy(self):
        """
        Deploys any models to Triton Inference Server.
        """

        self._init_nemo_model()

        try:
            if self.streaming:
                # TODO: can't set allow_http=True due to a bug in pytriton, will fix in latest pytriton
                triton_config = TritonConfig(
                    log_verbose=self.pytriton_log_verbose,
                    allow_grpc=self.allow_grpc,
                    allow_http=self.allow_http,
                    grpc_address=self.address,
                )
                self.triton = Triton(config=triton_config)
                self.triton.bind(
                    model_name=self.triton_model_name,
                    model_version=self.triton_model_version,
                    infer_func=self.model.triton_infer_fn_streaming,
                    inputs=self.model.get_triton_input,
                    outputs=self.model.get_triton_output,
                    config=ModelConfig(decoupled=True),
                )
            else:
                triton_config = TritonConfig(
                    http_address=self.address,
                    http_port=self.port,
                    allow_grpc=self.allow_grpc,
                    allow_http=self.allow_http,
                )
                self.triton = Triton(config=triton_config)
                self.triton.bind(
                    model_name=self.triton_model_name,
                    model_version=self.triton_model_version,
                    infer_func=self.model.triton_infer_fn,
                    inputs=self.model.get_triton_input,
                    outputs=self.model.get_triton_output,
                    config=ModelConfig(max_batch_size=self.max_batch_size),
                )
        except Exception as e:
            self.triton = None
            print(e)

    def serve(self):
        """
        Starts serving the model and waits for the requests
        """

        if self.triton is None:
            raise Exception("deploy should be called first.")

        try:
            self.triton.serve()
        except Exception as e:
            self.triton = None
            print(e)

    def run(self):
        """
        Starts serving the model asynchronously.
        """

        if self.triton is None:
            raise Exception("deploy should be called first.")

        self.triton.run()

    def stop(self):
        """
        Stops serving the model.
        """

        if self.triton is None:
            raise Exception("deploy should be called first.")

        self.triton.stop()
