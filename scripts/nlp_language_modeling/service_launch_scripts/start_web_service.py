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
"""
"""

from nemo.collections.nlp.modules.common.megatron_web_server import RetroDemoWebApp
from nemo.core.config import hydra_runner


@hydra_runner(config_path="conf", config_name="retro_web_server")
def main(cfg) -> None:

    demo = RetroDemoWebApp(cfg.text_service_ip, cfg.text_service_port, cfg.combo_service_ip, cfg.combo_service_port)
    demo.run_demo(cfg.share, cfg.username, cfg.password, cfg.port)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
