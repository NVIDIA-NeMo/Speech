# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from enum import Enum

from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape


class TemplateName(str, Enum):
    """Template file names used by the report renderer."""

    eval_report = "eval_report.jinja"
    eval_report_configuration = "eval_report_configuration.jinja"
    eval_report_header = "eval_report_header.jinja"
    eval_report_table = "eval_report_table.jinja"
    eval_report_stat_analysis = "eval_report_stat_analysis.jinja"
    eval_report_image = "eval_report_image.jinja"
    eval_report_block = "eval_report_block.jinja"

    audio_report = "audio_report.jinja"
    audio_report_header = "audio_report_header.jinja"
    audio_report_pair = "audio_report_pair.jinja"
    audio_report_block = "audio_report_block.jinja"


class Renderer:
    """Load and render Jinja templates used for report generation."""

    def __init__(self, templates_dir: Path) -> None:
        self.env = Environment(
            loader=FileSystemLoader(templates_dir.as_posix()),
            autoescape=select_autoescape(["html", "xml", "jinja"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.templates = {t: self.env.get_template(t.value) for t in TemplateName}

    def render(self, name: TemplateName, **kwargs) -> str:
        """Render the selected template with the provided context variables.

        Args:
            name: Template identifier to render.
            **kwargs: Context variables passed to the template.

        Returns:
            Rendered template as a string.
        """
        return self.templates[name].render(**kwargs)
