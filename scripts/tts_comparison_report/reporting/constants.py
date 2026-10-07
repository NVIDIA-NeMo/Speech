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
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


_ROOT: Path = Path(__file__).parent.parent


class ContextType(str, Enum):
    """Type of speaker context the model was conditioned on when a benchmark was generated."""

    # Speaker identity was conditioned on a context audio prompt.
    audio = "audio"
    # Speaker identity was conditioned on a text description; no context audio exists.
    text = "text"


@dataclass(frozen=True)
class BenchmarkMeta:
    """Static metadata of a benchmark supported by the comparison report pipeline."""

    # Language code used in results directory names and report navigation.
    lang: str
    # Context type used when the benchmark was generated. Metrics that compare generated
    # audio against the context audio are reported only for audio-context benchmarks.
    context_type: ContextType = ContextType.audio


# Benchmarks supported by the comparison report pipeline. Benchmark names ending with
# '_ct_text' were generated with text context and therefore have no context audio.
BENCHMARK_META: dict[str, BenchmarkMeta] = {
    'libritts': BenchmarkMeta('en'),
    'riva_en': BenchmarkMeta('en'),
    'riva_en_hard_sentences': BenchmarkMeta('en'),
    'riva_en_short_sentences': BenchmarkMeta('en'),
    'riva_en_qa': BenchmarkMeta('en'),
    'riva_en_qa_longform': BenchmarkMeta('en'),
    'King_ASR_sa_diacritics': BenchmarkMeta('ar'),
    'King_ASR_sa_no_diacritics': BenchmarkMeta('ar'),
    'King_ASR_uae_diacritics': BenchmarkMeta('ar'),
    'King_ASR_uae_no_diacritics': BenchmarkMeta('ar'),
    'cmltts_de': BenchmarkMeta('de'),
    'cmltts_es': BenchmarkMeta('es'),
    'cmltts_fr': BenchmarkMeta('fr'),
    'AI4bharat': BenchmarkMeta('hi'),
    'cmltts_it': BenchmarkMeta('it'),
    'jvs_jsut': BenchmarkMeta('ja'),
    'F5I9N7A1': BenchmarkMeta('ko'),
    'cmltts_pt': BenchmarkMeta('pt'),
    'vivos': BenchmarkMeta('vi'),
    'mscenespeech': BenchmarkMeta('zh'),
    'ar_MSA_qa_ct_text': BenchmarkMeta('ar', ContextType.text),
    'ar_MSA_qa': BenchmarkMeta('ar'),
    'de_qa_ct_text': BenchmarkMeta('de', ContextType.text),
    'de_qa': BenchmarkMeta('de'),
    'es_qa_ct_text': BenchmarkMeta('es', ContextType.text),
    'es_qa': BenchmarkMeta('es'),
    'fr_qa_ct_text': BenchmarkMeta('fr', ContextType.text),
    'fr_qa': BenchmarkMeta('fr'),
    'it_qa_ct_text': BenchmarkMeta('it', ContextType.text),
    'it_qa': BenchmarkMeta('it'),
    'ja_qa_ct_text': BenchmarkMeta('ja', ContextType.text),
    'ja_qa': BenchmarkMeta('ja'),
    'ko_qa_ct_text': BenchmarkMeta('ko', ContextType.text),
    'ko_qa': BenchmarkMeta('ko'),
    'pt_qa_ct_text': BenchmarkMeta('pt', ContextType.text),
    'pt_qa': BenchmarkMeta('pt'),
    'vi_qa_ct_text': BenchmarkMeta('vi', ContextType.text),
    'vi_qa': BenchmarkMeta('vi'),
    'zh_qa_ct_text': BenchmarkMeta('zh', ContextType.text),
    'zh_qa': BenchmarkMeta('zh'),
    'hi_qa_ct_text': BenchmarkMeta('hi', ContextType.text),
    'hi_qa': BenchmarkMeta('hi'),
    'hi_qa_expanded_pronunciation_ct_text': BenchmarkMeta('hi', ContextType.text),
    'hi_qa_expanded_pronunciation': BenchmarkMeta('hi'),
}

SUPPORTED_BENCHMARK_NAMES: list[str] = list(BENCHMARK_META.keys())

# Default width of tqdm progress bars in terminal columns.
TQDM_NCOLS: int = 80

# Random seed used for reproducible sampling of audio examples.
SEED: int = 42

# Number of decimal digits used when formatting p-values in statistical tests.
P_VAL_ROUND_DIGITS: int = 4

# Default signature version used to sign S3 client requests.
S3_SIGNATURE_VERSION: str = "s3"

# Default lifetime of generated S3 presigned links in seconds (one year).
S3_LINK_EXPIRES_IN: int = 31536000

# Subdirectory inside the S3 report prefix used for uploaded audio files.
S3_AUDIO_DIR: str = "audio"

# Subdirectory inside the S3 report prefix used for uploaded plot images.
S3_IMAGES_DIR: str = "images"

# Directory containing Jinja templates used for report rendering.
TEMPLATES_DIR: Path = _ROOT / "templates"

# Fallback task id used when no real Jira ticket is provided.
DUMMY_TASK_ID: str = "NEMOTTS-0000"

# URL prefix used to construct clickable Jira ticket links in reports.
JIRA_TICKET_URL_PREFIX: str = "https://jirasw.nvidia.com/browse"
