# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

import unittest
from unittest.mock import MagicMock, patch

from nemo.collections.llm.api import evaluate  # Replace 'your_module' with the actual module name


class TestEvaluateFunction(unittest.TestCase):

    def setUp(self):
        # Mocking EvaluationTarget and EvaluationConfig
        self.target_cfg = MagicMock()
        self.target_cfg.api_endpoint = MagicMock()
        self.target_cfg.api_endpoint.nemo_checkpoint_path = "path/to/checkpoint"
        self.target_cfg.api_endpoint.url = "http://example.com"
        self.target_cfg.api_endpoint.nemo_triton_http_port = 8000
        self.target_cfg.api_endpoint.model_id = "model_id"

        self.eval_cfg = MagicMock()
        self.eval_cfg.type = "gsm8k"
        self.eval_cfg.params = MagicMock()
        self.eval_cfg.params.batch_size = 16
        self.eval_cfg.params.max_new_tokens = 128
        self.eval_cfg.params.temperature = 1.0
        self.eval_cfg.params.top_p = 0.9
        self.eval_cfg.params.top_k = 50
        self.eval_cfg.params.add_bos = True
        self.eval_cfg.params.limit_samples = 100
        self.eval_cfg.params.num_fewshot = 4
        self.eval_cfg.params.bootstrap_iters = 1000

    @patch('nemo.lightning.io.load_context')
    @patch('nemo.collections.llm.evaluation.base.wait_for_server_ready')
    @patch('lm_eval.evaluator.simple_evaluate')
    @patch('nemo.collections.llm.evaluation.base.NeMoFWLMEval')
    def test_evaluate_success(
        self, mock_NeMoFWLMEval, mock_simple_evaluate, mock_wait_for_server_ready, mock_load_context
    ):
        # Mocking necessary methods
        mock_load_context.return_value = "tokenizer"
        mock_NeMoFWLMEval.return_value = "model"
        mock_simple_evaluate.return_value = {"results": {"gsm8k": "score"}}

        # Call the function
        evaluate(self.target_cfg, self.eval_cfg)

        # Asserts
        mock_load_context.assert_called_once_with("path/to/checkpoint/context", subpath="model.tokenizer")
        mock_wait_for_server_ready.assert_called_once_with(
            url="http://example.com", triton_http_port=8000, model_name="model_id"
        )
        mock_NeMoFWLMEval.assert_called_once_with(
            model_name="model_id",
            api_url="http://example.com",
            tokenizer="tokenizer",
            batch_size=16,
            max_tokens_to_generate=128,
            temperature=1.0,
            top_p=0.9,
            top_k=50,
            add_bos=True,
        )
        mock_simple_evaluate.assert_called_once_with(
            model="model",
            tasks="gsm8k",
            limit=100,
            num_fewshot=4,
            bootstrap_iters=1000,
        )

    @patch('nemo.lightning.io.load_context')
    @patch('nemo.collections.llm.evaluation.base.wait_for_server_ready')
    @patch('lm_eval.evaluator.simple_evaluate')
    @patch('nemo.collections.llm.evaluation.base.NeMoFWLMEval')
    def test_evaluate_nemo_checkpoint_path_none(
        self, mock_NeMoFWLMEval, mock_simple_evaluate, mock_wait_for_server_ready, mock_load_context
    ):
        # Set nemo_checkpoint_path to None
        self.target_cfg.api_endpoint.nemo_checkpoint_path = None

        # Call the function and assert it raises ValueError
        with self.assertRaises(ValueError):
            evaluate(self.target_cfg, self.eval_cfg)

        # No other methods should be called
        mock_load_context.assert_not_called()
        mock_wait_for_server_ready.assert_not_called()
        mock_NeMoFWLMEval.assert_not_called()
        mock_simple_evaluate.assert_not_called()

    @patch('nemo.lightning.io.load_context')
    @patch('nemo.collections.llm.evaluation.base.wait_for_server_ready')
    @patch('lm_eval.evaluator.simple_evaluate')
    @patch('nemo.collections.llm.evaluation.base.NeMoFWLMEval')
    def test_evaluate_import_error(
        self, mock_NeMoFWLMEval, mock_simple_evaluate, mock_wait_for_server_ready, mock_load_context
    ):
        # Mocking ImportError for lm-evaluation-harness
        with patch('builtins.__import__', side_effect=ImportError("Mocked ImportError")):
            # Call the function and assert it raises ImportError
            with self.assertRaises(ImportError):
                evaluate(self.target_cfg, self.eval_cfg)

            # No other methods should be called
            mock_load_context.assert_not_called()
            mock_wait_for_server_ready.assert_not_called()
            mock_NeMoFWLMEval.assert_not_called()
            mock_simple_evaluate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
