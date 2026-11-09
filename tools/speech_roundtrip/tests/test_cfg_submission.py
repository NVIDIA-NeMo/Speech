import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from speech_roundtrip.core import write_json


class CFGSubmissionTests(unittest.TestCase):
    def test_judging_waits_for_successful_single_gpu_job_and_submission_is_not_duplicated(self):
        path = Path(__file__).parents[1] / 'ord/spoken_low_cfg.py'
        spec = importlib.util.spec_from_file_location('cfg_eval_launcher', path)
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'run'
            write_json(
                run / 'control/runtime_paths.json',
                {'key_loader': '/example/key_loader.py', 'judge_python': '/example/python'},
            )
            plan = {'root': str(root), 'speech': str(root / 'Speech_v2')}
            with (
                patch.object(launcher, 'validate_sources', return_value=plan),
                patch.object(launcher.subprocess, 'check_output', side_effect=['123\n', '124\n']) as scheduler,
            ):
                launcher.submit(run, 'approved_account', 'polar4,interactive')
                calls = scheduler.call_args_list
                self.assertEqual(len(calls), 2)
                self.assertIn('--dependency=afterok:123', calls[1].args[0])
                self.assertIn('#SBATCH --gres=gpu:1', (run / 'control/gpu.sbatch').read_text())
                self.assertNotIn('--gres=gpu', (run / 'control/judges.sbatch').read_text())
                with self.assertRaisesRegex(ValueError, 'already submitted'):
                    launcher.submit(run, 'approved_account', 'polar4,interactive')
                self.assertEqual(scheduler.call_count, 2)


if __name__ == '__main__':
    unittest.main()
