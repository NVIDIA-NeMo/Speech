# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for the CI failure notification caller contract."""

import re
import unittest
from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / "workflows" / "cicd-main.yml"
ACTION = "NVIDIA-NeMo/FW-CI-templates/.github/actions/notify-ci-failure@" "631c404d00a9e60afc591cd071d35d2e18f82fc6"
JOB_ID = 'notify-failure'
WEBHOOK = 'SLACK_WEBHOOK'
EXPECTED_NEEDS = {
    'ephemeral-runner-routing',
    'pre-flight',
    'configure',
    'code-linting',
    'cicd-wait-in-queue',
    'cicd-test-container-build',
    'cicd-import-tests',
    'L0_Setup_Test_Data_And_Models',
    'cicd-main-unit-tests',
    'cicd-main-speech',
    'Nemo_CICD_Test',
}


class NotificationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW.read_text()
        match = re.search(r"^  " + re.escape(JOB_ID) + r":\n(.*?)(?=^  [\w-]+:|\Z)", cls.workflow, re.M | re.S)
        if match is None:
            raise AssertionError("Notification job is missing")
        cls.job = match.group(1)

    def test_immutable_action_and_complete_inputs(self) -> None:
        self.assertIn("uses: " + ACTION, self.job)
        self.assertIn("needs-json: ${{ toJSON(needs) }}", self.job)
        self.assertIn("webhook: ${{ secrets." + WEBHOOK + " }}", self.job)
        self.assertNotIn("message:", self.job)

    def test_direct_dependencies_include_preparation_and_test_jobs(self) -> None:
        dependencies = re.search(r"^    needs:\n((?:      - [\w-]+\n)+)", self.job, re.M)
        self.assertIsNotNone(dependencies)
        actual = set(re.findall(r"- ([\w-]+)", dependencies.group(1)))
        self.assertEqual(actual, EXPECTED_NEEDS)
        workflow_jobs = set(re.findall(r"^  ([\w-]+):$", self.workflow.split("jobs:\n", 1)[1], re.M))
        self.assertTrue(actual <= workflow_jobs)
        self.assertNotIn(JOB_ID, actual)

    def test_no_checkout_or_token_is_needed_to_notify(self) -> None:
        self.assertIn("    permissions: {}", self.job)
        self.assertNotIn("actions/checkout", self.job)
        self.assertNotIn("GH_TOKEN", self.job)
        self.assertNotIn("outputs.", self.job)
        self.assertIn("always()", self.job)
        self.assertIn("!cancelled()", self.job)

    def test_only_scheduled_or_opted_in_manual_runs_on_canonical_branches_alert(self) -> None:
        self.assertIn("github.repository == 'NVIDIA-NeMo/Speech'", self.job)
        self.assertIn('github.ref_name == github.event.repository.default_branch', self.job)
        self.assertIn("failure()", self.job)
        self.assertIn("github.event_name == 'schedule'", self.job)
        self.assertIn("github.event_name == 'workflow_dispatch' && inputs.send_notification", self.job)
        self.assertNotIn("github.event_name == 'push'", self.job)
        self.assertNotIn("github.event_name == 'merge_group'", self.job)
        dispatch = self.workflow.split("  workflow_dispatch:\n", 1)[1].split("\nconcurrency:", 1)[0]
        self.assertIn("      send_notification:\n", dispatch)
        self.assertIn("        default: true\n        type: boolean", dispatch)

    def test_contract_check_is_required_by_lint_gate(self) -> None:
        lint = WORKFLOW.with_name("code-linting.yml").read_text()
        self.assertIn("run: python3 .github/scripts/test_notify_ci_failure.py", lint)
        self.assertIn("needs: [linting, notification-tests]", lint)
        self.assertIn('test "${{ needs.notification-tests.result }}" = "success"', lint)


if __name__ == "__main__":
    unittest.main()
