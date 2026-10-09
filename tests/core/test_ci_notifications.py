# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml


WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/cicd-main.yml"
ACTION = "NVIDIA-NeMo/FW-CI-templates/.github/actions/notify-ci-failure@631c404d00a9e60afc591cd071d35d2e18f82fc6"


@pytest.mark.unit
def test_nightly_notification_covers_setup_tests_and_coverage():
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    notify = jobs["notify-nightly-failure"]
    assert set(notify["needs"]) == set(jobs) - {"notify-nightly-failure"}
    assert notify["runs-on"] == "ubuntu-latest"
    assert notify["permissions"] == {}
    (step,) = notify["steps"]
    assert step["uses"] == ACTION
    assert step["with"] == {
        "needs-json": "${{ toJSON(needs) }}",
        "webhook": "${{ secrets.SLACK_WEBHOOK }}",
    }
    # The notifier executes no checkout or code from the failed run.
    assert "run" not in step


@pytest.mark.unit
def test_nightly_notification_excludes_pr_push_dispatch_and_cancelled_runs():
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    condition = jobs["notify-nightly-failure"]["if"]
    assert "always() && !cancelled() && failure()" in condition
    assert "github.repository == 'NVIDIA-NeMo/Speech'" in condition
    assert "github.event_name == 'schedule'" in condition
    assert "github.ref_name == github.event.repository.default_branch" in condition
    assert "needs." not in condition
    assert "workflow_dispatch" not in condition
    assert "push" not in condition
