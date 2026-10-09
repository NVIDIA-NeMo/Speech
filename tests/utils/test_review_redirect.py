# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_review_redirect_is_least_privilege_and_does_not_execute_review() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/claude-review.yml").read_text())
    job = workflow["jobs"]["redirect-to-review"]
    condition = job["if"]
    assert " ".join(condition.split()) == (
        "github.event_name == 'issue_comment' && "
        "github.event.issue.pull_request && "
        "github.event.comment.user.type != 'Bot' && "
        'contains(fromJSON(\'["OWNER", "MEMBER", "COLLABORATOR"]\'), '
        "github.event.comment.author_association) && "
        "(github.event.comment.body == '/claude review' || "
        "github.event.comment.body == '/claude strict-review')"
    )
    assert job["permissions"] == {"pull-requests": "write"}
    assert "uses" not in job
    assert "secrets" not in job
    assert len(job["steps"]) == 1
    step = job["steps"][0]
    assert "uses" not in step
    assert step["run"] == 'gh pr comment "$PR_NUMBER" --repo "$REPO" --body "$NOTICE"'
    assert "${{" not in step["run"]
    assert "model=claude" in step["env"]["NOTICE"]
    assert "model=codex" in step["env"]["NOTICE"]
    assert "/review help" in step["env"]["NOTICE"]
    assert job["env"]["REVIEW_COMMAND"] == (
        "${{ github.event.comment.body == '/claude strict-review' && '/review mode=strict' || '/review' }}"
    )


def test_formal_review_rubric_is_inert_and_uses_formal_submission() -> None:
    rubric = (ROOT / ".claude/skills/nemo-speech-pr-review/SKILL.md").read_text()
    metadata = yaml.safe_load(rubric.split("---", 2)[1])
    assert metadata["name"] == "nemo-speech-pr-review"
    assert metadata["disable-model-invocation"] is True
    assert metadata["user-invocable"] is False
    assert "mode=light" in rubric
    assert "mode=strict" in rubric
    assert "Do not run GitHub commands or" in rubric
    assert "Never approve an incomplete" in rubric


def test_review_redirect_workflow_has_no_other_execution_or_permissions() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/claude-review.yml").read_text())
    assert workflow["permissions"] == {}
    assert set(workflow["jobs"]) == {"redirect-to-review"}
    events = workflow.get("on", workflow.get(True))
    assert events == {"issue_comment": {"types": ["created"]}}


@pytest.mark.parametrize(
    "notice", ["Use /review", "Use /review mode=strict", "$(touch injected) `touch injected` ' \" ;\n/review"]
)
def test_notice_is_passed_as_one_literal_argument(tmp_path: Path, notice: str) -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/claude-review.yml").read_text())
    script = workflow["jobs"]["redirect-to-review"]["steps"][0]["run"]
    gh = tmp_path / "gh"
    gh.write_text("#!/bin/sh\nprintf '%s\\0' \"$@\" > \"$CAPTURE\"\n")
    gh.chmod(0o755)
    capture = tmp_path / "args"
    env = dict(
        os.environ,
        PS1="",
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        PR_NUMBER="123",
        REPO="NVIDIA-NeMo/example",
        NOTICE=notice,
        CAPTURE=str(capture),
    )
    env.pop("BASH_ENV", None)
    env.pop("ENV", None)
    subprocess.run(["/bin/bash", "-eu", "-c", script], cwd=tmp_path, env=env, check=True)
    assert capture.read_bytes().split(b"\0") == [
        b"pr",
        b"comment",
        b"123",
        b"--repo",
        b"NVIDIA-NeMo/example",
        b"--body",
        notice.encode(),
        b"",
    ]
    assert not (tmp_path / "injected").exists()
