# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# dependencies = ["pyyaml>=6.0,<7"]
# ///
"""Exercise release guards without credentials, uploads, or package builds."""

import json
import os
import re
import subprocess
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load(next((ROOT / ".github/workflows").glob("release.y*ml")).read_text())
JOBS = WORKFLOW["jobs"]


def evaluate(expression: str, context: dict[str, object]) -> bool:
    """Evaluate the boolean and equality subset used by release job guards."""
    source = expression.strip().removeprefix("${{").removesuffix("}}")
    source = source.replace("!cancelled()", "True").replace("!failure()", "True")
    source = re.sub(
        r"\b(?:github|inputs|needs)\.[\w.-]+",
        lambda match: repr(context[match[0]]),
        source,
    )
    source = source.replace("&&", " and ").replace("||", " or ")
    source = re.sub(r"!(?!=)", "not ", source)
    return bool(eval(" ".join(source.split()), {"__builtins__": {}}, {}))


class ReleaseGuards(unittest.TestCase):
    """Cover publication eligibility and unsuccessful or skipped dependencies."""

    def setUp(self) -> None:
        """Create a successful manual release context for the canonical repo."""
        guard = JOBS["publish-wheel"]["if"]
        match = re.search(r"github.repository == '([^']+)'", guard)
        assert match is not None
        repository = match[1]
        self.context = {
            "github.repository": repository,
            "github.event_name": "workflow_dispatch",
            "inputs.dry-run": False,
            "needs.bump.result": "success",
            "needs.build-wheel.result": "success",
            "needs.publish-wheel.result": "success",
        }

    def test_publish_requires_manual_non_dry_run(self) -> None:
        """Pushes and dry runs cannot request the production upload job."""
        for event in ("push", "workflow_dispatch"):
            for dry_run in (False, True):
                with self.subTest(event=event, dry_run=dry_run):
                    context = dict(
                        self.context,
                        **{"github.event_name": event, "inputs.dry-run": dry_run},
                    )
                    self.assertEqual(
                        evaluate(JOBS["publish-wheel"]["if"], context),
                        event == "workflow_dispatch" and not dry_run,
                    )
        self.context["github.repository"] = "example/fork"
        self.assertFalse(evaluate(JOBS["publish-wheel"]["if"], self.context))

    def test_missing_build_blocks_upload(self) -> None:
        """Failed, cancelled and skipped builds cannot reach publication."""
        for state in ("failure", "cancelled", "skipped"):
            with self.subTest(state=state):
                self.context["needs.build-wheel.result"] = state
                self.assertFalse(evaluate(JOBS["publish-wheel"]["if"], self.context))

    def test_real_finalize_requires_upload(self) -> None:
        """A missing upload cannot finalize a real release; dry runs may continue."""
        for state in ("failure", "cancelled", "skipped"):
            with self.subTest(state=state):
                self.context["needs.publish-wheel.result"] = state
                self.context["inputs.dry-run"] = False
                self.assertFalse(evaluate(JOBS["finalize"]["if"], self.context))
                self.context["inputs.dry-run"] = True
                self.assertTrue(evaluate(JOBS["finalize"]["if"], self.context))

    def test_summary_rejects_missing_required_jobs(self) -> None:
        """Run the actual shell summary against failures and permitted skips."""
        script = JOBS["release-summary"]["steps"][0]["run"]
        for publish in (False, True):
            baseline = {name: {"result": "success"} for name in JOBS["release-summary"]["needs"]}
            baseline["publish-wheel"]["result"] = "success" if publish else "skipped"
            for job in (
                "validate-inputs",
                "bump",
                "build-wheel",
                "publish-wheel",
                "finalize",
            ):
                for state in ("success", "failure", "cancelled", "skipped"):
                    with self.subTest(publish=publish, job=job, state=state):
                        needs = {name: dict(value) for name, value in baseline.items()}
                        needs[job]["result"] = state
                        env = dict(
                            os.environ,
                            NEEDS=json.dumps(needs),
                            PUBLISH_EXPECTED=str(publish).lower(),
                            RELEASE_SKIPPED="false",
                        )
                        result = subprocess.run(
                            ["bash", "-euo", "pipefail", "-c", script],
                            env=env,
                            capture_output=True,
                        )
                        expected = state == ("skipped" if job == "publish-wheel" and not publish else "success")
                        self.assertEqual(result.returncode == 0, expected, result.stderr.decode())

    def test_validate_ref_treats_input_as_data(self) -> None:
        """Only full immutable SHAs pass; branch names and shell payloads fail."""
        script = JOBS["validate-inputs"]["steps"][0]["run"]
        for ref in ("a" * 40, "main", "abc1234", "$(exit 0)", "a" * 40 + "\n"):
            with self.subTest(ref=ref):
                result = subprocess.run(
                    ["bash", "-euo", "pipefail", "-c", script],
                    env=dict(os.environ, RELEASE_REF=ref),
                    capture_output=True,
                )
                self.assertEqual(result.returncode == 0, ref == "a" * 40)

    def test_upload_has_no_checkout_or_password(self) -> None:
        """Keep source execution and build credentials outside the OIDC job."""
        self.assertEqual(JOBS["build-wheel"]["permissions"], {"contents": "read"})
        self.assertNotIn("secrets", JOBS["build-wheel"])
        upload = JOBS["publish-wheel"]
        self.assertEqual(upload["permissions"], {"id-token": "write"})
        self.assertEqual(upload["environment"]["name"], "pypi")
        self.assertEqual(len(upload["steps"]), 2)
        for step in upload["steps"]:
            self.assertNotIn("run", step)
            self.assertNotIn("password", step.get("with", {}))
            self.assertNotIn("checkout", step["uses"])


if __name__ == "__main__":
    unittest.main()
