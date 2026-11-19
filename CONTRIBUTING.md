# Contributions are welcome!

We do all of NeMo's development in the open. Contributions from NeMo community are welcome.


# Pull Requests (PR) Guidelines

**Send your PRs to the `main` branch**

1) Make sure your PR does one thing. Have a clear answer to "What does this PR do?".
   Raising the minimum torch or Python version, Dockerfile and CI changes, and files that are not part of the change (local
   environment files, scratch scripts, generated files) belong in their own PR or stay out. A PR with a release-branch label
   (for example `r3.1.0`) must only touch files that exist on that branch.
2) Read General Principles and style guide below
3) Make sure you sign your commits. E.g. use ``git commit -s`` when before your commit
4) Make sure relevant unittests finish successfully before sending PR
5) Send your PR and request a review

## Signing Your Work

All contributions must be signed off to certify agreement with the Developer Certificate of Origin below. Add a
`Signed-off-by` trailer to each commit by using `git commit -s`.

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Unit tests
Quick tests (locally, while developing)
```
pytest
# If you don't have NVIDIA GPU do:
# pytest -m "not pleasefixme" --cpu path/to/relevant_tests
```
Full tests, including pre-trained model downloads
```
pytest -m "not pleasefixme" --with_downloads  path/to/relevant_tests
```

Replace `path/to/relevant_tests` with the test directory to run such as `tests/collections/asr`. Check the test scripts in `tests/functional_tests`
that begin with `L0_Unit_Tests_` for the specific test configuration used by different parts of the unit test suite. Different suites may expect
different environment variables to be set.

Each test has a 300-second timeout, set through `pytest-timeout` in `[tool.pytest.ini_options]` in `pyproject.toml`. A test that hangs or
runs longer fails on its own instead of exhausting the CI job timeout. Keep unit tests far below this limit. Put long-running checks, such as
inference with a large released checkpoint, behind `@pytest.mark.run_only_on('GPU')` or in a functional test.

### GPU tests

CI runs every unit test in the CPU jobs (`--cpu`). The GPU jobs on pull requests run only the tests marked
`@pytest.mark.gpu`, so mark every test that exercises CUDA code, for example a test parametrized over `cuda`, a CUDA
graph or kernel test, or a test that runs a model on the GPU. `@pytest.mark.run_only_on('GPU')` implies the marker.
Tests that run the same CPU code in both modes must not be marked.

```
pytest -m "gpu and not pleasefixme" path/to/relevant_tests   # what the GPU jobs run on a pull request
```

The scheduled nightly run executes the whole suite on GPU with `--check-gpu-markers`, which fails if a test without
the marker allocates CUDA memory. The audit sees only PyTorch's CUDA allocator, so a test that runs CUDA only through
Numba, DALI, or another library must be marked by hand. For a test parametrized over devices, mark only the CUDA case:
`pytest.param("cuda", marks=pytest.mark.gpu)`.

### Writing tests

A bug fix comes with a test that fails without the fix, and a behavior change with tests for the new behavior and its edge cases.

- Put the test in the directory that mirrors the source (``tests/collections/<collection>/``), and do not repeat tests that another
  file of the same module already has. Use ``pytest.mark.parametrize`` for variants of one test.
- Assert the values you expect, not only that the code runs or that a count matches. Where the exact value is hard to state (statistical or
  numerically noisy results), assert a property, or compare against a reference within a tolerance.
- Build the object under test through its constructor (with a small stub for heavy dependencies) rather than with ``object.__new__`` and
  hand-set attributes, and make fakes behave like the real class, so that the test fails for the reason you test.
- Name a test after the behavior it checks, not after the bug that was fixed, and keep PR numbers out of names and docstrings.
- Use the shared fixtures (for example ``test_data_dir``) instead of building paths by hand.

## Running the Github CI

CI is powered by [copy-pr-bot](https://github.com/apps/copy-pr-bot), which mirrors each PR to a `pull-request/<number>` branch and triggers the workflow on push.

**CI runs automatically** when all of the following are true:
- Every commit in the PR is [GPG-signed](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification)
- Every committer is a member of the NVIDIA-NeMo GitHub org (or listed as an `additional_trustee`)
- The PR has no more than 249 commits

If any of those conditions are not met, copy-pr-bot posts a comment and skips branch creation — CI will not run. A maintainer (anyone with write access or greater) can trigger it manually by commenting:
```
/ok to test <sha>
```
where `<sha>` is the full or abbreviated SHA of the PR's HEAD commit. The bot also accepts `/okay to test` and `/ok-to-test`.

To re-run CI after a new push, the bot will sync automatically if the PR is still trusted. Otherwise comment `/ok to test <sha>` again with the new HEAD SHA.

The CI test suites are selectively ran based on the files that are changed. In some cases, no tests may be ran such as when docs are updated.

Lint checks using flake8 and pylint are ran on the code based on the files that were changed. Please resolve any lint errors. It is possible but discouraged
to ignore the lint errors by adding the "skip-linting" label to the PR.

To run the nightly e2e test suite on a PR, add the "Run e2e nightly" label. Labels are read once by the `pre-flight` job at the start of each run, so the label must be present before CI starts. If CI is already running when you add the label, cancel the active workflow run and re-trigger by pushing a new commit.

## Whom should you ask for review:
Please tag @nithinraok for NeMo core and ASR related PRs and @blisc for TTS related PRs.

Note that some people may self-assign to review your PR - in which case, please wait for them to add a review.

Your  pull requests must pass all checks and peer-review before they can be merged.

# General principles
1. **User-oriented**: make it easy for end users, even at the cost of writing more code in the background
2. **Robust**: make it hard for users to make mistakes.
3. **Well-tested**: please add simple, fast unittests. Consider adding CI tests for end-to-end functionality.
4. **Reusable**: for every piece of code, think about how it can be reused in the future and make it easy to be reused.
5. **Readable**: code should be easier to read.
6. **Legal**: if you copy even one line of code from the Internet, make sure that the code allows the license that NeMo supports. Give credit and link back to the code.
7. **Sensible**: code should make sense. If you think a piece of code might be confusing, write comments.

## Class naming conventions
* No “I”, “Interface”, “NM” nor “NeMo” pre/postfixes anywhere
* Core interfaces have simple names: Typing, Cloud, Serialization, FileIO*
* Core classes have the simplest names ever: NeuralModule, Model, Graph, Dataset, Loss, Module*
* Abstract classes in the Model hierarchy have Model postfix
* A config class for MyModel should be called MyModelConfig
* Leaf Neural Module classes have simple names without any postfixes (e.g. AudioPreprocess)
* Leaf Datasets have Dataset postfix (e.g. AudioToSpeechLabelDataset)
* Leaf Losses have Loss postfix (e.g. CTCLoss)
* Leaf Models do not have any postfix, just name (e.g. QuartzNet)

## Python style
We use ``black`` and ``isort`` (see ``pyproject.toml``) as our style guide. Run ``pre-commit run`` on your staged files, or
``pre-commit run --from-ref origin/main --to-ref HEAD`` for the whole branch, before you push.

1. Include docstrings for every new public class, function and method, and document every new parameter. Follow the docstring format of
   the file you change (Google style with ``Args:`` and ``Returns:``).
2. Use Python 3 type hints for every new function and method. Prefer the builtin generics (``list``, ``dict``, ``X | None``) to ``typing.List`` and
   friends, and accept ``str | Path`` for paths where it costs nothing.
3. Avoid wild import: ``from X import *`` unless in ``X.py``, ``__all__`` is defined.
4. Minimize the use of ``**kwargs``.
5. ``RaiseError`` is preferred to ``assert``. Write: ```if X: raise Error``` instead of ```assert X```. Reject a bad option when the object
   is constructed and before expensive work such as loading a model, and do not replace an error with a warning that lets the run continue
   with a wrong result. Re-raise with ``raise NewError(...) from e`` to keep the cause.
6. Generic helpers are private functions, tied to a class only when they need its state. A loss is a class with ``input_types``,
   ``output_types``, ``__init__`` and ``forward`` (see ``nemo/collections/asr/losses/bce_loss.py``).
7. Methods should be atomic. A method shouldn't be longer than 75 lines, e.g. can be fit into the computer screen without scrolling.
8. If a method has arguments that don't fit into one line, each argument should be in its own line for readability.
9. Add ``__init__.py`` for every folder.
10. F-strings are prefered to formatted strings.
11. Loggers are preferred to print. In NeMo, you can use logger from ``from nemo.utils import logging``
12. Private functions (functions start with ``_``) shouldn't be called outside its host file. If you need one elsewhere, drop the underscore.
13. Order a file by importance: ``__all__`` after the imports, then the public functions and classes (the main one first), and the helpers
    they are built on after the code that uses them, private helpers at the bottom. Follow the order of the file if it already has one.
14. Keep imports at the top of the file, in tests as well. A local import is only for an optional dependency or an import cycle. A required
    dependency is a normal top-level import.
15. Give a literal that appears more than once, or whose meaning is not obvious (a model name, a shape, a magic number), a name or make it
    a parameter. Avoid mutable module-level state.
16. Choose names that say what the thing is and match the neighbours (option names, config keys, scripts). A broad word such as ``prompt``
    or ``config`` needs a qualifier when the repository already uses it for something else.
17. Comments are short and only for what the code cannot show: a non-obvious constraint or the reason for a workaround. Do not restate the
    code, do not mention PR numbers or earlier versions, and cite a paper by its link. Mark a workaround for another project with a ``TODO``
    that names the upstream issue, and a deprecated alias with ``TODO(<your login>)`` and the release in which it is removed.
18. Prefer the simplest solution that does the job. Search the code for an existing helper before writing a new one, and for the same
    pattern elsewhere: when it already exists twice, share it instead of writing it a third time. Do not add code for cases that cannot
    occur, or an option with one sensible value.
19. YAML configs use ASCII only. A new option also goes into the structured config (dataclass) that its siblings are in.

## Documentation, dependencies and compatibility

1. Update the documentation in the same PR when you change a public API, a configuration option, a script or a workflow. After a rename or
   removal, search ``docs/``, ``examples/`` and the configs for the old name.
2. Write documentation for readers: simple language, one consistent narrative, no list of what each commit did. Implementation details
   belong in code comments, not in a README or a user guide. Put a fact that several pages need on one page and link to it.
3. Examples and configs point to released models and public datasets. Do not mention internal datasets, internal paths or models that are
   not released yet.
4. Released checkpoints and existing configurations keep their behavior: add new behavior as a new class or an opt-in option. When you
   rename a public argument, keep the old name as a deprecated alias that warns and names the release in which it is removed.
5. Optional dependencies are imported inside the function that uses them. A new dependency goes into ``pyproject.toml`` with its version
   constraint; do not raise the minimum torch or Python version in a feature PR.
6. Do not load checkpoints with ``torch.load(..., weights_only=False)``.

# Collections
Collection is a logical grouping of related Neural Modules. It is a grouping of modules that share a domain area or semantics.
When contributing module to a collection, please make sure it belongs to that category.
If you would like to start a new one and contribute back to the platform, you are very welcome to do so.

## Pull-request reviews

Comment `/review` on a pull request for a comprehensive review from the formal
review service. Use `/review mode=strict` for an exhaustive, maximally strict
review, or add `model=claude` to select a
Claude reviewer instead of the default Codex reviewer. `/review help` lists
all options. The retired `/claude review` and `/claude strict-review` commands
only reply with migration instructions when posted as an exact command by an
owner, member, or collaborator. Quoted mentions, other commenters, and bots
do not trigger a notice. These commands never run or automatically request a review.

The repository policy lives in `.claude/skills/nemo-speech-pr-review/SKILL.md`. The review service
must load this rubric from protected `main`, not the pull-request branch.
Before deploying this migration, publish the rubric, register its repository
profile, and verify a Ready plugin snapshot containing it. Until those
prerequisites are verified, do not rely on the redirect as evidence that the
repository-specific review policy is active.
