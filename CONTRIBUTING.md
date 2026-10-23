# How to contribute to Eurybia

This guide explains how to report bugs, propose improvements, and submit contributions to Eurybia.

- [How to open an issue](#how-to-open-an-issue)
- [Set up your environment](#set-up-your-environment)
  - [Get the code](#get-the-code)
  - [Install dependencies](#install-dependencies)
  - [Set up pre-commit hooks](#set-up-pre-commit-hooks)
- [Create your contribution](#create-your-contribution)
  - [Start coding on a dedicated branch](#start-coding-on-a-dedicated-branch)
  - [Run tests and build checks](#run-tests-and-build-checks)
  - [Run code quality checks](#run-code-quality-checks)
  - [Commit and push your changes](#commit-and-push-your-changes)
  - [Create a pull request](#create-a-pull-request)
- [Contributing using AI](#contributing-using-ai)
- [Publishing a release (maintainers)](#publishing-a-release-maintainers)

## How to open an issue

Before opening an issue:

- Search existing issues to avoid duplicates and check the project roadmap.
- Explain whether you are reporting a bug or proposing a feature.
- For bugs, include your Python and Eurybia versions, a minimal reproducible
  example, and the expected and actual behavior.

Every pull request must be linked to an issue. Discuss substantial changes with
the maintainers before starting implementation.

## Set up your environment

### Get the code

Fork [MAIF/eurybia](https://github.com/MAIF/eurybia) to your GitHub account, then
clone your fork:

```bash
git clone https://github.com/YOUR_GITHUB_PROFILE/eurybia.git
cd eurybia
git remote add upstream https://github.com/MAIF/eurybia.git
```

Before starting a contribution, update your local `master` branch:

```bash
git checkout master
git fetch upstream
git merge --ff-only upstream/master
```

### Install dependencies

Python `>=3.10, <3.15` is required. From the repository root, create and activate
a virtual environment, then install Eurybia in editable mode with development
and test dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,test]"
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in
PowerShell instead.

Alternatively, with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev --extra test
source .venv/bin/activate
```

See [pyproject.toml](pyproject.toml) for the supported Python versions and
dependency extras.

### Set up pre-commit hooks

Install both the code checks and the commit message hook:

```bash
pre-commit install
pre-commit install --hook-type commit-msg
```

The hooks are defined in [.pre-commit-config.yaml](.pre-commit-config.yaml).

## Create your contribution

### Start coding on a dedicated branch

```bash
git checkout -b feature/my-contribution-branch
```

Use `feature/your_feature_name` for a feature or `hotfix/your_bug_fix` for a bug
fix. Add or update tests for behavior changes and update related documentation.

### Run tests and build checks

Run the test suite:

```bash
python -m pytest
```

During development, you can run only the relevant test file:

```bash
python -m pytest tests/unit_tests/core/test_smartplotter.py
```

To collect coverage:

```bash
python -m pytest --cov=eurybia --cov-report=term-missing
```

Check that the source distribution and wheel build successfully:

```bash
python -m pip install build
python -m build
```

If you use uv, `uv build` is an alternative.

CI runs the test suite on Python 3.10 through 3.14 and checks compatibility with
multiple pandas versions. See [the CI workflow](.github/workflows/main.yml) for
the current matrices.

### Run code quality checks

Check linting, formatting, and type annotations:

```bash
ruff check .
ruff format --check .
mypy eurybia
```

Apply formatting changes with `ruff format .`. You can also run the pre-commit
checks across all tracked files:

```bash
pre-commit run --all-files
```

### Commit and push your changes

Stage only files related to your contribution. Use a
[Conventional Commits](https://www.conventionalcommits.org/) message, as checked
by the commit message hook:

```bash
git add <file>
git commit -m "fix: describe the bug fix"
git push origin feature/my-contribution-branch
```

Hooks may modify files. Review those changes, stage the relevant files again,
and retry the commit if necessary.

### Create a pull request

On GitHub, click **Compare & pull request** and select:

- Base repository: `MAIF/eurybia`
- Base branch: `master`
- Head repository: your fork
- Head branch: your contribution branch

Complete the repository's pull request template, link the related issue, and
describe the changes and validation performed. Ensure CI passes and address
review feedback before the pull request is merged.

Thank you for contributing to Eurybia.

## Contributing using AI

You may use AI tools to help prepare a contribution, but you remain responsible
for its correctness and quality. The
[unitaryHACK guide to ethical AI contributions](https://unitaryhack.dev/ai-guide/)
provides useful guidance.

- Understand and review every change before submitting it. Be prepared to
  explain your implementation and respond to review feedback yourself.
- Validate generated code, examples, and API references. Run the relevant tests
  and quality checks, cover important edge cases, and report what you verified.
  Do not remove or weaken tests just to make generated code pass.
- Keep contributions focused. Remove unnecessary generated comments,
  boilerplate, and explanations that do not help maintain the code.
- Describe AI assistance in the pull request: identify the tool, what it helped
  with, and how you reviewed and validated the result.
- Do not automate bulk submissions of AI-generated issues or pull requests.
  Each contribution needs individual investigation and review.

## Publishing a release (maintainers)

Publishing a GitHub release triggers [the PyPI publication workflow](.github/workflows/publish.yaml).
Pushing a tag alone runs CI but does not publish a package.

To release:

1. Create a version tag such as `v1.4.1` on the commit to release.
   Package versions are derived from Git tags using `setuptools-scm`.
2. Publish a GitHub release using that tag.
3. The workflow runs the complete Python and pandas CI matrices on the release
   commit, builds the distributions, and checks that the wheel version matches
   the release tag before publishing to PyPI via OIDC.
4. A final smoke test installs the exact published version from PyPI and checks
   its import and version. A failure here occurs after publication; it does not
   roll back the release.

This process does not use TestPyPI.
