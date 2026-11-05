# MVRFDet-Lightweight

A reproduce and runnable **lightweight reproduction/variant of MVRFDet** for PyPI malicious-package detection. It preserves the three-view pipeline, offline sensitive-API workflow, feature-matrix construction, CNN, channel attention, and binary classifier while replacing the heaviest representation models with static features.

## Architecture

```text
OFFLINE
verified malicious packages
    -> Scalpel API call graphs
    -> six centralities (+1)
    -> aggregate by API name
    -> Top-500
    -> paper prompt + DeepSeek
    -> sensitive_apis.json

ONLINE
PyPI package
    |-- metadata: PKG-INFO -> 7 fields -> static/hash encoder -> Vm[768]
    |-- code: Scalpel API graph -> sensitive match -> <=4096-token contexts -> AST/API encoder -> mean pool -> Vc[768]
    `-- graph: Scalpel API call graph + context CFGs -> graph statistics -> Gcg[512] || Gcfg[512] -> Vg[1024]

Vm/Vc/Vg
    -> pad/L2/reshape/stack
    -> F[3,32,32]
    -> 3-layer CNN
    -> channel attention
    -> MLP
    -> benign / malicious
```

## Unified dataset runner

Run stages independently when you want to reuse saved results. Use `full` to
process packages through all three stages in one pass.

```powershell
# Extract features from one sample in the local dataset.
python scripts/run_pipeline.py stage1 official_dataset --only 'custohtkinter-1.0.0.tar.gz' --output-dir runs/smoke

# Reuse saved features, then classify the saved matrices.
python scripts/run_pipeline.py stage2 runs/smoke/stage1 --output-dir runs/smoke
python scripts/run_pipeline.py stage3 runs/smoke/stage2 --checkpoint checkpoints/model.pt --output-dir runs/smoke

# Or run the whole pipeline directly from the package.
python scripts/run_pipeline.py full official_dataset --only 'custohtkinter-1.0.0.tar.gz' --checkpoint checkpoints/model.pt --output-dir runs/smoke
```

| Command | Input | Main output per package |
| --- | --- | --- |
| `stage1` | Dataset directory, unpacked package, or `.tar.gz` source archive | `stage1/features.npz`, graph files, and HTML report |
| `stage2` | Stage 1 report directory or individual feature `.npz` file | `stage2/matrix.npy` and `stage2/matrix.json` |
| `stage3` | Stage 2 report directory or individual matrix `.npy` file, plus a trained `--checkpoint` | `stage3/inference.json` |
| `full` | Same inputs as `stage1`, plus a trained `--checkpoint` | All three stages and an HTML report |

`stage1` and `full` default to `dataset/` when the input is omitted. Directory
inputs to `stage2` and `stage3` are searched recursively for `features.npz`
and `matrix.npy`, respectively. `full` uses the configured sensitive API list
and runs locally. The runner reads package archives as source files only.

Existing flags `--stage1`, `--stage2`, `--stage3`, `--lite`, `--all`, and
`--showfig` still accept package inputs. Use `--all` with a DeepSeek API key to
include per-package LLM reasoning or `--showfig` for graph visuals.

Reports are written under `data/cache/pipeline_reports/<mode>/<package>/` by
default. `--output-dir` changes that base directory. Each run also writes a
`summary.json` with per-package status and report paths. Common options are:

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--sensitive-list` | `data/sensitive_api/sensitive_apis.json` | Static API rules used when processing packages. |
| `--output-dir` | `data/cache/pipeline_reports` | Base directory for mode-specific package reports. |
| `--backend` | `ast` | Static analysis backend: `ast`, `scalpel`, or `auto`. |
| `--checkpoint` | unset | Trained model checkpoint used for Stage 3 prediction. |
| `--only` | unset | Package/archive filename glob filter. |
| `--limit` | unset | Maximum packages to process; useful for a small trial. |
| `--llm-api-key-env` | `DEEPSEEK_API_KEY` | Environment variable name containing the API key (`--all`). |
| `--llm-model` | `DEEPSEEK_MODEL` or `deepseek-v4-pro` | DeepSeek model ID (`--all`). |
| `--llm-api-url` | `DEEPSEEK_API_URL` or the DeepSeek endpoint | Full Chat Completions URL override. |
| `--llm-api` | unset | Restrict LLM verification to one detected API name. |
| `--llm-max-apis` | `1` | Maximum API names verified per package. |
| `--llm-contexts-per-api` | `2` | Maximum source contexts sent per API. |
| `--llm-prompt` | `prompts/sensitive_api/paper_prompt.txt` | Prompt template path. |
| `--llm-timeout` | `120` | Timeout in seconds per DeepSeek request. |
| `--llm-retries` | `2` | Number of request attempts. |
| `--llm-max-tokens` | `4096` | Maximum output tokens. |
| `--llm-refresh` | off | Ignore cached reasoning and send a new request. |

Run `python scripts/run_pipeline.py --help` to see the current CLI options.

## Environment policy

The recommended environment for this project is **Kali Linux (or another Linux distribution) with a dedicated CPython 3.10 virtual environment**.

Do **not** use Kali's system Python 3.14 environment for Scalpel. Scalpel is an older research package and its dependency stack is much safer to isolate in Python 3.10. The virtual environment prevents the project from modifying Kali's system-managed Python packages.

The project targets:

```text
OS:      Kali Linux / Linux
Python:  CPython 3.10.x
Env:     venv
Scalpel: installed explicitly from the official SMAT-Lab/Scalpel GitHub repository
```

## System packages on Kali Linux

Install the native Graphviz executable and common build tools first:

```bash
sudo apt update
sudo apt install -y \
  graphviz \
  build-essential \
  git \
  curl
```

`graphviz` in `requirements.txt` is the Python wrapper. The `apt` package above provides the native `dot` executable used for rendering/Graphviz integration.

Verify it:

```bash
dot -V
```

## Create a Python 3.10 venv

### Option A - `python3.10` is already available

Check:

```bash
python3.10 --version
```

Then create the environment:

```bash
cd MVRFDet-Lightweight
python3.10 -m venv .venv
source .venv/bin/activate
```

Confirm that the venv is using the correct interpreter:

```bash
python --version
which python
```

Expected:

```text
Python 3.10.x
.../MVRFDet-Lightweight/.venv/bin/python
```

### Option B - Kali only provides Python 3.14

Do not replace or downgrade Kali's system Python. Install Python 3.10 separately and use it only for this project. A practical approach on a rolling distribution is `pyenv`.

Install Python build prerequisites:

```bash
sudo apt update
sudo apt install -y \
  make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev \
  wget curl llvm libncurses-dev xz-utils \
  tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

Install `pyenv` using its official installation method, open a new shell, then install a Python 3.10 release, for example:

```bash
pyenv install 3.10.21
pyenv local 3.10.21
python --version
```

Create the venv:

```bash
python -m venv .venv
source .venv/bin/activate
```

The important point is that `python --version` inside `.venv` must report Python 3.10.x. Kali's `/usr/bin/python3` can remain Python 3.14.

## Install Python dependencies

With `.venv` activated, first install the project's normal dependencies:

```bash
python -m pip install --upgrade pip wheel
python -m pip install "setuptools==80.10.2"
python -m pip install -r requirements.txt
```

`requirements.txt` intentionally **does not install Scalpel**. This prevents `pip install -r requirements.txt` from silently selecting a PyPI release when the intended setup is to install Scalpel directly from its official source repository.

### Install Scalpel directly from the official repository

Install Scalpel into the active Python 3.10 venv directly from the official `SMAT-Lab/Scalpel` GitHub repository:

```bash
python -m pip install "git+https://github.com/SMAT-Lab/Scalpel.git"
```

This command uses `pip`'s VCS support. It downloads/builds the package from the official repository and installs it into `.venv`; you do **not** need to manually clone the repository.

For exploratory development, installing the current repository revision is convenient. Once you identify a revision that passes the Scalpel acceptance tests, pin that exact commit for reproducible experiments:

```bash
python -m pip install \
  "git+https://github.com/SMAT-Lab/Scalpel.git@<COMMIT_HASH>"
```

Record the tested commit hash in your experiment notes/README before running the final corpus. This avoids future changes to the repository changing the static-analysis behavior.

After Scalpel is installed, install this project itself without re-resolving dependencies:

```bash
python -m pip install -e . --no-deps
```

Why `setuptools==80.10.2` is pinned: Scalpel's call-graph stack uses `pkg_resources`, so this project keeps a compatible Setuptools release inside the isolated venv.

Do not run these commands with `sudo pip`.

## Verify Scalpel and Graphviz

Run this immediately after installing Scalpel from GitHub:

```bash
python - <<'PY'
import scalpel
from scalpel.cfg import CFGBuilder
from scalpel.call_graph.pycg import CallGraphGenerator

print("Scalpel import: OK")
print("CFGBuilder: OK")
print("CallGraphGenerator: OK")
PY
```

You can also inspect which Scalpel distribution is installed:

```bash
python -m pip show python-scalpel
```

Verify the Graphviz binary as well:

```bash
dot -V
```

If the three Scalpel imports and `dot -V` pass, the environment is ready for the Scalpel integration tests.

## Recommended Scalpel acceptance tests

Before processing a large malicious-package corpus, verify Scalpel against small controlled samples in this order:

```text
1. Basic intra-file function call: a() -> b()
2. External API: socket.socket()
3. Import alias: import subprocess as sp; sp.Popen(...)
4. CFG containing if/else branches
5. Multi-file Python package
6. One unpacked real PyPI package
```

The important checks are call-graph topology, qualified API names, source mapping, and CFG construction.

## Static-analysis backends

The project keeps two backends by design:

```text
scalpel  -> intended reproduction backend
ast      -> lightweight/debug fallback
```

Use AST only for local development or when testing modules that do not depend on paper-style graph extraction. For an experimental run intended to match the MVRFDet methodology, use Scalpel and do not silently mix Scalpel graphs with AST fallback graphs.

> Current code note: the repository already has the `StaticAnalysisAdapter` boundary. Scalpel is an external environment dependency installed explicitly from the official `SMAT-Lab/Scalpel` repository. The version-specific conversion from Scalpel's raw call-graph/CFG objects into the project's `APICallGraph`/CFG contracts must still be completed and validated before `--backend scalpel` is treated as production-ready.

## What is already implemented

- PKG-INFO/METADATA locator and regex-based extraction of the seven fields.
- Lightweight metadata encoder producing exactly `Vm[768]`.
- Runnable AST fallback API call graph extractor plus a strict Scalpel adapter boundary.
- Sensitive API list loader and occurrence locator.
- Centered token-window context extraction with 4096-token cap and identical-snippet deduplication.
- AST + API + token lightweight code encoder producing `Vc[768]`, mean-pooled across contexts.
- Approximate AST CFG builder used as a development fallback.
- API-CG and CFG statistical encoders producing `512 + 512`, concatenated to `Vg[1024]`.
- Paper-style Stage 2 producing `F[3,32,32]`.
- CNN + channel attention + 2-class MLP.
- Training, evaluation, prediction scripts.
- Offline six-centrality aggregation, Top-K candidate generation, paper-prompt loading, DeepSeek verification, and response caching.
- Unit and end-to-end tests for the lightweight path.

See `REPRODUCTION_GAPS.md` for everything that is intentionally not claimed to be paper-exact.

## Run tests

```bash
pytest
```

## 1. Extract one package's features

Until the Scalpel conversion layer passes the acceptance tests, the AST backend remains useful for validating the rest of the pipeline:

```bash
python scripts/extract_features.py \
  --package /path/to/unpacked/package \
  --sensitive-list data/sensitive_api/sensitive_apis.example.json \
  --output data/cache/example.npz \
  --backend ast
```

After the Scalpel adapter is validated, the intended reproduction command becomes:

```bash
python scripts/extract_features.py \
  --package /path/to/unpacked/package \
  --sensitive-list data/sensitive_api/sensitive_apis.json \
  --output data/cache/example.npz \
  --backend scalpel
```

The output `.npz` contains `vm`, `vc`, `vg`. A neighboring `.json` contains extraction diagnostics.

### Per-package stage reports and graph visualizations

For an unpacked package, add `--report-dir` to save a browsable report alongside
the stage outputs. `--output` is optional when a report directory is provided.

```bash
python scripts/extract_features.py \
  --package dataset/malware/1inch-8.7 \
  --sensitive-list data/sensitive_api/sensitive_apis.json \
  --report-dir data/cache/package_reports/1inch-8.7 \
  --backend ast
```

Open `data/cache/package_reports/1inch-8.7/report.html`. Each package directory
contains `stage1/` (metadata, contexts, vectors, full API call graph and CFG JSON,
DOT, SVG), `stage2/` (the 3x32x32 matrix), and `stage3/` (inference status). API
graph SVGs show at most 160 nodes to remain readable; the JSON stores the full
graph. Supply `--checkpoint checkpoints/model.pt` to include Stage 3 prediction.
Without a checkpoint, Stage 3 is explicitly marked as not run.

To process local `.tar.gz` source archives into separate package directories:

```bash
python scripts/smoke_dataset.py \
  --dataset-dir dataset/malware \
  --sensitive-list data/sensitive_api/sensitive_apis.json \
  --output-dir data/cache/package_reports/archives \
  --timeout 40
```

The batch script reads only regular Python source and package metadata from
archives into temporary directories; it does not import or install the samples.
The batch summary is `archives/summary.json`. The figures reflect the selected
lightweight AST backend and do not claim Scalpel-equivalent graphs.

### Optional DeepSeek reasoning for package reports

Add `--llm-reasoning` to either report command to verify detected suspicious APIs
with the paper prompt in DeepSeek thinking mode. This is an **opt-in audit output**:
it writes `stage0/llm_reasoning.json` and a link in `report.html`; it does not
alter `Vm`, `Vc`, `Vg`, the sensitive API list, or the Stage 3 prediction. The
offline sensitive-list builder below is the separate workflow for constructing
that list.

The key is read from the environment variable named by `--llm-api-key-env`.
The default name is `DEEPSEEK_API_KEY`; a key stored in `ANTHROPIC_API_KEY` must
be selected explicitly. The key value is never written to the report. For the
current Windows checkout, with `.venv` already installed, a one-archive run is:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
.\.venv\Scripts\python.exe scripts/smoke_dataset.py `
  --dataset-dir dataset/malware `
  --sensitive-list data/sensitive_api/sensitive_apis.json `
  --output-dir data/cache/package_reports/archives `
  --only 1inch-8.7.tar.gz `
  --llm-reasoning `
  --llm-api-key-env ANTHROPIC_API_KEY `
  --llm-model deepseek-v4-pro `
  --llm-api subprocess.Popen `
  --llm-max-apis 1
```

Remove `--only` and `--llm-api` to process all archives and let the command
select up to `--llm-max-apis` detected API names per package. The complete
dataset run writes `archives/summary.json`; filtered runs write
`archives/summary.selection.json` so they do not replace the complete summary.
To process an unpacked package instead, use `scripts/extract_features.py` with
`--package`, `--sensitive-list`, `--report-dir`, `--backend ast`, and the same
`--llm-*` options.

| Parameter | Default | Purpose |
| --- | --- | --- |
| `--llm-reasoning` | off | Enable package-level DeepSeek thinking requests. Requires `--report-dir` for one package. |
| `--llm-api-key-env` | `DEEPSEEK_API_KEY` | Name of the environment variable holding the key. |
| `--llm-model` | `DEEPSEEK_MODEL` or `deepseek-v4-pro` | Model identifier available to your account. |
| `--llm-api-url` | `DEEPSEEK_API_URL` or DeepSeek Chat Completions URL | Endpoint override. |
| `--llm-api` | unset | Check just this API name if detected in the package. |
| `--llm-max-apis` | `1` | Maximum distinct API names checked per package. |
| `--llm-contexts-per-api` | `2` | Maximum distinct code snippets sent for each API. |
| `--llm-prompt` | `prompts/sensitive_api/paper_prompt.txt` | Prompt template path. |
| `--llm-timeout` | `120` | Seconds allowed for one request. |
| `--llm-retries` | `2` | Maximum attempts for a failed request. |
| `--llm-max-tokens` | `4096` | DeepSeek output token cap in thinking mode. |
| `--llm-refresh` | off | Ignore a matching saved case and request it again. |
| `--only` / `--limit` | unset | Batch only: select archive filename glob / cap archive count. |
| `--timeout` | computed | Batch only: whole-package timeout; grows with LLM request limits when omitted. |

Completed cases are reused when the model, endpoint, prompt, and token cap
match. `llm_cached` in the command summary counts cases reused without a new
request. A request failure remains visible as `partial_error` with its details
in the package's JSON report; it does not erase the static-analysis outputs.

## 2. Build the offline sensitive API list

First generate candidates without spending API calls. While developing the pipeline, `--backend ast` can still be used as a smoke-test. For the real experiment, switch to `--backend scalpel` after Scalpel integration is validated.

Development smoke-test:

```bash
python scripts/build_sensitive_api_list.py \
  --packages-dir /path/to/verified-malicious-packages \
  --output-dir data/sensitive_api/run_v1 \
  --backend ast \
  --skip-llm
```

Intended Scalpel run:

```bash
python scripts/build_sensitive_api_list.py \
  --packages-dir /path/to/verified-malicious-packages \
  --output-dir data/sensitive_api/run_v1 \
  --backend scalpel \
  --skip-llm
```

Then configure DeepSeek access. Do not commit the key:

```bash
export DEEPSEEK_API_KEY='...'
export DEEPSEEK_API_URL='https://api.deepseek.com/chat/completions'
export DEEPSEEK_MODEL='YOUR_EXACT_MODEL_ID'
```

Run LLM verification:

```bash
python scripts/build_sensitive_api_list.py \
  --packages-dir /path/to/verified-malicious-packages \
  --output-dir data/sensitive_api/run_v1 \
  --backend scalpel
```

For a small offline thinking-mode trial using the key in `ANTHROPIC_API_KEY`,
add `--top-k 1 --llm-thinking --llm-api-key-env ANTHROPIC_API_KEY
--llm-model deepseek-v4-pro`. The builder also accepts `--llm-api-url`,
`--llm-timeout`, `--llm-retries`, `--llm-max-tokens`, and `--llm-refresh`.

Every LLM response is cached under `llm_results/`, so interrupted runs can resume without re-querying completed candidates.

## 3. Train Stage 3

Create CSV manifests with columns:

```csv
feature_path,label
features/pkg_a.npz,0
features/pkg_b.npz,1
```

Then:

```bash
python scripts/train.py \
  --train-manifest train.csv \
  --val-manifest val.csv \
  --output checkpoints/mvrfdet_light.pt
```

Default starting point is Adam, learning rate `2e-5`, batch size `8`, 100 epochs.

## 4. Evaluate / predict

```bash
python scripts/evaluate.py --manifest test.csv --checkpoint checkpoints/mvrfdet_light.pt
```

```bash
python scripts/predict.py \
  --package /path/to/unpacked/package \
  --sensitive-list data/sensitive_api/run_v1/sensitive_apis.json \
  --checkpoint checkpoints/mvrfdet_light.pt
```

## Security note for malicious packages

Analyze packages as files only. Do not `pip install` an untrusted sample and do not execute its `setup.py`, package entry point, or imported modules. Prefer an isolated VM/container with no secrets and restricted network access for corpus processing.
