<div align="center">

<img src="header.png" alt="AlphaLab" width="600">

**Autonomous research agent.** Give it a dataset and a task, and it will explore the data end-to-end, build an evaluation framework, then run dozens of experiments on GPUs — all without human intervention.

[DETAILS.md](DETAILS.md)

*Released by the Morgan Stanley Machine Learning Research Team*

</div>

---

## What is AlphaLab?

AlphaLab automates quantitative, verifiable research end-to-end. You give it a dataset and a natural-language objective — "optimize these CUDA kernels", "forecast this time series", "train the best small language model you can in 20 minutes" — and it goes off for hours or days, autonomously exploring the data, building its own evaluation framework, running dozens of GPU experiments, and evolving its own methodology as it learns what works. No human in the loop.

The system is **domain-agnostic**: the same pipeline handles CUDA kernel optimization, LLM pretraining, time series forecasting, and any other domain with objective, easy-to-evaluate metrics. All domain-specific behavior lives in *adapters* that the model generates itself from the data — AlphaLab writes its own prompts, defines its own metrics, and builds its own evaluation harness. It is **self-evolving**: a persistent *playbook* accumulates knowledge experiment-by-experiment, functioning as online prompt optimization — by the end of a campaign, the system has discovered and encoded domain methodology that didn't exist anywhere in its code or prompts at launch. And it is **recursively LLM-powered**: agents can spawn sub-agents as needed, delegating subtasks without polluting the parent's context.

This repository accompanies the paper *AlphaLab: Autonomous Multi-Agent Research Across Optimization Domains with Frontier LLMs*. See the paper for full experimental details, ablations, and analysis.

<div align="center">
<img src="docs/pipeline_overview.png" alt="AlphaLab Pipeline" width="800">
</div>

### The Pipeline

AlphaLab is a *harness*: a combination of tools and a structured environment that converts a frontier LLM into an autonomous research agent. The system is LLM-agnostic — any model that supports tool use and multi-modal input can be dropped in with no changes to the infrastructure. We evaluate with GPT-5.2 and Claude Opus 4.6 in the paper; all differences in outcomes are attributable to the model, not the harness.

The pipeline runs in four phases:

- **Phase 0 — Adapter Resolution.** All domain-specific behavior is parameterized by a *domain adapter*: 11 files comprising prompt templates (one per agent role), metric definitions, experiment structure, and a `domain_knowledge.md` document that is injected into every agent's context for the entire campaign. Phase 0 generates or customizes this adapter by examining the actual data and searching the web for prior work. The key idea is that *prompt engineering is performed by the model*, grounded in the data.

- **Phase 1 — Data Exploration.** A single Explorer agent operates autonomously for 1-2 hours: it generates a plan, then works through it — writing and running Python scripts, generating plots, searching the web for relevant papers, and updating its notes after each finding. It produces a human-readable research report and a machine-readable `learnings.md` consumed by later phases.

- **Phase 2 — Adversarial Evaluation Construction.** Evaluation correctness is critical — if the metric is wrong, every experiment optimizes the wrong objective. Phase 2 addresses this through a multi-agent adversarial loop: a **Builder** writes the evaluation framework, a **Critic** (a fresh agent with no shared context) audits for data leakage, lookahead bias, and metric errors, and a **Tester** writes and runs an automated test suite. The loop repeats until all tests pass.

- **Phase 3 — GPU-Scale Experimentation.** The core of the system: a sustained experimental campaign where a **Strategist** proposes experiments, **Workers** implement and analyze them, and a **Dispatcher** (pure Python, no LLM) orchestrates GPU resources. A persistent **Playbook** accumulates domain knowledge across experiments, creating a feedback loop that functions as online prompt optimization — the system literally gets better at its task as the campaign progresses. The campaign runs until convergence (no improvement for 20 consecutive experiments) or budget exhaustion.

A **Supervisor** meta-agent monitors health across the pipeline, intervening when error rates spike — it can diagnose systemic issues and patch the adapter's domain knowledge on the fly.

### Tools

Every agent has access to the same core tool set:

| Tool | Description | Usage |
|------|-------------|-------|
| `shell_exec` | Full Unix shell access — write code, install packages, run training, manage jobs | ~50% of all tool calls |
| `web_search` | Search the web for papers, documentation, and prior work | Heavy in early phases |
| `spawn_agent` | Launch a sub-agent with its own context window and full tool access — enables recursive delegation | Used for complex subtasks |
| `view_image` | Read plots and visualizations the agent generates | Used to inspect results |

Additional tools include `read_file`, `grep_file`, `propose_experiment`, `report_to_user`, and adapter-specific tools for reading/patching domain configuration. The full tool set is defined in [`src/alpha_lab/tools.py`](src/alpha_lab/tools.py).

### Key Results (from the paper)

| Domain | Task | Best Result | vs. Baseline |
|--------|------|-------------|-------------|
| **CUDA Kernels** | Write optimized GPU kernels vs. `torch.compile` | **4.4x** mean speedup (up to **91x**) | Outperforms compiled PyTorch on 83% of tasks |
| **LLM Pretraining** | Minimize val BPB under 20-min budget, <100M params | **0.7578** BPB (Opus) | 22% lower loss than single-shot baseline |
| **Traffic Forecasting** | 24h ahead road occupancy (862 sensors) | **0.0214** RMSE (Opus) | 25% better than seasonal baseline |

Each campaign costs $150-200 in LLM API calls and completes in 12-48 hours on 4x H100 hardware. The two models (GPT-5.2 and Claude Opus 4.6) discover qualitatively different solutions in every domain — neither dominates uniformly — suggesting that multi-model campaigns provide complementary search coverage.

For detailed architecture documentation, see [DETAILS.md](DETAILS.md).

---

## 🚨🚨🚨 SAFETY WARNING 🚨🚨🚨

> **⚠️ READ THIS BEFORE RUNNING ⚠️**
>
> AlphaLab runs an LLM **in a loop** as **you** on your machine.
>
> Anything you can do from your shell, AlphaLab can do:
> - **Delete files and directories** — anything you have permission to `rm`, it can `rm`
> - **Overwrite files** — anything you can write to, it can overwrite
> - **Execute arbitrary code** — it writes and runs scripts autonomously
> - **Install packages, modify environments, make network calls**
>
> It is **not malicious**, but it is **autonomous** — and autonomous agents make mistakes. Assume that anything you have permission to destroy *could* be destroyed.
>
> **Before running:**
> - Run in an **isolated workspace** — not in your home directory
> - **Back up anything that matters**
> - Understand that you are giving an AI agent **the same access as your user account**

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/morganstanley/MSML.git
cd MSML/projects/alpha-lab

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install PyTorch for your hardware (see https://pytorch.org/get-started/locally/)
pip install torch  # CPU only
# pip install torch --index-url https://download.pytorch.org/whl/cu126  # CUDA 12.6
```

### Native Windows

Local CPU and GPU experiments run directly through Python on Windows. WSL
and Git Bash are not required. Agent `shell_exec` commands use PowerShell;
file search has a native Python implementation. Cancellation and timeouts
terminate the entire experiment process tree using Windows Job Objects.
POSIX hosts continue to use process groups, and SLURM remains Linux-specific.

Install the dependencies in a Windows Python 3.11+ virtual environment and
install a CUDA PyTorch wheel appropriate for your NVIDIA GPU. Keep the
experiment interpreter configured with a Windows path (`ALPHALAB_PYTHON`
or `pipeline.phase3.python_executable`). Linux virtual environments and
Linux wheels cannot be used by a Windows interpreter.

To verify local execution without API keys or LLM calls, run from the project
directory in the activated environment:

```powershell
python examples/verify_local.py --workspace ./workspace/native_check --gpu 0
```

This trains a small CPU regression model and a small CUDA model and writes
`workspace/native_check/verification.json` and individual experiment logs.
Omit `--gpu 0` to check CPU execution only. Automated lifecycle tests, including
child-process cancellation, timeouts and PID-safe GPU recovery, are available
with `python -m pytest tests/test_local_experiments.py tests/test_local_process.py`.

Before launching a paid campaign, load your API key into the process environment.
The repository's `.env` file is not automatically loaded by `run.py`. For example:

```powershell
python -c "from dotenv import load_dotenv; load_dotenv(); from alpha_lab.run import run_main; run_main()" --config data/demo_exchange_single_gpu.json --workspace ./workspace_demo
```

Generated code must also use portable Python paths and process APIs. CUDA kernel
compilation and Linux-only libraries may require additional domain-specific work;
the local runner support does not make every research domain Windows-compatible.

### Default market data: Sharadar first

`alpha-lab-data` (or `python -m alpha_lab.market_data.pipeline`) now uses
Sharadar as the primary source. All 14 bulk tables remain available, including
fundamentals, daily market-cap metrics, stock/fund prices, corporate actions,
S&P 500 membership, insiders and institutional holdings. The `daily_prices`
dataset chooses `stocks` or `funds` for each requested ticker.

The default command looks for ignored `data/market/pipeline.json`, then falls
back to `data/market.example.json`. The pipeline configuration references
existing provider configuration files; paths resolve relative to that file.
Keep both API keys in `.env`.

```json
{
  "sharadar_config": "sharadar.json",
  "tiingo_config": "tiingo.json",
  "tickers": ["SPY"]
}
```

```powershell
# Update the configured Sharadar tables; check configured tickers for gaps.
alpha-lab-data sync
alpha-lab-data status
alpha-lab-data route --ticker SPY
alpha-lab-data export --ticker SPY --output data/market/snapshots/spy-primary-v1.csv
# Explicit earlier-history request routes SPY's complete series to Tiingo.
alpha-lab-data sync --dataset daily_prices --ticker SPY --start-date 1993-01-29
alpha-lab-data export --ticker SPY --start-date 1993-01-29 --output data/market/snapshots/spy-long-v1.csv
# Other datasets always use their Sharadar table, preserving native fields.
alpha-lab-data export --dataset fundamentals --ticker AAPL --dimension ARQ --output data/market/snapshots/aapl-arq-v1.csv
alpha-lab-data catalog
```

With no requested start date, prices use Sharadar's available full history.
Tiingo EOD is used only when the ticker is absent from Sharadar's downloaded
stock/fund tables or the request starts before its Sharadar history. Tiingo
coverage is checked during sync; fallback is not a promise that Tiingo has
the missing data. Missing/corrupt local snapshots and authentication, network,
rate-limit or validation failures are errors and never trigger fallback.
Updates sync primary tables before checking gaps, so stale primary data is
refreshed through Sharadar. A requested end date beyond Sharadar's last bar
does not imply a missing dataset. Direct bulk table syncs always fetch the
full table; ticker/date options control routing and offline exports.

Daily-price exports accept one ticker at a time. Each export retains one
provider's native schema and a manifest with the source, routing reason,
adjustment conventions and content hash. No adjusted price series are
spliced across providers. `--asset-table stocks|funds` resolves a ticker
present in both Sharadar tables. `route`, `status` and `export` are offline;
sync a fallback request before exporting it. CSV snapshots are immutable.

The existing Tiingo adapter covers daily prices. News, intraday/overnight
prices, fund fees and detailed dividends are listed as Tiingo-only datasets
in `catalog`, with `not_implemented` status; requesting them gives an explicit
error. Their adapters and account entitlements must be added before ingestion.
No LLM campaign or recurring schedule is started by these commands.

### Read-only equity-risk handoff

The sibling `projects/equity-risk/docs/DATA_INTERFACE.md` defines an audited
session/security panel. Raw table access and a prepared `RiskPanel` are separate
stages. The infrastructure command below supplies immutable source bindings and
complete shared benchmark/cash/calendar arrays, with panel readiness explicitly
false until the historical preparation audits are completed.

```powershell
python -m pip install -e ".[risk-data]"
python -m alpha_lab.market_data.risk_handoff --config data/market/sharadar.json --interface ../equity-risk/docs/DATA_INTERFACE.md --output data/market/handoffs/risk-v1 --download-cash
# Offline repeat using the archived public cash CSV; always use a new directory.
python -m alpha_lab.market_data.risk_handoff --config data/market/sharadar.json --output data/market/handoffs/risk-v2 --cash-csv data/market/handoffs/risk-v1/cash-DGS3MO.csv
```

Each handoff pins and hashes the seven required Sharadar tables. It archives
the exact interface document, the raw FRED DGS3MO cash-yield CSV, a
`shared_series.npz` with pickle-free arrays, `sources.json`, and checksummed
`COMPLETE.json`. The command reads the source lake without syncing, publishing,
changing pointers or writing temporary files there. An interrupted output is
marked `INCOMPLETE` and cannot be loaded. Old pinned source versions must be
retained while consumer runs reference them.

The selected S&P 500 benchmark is SPY's Sharadar `closeadj` total-return proxy,
including ETF expenses and tracking differences; it is not the official SPTR
index. Cash uses the annual percentage yield in public FRED DGS3MO: simple
ACT/365F accrual across calendar days between adjacent closes. The rate is the
latest observation dated strictly before the preceding session, carried forward
at most seven calendar days. It is a documented Treasury-yield cash proxy,
not an actual deposit or traded bill return. These are shared input conventions
for all three risk models. Source references are embedded in `sources.json`.

XNYS sessions and early closes come from a recorded `exchange_calendars`
version. SPY must have a valid price for every session and its preceding session.
The default return range therefore starts on the session after the first
Sharadar price and ends at the last common complete stock/SPY date. No missing
benchmark/cash observations become zero, and requested unsupported date ranges
fail rather than silently changing the benchmark or splicing feeds.

Consumer access from the shared Python environment:

```python
import numpy as np
from alpha_lab.market_data.risk_handoff import load_handoff, iter_source_rows

bundle = load_handoff("data/market/handoffs/risk-v1", verify_sources=True)
rows = iter_source_rows(bundle, "stocks", tickers=["AAPL"],
                        start="2025-01-01", end="2025-12-31",
                        columns=["ticker", "date", "closeadj", "closeunadj", "close", "volume"])
# Iterate/aggregate rows without loading the entire stocks table into Python.
dictionary = list(iter_source_rows(bundle, "descriptions", columns=["table", "indicator", "unittype"]))
with np.load("data/market/handoffs/risk-v1/shared_series.npz", allow_pickle=False) as inputs:
    dates, benchmark, cash = inputs["dates"], inputs["market_returns"], inputs["cash_returns"]
```

Consumers in separate environments can query `sources[table].path` with
DuckDB directly. Filter dates/tickers, quote column identifiers (including the
reserved `table` field), and read pinned paths instead of following `latest.json`
again. The sibling reader's unquoted projection of `table` fails; reading the
complete dictionary or using `iter_source_rows` avoids that projection issue.

Current corrected vendor snapshots and conservative date lags do not establish
an original feed replay. The handoff does not attest point-in-time security
panels or reconciled terminal outcomes. Its readiness record enumerates issuer
links/cap treatment, dated eligibility/SIC, accounting/common-book checks,
listing-age evidence and terminal-action audits still required before producing
the model's `panel.npz` and `metadata.json`. A schema match alone does not justify
setting the contract's producer attestation booleans to true.

### Tiingo daily market data

The market data CLI runs independently of the research agents and makes no
LLM calls. It stores US stock/ETF daily bars in SQLite, including raw OHLCV,
split/dividend-adjusted OHLCV, `div_cash`, and `split_factor`.

Add `TIINGO_API_KEY=...` to your ignored local `.env`. Keep the key out of
configuration JSON and source control. Authentication uses the Authorization
header; the token is never placed in the request URL or printed in errors.

Edit `data/tiingo.example.json` for your ticker universe and start date, or
copy it to an ignored local file. Database paths resolve relative to the
configuration file. The example stores data under ignored `data/market/`.
It covers SPY from Tiingo's earliest reported date, January 29, 1993.
Omitting `end_date` requests through yesterday UTC to avoid a partial US day.

Run from the project directory in your Python environment:

```powershell
python -m alpha_lab.market_data.cli --config data/tiingo.example.json sync
python -m alpha_lab.market_data.cli --config data/tiingo.example.json status
python -m alpha_lab.market_data.cli --config data/tiingo.example.json export --output data/market/snapshots/daily-v1.csv
```

After reinstalling the editable package, `alpha-lab-tiingo` is an equivalent
entry point. `sync --env-file <path>` loads another local secret file.
Standard `HTTPS_PROXY` environment settings are respected.

Updates re-fetch a configurable overlap (seven calendar days by default) and
upsert `(ticker, date)` without duplicates. New dividends/splits or changed
overlapping bars trigger a full refresh of the cached coverage so historical
adjustments stay consistent. Requested coverage is recorded separately from
the first trading date, handling holidays and IPOs. Use `sync --full-refresh`
periodically to pick up older corrections outside the overlap. Requests are
paced at 30/minute by default with bounded retries for transient failures and
HTTP 429. Set the rate for your Tiingo account limits; no quota is assumed.

Each ticker update commits bars, metadata, response JSON hashes and an audit
record in one SQLite transaction. Failed updates preserve cached bars and
can be retried. A refresh missing previously cached dates is rejected for
review rather than silently deleting history. Sync exits nonzero if any
ticker fails. `status` and `export` operate entirely offline.

CSV snapshots include a SHA-256/provenance manifest and cannot overwrite an
existing snapshot. Use a new filename after updates. Research code can read
them with pandas, or query the local store directly:

```python
from alpha_lab.market_data import MarketStore
import pandas as pd

with MarketStore("data/market/tiingo.sqlite3") as store:
    bars = pd.DataFrame(store.prices(["SPY", "QQQ"], start="2020-01-01"))
closes = bars.pivot(index="date", columns="ticker", values="adj_close")
returns = closes.pct_change(fill_method=None)
```

Tiingo's adjusted history is the current back-adjusted series, not a
point-in-time dataset. `fetched_at` records retrieval time, not historical
availability. Preserve snapshots for reproducibility. No exchange calendar,
survivorship-free universe, forward filling, or trading signals are inferred.
Choose historical universes and chronological train/test splits explicitly
when building experiments.

Provider references: [EOD fields](https://www.tiingo.com/documentation/end-of-day),
[header authentication](https://www.tiingo.com/documentation/general/connecting),
and [adjustment refresh guidance](https://www.tiingo.com/kb/article/the-fastest-method-to-ingest-tiingo-end-of-day-stock-api-data/).
Tests run without network access: `python -m pytest tests/test_market_data.py`.

### Direct Sharadar full-history bundle

The direct `sharadar.com` connector uses `SHARADAR_API_KEY` in your ignored
`.env`, independently of Tiingo and LLM credentials. It supports all 14 bundle
tables: descriptions, tickers, S&P 500 membership, actions, events, metrics,
fundamentals, daily metrics, stock/fund prices, insider transactions, and the
three institutional holdings tables. This is the direct API, with canonical
table names such as `stocks` and `fundamentals`; legacy aliases `SEP`, `SF1`,
`SFP`, `SF2`, `SF3`, `SF3A`, and `SF3B` also work in `--table`.

Install the updated requirements (including DuckDB) in your environment.
Copy `data/sharadar.example.json` to `data/market/sharadar.json` and set its
`root` to `sharadar`. Paths resolve relative to the config file. The example
requests the full history of every table, including delisted securities.

```powershell
python -m alpha_lab.market_data.sharadar_cli --config data/market/sharadar.json probe
python -m alpha_lab.market_data.sharadar_cli --config data/market/sharadar.json sync
python -m alpha_lab.market_data.sharadar_cli --config data/market/sharadar.json status
python -m alpha_lab.market_data.sharadar_cli --config data/market/sharadar.json export --table funds --ticker SPY --output data/market/snapshots/sharadar-spy-v1.csv
python -m alpha_lab.market_data.sharadar_cli --config data/market/sharadar.json export --table fundamentals --ticker AAPL --dimension ARQ --output data/market/snapshots/aapl-arq-v1.csv
```

The installed entry point is `alpha-lab-sharadar`. Use `sync --table stocks`
to update one table, or `sync --force` to rebuild snapshots even if source
metadata is unchanged. Export accepts repeated `--ticker`, `--start-date`,
`--end-date`, and comma-separated `--columns`. Status and export work offline.
No research campaign or LLM is started by any of these commands.

Sync downloads provider ZIPs with byte-range resumption, reissues expiring
signed links, verifies ZIP CRCs, casts against the public schema and checks
primary-key uniqueness. It publishes a table's new Parquet version only after
validation succeeds. Failed tables keep their previous snapshot; rerunning
sync resumes downloads and skips unchanged completed tables. An OS file lock
rejects concurrent writers to the same storage root. Daily updates download
a new full file when the source changes; this initial connector does not merge
incremental query results. Budget disk for ZIP archives, staged CSVs and retained
Parquet versions. The default DuckDB memory limit is 4GB with four threads;
large validations may spill into the local temporary directory.

Original columns and schema types are retained. Each snapshot records source
modification time, row/date coverage, schemas, and archive/Parquet SHA-256 hashes.
Exports include a source-version manifest and refuse to overwrite files.
Research code can query a pinned Parquet snapshot without loading the universe:

```python
import duckdb
from alpha_lab.market_data.sharadar_store import SharadarStore

store = SharadarStore("data/market/sharadar")
snapshot = store.latest("funds")
with duckdb.connect() as db:
    spy = db.execute(
        "SELECT date, closeadj, volume FROM read_parquet(?) WHERE ticker = ? ORDER BY date",
        [snapshot["path"], "SPY"],
    ).df()
```

Sharadar stock/fund OHLC and volume are split-adjusted; `closeadj` also adjusts
for dividends and spinoffs, while `closeunadj` is the raw close. They retain
their provider names instead of being relabeled as Tiingo raw bars. `daily`
market cap is reported in **USD millions**, whereas `fundamentals` market cap
uses USD. Consult the `descriptions` table for each field's units. Field
descriptions and ticker metadata retain legacy table labels such as `SF1`, `SEP`, `SFP`,
and `DAILY`; use those labels when joining the metadata to canonical API tables.
Fundamentals ARQ/ARY/ART are as reported; MRQ/MRY/MRT are restated. Preserve dimensions and
report dates, and explicitly establish availability dates before backtesting.
S&P 500 data provides membership events/snapshots, without official weights.
Institutional holdings are quarterly 13F data, rather than daily ETF holdings.
Bulk files are the provider's current history, not an archive of what was
available on every historical day. Keep pinned versions for reproducibility.

Provider documentation: [bundle](https://sharadar.com/bundle),
[authentication](https://sharadar.com/docs/auth),
[stock prices](https://sharadar.com/docs/stocks),
[daily metrics](https://sharadar.com/docs/daily),
[fundamentals](https://sharadar.com/docs/fundamentals),
[S&P 500 membership](https://sharadar.com/docs/sp500), and
[institutional holdings](https://sharadar.com/docs/holdings).
Offline tests: `python -m pytest tests/test_sharadar_market_data.py`.

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env — add your API key:
#   OPENAI_API_KEY=sk-...        (for OpenAI)
#   or ANTHROPIC_API_KEY       (for Claude)
source .env
```

### 3. Run the demo

A working demo is included using synthetic exchange rate data. Pick the config that matches your hardware:

**CPU only** (no GPU needed — runs tree-based models like XGBoost, LightGBM, Random Forest):
```bash
python data/generate_synthetic.py
python run.py --config data/demo_exchange_cpu_only.json --workspace ./workspace_demo
```

**Single GPU** (one GPU, experiments run sequentially):
```bash
python data/generate_synthetic.py
python run.py --config data/demo_exchange_single_gpu.json --workspace ./workspace_demo
```

**Multi-GPU** (4x GPUs in parallel — the full experience):
```bash
python data/generate_synthetic.py
python run.py --config data/demo_exchange_config.json --workspace ./workspace_demo
```

All three run the full pipeline autonomously:
- **Phase 0**: Customize the domain adapter for your data (~5 min)
- **Phase 1**: Explore the dataset, write scripts, generate plots, build a report (~30-90 min)
- **Phase 2**: Build an evaluation framework with tests (~20-60 min)
- **Phase 3**: Run experiments with different ML models (~1-3 hours depending on hardware)

### 4. Watch it work — Web Dashboard

**In a separate terminal:**

```bash
# First time only — build the frontend
cd frontend && npm install && npm run build && cd ..

# Start the dashboard
python serve.py --workspace ./workspace_demo --port 8000
# Open http://localhost:8000
```

The dashboard is a passive viewer — it streams live events from the running pipeline. You can start it before, during, or after a run.

---

## Reproducing Paper Results

The paper evaluates AlphaLab on four datasets. Each has a config and download instructions.

### Download datasets

```bash
pip install huggingface_hub  # needed for Traffic and LLM Speedrun

# Download all datasets
python data/download_datasets.py

# Or download individually
python data/download_datasets.py --dataset traffic
python data/download_datasets.py --dataset llm_speedrun
python data/download_datasets.py --dataset kernelbench
python data/download_datasets.py --dataset exchange  # already included
```

### Run experiments

Each dataset has a pre-configured JSON. Adjust `gpu_ids` and `worker_count` for your hardware.

| Dataset | Config | Domain | Metric | GPUs |
|---------|--------|--------|--------|------|
| Exchange Rates | `data/demo_exchange_config.json` | time_series | Sharpe ratio (maximize) | 4x |
| Traffic | `data/paper_traffic_gpt.json` | time_series | RMSE (minimize) | 4x |
| LLM Speedrun | `data/paper_llm_speedrun_gpt.json` | llm_speedrun | val BPB (minimize) | 4x |
| CUDA KernelBench | `data/paper_cuda_kernelbench_gpt.json` | cuda_kernel | GFLOPS (maximize) | 4x |

```bash
# Exchange rates (synthetic, included)
python run.py --config data/demo_exchange_config.json --workspace ./results/exchange

# Traffic forecasting
python run.py --config data/paper_traffic_gpt.json --workspace ./results/traffic

# LLM pretraining speedrun
python run.py --config data/paper_llm_speedrun_gpt.json --workspace ./results/llm_speedrun

# CUDA kernel optimization
python run.py --config data/paper_cuda_kernelbench_gpt.json --workspace ./results/cuda
```

### Using Claude instead of GPT

Change `provider` and `model` in any config:

```json
{
  "provider": "anthropic",
  "model": "us.anthropic.claude-sonnet-4-20250514",
  "reasoning_effort": "medium"
}
```

Requires ANTHROPIC_API_KEY in your `.env`.

---

## Running Your Own Experiments

### Use an agent to set up your config

This codebase was largely built with AI coding agents, and while it aims to be plug-and-play, it may need some tweaking for your setup. We recommend using an AI coding agent:

1. Open [Claude Code](https://claude.ai/code) (or your preferred agent) in this repo
2. Prompt it to **explore the repository and become an expert in it**
3. Tell it: **where your data is**, **what you want to do**, **which model**, **what GPUs you have**
4. Ask it to **write a config and run script** for you

### Config format

```json
{
  "data_path": "path/to/your/data.csv",
  "description": "What this dataset is...",
  "target": "What to predict/optimize...",
  "provider": "openai",
  "model": "gpt-5.2",
  "domain": "",
  "pipeline": {
    "phases": ["phase1", "phase2", "phase3"],
    "phase3": {
      "executor": "local",
      "max_experiments": 50,
      "gpu_ids": [0, 1, 2, 3],
      "max_per_gpu": 1,
      "worker_count": 4,
      "time_limit_seconds": 21600,
      "python_executable": ""
    }
  }
}
```

### Hardware configuration

The key settings to adjust for your hardware:

| Setting | What it does | Examples |
|---------|-------------|---------|
| `gpu_ids` | Which GPUs to use. Set `[]` for CPU-only mode | `[0]` (one GPU), `[0,1,2,3]` (four), `[]` (CPU only) |
| `max_per_gpu` | Experiments per GPU (1 = exclusive, 2+ = packing) | `1` for large models, `2` if models fit |
| `worker_count` | Number of LLM worker agents running in parallel | Usually matches number of GPUs |
| `max_experiments` | Total experiments before stopping | `10` for quick test, `50` for full run |
| `time_limit_seconds` | Kill experiments exceeding this | `3600` (1h), `21600` (6h) |
| `cpu_enabled` | Run tree-based models on CPU in parallel with GPU | `true` (default) |
| `cpu_max_parallel` | Max concurrent CPU experiments | `4` |
| `python_executable` | Python binary for experiment subprocesses | Empty = uses `ALPHALAB_PYTHON` env var or `sys.executable` |

**Domain options:** `""` for time series (default), `"cuda_kernel"`, `"nanogpt"`, `"llm_speedrun"`, or any free-text description to generate a custom adapter from scratch.

**Provider options:** `"openai"` (GPT-5.2) or `"anthropic"` (Claude). Set the corresponding API key in `.env`.

---

## Citation

If you use AlphaLab in your research, please cite:

```bibtex
@article{alphalab2026,
  title={AlphaLab: Autonomous Research Agent for End-to-End Machine Learning Experimentation},
  author={Morgan Stanley Machine Learning Research},
  year={2026},
  note={Technical Report}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
