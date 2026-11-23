# WhyTrend

[CI](https://github.com/AlexProvatorov/WhyTrend/actions/workflows/ci.yml)
[License](LICENSE)
[Python](https://www.python.org/downloads/)

**Open Source Framework for Explainable Time Series Analysis**

WhyTrend detects anomalies, change points, and trend shifts in time series — then automatically explains *why* they happened using external context and LLMs.

If this project is useful to you, consider giving it a **star** on GitHub — it helps others discover the project. Forks and pull requests are welcome.

## Features

- Fluent `Pipeline` API for source → detection → collection → ranking → explanation
- Pluggable sources, detectors, collectors, rankers, and LLM providers
- Plugin registry via Python entry points (`whytrend.collectors`, …)
- Structured `Report` output (JSON, Markdown, HTML, and PDF)
- Async collectors and LLM calls
- Works offline with `MockLLMProvider` and local models via `OllamaExplainer`
- `whytrend` command-line interface over the same plugin registry as `Pipeline`

## Installation

```bash
git clone https://github.com/AlexProvatorov/WhyTrend.git
cd WhyTrend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Optional extras

```bash
pip install -e ".[openai]"      # OpenAI API
pip install -e ".[anthropic]"   # Anthropic Claude
pip install -e ".[deepseek]"    # DeepSeek (OpenAI-compatible client)
pip install -e ".[azure]"       # Azure OpenAI
pip install -e ".[openrouter]"  # OpenRouter
pip install -e ".[gemini]"      # Gemini marker (uses core httpx; no extra package)
pip install -e ".[trends]"      # Google Trends
pip install -e ".[prophet]"       # Prophet detector
pip install -e ".[ruptures]"      # change-point detector
pip install -e ".[river]"         # online / streaming detector
pip install -e ".[ranking]"       # Embedding and hybrid rankers
pip install -e ".[pdf]"           # PDF reports (xhtml2pdf)
pip install -e ".[reports]"       # report extras (currently PDF)
pip install -e ".[all]"           # everything
```

## Quickstart CLI

After install, analysis is one command. The mock explainer does not call the network when collectors are omitted:

```bash
whytrend analyze \
  --csv tests/fixtures/python_interest.csv \
  --keyword Python \
  --detector zscore \
  --explainer mock \
  --out report.md
```

Omit `--out` to print the report. `--format` is `markdown`, `json`, `html`, or `pdf`. PDF writes bytes and requires `--out` (`pip install 'whytrend[pdf]'`). `--parquet` reads a Parquet file instead of CSV. `--trends KEYWORD` loads Google Trends (`pip install 'whytrend[trends]'`).

`whytrend --help` and `whytrend analyze --help` list the plugins installed in this environment. A plugin that needs constructor arguments stays in that list and is created when every missing value is passed with `--set`. `llm` takes `--provider` instead. Collectors call external sources, so the offline path is the command above with no `--collectors`. `hn` is an alias of `hacker_news`:

```bash
whytrend analyze \
  --csv tests/fixtures/python_interest.csv \
  --keyword Python \
  --collectors hn,wikipedia \
  --ranker bm25 \
  --format json \
  --out report.json
```

Repeat `--set` as `detector.<param>`, `explainer.<param>`, `ranker.<param>`, `collector.<name>.<param>`, `source.<param>`, or `provider.<param>`. `none` clears an optional value (`ranker.top_k=none`). Lists are comma-separated (`collector.github_releases.repos=owner/name`) and dicts are JSON (`detector.prophet_kwargs={"seasonality_mode":"multiplicative"}`).

Cloud explainers read the same environment variables as the Python API below (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and the rest). Ollama needs a local server instead of a key.

RSS, an LLM provider, and an HTML report:

```bash
whytrend analyze \
  --csv tests/fixtures/python_interest.csv \
  --keyword Python \
  --collectors rss \
  --set collector.rss.feed_urls=https://blog.python.org/feeds/posts/default \
  --explainer llm \
  --provider mock \
  --format html \
  --out report.html
```

Azure OpenAI needs its deployment name: `--explainer azure_openai --set explainer.deployment=my-deploy` (`AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT`). In-memory dataframes and `client` objects stay in the Python API.

## Quickstart (CSV, no API keys)

```python
from whytrend import (
    BM25Ranker,
    CSVSource,
    LLMExplainer,
    MockLLMProvider,
    Pipeline,
    ZScoreDetector,
)

pipeline = (
    Pipeline(window_days=3)
    .add_source(CSVSource("tests/fixtures/python_interest.csv", keyword="Python"))
    .add_detector(ZScoreDetector())
    .add_ranker(BM25Ranker(top_k=5))
    .add_explainer(LLMExplainer(MockLLMProvider()))
)

report = pipeline.run()
print(report.executive_summary)
print(report.to_markdown())
print(report.to_html())
```

Export a shareable HTML page or PDF attachment (`whytrend[pdf]` for PDF):

```python
from pathlib import Path

Path("report.html").write_text(report.to_html(), encoding="utf-8")
Path("report.pdf").write_bytes(report.to_pdf())  # pip install 'whytrend[pdf]'
```

Run the full demo:

```bash
pip install -e ".[dev]"
python examples/mvp_demo.py
```

## Production-style example

```python
from whytrend import (
    GoogleTrends,
    OpenAIExplainer,
    Pipeline,
    ProphetDetector,
)
from whytrend.collectors import (
    GitHubReleasesCollector,
    GoogleNewsCollector,
    HackerNewsCollector,
    RSSFeedCollector,
    RedditCollector,
    StackOverflowCollector,
    WikipediaCollector,
)
from whytrend.rankers import BM25Ranker

pipeline = (
    Pipeline()
    .add_source(GoogleTrends("Python"))
    .add_detector(ProphetDetector())
    .add_collector(GoogleNewsCollector())
    .add_collector(HackerNewsCollector())
    .add_collector(RedditCollector())  # REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET
    .add_collector(GitHubReleasesCollector())  # optional GITHUB_TOKEN
    .add_collector(StackOverflowCollector())  # optional STACKEXCHANGE_KEY
    .add_collector(
        RSSFeedCollector(
            [
                "https://blog.python.org/feeds/posts/default",
                "https://pyfound.blogspot.com/feeds/posts/default",
            ]
        )
    )
    .add_collector(WikipediaCollector())
    .add_ranker(BM25Ranker())
    .add_explainer(OpenAIExplainer())  # OPENAI_API_KEY env var or api_key="..."
)

report = pipeline.run()
print(report.executive_summary)
```

Use `OllamaExplainer(model="llama3.2")` for a local LLM instead of OpenAI.

Other cloud providers (same explainability pipeline, different backends):

```python
from whytrend import (
    AnthropicExplainer,
    AzureOpenAIExplainer,
    DeepSeekExplainer,
    GeminiExplainer,
    OpenRouterExplainer,
)

# ANTHROPIC_API_KEY — default model: claude-sonnet-4-20250514
AnthropicExplainer()

# GEMINI_API_KEY or GOOGLE_API_KEY — default model: gemini-2.0-flash
GeminiExplainer()

# DEEPSEEK_API_KEY — default model: deepseek-chat
DeepSeekExplainer()

# AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT (or pass api_key= / azure_endpoint=)
AzureOpenAIExplainer(deployment="gpt-4o-mini")

# OPENROUTER_API_KEY — default model: openai/gpt-4o-mini
# Optional site_url= / app_title= set OpenRouter ranking headers
OpenRouterExplainer(model="anthropic/claude-sonnet-4")
```

Pass `api_key="..."` to any of these constructors if you prefer not to use env vars. Swap them into `.add_explainer(...)` the same way as `OpenAIExplainer`.

`OpenAIExplainer(base_url="https://openrouter.ai/api/v1", api_key=...)` also works for OpenRouter; the dedicated classes improve DX (env vars, Azure deployment/endpoint, OpenRouter headers).

Reddit credentials (create at [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)):

```bash
export REDDIT_CLIENT_ID="..."
export REDDIT_CLIENT_SECRET="..."
```

Or pass them explicitly:

```python
RedditCollector(client_id="...", client_secret="...", subreddits=["Python", "MachineLearning"])
```

GitHub Releases searches starred repositories whose name contains the keyword, then keeps releases inside the event window. Pass `repos` to pin the list. It works without a token for light use. For higher rate limits:

```bash
export GITHUB_TOKEN="ghp_..."
```

```python
GitHubReleasesCollector(token="ghp_...", repos=["python/cpython"])
```

Stack Overflow searches questions tagged with the keyword that also use it in the title. It works without a key for light use. For a higher daily quota
(register at [https://stackapps.com/](https://stackapps.com/)):

```bash
export STACKEXCHANGE_KEY="..."
```

```python
StackOverflowCollector(api_key="...", site="stackoverflow")
```

Custom RSS/Atom feeds (keyword + time-window filtered):

```python
RSSFeedCollector(
    [
        "https://blog.python.org/feeds/posts/default",
        "https://hnrss.org/frontpage",
    ]
)
```

## Architecture

```
Source → Detector → Event Builder → Collectors → Ranker → Explainer → Report
```

### Choosing a detector


| Detector             | Best for                                             | Notes                                           |
| -------------------- | ---------------------------------------------------- | ----------------------------------------------- |
| **ZScoreDetector**   | Sudden spikes/drops vs the series mean               | Fast, no extra deps                             |
| **ProphetDetector**  | Points outside a forecast band (trend + seasonality) | Needs `whytrend[prophet]`                       |
| **RupturesDetector** | Structural breaks / regime changes                   | Needs `whytrend[ruptures]`; emits `changepoint` |
| **RiverDetector**    | Online / streaming point-by-point scores             | Needs `whytrend[river]`; batch + `update()`     |


```python
from whytrend import RiverDetector, RupturesDetector

RupturesDetector(algorithm="pelt", model="rbf", penalty=10.0)
RiverDetector(model="gaussian", threshold=0.95, min_points=10)
```

### Choosing a ranker

| Ranker              | Best for                             | Notes                                                         |
| ------------------- | ------------------------------------ | ------------------------------------------------------------- |
| **BM25Ranker**      | Exact keyword overlap                | No extra dependencies                                         |
| **EmbeddingRanker** | Paraphrases of the event keyword     | Needs `whytrend[ranking]`                                     |
| **HybridRanker**    | Exact terms and paraphrases together | BM25 and embeddings merged with RRF; needs `whytrend[ranking]` |


```python
from whytrend import BM25Ranker, EmbeddingRanker, HybridRanker

BM25Ranker(top_k=5)
EmbeddingRanker(top_k=5)  # pip install 'whytrend[ranking]'
HybridRanker(top_k=5)  # pip install 'whytrend[ranking]'
```

`whytrend analyze --ranker bm25`, `--ranker embedding`, and `--ranker hybrid` each fill the pipeline's single ranker slot. BM25 scores are a fraction of a perfect match for the query, not a stretch that makes the best item `1.0`. Embedding relevance is cosine clamped to `[0, 1]`, so an unrelated page scores near `0` rather than `0.5`. Hybrid orders with reciprocal rank fusion and keeps the mean of the two scores, so a weak mention stays below `cause_threshold`.

Each ranker takes `time_weight` (default `0.0`). At `0`, scores stay on the text match. Above that, an item published at `event.timestamp` keeps its score, and one at `window_start` or `window_end` keeps less:

```text
adjusted = relevance * ((1 - time_weight) + time_weight * time_factor)
```

`time_factor` is 1 on the event timestamp and 0 at either edge of the window. The result stays in `[0, 1]`. A missing `published_at` is not penalised, so an undated page stays eligible and the evidence filter decides whether it is a cause. An item outside the window scores lower as `time_weight` grows; the ranker does not drop it. `HybridRanker` passes `time_weight` into BM25 and embeddings before reciprocal rank fusion, and the fusion formula itself stays rank positions only.

```python
BM25Ranker(top_k=5, time_weight=0.5)
HybridRanker(top_k=5, time_weight=0.5)  # pip install 'whytrend[ranking]'
```

```bash
whytrend analyze \
  --csv tests/fixtures/python_interest.csv \
  --keyword Python \
  --ranker hybrid \
  --set ranker.time_weight=0.5
```

### Evidence filter

Ranking only reorders evidence. `EvidenceFilter` runs after the ranker and before the explainer. It drops items below `keep_threshold` (default `0.2`), sorts what remains by `relevance_score`, and sets `Explanation.verdict`:

| Verdict | When | Confidence ceiling |
| --- | --- | --- |
| `likely_cause` | At least one kept item is at or above `cause_threshold` (default `0.6`) | `1.0` (`likely_cause_cap`) |
| `correlation_only` | Items remain, and every one of them is below `cause_threshold` | `0.6` (`correlation_only_cap`) |
| `evidence_insufficient` | Nothing remains | `0.3` (`evidence_insufficient_cap`) |

`likely_cause` is the only verdict whose summary may say "likely cause", including inside a longer phrase such as "The likely cause is". `evidence_insufficient` has no cause. After the explainer returns, the filter clamps `Explanation.confidence` and every `Cause.score` to that verdict's ceiling. A likely cause can still be `1.0`. A pipeline with no filter leaves the explainer's confidence unchanged. When a number is lowered, the explanation metadata includes `confidence_capped: true`. The filter is optional and works with `bm25`, `embedding`, and `hybrid`.

```python
from whytrend import EvidenceFilter

EvidenceFilter(keep_threshold=0.2, cause_threshold=0.6)
```

```bash
whytrend analyze \
  --csv tests/fixtures/python_interest.csv \
  --keyword Python \
  --ranker bm25 \
  --filter evidence \
  --set filter.keep_threshold=0.3 \
  --set filter.cause_threshold=0.7 \
  --set filter.correlation_only_cap=0.5
```

## Writing a WhyTrend plugin

Third-party packages can register components without forking WhyTrend. Direct imports (`from whytrend import ZScoreDetector`) remain the supported API.

Declare an entry point in your package:

```toml
[project.entry-points."whytrend.collectors"]
my_news = "mypkg.collectors:MyNewsCollector"
```

Groups:

- `whytrend.sources`
- `whytrend.detectors`
- `whytrend.collectors`
- `whytrend.rankers`
- `whytrend.filters`
- `whytrend.explainers`

Discover and load plugins at runtime:

```python
from whytrend import list_plugins, load_plugin

detectors = list_plugins("whytrend.detectors")
Ruptures = load_plugin("whytrend.detectors", "ruptures")
MyNews = load_plugin("whytrend.collectors", "my_news")
```

`load_plugin` raises `UnknownPluginError` if the name is missing. Built-in classes are also registered in WhyTrend's own `pyproject.toml`, so names like `zscore` and `hacker_news` resolve the same way.

## Roadmap

Core MVP is in place. Next focus: **ranking quality**, then explanations you can check, then analyst decisions.

### v0.2 — More collectors
- [x] Google News
- [x] Reddit
- [x] GitHub Releases
- [x] RSS / Stack Overflow

### v0.3 — More detectors
- [x] Ruptures (change-point)
- [x] River / streaming detectors

### v0.4 — More LLM providers
- [x] Anthropic, Gemini, DeepSeek
- [x] Azure OpenAI, OpenRouter

### v0.5 — Reports and DX
- [x] HTML / PDF reports
- [x] CLI (`whytrend analyze ...`)
- [x] Plugin registry (entry points)

### v0.6 — Ranking quality
- [x] Hybrid ranker (BM25 + embeddings merged with RRF)
- [x] Post-ranker evidence filter (relevance thresholds, stronger rerank, explicit verdicts like `evidence_insufficient` / `correlation_only` / `likely_cause`)
- [x] Time proximity in the relevance score (`published_at` inside the event window, not only lexical match)
- [x] Confidence cap: explainer confidence cannot exceed the evidence-filter verdict

### v0.7 — Explanations you can check
- [ ] Known-event eval set (expected verdicts for releases, news spikes, and empty background)
- [ ] Ground every cause URL in collected evidence; drop invented links before the report
- [ ] Cluster adjacent detections into one event instead of one LLM call per point
- [ ] Surface collector failures in report metadata instead of dropping them silently
- [ ] HTML report chart of the series, the event window, and the detection

### v0.8 — Analyst decisions
- [ ] Movement shape: one-off spike, level shift, seasonality, or sustained growth (`seasonality_break` is declared but unused)
- [ ] Compare several series: a move shared by all of them, or isolated to one keyword
- [ ] Example private-context collector (changelog or event CSV), not only public web sources
- [ ] Stable `Report` contract and leave Pre-Alpha once the known-event set passes

Track progress in [GitHub Issues](https://github.com/AlexProvatorov/WhyTrend/issues).

## Development

```bash
make install   # install with detected tool (uv / poetry / pip)
make check     # ruff + format check + mypy + pytest
```

Supports the three common workflows:


| Tool                          | Install                    | Run checks               |
| ----------------------------- | -------------------------- | ------------------------ |
| **uv** (default if installed) | `make install`             | `make check`             |
| **poetry**                    | `make install TOOL=poetry` | `make check TOOL=poetry` |
| **pip** / venv                | `make install TOOL=pip`    | `make check TOOL=pip`    |


Auto-detect order: `uv` → `poetry` → `pip`. Override anytime with `TOOL=...`.

Useful targets:

```bash
make lint          # ruff check
make format        # ruff format + autofix
make format-check  # ruff format --check
make typecheck     # mypy
make test          # pytest
make help          # list all targets
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).
Security reports: [SECURITY.md](SECURITY.md).

## Author

**Alexander Provatorov** — [GitHub @AlexProvatorov](https://github.com/AlexProvatorov)

## License

Licensed under the [Apache License, Version 2.0](LICENSE).