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
- Structured `Report` output (JSON and Markdown)
- Async collectors and LLM calls
- Works offline with `MockLLMProvider` and local models via `OllamaExplainer`

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
pip install -e ".[ranking]"       # Embedding ranker
pip install -e ".[all]"           # everything
```

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
    .add_detector(ZScoreDetector(threshold=1.0))
    .add_ranker(BM25Ranker(top_k=5))
    .add_explainer(LLMExplainer(MockLLMProvider()))
)

report = pipeline.run()
print(report.executive_summary)
print(report.to_markdown())
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

GitHub Releases works without a token for light use. For higher rate limits:

```bash
export GITHUB_TOKEN="ghp_..."
```

```python
GitHubReleasesCollector(token="ghp_...", repos=["python/cpython"])
```

Stack Overflow works without a key for light use. For a higher daily quota
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

## Roadmap

Core MVP is in place. Next focus: **integrations and ecosystem**.

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
- [ ] HTML / PDF reports
- [ ] CLI (`whytrend analyze ...`)
- [ ] Plugin registry (entry points)

### v0.6 — Ranking quality
- [ ] Hybrid ranker (BM25 + embeddings merged with RRF)
- [ ] Post-ranker evidence filter (relevance thresholds, stronger rerank, explicit verdicts like `evidence_insufficient` / `correlation_only` / `likely_cause`)

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