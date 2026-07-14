# WhyTrend

[![CI](https://github.com/AlexProvatorov/WhyTrend/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexProvatorov/WhyTrend/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

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
pip install -e ".[trends]"      # Google Trends
pip install -e ".[prophet]"       # Prophet detector
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
    GoogleNewsCollector,
    HackerNewsCollector,
    RedditCollector,
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
    .add_collector(WikipediaCollector())
    .add_ranker(BM25Ranker())
    .add_explainer(OpenAIExplainer())  # OPENAI_API_KEY env var or api_key="..."
)

report = pipeline.run()
print(report.executive_summary)
```

Use `OllamaExplainer(model="llama3.2")` for a local LLM instead of OpenAI.

Reddit credentials (create at https://www.reddit.com/prefs/apps):

```bash
export REDDIT_CLIENT_ID="..."
export REDDIT_CLIENT_SECRET="..."
```

Or pass them explicitly:

```python
RedditCollector(client_id="...", client_secret="...", subreddits=["Python", "MachineLearning"])
```

## Architecture

```
Source → Detector → Event Builder → Collectors → Ranker → Explainer → Report
```

## Roadmap

Core MVP is in place. Next focus: **integrations and ecosystem**.

### v0.2 — More collectors
- [x] Google News
- [x] Reddit
- [ ] GitHub Releases
- [ ] RSS / Stack Overflow

### v0.3 — More detectors
- [ ] Ruptures (change-point)
- [ ] River / streaming detectors

### v0.4 — More LLM providers
- [ ] Anthropic, Gemini, DeepSeek
- [ ] Azure OpenAI, OpenRouter

### v0.5 — Reports and DX
- [ ] HTML / PDF reports
- [ ] CLI (`whytrend analyze ...`)
- [ ] Plugin registry (entry points)

Track progress in [GitHub Issues](https://github.com/AlexProvatorov/WhyTrend/issues).

## Development

```bash
pip install -e ".[dev]"
ruff check src tests
mypy src/whytrend
pytest
```

## Author

**Alexander Provatorov** — [GitHub @AlexProvatorov](https://github.com/AlexProvatorov)

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
