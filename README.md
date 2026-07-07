# WhyTrend

**Open Source Framework for Explainable Time Series Analysis**

WhyTrend detects anomalies, change points, and trend shifts in time series — then automatically explains *why* they happened using external context and LLMs.

```python
from whytrend import Pipeline, GoogleTrends, ProphetDetector, OpenAIExplainer

pipeline = (
    Pipeline()
    .add_source(GoogleTrends("Python"))
    .add_detector(ProphetDetector())
    .add_explainer(OpenAIExplainer())
)

report = pipeline.run()
print(report.executive_summary)
# "Interest in 'Python' spiked on March 15 due to the release of Python 3.13."
```

## Author

**Alexander Provatorov** — [GitHub @AlexProvatorov](https://github.com/AlexProvatorov)

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
