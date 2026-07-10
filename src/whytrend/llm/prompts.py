"""Prompt templates for LLM explanation."""

from __future__ import annotations

import json

from whytrend.core.models import Event, Evidence

SYSTEM_PROMPT = """You explain anomalies in time series data.

Rules:
- Use only the event details and evidence provided by the user.
- Do not invent sources, URLs, or facts.
- If evidence is weak or missing, say that clearly and lower confidence.
- Respond with valid JSON only.
"""


def build_user_prompt(event: Event, evidences: list[Evidence]) -> str:
    evidence_lines = []
    for index, evidence in enumerate(evidences, start=1):
        evidence_lines.append(
            "\n".join(
                [
                    f"{index}. source={evidence.source_name}",
                    f"   title={evidence.title}",
                    f"   url={evidence.url}",
                    f"   snippet={evidence.snippet or 'n/a'}",
                    f"   relevance_score={evidence.relevance_score:.2f}",
                ]
            )
        )

    evidence_block = "\n\n".join(evidence_lines) if evidence_lines else "No external evidence was collected."

    payload = {
        "event": {
            "keyword": event.keyword,
            "anomaly_type": event.anomaly_type.value,
            "timestamp": event.timestamp.isoformat(),
            "value": event.value,
            "detection_score": event.detection_score,
            "window_start": event.window_start.isoformat(),
            "window_end": event.window_end.isoformat(),
        },
        "evidence": evidence_block,
        "response_schema": {
            "summary": "One concise executive sentence explaining the likely cause.",
            "confidence": "Float between 0 and 1.",
            "causes": [
                {
                    "source": "Evidence source name",
                    "score": "Float between 0 and 1",
                    "url": "Supporting URL from evidence",
                    "title": "Short title",
                    "summary": "Why this evidence matters",
                }
            ],
        },
    }
    return json.dumps(payload, indent=2)
