"""Unit tests for custom local endpoint verification with TypeSafeClassifier."""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr

from langchain_typesafe import (
    Choice,
    ChoiceAnswer,
    ClassifierRequest,
    Noul,
    NoulAnswer,
    NoulCriteria,
    Score,
    ScoreAnswer,
    TypeSafeClassifier,
)


def test_local_endpoint_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that environment variables configure base URL and API key."""
    monkeypatch.setenv("TYPESAFE_API_BASE", "http://localhost:8080")
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-test-key")

    classifier = TypeSafeClassifier()

    assert classifier.base_url == "http://localhost:8080"
    assert classifier._endpoint == "http://localhost:8080/v1/systemone"
    assert isinstance(classifier.api_key, SecretStr)
    assert classifier.api_key.get_secret_value() == "fake-test-key"


def test_local_endpoint_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify request structure and response deserialization for local endpoint."""
    monkeypatch.setenv("TYPESAFE_API_BASE", "http://localhost:8080")
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-test-key")

    captured_requests: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        payload = json.loads(request.content)
        captured_requests.append(
            {
                "url": str(request.url),
                "method": request.method,
                "headers": dict(request.headers),
                "payload": payload,
            }
        )

        mock_body = {
            "model": "jev-latest",
            "answers": {
                "urgent": {"type": "noul", "noul": 0.95},
                "department": {
                    "type": "choice",
                    "choice": "technical",
                    "probabilities": {"billing": 0.05, "technical": 0.95},
                    "confidence": 0.92,
                },
                "frustration": {
                    "type": "score",
                    "score": 1.8,
                    "legend": {"0": "calm", "1": "annoyed", "2": "angry"},
                    "probabilities": {"0": 0.05, "1": 0.1, "2": 0.85},
                    "confidence": 0.9,
                },
            },
            "usage": {"input_tokens": 100, "output_tokens": 30},
        }
        return httpx2.Response(
            200,
            json=mock_body,
            headers={"x-typesafe-request-id": "req_mock_123"},
        )

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    classifier = TypeSafeClassifier(client=client)

    request: ClassifierRequest = {
        "state": "Customer system is experiencing Stripe connection errors.",
        "questions": {
            "urgent": Noul(
                instructions="Is this request urgent?",
                criteria=NoulCriteria(true="Critical failure", false="Normal inquiry"),
            ),
            "department": Choice(
                instructions="Which team should handle this?",
                criteria={"billing": "Payment issues", "technical": "Product errors"},
            ),
            "frustration": Score(
                instructions="How frustrated is the customer?",
                criteria=["calm", "annoyed", "angry"],
            ),
        },
    }

    response = classifier.invoke(request)

    # 1. Assert request capture
    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req["url"] == "http://localhost:8080/v1/systemone"
    assert req["method"] == "POST"
    assert req["headers"]["authorization"] == "Bearer fake-test-key"
    assert req["headers"]["content-type"] == "application/json"
    assert "langchain-typesafe" in req["headers"]["user-agent"]

    # 2. Assert payload contents
    assert req["payload"]["model"] == "jev-latest"
    assert (
        req["payload"]["state"]
        == "Customer system is experiencing Stripe connection errors."
    )
    assert "urgent" in req["payload"]["questions"]
    assert "department" in req["payload"]["questions"]
    assert "frustration" in req["payload"]["questions"]

    # 3. Assert deserialized response answers
    assert response.request_id == "req_mock_123"
    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 30

    urgent_ans = response.nouls["urgent"]
    assert isinstance(urgent_ans, NoulAnswer)
    assert urgent_ans.noul == 0.95

    dept_ans = response.choices["department"]
    assert isinstance(dept_ans, ChoiceAnswer)
    assert dept_ans.choice == "technical"
    assert dept_ans.probabilities == {"billing": 0.05, "technical": 0.95}
    assert dept_ans.confidence == 0.92

    frust_ans = response.scores["frustration"]
    assert isinstance(frust_ans, ScoreAnswer)
    assert frust_ans.score == 1.8
    assert frust_ans.legend == {0: "calm", 1: "annoyed", 2: "angry"}
    assert frust_ans.probabilities == {0: 0.05, 1: 0.1, 2: 0.85}
    assert frust_ans.confidence == 0.9

    client.close()
