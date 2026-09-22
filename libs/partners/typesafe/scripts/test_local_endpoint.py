"""Test script to verify TypeSafeClassifier against a custom local endpoint."""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Callable

# 1. Environment Setup: Configure environment variables programmatically
os.environ["TYPESAFE_API_BASE"] = "http://localhost:8080"
os.environ["TYPESAFE_API_KEY"] = "fake-test-key"

import httpx2

# 2. Import All Supported Primitives from langchain_typesafe
from langchain_typesafe import (
    Choice,
    ChoiceAnswer,
    ClassifierRequest,
    ClassifierResponse,
    Noul,
    NoulAnswer,
    NoulCriteria,
    Score,
    ScoreAnswer,
    TypeSafeClassifier,
)

# Configure logging to display debug info if enabled
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_mock_interceptor() -> Callable[[httpx2.Request], httpx2.Response]:
    """Create an httpx2.MockTransport handler to intercept requests."""

    def mock_handler(request: httpx2.Request) -> httpx2.Response:
        print("\n==================================================")
        print(" [DEBUG] OUTGOING HTTP REQUEST INTERCEPTED")
        print("==================================================")
        print(f"Target URL:  {request.url}")
        print(f"HTTP Method: {request.method}")
        print("\nRequest Headers:")
        for header, value in request.headers.items():
            print(f"  {header}: {value}")

        print("\nRequest JSON Body:")
        try:
            body_json = json.loads(request.content.decode("utf-8"))
            print(json.dumps(body_json, indent=2))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  Failed to parse body as JSON: {e}")
        print("==================================================\n")

        # Return a valid Jev API mock response containing all 3 question types
        mock_response_data = {
            "model": "jev-latest",
            "answers": {
                "urgent": {
                    "type": "noul",
                    "noul": 0.92,
                },
                "department": {
                    "type": "choice",
                    "choice": "technical",
                    "probabilities": {
                        "billing": 0.05,
                        "technical": 0.88,
                        "sales": 0.07,
                    },
                    "confidence": 0.91,
                },
                "frustration": {
                    "type": "score",
                    "score": 1.85,
                    "legend": {
                        "0": "Calm and neutral",
                        "1": "Concerned but civil",
                        "2": "Very angry or frustrated",
                    },
                    "probabilities": {
                        "0": 0.02,
                        "1": 0.11,
                        "2": 0.87,
                    },
                    "confidence": 0.89,
                },
            },
            "usage": {
                "input_tokens": 128,
                "output_tokens": 42,
            },
        }

        return httpx2.Response(
            200,
            json=mock_response_data,
            headers={"x-typesafe-request-id": "req_local_test_12345"},
        )

    return mock_handler


def main() -> None:
    """Run end-to-end verification of TypeSafeClassifier against local endpoint."""
    print("Initializing TypeSafe Local Endpoint Test...")
    print(f"TYPESAFE_API_BASE: {os.environ.get('TYPESAFE_API_BASE')}")
    print(f"TYPESAFE_API_KEY:  {os.environ.get('TYPESAFE_API_KEY')}")

    # Create HTTP client with mock transport to intercept outgoing calls
    mock_handler = create_mock_interceptor()
    client = httpx2.Client(transport=httpx2.MockTransport(mock_handler))

    # Instantiate TypeSafeClassifier reading from env vars + injected client
    classifier = TypeSafeClassifier(client=client)

    # Verify constructor resolved base_url correctly
    print(f"Resolved Base URL: {classifier.base_url}")
    print(f"Resolved Endpoint: {classifier._endpoint}")

    # 3. Comprehensive Test Invocation:
    # Build a single state payload containing all three question types in parallel
    request_payload: ClassifierRequest = {
        "state": {
            "customer_id": "cust_9988",
            "issue_description": (
                "Stripe payment connection has been failing for 3 days. "
                "I cannot accept customer payments! Fix this immediately!"
            ),
            "account_tier": "enterprise",
        },
        "questions": {
            # Noul question (boolean probability, e.g., checking urgency)
            "urgent": Noul(
                instructions="Does this message require an urgent response?",
                criteria=NoulCriteria(
                    true="Payment or critical infrastructure is down.",
                    false="General inquiry or minor issue.",
                ),
            ),
            # Choice question (multiple-choice selection across categories)
            "department": Choice(
                instructions="Which team should handle this support ticket?",
                criteria={
                    "billing": ("Payment processing, invoice, or subscription issues."),
                    "technical": "API failures, integrations, or product bugs.",
                    "sales": "Pricing, plan upgrades, or purchasing questions.",
                },
            ),
            # Score question (continuous scale rating against ordered levels)
            "frustration": Score(
                instructions=(
                    "How frustrated does the customer appear on a 3-level scale?"
                ),
                criteria=[
                    "Calm and neutral",
                    "Concerned but civil",
                    "Very angry or frustrated",
                ],
            ),
        },
    }

    # Execute invocation with try-except for robust debugging
    try:
        print("\nInvoking TypeSafeClassifier...")
        response: ClassifierResponse = classifier.invoke(request_payload)
        print("Classifier invocation successful!\n")

        # 4. Validation and Debugging:
        # Print structured response fields for each question type
        print("==================================================")
        print(" [RESULT] STRUCTURED RESPONSE VALIDATION")
        print("==================================================")
        print(f"Model Name: {response.model}")
        print(f"Request ID: {response.request_id}")
        print(
            f"Token Usage: Input={response.usage.input_tokens}, "
            f"Output={response.usage.output_tokens}"
        )
        print("--------------------------------------------------")

        # Validate Noul question response
        urgent_answer: NoulAnswer = response.nouls["urgent"]
        print("\n1. NOUL QUESTION (Urgency):")
        print(f"   Answer Type:  {urgent_answer.type}")
        print(f"   Probability:  {urgent_answer.noul}")
        assert isinstance(urgent_answer, NoulAnswer)
        assert 0.0 <= urgent_answer.noul <= 1.0

        # Validate Choice question response
        dept_answer: ChoiceAnswer = response.choices["department"]
        print("\n2. CHOICE QUESTION (Department):")
        print(f"   Answer Type:   {dept_answer.type}")
        print(f"   Selected Choice:{dept_answer.choice}")
        print(f"   Probabilities: {dept_answer.probabilities}")
        print(f"   Confidence:    {dept_answer.confidence}")
        assert isinstance(dept_answer, ChoiceAnswer)
        assert dept_answer.choice in dept_answer.probabilities
        assert 0.0 <= dept_answer.confidence <= 1.0

        # Validate Score question response
        frust_answer: ScoreAnswer = response.scores["frustration"]
        print("\n3. SCORE QUESTION (Frustration):")
        print(f"   Answer Type:   {frust_answer.type}")
        print(f"   Score Rating:  {frust_answer.score}")
        print(f"   Legend Rubric: {frust_answer.legend}")
        print(f"   Probabilities: {frust_answer.probabilities}")
        print(f"   Confidence:    {frust_answer.confidence}")
        assert isinstance(frust_answer, ScoreAnswer)
        assert 0.0 <= frust_answer.score <= 2.0
        assert 0.0 <= frust_answer.confidence <= 1.0

        print("==================================================")
        print(" ALL CHECKS AND DESERIALIZATIONS PASSED SUCCESSFULLY")
        print("==================================================\n")

    except Exception as exc:
        print(f"\n[ERROR] Classifier invocation failed: {exc}", file=sys.stderr)
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()
