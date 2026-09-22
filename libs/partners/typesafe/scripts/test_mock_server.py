"""Integration test script with a built-in HTTP server for TypeSafeClassifier."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# 1. Environment Setup
os.environ["TYPESAFE_API_BASE"] = "http://localhost:8080"
os.environ["TYPESAFE_API_KEY"] = "test-mock-api-key"

from typing import Any

from langchain_typesafe import (
    Choice,
    ClassifierRequest,
    ClassifierResponse,
    Noul,
    NoulCriteria,
    Score,
    TypeSafeClassifier,
)

# Shared dictionary to store captured server state
CAPTURED_SERVER_DATA: dict[str, Any] = {
    "url_path": None,
    "method": None,
    "headers": {},
    "body_json": None,
    "received": False,
}


class MockTypeSafeHandler(BaseHTTPRequestHandler):
    """Custom HTTP handler that intercepts and responds to TypeSafe API calls."""

    def do_POST(self) -> None:  # noqa: N802
        """Handle POST request to mock endpoint."""
        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length)

        try:
            body_json = json.loads(body_bytes.decode("utf-8"))
        except Exception:  # noqa: BLE001
            body_json = None

        header_dict = dict(self.headers.items())

        CAPTURED_SERVER_DATA["url_path"] = self.path
        CAPTURED_SERVER_DATA["method"] = self.command
        CAPTURED_SERVER_DATA["headers"] = header_dict
        CAPTURED_SERVER_DATA["body_json"] = body_json
        CAPTURED_SERVER_DATA["received"] = True

        # Validate path and headers
        if self.path != "/v1/systemone":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "Not Found"}')
            return

        auth_header = self.headers.get("Authorization", "")
        if auth_header != "Bearer test-mock-api-key":
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"error": "Unauthorized"}')
            return

        # Success response payload
        response_payload = {
            "model": "jev-latest",
            "answers": {
                "urgent": {
                    "type": "noul",
                    "noul": 0.88,
                },
                "department": {
                    "type": "choice",
                    "choice": "technical",
                    "probabilities": {
                        "billing": 0.10,
                        "technical": 0.80,
                        "sales": 0.10,
                    },
                    "confidence": 0.85,
                },
                "frustration": {
                    "type": "score",
                    "score": 1.6,
                    "legend": {
                        "0": "Calm",
                        "1": "Frustrated",
                        "2": "Furious",
                    },
                    "probabilities": {
                        "0": 0.05,
                        "1": 0.30,
                        "2": 0.65,
                    },
                    "confidence": 0.82,
                },
            },
            "usage": {
                "input_tokens": 150,
                "output_tokens": 50,
            },
        }

        response_bytes = json.dumps(response_payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("x-typesafe-request-id", "req_mock_server_8080")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def log_message(self, format_str: str, *args: object) -> None:
        """Suppress default HTTP server stderr logging."""


def run_mock_server(server: HTTPServer) -> None:
    """Server runner thread loop."""
    server.serve_forever()


def main() -> None:
    """Start local mock server, invoke classifier, and write Markdown report."""
    print("Starting local HTTP Mock Server on port 8080...")
    server_address = ("localhost", 8080)
    httpd = HTTPServer(server_address, MockTypeSafeHandler)

    server_thread = threading.Thread(target=run_mock_server, args=(httpd,), daemon=True)
    server_thread.start()
    time.sleep(0.1)  # Allow server time to bind

    test_passed = False
    error_message: str | None = None
    response: ClassifierResponse | None = None

    try:
        print("Instantiating TypeSafeClassifier with environment variables...")
        classifier = TypeSafeClassifier()

        request_payload: ClassifierRequest = {
            "state": {
                "customer_id": "cust_12345",
                "message": (
                    "Payment system returned 500 error on checkout page. "
                    "Need immediate assistance!"
                ),
            },
            "questions": {
                "urgent": Noul(
                    instructions="Is this issue urgent?",
                    criteria=NoulCriteria(
                        true="Checkout or payment processing is blocked.",
                        false="General feature request or questions.",
                    ),
                ),
                "department": Choice(
                    instructions="Which team should handle this ticket?",
                    criteria={
                        "billing": "Invoice or subscription issues.",
                        "technical": "Software bugs or system downtime.",
                        "sales": "Pricing or contract inquiries.",
                    },
                ),
                "frustration": Score(
                    instructions="Rate customer frustration level.",
                    criteria=["Calm", "Frustrated", "Furious"],
                ),
            },
        }

        print("Invoking classifier over HTTP to http://localhost:8080/v1/systemone...")
        response = classifier.invoke(request_payload)
        test_passed = True
        print("Invocation completed successfully!")

    except Exception as exc:  # noqa: BLE001
        error_message = str(exc)
        print(f"Test failed with error: {exc}", file=sys.stderr)
    finally:
        httpd.shutdown()
        httpd.server_close()

    # Generate test_results.md
    print("Generating Markdown report 'test_results.md'...")
    report_lines: list[str] = [
        "# TypeSafe Local Endpoint Integration Test Report",
        "",
        "## Executive Summary",
        "",
        f"- **Test Status:** {'✅ **PASSED**' if test_passed else '❌ **FAILED**'}",
        f"- **Endpoint Base URL:** `{os.environ.get('TYPESAFE_API_BASE')}`",
        "- **Target Endpoint:** `http://localhost:8080/v1/systemone`",
        f"- **API Key Used:** `{os.environ.get('TYPESAFE_API_KEY')}`",
        (
            "- **Execution Timestamp:**"
            f" `{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}`"
        ),
        "",
        "## Base URL Resolution & Header Verification",
        "",
        (
            "The local HTTP mock server successfully captured the outgoing"
            " request sent by `TypeSafeClassifier`."
        ),
        "",
        "### Captured HTTP Request Details",
        f"- **HTTP Method:** `{CAPTURED_SERVER_DATA.get('method')}`",
        f"- **Request Path:** `{CAPTURED_SERVER_DATA.get('url_path')}`",
        "- **HTTP Headers:**",
        "```json",
        json.dumps(CAPTURED_SERVER_DATA.get("headers", {}), indent=2),
        "```",
        "",
        "### Request JSON Body Payload",
        "```json",
        json.dumps(CAPTURED_SERVER_DATA.get("body_json", {}), indent=2),
        "```",
        "",
        "## Response Serialization & Deserialization Logs",
        "",
    ]

    if response is not None:
        report_lines.extend(
            [
                "### Parsed Response Metadata",
                f"- **Model:** `{response.model}`",
                f"- **Request ID:** `{response.request_id}`",
                f"- **Input Tokens:** `{response.usage.input_tokens}`",
                f"- **Output Tokens:** `{response.usage.output_tokens}`",
                "",
                "### Question 1: Noul (Boolean Probability)",
                "```python",
                f"Answer Object: {response.nouls.get('urgent')!r}",
                f"Probability (noul): {response.nouls['urgent'].noul}",
                "```",
                "",
                "### Question 2: Choice (Categorical Selection)",
                "```python",
                f"Answer Object: {response.choices.get('department')!r}",
                f"Selected Choice: {response.choices['department'].choice}",
                (f"Probabilities: {response.choices['department'].probabilities}"),
                (f"Confidence Metric: {response.choices['department'].confidence}"),
                "```",
                "",
                "### Question 3: Score (Ordinal Rating Spectrum)",
                "```python",
                f"Answer Object: {response.scores.get('frustration')!r}",
                f"Expected Score: {response.scores['frustration'].score}",
                f"Legend Rubric: {response.scores['frustration'].legend}",
                (f"Probabilities: {response.scores['frustration'].probabilities}"),
                (f"Confidence Metric: {response.scores['frustration'].confidence}"),
                "```",
                "",
            ]
        )

    report_lines.extend(
        [
            "## Errors & Unexpected Behaviors",
            "",
            error_message
            or (
                "No errors or unexpected behaviors were encountered during"
                " the test execution."
            ),
            "",
        ]
    )

    report_content = "\n".join(report_lines)
    Path("test_results.md").write_text(report_content, encoding="utf-8")

    print("Markdown report written successfully to 'test_results.md'.")


if __name__ == "__main__":
    main()
