# TypeSafe Local Endpoint Integration Test Report

## Executive Summary

- **Test Status:** ✅ **PASSED**
- **Endpoint Base URL:** `http://localhost:8080`
- **Target Endpoint:** `http://localhost:8080/v1/systemone`
- **API Key Used:** `test-mock-api-key`
- **Execution Timestamp:** `2026-09-22 02:43:02 UTC`

## Base URL Resolution & Header Verification

The local HTTP mock server successfully captured the outgoing request sent by `TypeSafeClassifier`.

### Captured HTTP Request Details
- **HTTP Method:** `POST`
- **Request Path:** `/v1/systemone`
- **HTTP Headers:**
```json
{
  "Host": "localhost:8080",
  "Accept": "*/*",
  "Accept-Encoding": "gzip, deflate",
  "Connection": "keep-alive",
  "Authorization": "Bearer test-mock-api-key",
  "Content-Type": "application/json",
  "User-Agent": "langchain-typesafe/0.0.1a3",
  "Content-Length": "697"
}
```

### Request JSON Body Payload
```json
{
  "state": {
    "customer_id": "cust_12345",
    "message": "Payment system returned 500 error on checkout page. Need immediate assistance!"
  },
  "model": "jev-latest",
  "questions": {
    "urgent": {
      "type": "noul",
      "instructions": "Is this issue urgent?",
      "criteria": {
        "true": "Checkout or payment processing is blocked.",
        "false": "General feature request or questions."
      }
    },
    "department": {
      "type": "choice",
      "criteria": {
        "billing": "Invoice or subscription issues.",
        "technical": "Software bugs or system downtime.",
        "sales": "Pricing or contract inquiries."
      },
      "instructions": "Which team should handle this ticket?"
    },
    "frustration": {
      "type": "score",
      "criteria": [
        "Calm",
        "Frustrated",
        "Furious"
      ],
      "instructions": "Rate customer frustration level."
    }
  }
}
```

## Response Serialization & Deserialization Logs

### Parsed Response Metadata
- **Model:** `jev-latest`
- **Request ID:** `req_mock_server_8080`
- **Input Tokens:** `150`
- **Output Tokens:** `50`

### Question 1: Noul (Boolean Probability)
```python
Answer Object: NoulAnswer(type='noul', noul=0.88)
Probability (noul): 0.88
```

### Question 2: Choice (Categorical Selection)
```python
Answer Object: ChoiceAnswer(type='choice', choice='technical', probabilities={'billing': 0.1, 'technical': 0.8, 'sales': 0.1}, confidence=0.85)
Selected Choice: technical
Probabilities: {'billing': 0.1, 'technical': 0.8, 'sales': 0.1}
Confidence Metric: 0.85
```

### Question 3: Score (Ordinal Rating Spectrum)
```python
Answer Object: ScoreAnswer(type='score', score=1.6, legend={0: 'Calm', 1: 'Frustrated', 2: 'Furious'}, probabilities={0: 0.05, 1: 0.3, 2: 0.65}, confidence=0.82)
Expected Score: 1.6
Legend Rubric: {0: 'Calm', 1: 'Frustrated', 2: 'Furious'}
Probabilities: {0: 0.05, 1: 0.3, 2: 0.65}
Confidence Metric: 0.82
```

## Errors & Unexpected Behaviors

No errors or unexpected behaviors were encountered during the test execution.
