<div align="center">

![logo](https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant/blob/main/assets/logo_rounded.png)

<h1 align="center"><strong>Multi-Agent Medical Assistant</strong></h1>
<h6 align="center">AI-powered multi-agent system for medical diagnosis and assistance — with a DeepEval-based evaluation suite</h6>

![Python - Version](https://img.shields.io/badge/PYTHON-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)
![LangGraph - Version](https://img.shields.io/badge/LangGraph-0.3+-teal?style=for-the-badge&logo=langgraph)
![FastAPI - Version](https://img.shields.io/badge/FastAPI-0.115+-teal?style=for-the-badge&logo=fastapi)
[![Generic badge](https://img.shields.io/badge/License-Apache-<COLOR>.svg?style=for-the-badge)](https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant/blob/main/LICENSE)

</div>

---

## Table of Contents

- [What Is This Application?](#what-is-this-application)
- [Quick Start — Run the Application](#quick-start--run-the-application)
- [Evaluation Overview](#evaluation-overview)
- [Evaluation Setup](#evaluation-setup)
- [Evaluation Scripts](#evaluation-scripts)
  - [Answer Relevancy](#1-answer-relevancy-test_answer_relevancypy)
  - [Tool / Agent Correctness](#2-tool--agent-correctness-test_tool_correctnesspy)
  - [Safety Metrics](#3-safety-metrics-safety_metricspy)
  - [Conversation Completeness](#4-conversation-completeness-conversation_completness_metricpy)
- [Running All Evaluations](#running-all-evaluations)
- [Understanding Results](#understanding-results)
- [Multi-Agent Evaluation Concepts](#multi-agent-evaluation-concepts)
- [Project Structure](#project-structure)
- [Further Reading](#further-reading)
- [License & Contact](#license--contact)

---

## What Is This Application?

The **Multi-Agent Medical Assistant** is a FastAPI-based chatbot that routes user queries through specialized AI agents orchestrated with **LangGraph**. It is designed for medical information retrieval, image analysis, and safe conversational assistance.

| Agent | Purpose |
|-------|---------|
| `CONVERSATION_AGENT` | Greetings and general chat |
| `RAG_AGENT` | Answers from ingested medical literature (Qdrant vector DB) |
| `WEB_SEARCH_PROCESSOR_AGENT` | Recent/time-sensitive medical information (Tavily / PubMed) |
| `CHEST_XRAY_AGENT` | Chest X-ray classification |
| `SKIN_LESION_AGENT` | Skin lesion segmentation |
| `BRAIN_TUMOR_AGENT` | Brain MRI analysis (stub / TBD) |
| `INPUT_GUARDRAILS` / `OUTPUT_GUARDRAILS` | Block unsafe or irrelevant input/output |
| `HUMAN_VALIDATION` | Human-in-the-loop review for computer-vision diagnoses |

**Key behaviors relevant to evaluation:**

- **Confidence-based routing** — low RAG confidence can trigger a handoff to web search (`RAG_AGENT, WEB_SEARCH_PROCESSOR_AGENT`).
- **Composite agent names** — the API `agent` field may return multiple agents (e.g. `SKIN_LESION_AGENT, HUMAN_VALIDATION`).
- **Black-box API** — evaluations call `POST /chat` (text) and `POST /upload` (image + text) on a running server.

Architecture diagram: [`assets/final-medical-assistant-flowchart-code.mermaid`](assets/final-medical-assistant-flowchart-code.mermaid)

Agent details: [`agents/README.md`](agents/README.md)

---

## Quick Start — Run the Application

Evaluations require the application to be **running locally** on `http://127.0.0.1:8000`.

### 1. Clone and install

```bash
git clone https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant.git
cd Multi-Agent-Medical-Assistant

# Create and activate a virtual environment (Python 3.11+)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root with your API keys (LLM, embeddings, Tavily, Hugging Face, etc.). See the [Docker setup section in the previous docs](https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant) or copy the template from `config.py` requirements.

Minimum keys needed for evaluation:

- Azure OpenAI / OpenAI (LLM + embeddings)
- `TAVILY_API_KEY` (for web search routing tests)
- `HUGGINGFACE_TOKEN` (for RAG reranker)

### 3. Ingest RAG data (optional but recommended)

```bash
python ingest_rag_data.py --dir ./data/raw
```

### 4. Start the server

```bash
python app.py
```

The app will be available at **http://localhost:8000**.

> **Note:** The first run may download models (CV agents, reranker, etc.) and can be slow. Retry once downloads complete.

### Docker alternative

```bash
docker build -t medical-assistant .
docker run -d --name medical-assistant-app -p 8000:8000 --env-file .env medical-assistant
```

---

## Evaluation Overview

This repository includes a **black-box evaluation suite** built with [DeepEval](https://github.com/confident-ai/deepeval). Scripts send HTTP requests to the running FastAPI app, collect responses (and agent routing metadata), and score them with LLM-as-judge metrics.

| Script | Metric | What it measures |
|--------|--------|------------------|
| `test_scripts/test_answer_relevancy.py` | `AnswerRelevancyMetric` | Is the response relevant to the user's question? |
| `test_scripts/test_tool_correctness.py` | `ToolCorrectnessMetric` | Did the correct agent handle the request? |
| `test_scripts/safety_metrics.py` | `BiasMetric`, `ToxicityMetric`, `GEval` | Bias, toxicity, and medical accuracy |
| `test_scripts/conversation_completness_metric.py` | `ConversationCompletenessMetric` | Multi-turn intent coverage |

**Judge model:** `gpt-4o-mini` (configured in each script).

**Tracing:** `app.py` uses DeepEval `@observe()` on `/chat` and `/upload` for runtime tracing.

---

## Evaluation Setup

Install DeepEval separately (not included in `requirements.txt`):

```bash
pip install deepeval
```

Ensure:

1. The app is running (`python app.py`).
2. Your `.env` has valid API keys for both the app **and** the DeepEval judge model.
3. For image-based tests, run tool-correctness from the `test_scripts/` directory (sample images live there).

```bash
cd test_scripts
python test_tool_correctness.py
```

---

## Evaluation Scripts

### 1. Answer Relevancy (`test_answer_relevancy.py`)

**Purpose:** Measure whether responses actually address the user's question — especially for ambiguous or tricky prompts where routing alone is not enough.

**Metric:** `AnswerRelevancyMetric` (threshold: **0.7**)

**Test cases (5):**

| # | Question | Why it's tricky |
|---|----------|-----------------|
| 1 | "How does it spread?" | Ambiguous referent ("it") |
| 2 | "What are the latest FDA approved drugs for glioblastoma in 2024?" | May need web search; risk of hallucination |
| 3 | "How is brain tumor diagnosis different from COVID-19 diagnosis using deep learning?" | Multi-part question |
| 4 | "Hey, can you tell me what happens when brain cells grow uncontrollably?" | Greeting-framed medical question |
| 5 | "What does deep learning NOT do well when detecting brain tumors?" | Negation — system may ignore "NOT" |

**How to run:**

```bash
python test_scripts/test_answer_relevancy.py
```

**Output:** Console scores + JSON saved to `results/answer_relevancy_<timestamp>.json`

---

### 2. Tool / Agent Correctness (`test_tool_correctness.py`)

**Purpose:** Verify the orchestrator routes each query to the **correct specialized agent**. In multi-agent systems, routing correctness is as important as answer quality.

**Metric:** `ToolCorrectnessMetric` (threshold: **0.7**)

Agents are treated as tools via DeepEval's `ToolCall` — the API `agent` field is compared against `expected_tools`.

**Test cases (4):**

| Input | Endpoint | Expected agent(s) |
|-------|----------|-------------------|
| "What are the latest COVID treatments?" | `/chat` | `WEB_SEARCH_PROCESSOR_AGENT` |
| "What does the literature say about brain tumor?" | `/chat` | `RAG_AGENT` |
| "Hello, how are you?" | `/chat` | `CONVERSATION_AGENT` |
| Skin lesion image + "Analyze" | `/upload` | `SKIN_LESION_AGENT`, `HUMAN_VALIDATION` |

**How to run:**

```bash
cd test_scripts
python test_tool_correctness.py
```

**Sample images:** `test_scripts/sample_images/skin_lesion_images/`

> **Tip:** A "failure" may mean correct multi-agent behavior (e.g. `RAG_AGENT, WEB_SEARCH_PROCESSOR_AGENT` handoff) if expectations are too strict. Review the actual `agent` field before changing test cases.

---

### 3. Safety Metrics (`safety_metrics.py`)

**Purpose:** Evaluate bias, toxicity, and factual medical accuracy — critical for healthcare applications.

**Metrics:**

| Metric | Threshold | Lower/higher is better |
|--------|-----------|------------------------|
| `BiasMetric` | 0.5 | Lower score = less bias |
| `ToxicityMetric` | 0.5 | Lower score = less toxic |
| `GEval` (Medical Accuracy) | default | Higher score = more accurate/safe |

**Test categories:**

**Bias (5 questions)** — gender stereotypes, age-based treatment bias, socioeconomic assumptions, race/gender in clinical vignettes.

**Toxicity (5 questions)** — hostile prompts, self-harm, requests for dangerous procedures, dismissive partner advice.

**Medical Accuracy (4 questions)** — juice fasting cures, alkaline water claims, pharma conspiracy, misinterpreted study conclusions.

**How to run:**

```bash
python test_scripts/safety_metrics.py
```

**Output:** Console scores with pass/fail and reasons per metric block.

---

### 4. Conversation Completeness (`conversation_completness_metric.py`)

**Purpose:** Evaluate whether a **multi-turn conversation** collectively addresses the user's intents — not just the final turn.

**Metric:** `ConversationCompletenessMetric` (threshold: **0.7**)

**Episodes (3):**

| Episode | Flow |
|---------|------|
| **1** | Brain tumor basics → latest treatments → immunotherapy → COVID brain impact → COVID treatments |
| **2** | Greeting → COVID X-ray deep learning literature → models used → vs PCR testing |
| **3** | Current COVID variant → danger level → CNN for X-rays → skin cancer AI crossover |

**How to run:**

```bash
python test_scripts/conversation_completness_metric.py
```

**Output:** Console scores; may sync to the [Confident AI](https://app.confident-ai.com) dashboard if configured.

> **Limitation:** Each turn calls `/chat` with an empty `conversation_history`, so the app does not maintain session context between turns. The metric evaluates the collected transcript as a whole, not true stateful multi-turn behavior.

---

## Running All Evaluations

**Terminal 1 — start the app:**

```bash
python app.py
```

**Terminal 2 — run evaluations (from project root):**

```bash
python test_scripts/test_answer_relevancy.py
python test_scripts/safety_metrics.py
python test_scripts/conversation_completness_metric.py

cd test_scripts
python test_tool_correctness.py
```

---

## Understanding Results

### Pass / fail thresholds

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Answer Relevancy | ≥ 0.7 | Response addresses the question |
| Tool Correctness | ≥ 0.7 | Correct agent(s) were invoked |
| Conversation Completeness | ≥ 0.7 | Multi-turn intents were covered |
| Bias / Toxicity | ≤ 0.5 | Response is not biased or toxic |
| GEval Medical Accuracy | varies | Factually sound, no dangerous advice |

### Common failure patterns

| Symptom | Possible cause |
|---------|----------------|
| Wrong agent in tool correctness | Router ambiguity, confidence handoff, or outdated expected agent |
| Low answer relevancy | Vague prompt handling, partial multi-part answers |
| High bias/toxicity score | Unsafe response content (metric failure) |
| Guardrail block | `INPUT_GUARDRAILS` in agent field — routing "failure" may be correct safety behavior |
| Composite agent name | e.g. `RAG_AGENT, WEB_SEARCH_PROCESSOR_AGENT` — update expectations if handoff is intended |

### Saved artifacts

- `results/answer_relevancy_*.json` — timestamped relevancy results (gitignored)
- Console output — all other scripts print scores and reasons inline

---

## Multi-Agent Evaluation Concepts

When evaluating this system, keep these ideas in mind:

1. **Two axes of correctness** — output quality (relevancy, safety) and routing correctness (tool correctness) are independent; both must pass.
2. **Agents as tools** — DeepEval's `ToolCorrectnessMetric` maps each API `agent` name to a `ToolCall`.
3. **Black-box testing** — scripts treat the system as a deployed service via HTTP; no internal code imports required.
4. **LLM-as-judge** — DeepEval uses `gpt-4o-mini` to score subjective criteria (relevancy, completeness, GEval).
5. **Medical domain risk** — safety metrics (bias, toxicity, medical accuracy) are essential alongside functional metrics.

---

## Project Structure

```
Multi-Agent-Medical-Assistant/
├── app.py                              # FastAPI server (evaluation target)
├── agents/
│   └── agent_decision.py               # LangGraph orchestration & routing
├── test_scripts/
│   ├── test_answer_relevancy.py        # Answer relevancy evaluation
│   ├── test_tool_correctness.py        # Agent routing evaluation
│   ├── safety_metrics.py               # Bias, toxicity, medical accuracy
│   ├── conversation_completness_metric.py  # Multi-turn completeness
│   └── sample_images/                  # Test images for /upload tests
├── results/                            # Generated eval JSON (gitignored)
├── ingest_rag_data.py                  # RAG document ingestion
└── config.py                           # App configuration
```

---

## Further Reading

- Agent workflow details: [`agents/README.md`](agents/README.md)
- Architecture flowchart: [`assets/final-medical-assistant-flowchart-code.mermaid`](assets/final-medical-assistant-flowchart-code.mermaid)
- DeepEval docs: https://docs.confident-ai.com

---

## License & Contact

This project is licensed under the **Apache-2.0 License**. See [LICENSE](LICENSE).

**Souvik Majumder**

- LinkedIn: [linkedin.com/in/souvikmajumder26](https://www.linkedin.com/in/souvikmajumder26)
- GitHub: [github.com/souvikmajumder26](https://github.com/souvikmajumder26)
