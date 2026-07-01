import os
from dotenv import load_dotenv
load_dotenv()

import requests
import json
from datetime import datetime
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

# ── Helper ────────────────────────────────────────────────────────────────────
def call_chat(question: str) -> dict:
    res = requests.post(
        "http://127.0.0.1:8000/chat",
        json={"query": question, "conversation_history": []}
    )
    return res.json()

# ── 5 Tricky Test Cases ───────────────────────────────────────────────────────

# Case 1: Vague question — system might give generic answer instead of medical answer
# Trick: "it" is ambiguous, system might not know what "it" refers to
q1 = "How does it spread?"
r1 = call_chat(q1)
print(f"Case 1 | Agent: {r1['agent']} | Response: {r1['response'][:80]}...")

# Case 2: Question outside RAG knowledge base
# Trick: system might hallucinate or give irrelevant filler instead of saying it doesn't know
q2 = "What are the latest FDA approved drugs for glioblastoma in 2024?"
r2 = call_chat(q2)
print(f"Case 2 | Agent: {r2['agent']} | Response: {r2['response'][:80]}...")

# Case 3: Multi-part question — system might only answer one part
# Trick: system answers brain tumor part but ignores COVID comparison
q3 = "How is brain tumor diagnosis different from COVID-19 diagnosis using deep learning?"
r3 = call_chat(q3)
print(f"Case 3 | Agent: {r3['agent']} | Response: {r3['response'][:80]}...")

# Case 4: Misleading phrasing — sounds like a greeting but is actually a medical question
# Trick: conversation agent might handle it as small talk instead of routing to RAG
q4 = "Hey, can you tell me what happens when brain cells grow uncontrollably?"
r4 = call_chat(q4)
print(f"Case 4 | Agent: {r4['agent']} | Response: {r4['response'][:80]}...")

# Case 5: Negation question — system might answer the positive version instead
# Trick: system might ignore the "not" and give standard brain tumor info
q5 = "What does deep learning NOT do well when detecting brain tumors?"
r5 = call_chat(q5)
print(f"Case 5 | Agent: {r5['agent']} | Response: {r5['response'][:80]}...")

# ── Build Test Cases ──────────────────────────────────────────────────────────
test_cases = [
    LLMTestCase(input=q1, actual_output=r1["response"]),
    LLMTestCase(input=q2, actual_output=r2["response"]),
    LLMTestCase(input=q3, actual_output=r3["response"]),
    LLMTestCase(input=q4, actual_output=r4["response"]),
    LLMTestCase(input=q5, actual_output=r5["response"]),
]

# ── Metric ────────────────────────────────────────────────────────────────────
metric = AnswerRelevancyMetric(
    threshold=0.7,
    model="gpt-4o-mini",
    include_reason=True
)

# ── Run Evaluation ────────────────────────────────────────────────────────────
results = evaluate(
    test_cases=test_cases,
    metrics=[metric]
)

# ── Save Results to JSON ──────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
output_path = f"results/answer_relevancy_{timestamp}.json"

save_data = []
for i, test_result in enumerate(results.test_results):
    save_data.append({
        "test_case": i + 1,
        "input": test_result.input,
        "actual_output": test_result.actual_output,
        "metrics": [
            {
                "name": m.name,
                "score": m.score,
                "passed": m.passed,
                "reason": m.reason
            }
            for m in test_result.metrics_data
        ]
    })

with open(output_path, "w") as f:
    json.dump(save_data, f, indent=2)

print(f"\n✅ Results saved to: {output_path}")
