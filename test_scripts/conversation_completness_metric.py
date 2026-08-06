import os
from dotenv import load_dotenv
load_dotenv()

import requests
from deepeval import evaluate
from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.metrics import ConversationCompletenessMetric


def call_chat(question: str) -> str:
    res = requests.post(
        "http://127.0.0.1:8000/chat",
        json={"query": question, "conversation_history": []}
    )
    return res.json()["response"]


metric = ConversationCompletenessMetric(threshold=0.7, model="gpt-4o-mini", include_reason=True)


# ─────────────────────────────────────────────────────────────────────────────
# Episode 1 — Brain tumor (RAG + Web Search)
# ─────────────────────────────────────────────────────────────────────────────
print("Building Episode 1...")

q1 = "What is a brain tumor?"
q2 = "What are the latest treatments for brain tumors in 2024?"
q3 = "Is immunotherapy effective for brain tumors?"
q4 = "Can COVID affect brain health?"
q5 = "What are the newest COVID treatments available?"

r1 = call_chat(q1)
r2 = call_chat(q2)
r3 = call_chat(q3)
r4 = call_chat(q4)
r5 = call_chat(q5)

episode_1 = ConversationalTestCase(
    turns=[
        Turn(role="user", content=q1),
        Turn(role="assistant", content=r1),
        Turn(role="user", content=q2),
        Turn(role="assistant", content=r2),
        Turn(role="user", content=q3),
        Turn(role="assistant", content=r3),
        Turn(role="user", content=q4),
        Turn(role="assistant", content=r4),
        Turn(role="user", content=q5),
        Turn(role="assistant", content=r5),
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# Episode 2 — Conversation agent + RAG (COVID chest X-ray literature)
# ─────────────────────────────────────────────────────────────────────────────
print("Building Episode 2...")

q1 = "Hi, how are you today?"
q2 = "What does the literature say about detecting COVID from chest X-rays using deep learning?"
q3 = "What models were used for that?"
q4 = "Does that work better than PCR testing?"

r1 = call_chat(q1)
r2 = call_chat(q2)
r3 = call_chat(q3)
r4 = call_chat(q4)

episode_2 = ConversationalTestCase(
    turns=[
        Turn(role="user", content=q1),
        Turn(role="assistant", content=r1),
        Turn(role="user", content=q2),
        Turn(role="assistant", content=r2),
        Turn(role="user", content=q3),
        Turn(role="assistant", content=r3),
        Turn(role="user", content=q4),
        Turn(role="assistant", content=r4),
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# Episode 3 — Web search + cross-domain trick
# ─────────────────────────────────────────────────────────────────────────────
print("Building Episode 3...")

q1 = "What's the current COVID variant spreading right now?"
q2 = "Is it more dangerous than previous variants?"
q3 = "How does CNN architecture help detect this from X-rays?"
q4 = "Can the same AI model also work for skin cancer detection?"

r1 = call_chat(q1)
r2 = call_chat(q2)
r3 = call_chat(q3)
r4 = call_chat(q4)

episode_3 = ConversationalTestCase(
    turns=[
        Turn(role="user", content=q1),
        Turn(role="assistant", content=r1),
        Turn(role="user", content=q2),
        Turn(role="assistant", content=r2),
        Turn(role="user", content=q3),
        Turn(role="assistant", content=r3),
        Turn(role="user", content=q4),
        Turn(role="assistant", content=r4),
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# Run Evaluation (syncs to Confident AI dashboard)
# ─────────────────────────────────────────────────────────────────────────────
print("Running evaluation on all three episodes...")

evaluate(
    test_cases=[episode_1, episode_2, episode_3],
    metrics=[metric]
)

