from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from typing import Literal

from query_analyzer_agent import load_llm

llm = load_llm()


# ========================= SCHEMA =========================

class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not-useful"]
    reason: str = Field(..., description="Short reason in 1 line")


# ========================= PROMPT =========================

isuse_prompt = PromptTemplate.from_template("""
You are judging USEFULNESS of the ANSWER for the QUESTION.

GOAL:
- Decide if the answer actually addresses what the user asked.

Return JSON with keys: isuse, reason.
isuse must be one of: useful, not-useful.

RULES:
- useful: The answer directly answers the question or provides the requested specific information.
- not-useful: The answer is generic, off-topic, incomplete, or avoids the core question.
- If the answer only partially addresses the question, choose not-useful.
- Do NOT use outside knowledge.
- Do NOT re-check grounding (IsSUP already verified that).
- Keep reason to 1 short line.

QUESTION:
{question}

ANSWER:
{answer}
""")


def is_useful(question: str, answer: str) -> IsUSEDecision:

    if not answer:
        return IsUSEDecision(
            isuse="not-useful",
            reason="No answer provided."
        )

    use_checker_llm = llm.with_structured_output(IsUSEDecision)

    chain = isuse_prompt | use_checker_llm

    decision = chain.invoke({
        "question": question,
        "answer": answer
    })

    return decision