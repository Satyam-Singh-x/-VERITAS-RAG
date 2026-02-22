from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from typing import Literal
from query_analyzer_agent import load_llm

llm = load_llm()

# ============================= SCHEMA =============================

class RewriteDecision(BaseModel):
    retrieval_query: str = Field(
        ..., description="Rewritten query optimized for vector retrieval"
    )

# ============================= PROMPT =============================

rewrite_prompt = PromptTemplate.from_template("""
You are rewriting a QUESTION into a query optimized for vector retrieval
over Chemical Engineering PDF textbooks.

Output ONLY valid JSON with key:
- retrieval_query

RULES:
- Keep it short (8–16 words).
- Preserve key entities from the QUESTION (e.g., Distillation, Reactor, Heat Exchanger).
- Add 2–5 high-signal domain-specific technical keywords likely to appear in textbooks
  (e.g., equations, thermodynamics terms, mass transfer, McCabe-Thiele, enthalpy balance).
- Remove filler words.
- Do NOT answer the question.
- Do NOT explain your reasoning.

EXAMPLE:
Q: Derive operating line equation in distillation.
Output:
{"retrieval_query": "Operating line equation distillation McCabe Thiele mass balance"}

QUESTION:
{question}

Previous retrieval query:
{retrieval_query}

Answer (if any):
{answer}
""")

# ============================= FUNCTION =============================

def rewrite_question(question: str, retrieval_question: str, answer: str) -> RewriteDecision:

    if not question:
        return RewriteDecision(retrieval_query="")

    rewrite_llm = llm.with_structured_output(RewriteDecision)

    chain = rewrite_prompt | rewrite_llm

    decision = chain.invoke({
        "question": question,
        "retrieval_query": retrieval_question,
        "answer": answer
    })

    return decision