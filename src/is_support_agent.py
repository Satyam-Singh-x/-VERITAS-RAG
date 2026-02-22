from typing import Literal, List
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

from query_analyzer_agent import load_llm

# Initialize once (good practice)
llm = load_llm()


# ============================== SCHEMA ==============================

class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list)


# ============================== ISSUP CHECKER ==============================

def issup_checker(query: str, context: str, answer: str) -> IsSUPDecision:
    # 🔒 Guard: if no context, automatically no_support
    if not context:
        return IsSUPDecision(
            issup="no_support",
            evidence=[]
        )

    issup_llm = llm.with_structured_output(IsSUPDecision)

    issup_prompt = PromptTemplate.from_template("""
You are a STRICT verification agent inside a grounded multi-agent RAG system.

Your task is to determine whether the ANSWER is supported by the provided CONTEXT.

Output ONLY valid JSON with the following keys:
- issup
- evidence

The value of "issup" MUST be exactly one of:
  - "fully_supported"
  - "partially_supported"
  - "no_support"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEFINITION OF SUPPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1) fully_supported:
   • Every meaningful technical claim in the ANSWER is explicitly stated in the CONTEXT.
   • No new explanations, interpretations, abstractions, or qualitative phrasing are introduced.
   • No additional meaning beyond what is directly written in CONTEXT.

2) partially_supported:
   • Core facts are supported,
   BUT
   • The ANSWER introduces ANY additional explanation, abstraction,
     interpretation, qualitative phrasing, or inferred reasoning
     not explicitly present in the CONTEXT.
   • If even one unsupported interpretive statement exists,
     choose "partially_supported".

3) no_support:
   • Key claims are not found in the CONTEXT.
   • The answer is mostly unrelated.
   • The answer relies on outside knowledge.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Be conservative and strict.
• Do NOT assume implicit knowledge.
• Do NOT infer missing logical steps.
• If unsure between categories, choose the lower support level.
• Evidence must be exact verbatim substrings copied directly from CONTEXT.
• Do NOT paraphrase evidence.
• Provide up to 3 short direct quotes.
• If nothing is supported, return:
    issup = "no_support"
    evidence = []

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUESTION:
{query}

ANSWER:
{answer}

CONTEXT:
{context}
""")

    chain = issup_prompt | issup_llm

    result = chain.invoke({
        "query": query,
        "answer": answer,
        "context": context
    })

    return result