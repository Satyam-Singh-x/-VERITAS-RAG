from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from query_analyzer_agent import load_llm

llm = load_llm()

reviser_prompt = PromptTemplate.from_template("""
You are a STRICT Grounded Answer Rewriter Agent inside a multi-agent RAG system.

Your task is to REVISE the ANSWER so that it becomes FULLY SUPPORTED by the provided CONTEXT.

The previous answer was classified as not fully supported.
You must correct it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OBJECTIVE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Rewrite the ANSWER so that:

• Every meaningful technical claim is explicitly supported by CONTEXT.
• No interpretation, abstraction, inferred reasoning, or qualitative expansion is added.
• No external knowledge is introduced.
• No logical steps are filled in unless they appear explicitly in CONTEXT.
• All statements must be directly traceable to the CONTEXT.

If something in the previous answer cannot be supported directly,
REMOVE it.

If information is incomplete, explicitly state:
"The provided context does not contain sufficient detail on this aspect."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT GROUNDING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Use wording that stays as close as possible to the CONTEXT.
2. Prefer quoting or lightly rephrasing exact sentences from CONTEXT.
3. Do NOT summarize beyond what is written.
4. Do NOT combine information from multiple sources in a single paragraph.
5. Every paragraph MUST end with exactly one citation.
6. Citation format MUST be exactly:

   (Source: filename, Page: page_number)

7. Filenames and page numbers MUST match those provided in CONTEXT.
8. Do NOT invent sources.
9. If a paragraph cannot be directly supported by one source, do NOT generate it.
10. If unsure, remove the statement rather than risk unsupported content.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Begin with a concise conceptual overview.
• Follow with structured explanation using clear sections or bullet points.
• Each paragraph must end with exactly one citation.
• Do NOT explain your reasoning.
• Output ONLY the revised answer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUESTION:
{query}

PREVIOUS ANSWER:
{answer}

VALIDATED CONTEXT:
{context}

Rewrite the answer so that it is FULLY SUPPORTED.
""")


def revise_answer(answer: str, context: str, query: str) -> str:

    if not context:
        return answer  # no context → cannot revise

    chain = reviser_prompt | llm | StrOutputParser()

    result = chain.invoke({
        "query": query,
        "context": context,
        "answer": answer
    })

    return result.strip()