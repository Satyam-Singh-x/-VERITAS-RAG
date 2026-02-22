from pydantic import BaseModel,Field
import json
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from query_analyzer_agent import load_llm



#=================================SCHEMA=====================================================
class RetrieveDecision(BaseModel):
    should_retrieve: bool= Field(
        ...,description="True if external documents are needed to answer reliably , else False."

    )

def decide_retrieval_func(user_query):
    llm = load_llm()
    user_query = user_query.strip()

    decide_retrieval_prompt = PromptTemplate.from_template(
        """
        You decide whether retrieval is needed for the given query.

        Return JSON that matches this schema:
        {"should_retrieve": boolean}

        GUIDELINES:
        - should_retrieve=True if answering requires specific facts, citations, or info likely not in the model.
        - should_retrieve=False for general explanations, definitions, or reasoning that doesn't need sources.
        - If unsure choose True.

        Query:
        {user_query}
        """
    )

    should_retrieve_llm = llm.with_structured_output(RetrieveDecision)

    chain = decide_retrieval_prompt | should_retrieve_llm

    decision = chain.invoke({'user_query': user_query})

    return decision

