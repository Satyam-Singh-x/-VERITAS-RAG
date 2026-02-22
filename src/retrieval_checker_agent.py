from pydantic import BaseModel,Field
import json
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from query_analyzer_agent import load_llm

#=========================SCHEMA==============================================================
class RelevanceDecision(BaseModel):
    is_relevant: bool=Field(...,description="True if the document helps answer the question ,else False")

def relevance_checker(doc:str,user_query:str):
    llm = load_llm()
    is_relevant_prompt=PromptTemplate.from_template(
        """
        You are an expert document checker. Your job is to check each of the given documents based on their relevance
        and their ability to answer the given question.
        Return JSON that matches this schema: {{'is_relevant:boolean'}}
        A document is relevant if it contains information useful for answering the question.
        
        Documents:
        {doc}
        
        Question:
        {user_query}
        
        """
    )

    relevance_llm=llm.with_structured_output(RelevanceDecision)
    chain= is_relevant_prompt | relevance_llm
    result= chain.invoke({'doc':doc,'user_query':user_query})
    return result.is_relevant
