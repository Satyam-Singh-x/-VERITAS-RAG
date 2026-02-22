from pydantic import BaseModel,Field
import json
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from query_analyzer_agent import load_llm

def direct_generation_func(user_query):
    direct_generation_prompt=PromptTemplate.from_template(
        """
        Answer the given question using your general knowledge.
        Do not assume access to external documents.
        If you are unsure or the answer requires specific sources, say:
        "I dont know based on my general knowledge"
        
        Question:
        {user_query}
        """
    )
    llm=load_llm()
    chain= direct_generation_prompt | llm | StrOutputParser()
    result = chain.invoke({'user_query': user_query})
    return result.strip()

