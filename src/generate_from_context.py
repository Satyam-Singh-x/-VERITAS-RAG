from pydantic import BaseModel,Field
import json
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from query_analyzer_agent import load_llm

def generate_from_context(user_query, context):
    llm = load_llm()

    rag_generation_prompt=PromptTemplate.from_template(
        """
      You are a Chemical Engineering Knowledge Generator Agent operating 
      inside a strictly grounded multi-agent RAG system.
  
      Your task is to generate a fully grounded explanation using ONLY 
      the provided validated context.
  
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      STRICT GROUNDING REQUIREMENTS
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  
      1. You MUST use ONLY information explicitly present in the provided context.
      2. You MUST NOT introduce external knowledge.
      3. You MUST NOT infer missing equations, steps, or assumptions.
      4. If information is incomplete, explicitly state:
         "The provided context does not contain sufficient detail on this aspect."
      5. Every major paragraph MUST end with a citation.
      6. The citation MUST exactly follow this format:
  
         (Source: filename, Page: page_number)
  
         Example:
         (Source: chemical process equipment.pdf, Page: 425)
  
      7. Do NOT combine multiple sources inside one citation.
      8. Do NOT invent filenames or page numbers.
      9. Do NOT place all citations at the end — citations must appear 
         immediately after the paragraph they support.
      10. If no valid citation can be attached to a paragraph, 
          DO NOT generate that paragraph.
  
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      OUTPUT STRUCTURE
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  
      • Start with a concise conceptual overview paragraph.
      • Follow with structured explanation using clear sections or bullet points.
      • Each paragraph must end with one citation in the exact format:
  
        (Source: filename, Page: page_number)
  
      • Keep explanation suitable for undergraduate chemical engineering students.
      • Prefer clarity over excessive complexity.
  
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  
      Query:
      {query}
  
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      VALIDATED CONTEXT
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  
      {context}
  
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  
      Generate a fully grounded, citation-backed explanation now.
      """


    )

    chain=rag_generation_prompt | llm | StrOutputParser()
    result=chain.invoke({"query":user_query,"context":context})

    return result



