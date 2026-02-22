# 🧪 VERITAS-RAG  
### A Self-Correcting Multi-Agent Grounded Retrieval-Augmented Generation Framework

VERITAS-RAG is a research-grade, multi-agent Retrieval-Augmented Generation (RAG) system designed to produce **strictly grounded, citation-verified answers** over domain-specific Chemical Engineering PDFs.

Unlike conventional RAG pipelines, VERITAS-RAG introduces:

- Hybrid retrieval (Dense + BM25)
- Multi-stage relevance filtering
- Strict grounding enforcement
- Post-generation support verification
- Iterative answer revision
- Usefulness evaluation
- Intelligent query rewriting
- Bounded corrective loops
- State-machine orchestration using LangGraph

This transforms RAG from a simple retrieval pipeline into a **verified reasoning architecture**.

---

## 📌 Motivation

Traditional RAG systems often:

- Hallucinate citations  
- Introduce unsupported interpretations  
- Fail silently when retrieval is weak  
- Provide generic answers that don’t address the question  

VERITAS-RAG addresses these limitations through:

1. Explicit grounding constraints  
2. Post-generation validation  
3. Iterative correction loops  
4. Structured state-machine control  

---

## 🏗 System Architecture

The system is orchestrated using **LangGraph**, implementing a bounded multi-agent control flow.

### High-Level Flow

User Query
   ↓
Decide Retrieval
   ↓
Hybrid Retrieval (Dense + BM25)
   ↓
Relevance Filtering
   ↓
Grounded Generation
   ↓
Support Verification (IsSUP)
   ↙            ↘
Revise        Rewrite Query
   ↓                ↓
Re-Verify        Re-Retrieve
   ↓
Usefulness Check (IsUSE)
   ↓
Final Answer
🧠 Multi-Agent Design

Each responsibility is isolated into a dedicated agent:

Agent File	Responsibility

decide_retrieval_agent.py	Determines whether external retrieval is necessary

hybrid_retrieval_agent.py	Performs FAISS (dense) + BM25 retrieval

retrieval_checker_agent.py	Filters documents for query-level relevance

generate_from_context.py	Generates strictly grounded, citation-backed answers

is_support_agent.py	Verifies grounding (fully_supported / partially_supported / no_support)

rewrite_answer_agent.py	Revises partially supported answers

useful_answer_checker.py	Checks if answer actually addresses the question

query_rewriter_agent.py	Reformulates retrieval query to improve recall

Direct_generation_agent.py	Direct LLM response when retrieval is not required

Orchestration logic is implemented in:

improved_rag_system.py

🔁 Self-Correcting Control Logic

The system enforces bounded corrective loops:

partially_supported → revise answer (max 2 retries)

no_support → rewrite retrieval query

not-useful → rewrite query and re-retrieve

rewrite attempts exceed threshold → safe termination


This ensures:

No infinite loops

Deterministic behavior

Controlled failure handling

Reduced hallucination risk

🔎 Hybrid Retrieval Strategy

Dense Retrieval

FAISS

HuggingFace Sentence Transformers

Semantic similarity search


Keyword Retrieval

BM25 (rank_bm25)

Exact term matching

Technical phrase recall

Results are:

Merged

Deduplicated

Relevance-filtered

📂 Project Structure

src/
│
├── Direct_generation_agent.py

├── decide_retrieval_agent.py

├── generate_from_context.py

├── hybrid_retrieval_agent.py

├── improved_rag_system.py

├── ingest.py

├── is_support_agent.py

├── query_rewriter_agent.py

├── retrieval_checker_agent.py

├── rewrite_answer_agent.py

├── useful_answer_checker.py

│

vectorstore/

│   (FAISS embeddings)

│
final_app.py  → Streamlit application



💬 Streamlit Application



final_app.py provides:


Chat-based interface

Multi-agent execution pipeline

System log visualization

Node-by-node state trace

Reset functionality

Run the application:

streamlit run final_app.py



🛠 Technology Stack


Python

LangChain

LangGraph

FAISS

BM25 (rank_bm25)

HuggingFace Embeddings

Ollama LLM

Streamlit



📊 Key Contributions

VERITAS-RAG introduces:

Multi-stage verification (Support + Usefulness)

Strict grounding enforcement with structured citations

Iterative answer revision

Intelligent retrieval query rewriting

Bounded self-correcting loops

State-machine-based RAG orchestration


📈 Future Improvements

Cross-encoder reranking

Citation alignment validation

Confidence scoring

Latency profiling per node

Retrieval evaluation (Recall@K, MRR)

Automated benchmarking vs baseline RAG

🎯 Intended Use Cases

Domain-specific QA over technical PDFs

Chemical Engineering knowledge systems

Research-grade RAG experimentation

Hallucination-resistant LLM pipelines

Educational AI systems
