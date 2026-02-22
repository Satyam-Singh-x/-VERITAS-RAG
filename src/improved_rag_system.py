from typing import TypedDict, Optional, List, Literal
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from decide_retrieval_agent import decide_retrieval_func
from Direct_generation_agent import direct_generation_func
from hybrid_retrieval_agent import HybridRetrievalAgent
from retrieval_checker_agent import relevance_checker
from generate_from_context import generate_from_context
from is_support_agent import issup_checker
from rewrite_answer_agent import revise_answer
from useful_answer_checker import is_useful
from query_rewriter_agent import rewrite_question

retriever = HybridRetrievalAgent()

# ==========================================================
# STATE
# ==========================================================

class AgentState(TypedDict):
    user_query: str
    retrieval_query: Optional[str]
    rewrite_tries: Optional[int]
    needs_retrieval: Optional[bool]
    docs: Optional[List[Document]]
    relevant_docs: Optional[List[Document]]
    context: Optional[str]
    answer: Optional[str]
    issup: Optional[Literal["fully_supported","partially_supported","no_support"]]
    evidence: Optional[List[str]]
    retries: Optional[int]
    isuse: Optional[Literal["useful","not-useful"]]
    use_reason: Optional[str]

# ==========================================================
# NODES
# ==========================================================

def decide_retrieval(state: AgentState):
    query = state.get("retrieval_query") or state["user_query"]
    decision = decide_retrieval_func(query)
    return {"needs_retrieval": decision.should_retrieve}

def direct_generation(state: AgentState):
    result = direct_generation_func(state["user_query"])
    return {"answer": result}

def retrieval(state: AgentState):
    query = state.get("retrieval_query") or state["user_query"]
    merged_docs = retriever.retrieve(query)
    return {"docs": merged_docs}

def route_after_decide(state: AgentState):
    return "retrieve" if state["needs_retrieval"] else "generate_direct"

def is_relevant(state: AgentState):
    docs = state.get("docs") or []
    query = state["user_query"]

    relevant_docs = []
    for doc in docs:
        if relevance_checker(doc.page_content, query):
            relevant_docs.append(doc)

    return {"relevant_docs": relevant_docs}

def route_after_relevance(state: AgentState):
    return "generate_from_context" if state.get("relevant_docs") else "no_relevant_docs"

def generate_from_context_agent(state: AgentState):
    relevant_docs = state.get("relevant_docs") or []
    query = state["user_query"]

    context = "\n\n".join(
        f"{doc.page_content}\n(Source: {doc.metadata['source_file']}, Page: {doc.metadata['page_number']})"
        for doc in relevant_docs
    )

    result = generate_from_context(query, context)

    return {"answer": result, "context": context, "retries": 0}

def no_relevant_docs(state: AgentState):
    return {"answer": "No relevant documents found.", "context": ""}

def is_sup(state: AgentState):
    decision = issup_checker(
        state["user_query"],
        state.get("context") or "",
        state.get("answer") or ""
    )
    return {"issup": decision.issup, "evidence": decision.evidence}

def route_after_issup(state: AgentState):
    retries = state.get("retries", 0)

    if state["issup"] == "fully_supported":
        return "is_use"
    elif state["issup"] == "partially_supported" and retries < 2:
        return "revise_answer"
    else:
        return "rewrite_question"

def revise_answer_node(state: AgentState):
    result = revise_answer(
        state["user_query"],
        state.get("context") or "",
        state.get("answer") or ""
    )
    return {
        "answer": result,
        "retries": state.get("retries", 0) + 1
    }

def is_use(state: AgentState):
    decision = is_useful(
        state["user_query"],
        state.get("answer") or ""
    )
    return {"isuse": decision.isuse, "use_reason": decision.reason}

def route_after_isuse(state: AgentState):
    return "finalize" if state["isuse"] == "useful" else "rewrite_question"

def rewrite_question_node(state: AgentState):
    rewrite_tries = state.get("rewrite_tries", 0)

    if rewrite_tries >= 2:
        return {}

    decision = rewrite_question(
        state["user_query"],
        state.get("retrieval_query", ""),
        state.get("answer", "")
    )

    return {
        "retrieval_query": decision.retrieval_query,
        "rewrite_tries": rewrite_tries + 1
    }

def finalize(state: AgentState):
    return state

# ==========================================================
# GRAPH BUILD
# ==========================================================

builder = StateGraph(AgentState)

builder.add_node("decide_retrieval", decide_retrieval)
builder.add_node("generate_direct", direct_generation)
builder.add_node("retrieve", retrieval)
builder.add_node("is_relevant", is_relevant)
builder.add_node("generate_from_context", generate_from_context_agent)
builder.add_node("no_relevant_docs", no_relevant_docs)
builder.add_node("is_sup", is_sup)
builder.add_node("revise_answer", revise_answer_node)
builder.add_node("is_use", is_use)
builder.add_node("rewrite_question", rewrite_question_node)
builder.add_node("finalize", finalize)

builder.set_entry_point("decide_retrieval")

builder.add_conditional_edges("decide_retrieval", route_after_decide)
builder.add_edge("retrieve", "is_relevant")
builder.add_conditional_edges("is_relevant", route_after_relevance)
builder.add_edge("generate_from_context", "is_sup")
builder.add_conditional_edges("is_sup", route_after_issup)
builder.add_edge("revise_answer", "is_sup")
builder.add_conditional_edges("is_use", route_after_isuse)
builder.add_edge("rewrite_question", "retrieve")

builder.add_edge("generate_direct", END)
builder.add_edge("no_relevant_docs", END)
builder.add_edge("finalize", END)

graph = builder.compile()

