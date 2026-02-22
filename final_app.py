import streamlit as st
from improved_rag_system import graph

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Chemical Engineering RAG Assistant",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Chemical Engineering Multi-Agent RAG System")

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "logs" not in st.session_state:
    st.session_state.logs = []

# ---------------- TABS ----------------
chat_tab, log_tab = st.tabs(["💬 Chat", "📊 System Logs"])

# ==========================================================
# CHAT TAB
# ==========================================================

with chat_tab:

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a Chemical Engineering question...")

    if user_input:

        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        # Run graph
        with st.chat_message("assistant"):
            with st.spinner("Thinking... Running multi-agent pipeline..."):

                logs = []

                final_state = None

                for event in graph.stream(
                    {
                        "user_query": user_input,
                        "retries": 0,
                        "rewrite_tries": 0
                    },
                    stream_mode="values"
                ):
                    logs.append(event)
                    final_state = event

                answer = final_state.get("answer", "No answer generated.")

                st.markdown(answer)

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        # Save logs
        st.session_state.logs = logs


# ==========================================================
# LOG TAB
# ==========================================================

with log_tab:

    st.subheader("Execution Trace")

    if not st.session_state.logs:
        st.info("No logs yet. Ask a question first.")
    else:
        for i, log in enumerate(st.session_state.logs):
            with st.expander(f"Step {i+1}"):
                st.json(log)


# ---------------- FOOTER ----------------

st.divider()

col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Reset Chat"):
        st.session_state.messages = []
        st.session_state.logs = []
        st.rerun()

with col2:
    st.markdown("Built with LangGraph + Hybrid RAG + Self-Correction")