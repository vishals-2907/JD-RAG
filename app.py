import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Campus Placement JD Chatbot",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Campus Placement JD Chatbot")
st.caption("Ask any question about the job descriptions uploaded by your placement team.")


# -----------------------------
# Load models and data
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_faiss_index():
    return faiss.read_index("jd_faiss.index")


@st.cache_data
def load_jds():
    with open("jds.json", "r", encoding="utf-8") as f:
        return json.load(f)


embedder = load_embedding_model()
index = load_faiss_index()
jds_data = load_jds()


# -----------------------------
# Groq LLM
# -----------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=1024
)


# -----------------------------
# Helpers
# -----------------------------
def retrieve_context(query, k=6):
    query_vec = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, k)

    results = []
    for idx in indices[0]:
        if idx < len(jds_data):
            results.append(jds_data[idx])

    return results


def build_context_text(retrieved_chunks):
    blocks = []

    for c in retrieved_chunks:
        block = f"""
Company: {c.get("company","")}
Source file: {c.get("source_file","")}
JD text:
{c.get("text","")}
"""
        blocks.append(block.strip())

    return "\n\n---\n\n".join(blocks)


# -----------------------------
# Chat memory
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Display history
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -----------------------------
# Chat input
# -----------------------------
user_query = st.chat_input("Ask anything about the campus job descriptions…")


if user_query:

    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    # -----------------------------
    # Retrieve relevant JDs
    # -----------------------------
    retrieved = retrieve_context(user_query, k=6)

    context_text = build_context_text(retrieved)

    system_prompt = f"""
You are a placement assistant for MBA and undergraduate students.

You must answer ONLY using the information present in the provided job descriptions.

Your goal:
Help students clearly understand:
- what roles are being offered
- what companies are hiring
- what skills and expectations are mentioned
- what kind of work freshers will actually do

If the information is not present in the provided context, clearly say:
"I could not find this information in the available job descriptions."

Do NOT hallucinate.

Use a student-friendly tone.

Here are the job descriptions:

{context_text}
""".strip()

    with st.chat_message("assistant"):
        with st.spinner("Reading job descriptions..."):

            response = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_query)
                ]
            )

            answer = response.content
            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )


# -----------------------------
# Optional transparency section
# -----------------------------
with st.expander("🔎 Show retrieved JD sources used for this answer"):
    if user_query:
        for i, c in enumerate(retrieved, 1):
            st.markdown(
                f"""
**{i}. {c.get("company","Unknown")}**  
File: `{c.get("source_file","")}`
"""
            )
    else:
        st.write("Ask a question to see which job descriptions were used.")
