import os
import json
import streamlit as st
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


# -------------------------
# Config
# -------------------------

JDS_PATH = "jds.json"
FAISS_PATH = "jd_faiss.index"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Groq model (new working models – old llama3-8b is deprecated)
GROQ_MODEL = "llama3-8b-8192"


# -------------------------
# UI
# -------------------------

st.set_page_config(page_title="Campus JD Chatbot", layout="wide")

st.title("🎓 Campus Placement JD Chatbot")
st.caption("Ask any question based only on the uploaded job descriptions")


# -------------------------
# Load resources
# -------------------------

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource
def load_index():
    return faiss.read_index(FAISS_PATH)


@st.cache_data
def load_jds():
    with open(JDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


embedder = load_embedder()
index = load_index()
jds = load_jds()


# -------------------------
# LLM
# -------------------------

llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model=GROQ_MODEL,
    temperature=0.2
)


# -------------------------
# Session chat memory
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------
# Display history
# -------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -------------------------
# Chat input
# -------------------------

query = st.chat_input("Ask anything about the available job descriptions...")

if query:

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    # -------------------------
    # Retrieval
    # -------------------------

    q_emb = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    k = 6
    scores, indices = index.search(q_emb, k)

    retrieved_chunks = []
    used_sources = set()

    for idx in indices[0]:
        rec = jds[idx]

        retrieved_chunks.append(
            f"""
Company: {rec['company']}
Role: {rec['role']}
Year: {rec['year']}
Source file: {rec['source_file']}

{rec['text']}
""".strip()
        )

        used_sources.add(
            f"{rec['company']} – {rec['source_file']}"
        )

    context = "\n\n---\n\n".join(retrieved_chunks)


    # -------------------------
    # System prompt
    # -------------------------

    system_prompt = """
You are a campus placement assistant.

You must answer ONLY using the information provided in the job description context.

The users are students who want to understand:
- what roles are available
- which companies are offering them
- what skills, background and expectations are mentioned
- what the work looks like in the first year

If the answer is not present in the context, clearly say:
"I could not find this information in the available job descriptions."

Do NOT hallucinate.
Do NOT use external knowledge.

Be structured and student-friendly.
"""

    final_prompt = f"""
Context from job descriptions:

{context}

Student question:
{query}
"""


    # -------------------------
    # LLM call
    # -------------------------

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=final_prompt)
        ]
    )

    answer = response.content


    # -------------------------
    # Show answer
    # -------------------------

    with st.chat_message("assistant"):
        st.markdown(answer)

        if used_sources:
            with st.expander("📄 Sources used"):
                for s in sorted(used_sources):
                    st.write(s)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
