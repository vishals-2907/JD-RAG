import os
import json
import streamlit as st
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


# -------------------------
# Config
# -------------------------

JDS_PATH = "jds.json"
FAISS_PATH = "jd_faiss.index"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Groq model (new working models – old llama3-8b is deprecated)
GROQ_MODEL = "llama-3.1-8b-instant"


# -------------------------
# UI Configuration & Styling
# -------------------------

st.set_page_config(
    page_title="Placement Intelligence | JD Copilot", 
    page_icon="💼", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to inject a professional look, hide default Streamlit branding
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Avatars for a polished chat interface
USER_AVATAR = "👤"
BOT_AVATAR = "🏢"

# Sidebar layout for context and instructions
with st.sidebar:
    st.header("💼 JD Copilot")
    st.markdown("---")
    st.markdown(
        "Welcome to the **Placement Intelligence System**. "
        "This tool is designed to help you quickly navigate and extract "
        "requirements from available job descriptions (JDs)."
    )
    st.markdown("### 💡 Example Queries")
    st.markdown("- *What roles are available in Product Management?*")
    st.markdown("- *What are the technical prerequisites for the FedEx role?*")
    st.markdown("- *Which companies are hiring for Customer Success?*")
    st.markdown("---")
    st.caption("Powered by Vector Search & Llama 3.1")

# Main Header
st.title("Placement Intelligence Hub")
st.markdown("Query the centralized job description database to find role specifics, company expectations, and skill requirements.")
st.divider()


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
    avatar = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# -------------------------
# Chat input
# -------------------------

query = st.chat_input("Ask anything about the available job descriptions...")

if query:

    # 1. Append user's original query to the visual chat history
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(query)

    # -------------------------
    # Query Reformulation (Fixing the RAG Blindspot)
    # -------------------------
    
    search_query = query # Default to the raw query
    
    # If there is conversational history (more than just the current prompt)
    if len(st.session_state.messages) > 1:
        
        # Build a text block of the previous conversation
        history_str = ""
        for msg in st.session_state.messages[:-1]: 
            role_label = "Student" if msg["role"] == "user" else "Assistant"
            history_str += f"{role_label}: {msg['content']}\n"
            
        rephrase_prompt = f"""
Given the following conversation history and the user's next question, rephrase the question to be a standalone question that can be understood without the chat history.
If the question is already standalone or doesn't need context, just return the original question.
Do NOT answer the question, just provide the standalone version.

Conversation History:
{history_str}

User's Next Question:
{query}

Standalone Question:
"""
        # Call LLM just to rewrite the query
        rephrase_response = llm.invoke([
            SystemMessage(content="You are a query reformulation assistant. Return ONLY the rewritten question."),
            HumanMessage(content=rephrase_prompt)
        ])
        
        search_query = rephrase_response.content.strip()
        
        # Optional: Show the rewritten query in the UI so you know it worked
        st.caption(f"*(System searched for: {search_query})*")


    # -------------------------
    # Retrieval (Using the Reformulated Query)
    # -------------------------

    q_emb = embedder.encode(
        [search_query], # <--- Pass the STANDALONE query to FAISS here
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
    # Final LLM call with Memory
    # -------------------------
    
    messages_to_pass = [SystemMessage(content=system_prompt)]
    
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            messages_to_pass.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages_to_pass.append(AIMessage(content=msg["content"]))
            
    messages_to_pass.append(HumanMessage(content=final_prompt))

    response = llm.invoke(messages_to_pass)
    answer = response.content


    # -------------------------
    # Show answer
    # -------------------------

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.markdown(answer)

        if used_sources:
            with st.expander("📄 View Source Documents"):
                for s in sorted(used_sources):
                    st.write(f"- {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )