import os
import json
import streamlit as st
import faiss
import numpy as np
import PyPDF2

from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


# -------------------------
# Config
# -------------------------

JDS_PATH = "jds.json"
FAISS_PATH = "jd_faiss.index"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"


# -------------------------
# UI Configuration & Styling
# -------------------------

st.set_page_config(
    page_title="Placement Intelligence Hub", 
    page_icon="💼", 
    layout="centered"
)

# Lightweight CSS to hide Streamlit branding without breaking Dark/Light mode
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

USER_AVATAR = "👤"
BOT_AVATAR = "💼"


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
# Helper: PDF Extractor
# -------------------------

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "
    return text


# -------------------------
# App Header
# -------------------------

st.title("💼 Placement Intelligence Hub")
st.markdown("Explore available job descriptions, ask questions about specific companies, or upload your resume to discover your best matches.")
st.write("") # Small spacing


# -------------------------
# Feature 1: CV Matcher (Moved to Main Page)
# -------------------------

with st.expander("🎯 **Upload your CV to find matching roles**", expanded=False):
    st.info("Our AI will analyze your skills and experience to find the top 5 most aligned roles in our database.")
    
    uploaded_cv = st.file_uploader("Select your resume (PDF format)", type=["pdf"])
    
    if uploaded_cv is not None:
        # Use a bold, primary button 
        if st.button("Analyze & Find Matches", type="primary", use_container_width=True):
            with st.spinner("Analyzing CV Profile and searching database..."):
                cv_text = extract_text_from_pdf(uploaded_cv)
                
                # Compress CV into search string
                extraction_prompt = f"""
                You are an expert HR recruiter. Analyze the following CV text and extract the candidate's core profile.
                Return ONLY a dense, comma-separated paragraph containing their key skills, total experience, target roles, and domain knowledge. Do not include introductory text.
                
                CV Text:
                {cv_text[:4000]}
                """
                
                compressed_profile = llm.invoke([HumanMessage(content=extraction_prompt)]).content
                
                # Embed the profile
                cv_emb = embedder.encode(
                    [compressed_profile], 
                    convert_to_numpy=True, 
                    normalize_embeddings=True
                )
                
                # Search FAISS
                scores, indices = index.search(cv_emb, 5)
                
                # Format response
                match_response = "### 🎯 Top 5 Roles Aligned With Your CV\n\nI analyzed your resume and found these roles in our database that best match your skills and experience:\n\n"
                
                for rank, idx in enumerate(indices[0]):
                    rec = jds[idx]
                    match_response += f"**{rank + 1}. {rec['role'].title()}**\n"
                    match_response += f"* **Company:** {rec['company']}\n"
                    match_response += f"* **Source Document:** `{rec['source_file']}`\n\n"
                
                match_response += "---\n*Feel free to ask me specific questions about any of these roles!*"
                
                # Inject directly into chat and refresh
                st.session_state.messages.append({"role": "assistant", "content": match_response})
                st.rerun()

st.divider()


# -------------------------
# Feature 2: Chat Interface
# -------------------------

# Display history
for msg in st.session_state.messages:
    avatar = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# Chat input
query = st.chat_input("Ask anything about the available job descriptions...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(query)

    search_query = query 
    
    if len(st.session_state.messages) > 1:
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
        rephrase_response = llm.invoke([
            SystemMessage(content="You are a query reformulation assistant. Return ONLY the rewritten question."),
            HumanMessage(content=rephrase_prompt)
        ])
        
        search_query = rephrase_response.content.strip()
        st.caption(f"*(System searched for: {search_query})*")

    q_emb = embedder.encode(
        [search_query], 
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
            f"Company: {rec['company']}\nRole: {rec['role']}\nYear: {rec['year']}\nSource file: {rec['source_file']}\n\n{rec['text']}".strip()
        )
        used_sources.add(f"{rec['company']} – {rec['source_file']}")

    context = "\n\n---\n\n".join(retrieved_chunks)

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

    final_prompt = f"Context from job descriptions:\n\n{context}\n\nStudent question:\n{query}"
    
    messages_to_pass = [SystemMessage(content=system_prompt)]
    
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            messages_to_pass.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages_to_pass.append(AIMessage(content=msg["content"]))
            
    messages_to_pass.append(HumanMessage(content=final_prompt))

    response = llm.invoke(messages_to_pass)
    answer = response.content

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        st.markdown(answer)

        if used_sources:
            with st.expander("📄 View Source Documents"):
                for s in sorted(used_sources):
                    st.write(f"- {s}")

    st.session_state.messages.append({"role": "assistant", "content": answer})