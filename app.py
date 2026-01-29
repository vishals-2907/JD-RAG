import streamlit as st
import faiss
import json
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# ----------------------------------
# Streamlit config
# ----------------------------------
st.set_page_config(
    page_title="JD Explorer – Multi-JD RAG",
    layout="wide"
)

st.title("🔍 Job Market Explorer (FAISS + Groq)")
st.caption("Ask questions across hundreds of Job Descriptions")

# ----------------------------------
# Load models & data (cached)
# ----------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_faiss_index():
    return faiss.read_index("jd_faiss.index")

@st.cache_resource
def load_jd_data():
    with open("jds.json", "r", encoding="utf-8") as f:
        jd_texts = json.load(f)

    with open("jd_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return jd_texts, metadata

embedding_model = load_embedding_model()
index = load_faiss_index()
jd_texts, metadata = load_jd_data()

# ----------------------------------
# Load Groq LLM
# ----------------------------------
llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama-3.1-8b-instant"
)

# ----------------------------------
# User query
# ----------------------------------
query = st.text_input(
    "Ask about companies, roles, or skills (e.g. 'Which companies are hiring for analytics roles?')"
)

if query:
    # ----------------------------------
    # Embed query & search FAISS
    # ----------------------------------
    query_embedding = embedding_model.encode([query]).astype("float32")

    TOP_K = 5
    distances, indices = index.search(query_embedding, TOP_K)

    # ----------------------------------
    # Collect retrieved context
    # ----------------------------------
    retrieved_chunks = []
    company_role_map = {}

    for idx in indices[0]:
        text = jd_texts[idx]["text"]
        meta = metadata[idx]

        company = meta.get("company", "Unknown Company")
        source = meta.get("source_file", "Unknown Source")

        retrieved_chunks.append(
            f"Company: {company}\nSource: {source}\n{text}"
        )

        if company not in company_role_map:
            company_role_map[company] = set()
        company_role_map[company].add(source)

    # ----------------------------------
    # Display retrieved companies
    # ----------------------------------
    st.subheader("🏢 Companies Found")
    for company, sources in company_role_map.items():
        st.write(f"**{company}**")
        for src in sources:
            st.write(f"- {src}")

    # ----------------------------------
    # Prepare LLM prompt (Explorer Mode)
    # ----------------------------------
    context = "\n\n---\n\n".join(retrieved_chunks)
    MAX_CONTEXT_CHARS = 6000
    context = context[:MAX_CONTEXT_CHARS]

    prompt = f"""
You are a Job Description analyst.

Using ONLY the information present in the context below, extract
DETAILED role-level information.

For EACH role you identify, clearly provide:

1. Company Name
2. Role Title
3. Key Responsibilities (bullet points)
4. Mandatory Skills / Requirements
5. Preferred Skills (if mentioned)
6. Experience Level (if mentioned)

Rules:
- Do NOT generalize
- Do NOT add information not present in the context
- If a field is not mentioned, explicitly say "Not specified"
- Separate roles clearly
- Be factual and structured

Context:
{context}

User Question:
{query}

Answer:
"""


    response = llm.invoke([HumanMessage(content=prompt)])

    # ----------------------------------
    # Final answer
    # ----------------------------------
    st.subheader("🧠 Market Insight")
    st.write(response.content)





