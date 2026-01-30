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
You are a placement mentor helping freshers understand campus job opportunities.

Your audience:
- Final-year students
- Fresh graduates
- Candidates new to placements

Using ONLY the information present in the context below, explain:

1. What TYPES of roles are being offered (group similar roles together)
2. Which COMPANIES are offering these roles
3. What these roles GENERALLY EXPECT from candidates
4. Common SKILLS and KNOWLEDGE areas students should prepare for
5. Any EXPERIENCE or background expectations mentioned (if any)

Guidelines:
- Use simple, student-friendly language
- Avoid HR jargon
- Do NOT invent information
- If something is not mentioned, say "Not clearly specified in the JDs"
- Focus on helping students understand how to prepare

Context (extracted from multiple job descriptions):
{context}

Student Question:
{query}

Answer:
"""


    response = llm.invoke([HumanMessage(content=prompt)])

    # ----------------------------------
    # Final answer
    # ----------------------------------
    st.subheader("🧠 Market Insight")
    st.write(response.content)






