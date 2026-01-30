import streamlit as st
import faiss
import json
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# ---------------------------------------------------
# Streamlit config
# ---------------------------------------------------
st.set_page_config(
    page_title="JD Explorer – Campus Placement Intelligence",
    layout="wide"
)

st.title("🎓 Campus Placement Explorer")
st.caption("Understand what roles come to campus and how freshers should prepare")

# ---------------------------------------------------
# Load assets (cached)
# ---------------------------------------------------
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

# ---------------------------------------------------
# Groq LLM
# ---------------------------------------------------
llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama-3.1-8b-instant"
)

# ---------------------------------------------------
# UI
# ---------------------------------------------------
query = st.text_input(
    "Ask about companies, roles, or skills "
    "(e.g. 'Which roles come for campus placements?', "
    "'Do not include Accenture', "
    "'Tell me about Deloitte')"
)

# ---------------------------------------------------
# Role focus (interactivity)
# ---------------------------------------------------
st.subheader("🎯 Optional role focus")

role_focus = st.selectbox(
    "Choose a role category to focus on",
    [
        "All roles",
        "Analyst / Data",
        "Product / Business",
        "Consulting",
        "Operations / Support",
        "Technology"
    ]
)

# ---------------------------------------------------
# Query intent: include / exclude companies
# ---------------------------------------------------
if query:
    query_lower = query.lower()

    all_companies = list({
        m.get("company", "").strip()
        for m in metadata
        if m.get("company")
    })

    include_companies = []
    exclude_companies = []

    for c in all_companies:
        c_l = c.lower()

        # include intent
        if c_l in query_lower:
            include_companies.append(c)

        # exclude intent
        if f"not include {c_l}" in query_lower or f"exclude {c_l}" in query_lower:
            exclude_companies.append(c)

    # also catch "without X"
    for c in all_companies:
        c_l = c.lower()
        if f"without {c_l}" in query_lower:
            exclude_companies.append(c)

    # ---------------------------------------------------
    # Embed query & search FAISS
    # ---------------------------------------------------
    query_embedding = embedding_model.encode([query]).astype("float32")

    # retrieve wider set for better coverage
    TOP_K = 30
    distances, indices = index.search(query_embedding, TOP_K)

    # ---------------------------------------------------
    # Diversified collection
    # ---------------------------------------------------
    MAX_CHUNKS_PER_COMPANY = 2
    MAX_CHUNKS_PER_SOURCE = 1

    company_counter = {}
    source_counter = {}

    retrieved_chunks = []
    company_role_map = {}

    for idx in indices[0]:

        meta = metadata[idx]
        company = meta.get("company", "Unknown Company")
        source = meta.get("source_file", "Unknown Source")

        # -------- include / exclude logic ----------
        if include_companies and company not in include_companies:
            continue

        if company in exclude_companies:
            continue
        # -------------------------------------------

        if company not in company_counter:
            company_counter[company] = 0
        if source not in source_counter:
            source_counter[source] = 0

        if company_counter[company] >= MAX_CHUNKS_PER_COMPANY:
            continue

        if source_counter[source] >= MAX_CHUNKS_PER_SOURCE:
            continue

        text = jd_texts[idx]["text"]

        retrieved_chunks.append(
            f"Company: {company}\nSource: {source}\n{text}"
        )

        company_counter[company] += 1
        source_counter[source] += 1

        if company not in company_role_map:
            company_role_map[company] = set()
        company_role_map[company].add(source)

    # ---------------------------------------------------
    # Feedback to user about filters
    # ---------------------------------------------------
    if include_companies:
        st.info(f"Showing results only for: {', '.join(include_companies)}")

    if exclude_companies:
        st.info(f"Excluding: {', '.join(exclude_companies)}")

    # ---------------------------------------------------
    # Show companies found
    # ---------------------------------------------------
    st.subheader("🏢 Companies Found")

    if not company_role_map:
        st.warning("No matching companies found for your query.")
        st.stop()

    for company, sources in company_role_map.items():
        st.write(f"**{company}**")
        for src in sources:
            st.write(f"- {src}")

    # ---------------------------------------------------
    # Build context (safety capped)
    # ---------------------------------------------------
    context = "\n\n---\n\n".join(retrieved_chunks)

    MAX_CONTEXT_CHARS = 6000
    context = context[:MAX_CONTEXT_CHARS]

    # ---------------------------------------------------
    # Student-centric mentor prompt
    # ---------------------------------------------------
    prompt = f"""
You are a campus placement mentor helping freshers understand job opportunities.

Your audience:
- final year students
- fresh graduates
- candidates new to placements

Use ONLY the information present in the context below.

If a role focus is provided, prioritise and structure your answer mainly
around that role category.

Role focus selected by the student:
{role_focus}

Explain clearly and in simple language:

1. What TYPES of roles are appearing in these campus placements
2. Which COMPANIES are offering these roles
3. What students are GENERALLY EXPECTED to know or be good at
4. Typical responsibilities students will handle in their first year
5. A practical 4-week preparation plan for a student

Rules:
- Do not invent information
- Avoid HR jargon
- Be realistic and helpful for freshers
- Use bullet points
- If something is not clearly present in the context, say
  "Not clearly specified in the JDs"

Context:
{context}

Student question:
{query}

Answer:
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    # ---------------------------------------------------
    # Final output
    # ---------------------------------------------------
    st.subheader("🧠 Placement Insight for Students")
    st.write(response.content)
