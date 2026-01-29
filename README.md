# Job Description Q&A using RAG (FAISS + Groq)

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system
that enables semantic question answering over Job Descriptions.
It uses FAISS for vector-based retrieval and Groq-hosted LLMs for
low-latency, context-aware answer generation.

Users can upload a Job Description and ask natural language questions
to get accurate answers grounded strictly in the document content.

---

## Tech Stack
- **FAISS** – Vector similarity search
- **SentenceTransformers** – Text embeddings
- **Groq (LLaMA 3)** – Large Language Model inference
- **LangChain** – LLM orchestration
- **Streamlit** – Web application hosting

---

## Architecture
1. Job Description is uploaded by the user
2. Text is split into overlapping chunks
3. Embeddings are generated locally
4. FAISS performs top-K semantic retrieval
5. Retrieved context is passed to Groq LLM
6. LLM generates grounded answers

