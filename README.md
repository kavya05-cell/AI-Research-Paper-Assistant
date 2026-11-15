# 📚 AI Research Paper Assistant

### An intelligent research paper understanding and analysis system using RAG and LLMs

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![AI](https://img.shields.io/badge/Domain-RAG%20%7C%20LLMs-purple)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)

## Overview

AI Research Paper Assistant is an application designed to help students and researchers read, understand, and analyse scientific research papers.

Users will be able to upload PDF papers and ask questions in natural language. The system will retrieve relevant passages from the uploaded documents and provide them to a Large Language Model (LLM), which generates an answer grounded in the retrieved context.

The project uses **Retrieval-Augmented Generation (RAG)** rather than relying only on an LLM's pre-trained knowledge. Planned capabilities include paper summarisation, structured information extraction, multi-paper comparison, and source references.

> **Project status:** This repository is being built incrementally. The RAG pipeline and research-analysis features are not implemented yet.

## Objectives

- Build a RAG-based question-answering system for research papers.
- Enable semantic search over uploaded documents.
- Generate context-grounded answers using an LLM.
- Summarise research papers and extract key information.
- Compare multiple papers using their documented contents.
- Show document and page references so users can verify answers.

## Planned Features

| Feature | Description |
|---|---|
| PDF upload | Upload one or more research papers. |
| Research chat | Ask natural-language questions about uploaded papers. |
| Semantic retrieval | Find relevant passages using vector embeddings. |
| Paper summarisation | Generate structured summaries of research papers. |
| Information extraction | Extract objectives, methods, datasets, findings, and limitations. |
| Paper comparison | Compare the approaches and findings of multiple papers. |
| Research-gap exploration | Surface possible gaps based on stated limitations and future-work sections. |
| Source references | Show the source document and page number supporting retrieved context. |

Research-gap suggestions are intended as preliminary reading assistance, not proof of a novel research contribution.

## System Architecture

### 1. Document ingestion

~~~text
Research paper (PDF)
        |
        v
   Text extraction
        |
        v
  Text preprocessing
        |
        v
    Text chunking
        |
        v
 Embedding generation
        |
        v
    FAISS index
~~~ 

### 2. Question answering

~~~text
    User question
          |
          v
    Query embedding
          |
          v
   Semantic retrieval
          |
          v
  Relevant text chunks
          |
          v
   Prompt construction
          |
          v
       Gemini LLM
          |
          v
   Grounded response
          |
          v
    Source references
~~~

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web interface |
| LangChain | RAG pipeline components and orchestration |
| Google Gemini API | LLM-based response generation |
| Sentence Transformers | Text embedding generation |
| FAISS | Local vector indexing and similarity search |
| PyMuPDF | PDF text extraction |
| Git and GitHub | Version control and team collaboration |

The exact model, package versions, chunking strategy, and retrieval configuration will be finalised and documented during implementation.

## Repository Structure

~~~text
AI-Research-Paper-Assistant/
├── app.py
├── config/
│   └── settings.py
├── data/
│   ├── uploaded_papers/
│   └── vector_store/
├── modules/
│   ├── pdf_processor.py
│   ├── text_chunker.py
│   ├── embeddings.py
│   ├── vector_database.py
│   ├── retriever.py
│   ├── llm_handler.py
│   ├── summarizer.py
│   └── paper_comparator.py
├── prompts/
│   └── templates.py
├── utils/
│   └── helpers.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
~~~

Some modules shown above are planned and will be added as their functionality is implemented.

## Setup

### Prerequisites

- Python 3.10 or later
- Git
- A Google AI Studio API key for Gemini-based generation

### 1. Clone the repository

~~~bash
git clone https://github.com/kavya05-cell/AI-Research-Paper-Assistant.git
cd AI-Research-Paper-Assistant
~~~

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

~~~powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
~~~

**macOS / Linux:**

~~~bash
python3 -m venv .venv
source .venv/bin/activate
~~~

### 3. Install dependencies

~~~bash
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

### 4. Configure environment variables

Copy .env.example to .env and add your API key:

~~~env
GOOGLE_API_KEY=your_google_ai_studio_api_key
~~~

Never commit the real .env file or expose API keys in source code, screenshots, logs, or public issues.

### 5. Run the application

~~~bash
streamlit run app.py
~~~

## Team Responsibilities

| Team member | Area |
|---|---|
| Member 1 | PDF processing and document metadata |
| Member 2 | Embeddings and vector database |
| Member 3 | Retrieval, prompts, and LLM integration |
| Member 4 | Streamlit interface and system integration |

All team members should participate in code reviews, integration, testing, and project documentation.

## Development Roadmap

- [ ] Repository setup and application scaffold
- [ ] PDF upload and page-aware text extraction
- [ ] Text cleaning and chunking
- [ ] Embedding generation
- [ ] FAISS indexing and persistence
- [ ] Semantic retrieval
- [ ] Gemini integration and grounded question answering
- [ ] Source and page references
- [ ] Research paper summarisation
- [ ] Multi-paper comparison
- [ ] Research-gap exploration
- [ ] Retrieval and answer evaluation
- [ ] Final testing and deployment

## Evaluation Plan

The system will be tested using a small set of research papers and manually prepared questions with known supporting passages. Evaluation will consider:

- Relevance of retrieved passages (such as Precision@K and Recall@K).
- Whether generated claims are supported by retrieved context (faithfulness).
- Relevance and completeness of answers.
- Correctness of document and page references.
- Response time and error handling.

## Future Enhancements

- OCR for scanned PDFs.
- Improved extraction of tables, equations, and figures.
- Hybrid keyword and vector search.
- Reranking of retrieved passages.
- Exportable summaries and comparison reports.
- Integration with open academic-paper repositories.

## Academic Use

This project is developed as an academic mini project in the areas of Artificial Intelligence, Machine Learning, Natural Language Processing, Generative AI, Large Language Models, and Retrieval-Augmented Generation.

## License

No open-source licence has been selected yet. Until a licence is added, all rights remain with the repository contributors.
