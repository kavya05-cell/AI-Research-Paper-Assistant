"""Streamlit entry point for the AI Research Paper Assistant."""

from pathlib import Path

import streamlit as st

from modules.embedding_manager import DEFAULT_MODEL_NAME, generate_embeddings
from modules.pdf_processor import PDFProcessingError, extract_pdf_pages
from modules.rag_chain import answer_question
from modules.text_chunker import chunk_pages
from modules.vector_store import FaissVectorStore


VECTOR_STORE_DIR = Path("data/vector_store")


def main() -> None:
    st.set_page_config(
        page_title="AI Research Paper Assistant",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 AI Research Paper Assistant")
    st.caption("Understand research papers with Retrieval-Augmented Generation.")

    st.info(
        "Upload papers, build a searchable vector index, and ask questions. "
        "Answers are generated from retrieved passages with page citations."
    )

    with st.sidebar:
        st.header("About")
        st.write(
            "A source-grounded research assistant. Answers should be supported "
            "by passages from your uploaded PDFs."
        )
        st.divider()
        st.caption("Embedding model: all-MiniLM-L6-v2")
        st.caption("LLM: Gemini 2.5 Flash")
        st.caption("Development build")

    st.subheader("Research library")
    uploaded_files = st.file_uploader(
        "Select research papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} PDF file(s).")
        if st.button("Process PDFs", type="primary"):
            all_pages = []
            errors = []
            with st.spinner("Extracting text from your PDFs..."):
                for uploaded_file in uploaded_files:
                    try:
                        all_pages.extend(
                            extract_pdf_pages(
                                uploaded_file.getvalue(),
                                uploaded_file.name,
                            )
                        )
                    except PDFProcessingError as exc:
                        errors.append(str(exc))

            st.session_state["extracted_pages"] = all_pages
            st.session_state.pop("text_chunks", None)
            st.session_state.pop("vector_store", None)
            st.session_state.pop("last_answer", None)

            if all_pages:
                st.success(f"Extracted text from {len(all_pages)} page(s).")
            for error in errors:
                st.error(error)
    else:
        st.write("Upload one or more research papers to get started.")

    extracted_pages = st.session_state.get("extracted_pages", [])
    if extracted_pages:
        with st.expander(f"Extracted pages ({len(extracted_pages)})"):
            for page_record in extracted_pages:
                metadata = page_record["metadata"]
                st.markdown(
                    f'**{metadata["source"]} — Page {metadata["page"]}**'
                )
                st.text(page_record["text"])

        st.subheader("Text chunking")
        st.caption(
            "800-token chunks with 120-token overlap; pages are split independently."
        )
        if st.button("Generate text chunks"):
            st.session_state["text_chunks"] = chunk_pages(extracted_pages)
            st.session_state.pop("vector_store", None)
            st.session_state.pop("last_answer", None)

        text_chunks = st.session_state.get("text_chunks", [])
        if text_chunks:
            st.success(f"Generated {len(text_chunks)} text chunk(s).")
            with st.expander("Preview first 10 chunks"):
                for index, chunk in enumerate(text_chunks[:10], start=1):
                    metadata = chunk["metadata"]
                    st.markdown(
                        f'**Chunk {index} · {metadata.get("source")} · '
                        f'Page {metadata.get("page")}**'
                    )
                    st.write(chunk["text"])

            st.subheader("Embedding generation and vector index")
            st.write(
                "Each chunk is converted into a numerical vector for semantic search."
            )

            if st.button("Generate embeddings and build search index", type="primary"):
                try:
                    texts = [chunk["text"] for chunk in text_chunks]
                    with st.spinner(
                        "Generating embeddings. The first run may download model files..."
                    ):
                        embeddings = generate_embeddings(
                            texts,
                            model_name=DEFAULT_MODEL_NAME,
                        )
                        store = FaissVectorStore()
                        store.build(embeddings, text_chunks)
                        store.save(VECTOR_STORE_DIR)

                    st.session_state["vector_store"] = store
                    st.success(
                        f"Created {embeddings.shape[0]} embeddings "
                        f"with {embeddings.shape[1]} dimensions and saved "
                        "the vector index locally."
                    )
                except Exception as exc:
                    st.error(f"Could not build the vector index: {exc}")

            store = st.session_state.get("vector_store")
            if store is not None:
                st.subheader("Ask your research papers")
                st.write(
                    "Gemini answers using retrieved passages. Each answer should "
                    "include document and page citations."
                )
                question = st.text_area(
                    "Your question",
                    placeholder="e.g. What method does the paper propose for detecting network intrusions?",
                    height=100,
                )
                top_k = st.slider(
                    "Retrieved passages",
                    min_value=1,
                    max_value=10,
                    value=5,
                    key="rag_top_k",
                )

                if st.button("Generate grounded answer", type="primary"):
                    if not question.strip():
                        st.warning("Enter a question first.")
                    else:
                        try:
                            with st.spinner(
                                "Retrieving relevant passages and asking Gemini..."
                            ):
                                query_vector = generate_embeddings(
                                    [question],
                                    model_name=DEFAULT_MODEL_NAME,
                                )
                                retrieved = store.search(
                                    query_vector,
                                    top_k=top_k,
                                )
                                answer = answer_question(question, retrieved)

                            st.session_state["last_answer"] = {
                                "question": question,
                                "answer": answer,
                                "sources": retrieved,
                            }
                        except Exception as exc:
                            st.error(f"Could not generate an answer: {exc}")

                last_answer = st.session_state.get("last_answer")
                if last_answer:
                    st.markdown("### Answer")
                    st.markdown(last_answer["answer"])

                    with st.expander("Retrieved source passages"):
                        for rank, result in enumerate(
                            last_answer["sources"], start=1
                        ):
                            metadata = result.get("metadata", {})
                            st.markdown(
                                f'**{rank}. {metadata.get("source", "Unknown")} '
                                f'— Page {metadata.get("page", "?")} '
                                f'— Similarity {result.get("similarity", 0):.3f}**'
                            )
                            st.write(result.get("text", ""))


if __name__ == "__main__":
    main()
