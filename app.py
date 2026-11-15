"""Streamlit entry point for the AI Research Paper Assistant."""

from pathlib import Path

import streamlit as st

from modules.embedding_manager import (
    DEFAULT_MODEL_NAME,
    generate_embeddings,
)
from modules.pdf_processor import PDFProcessingError, extract_pdf_pages
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
        "PDF extraction, chunking, embeddings, and vector-based semantic search "
        "are available. LLM-based answer generation is the next milestone."
    )

    with st.sidebar:
        st.header("About")
        st.write(
            "Upload research papers, create a searchable vector index, "
            "and retrieve relevant passages."
        )
        st.divider()
        st.caption(f"Embedding model: all-MiniLM-L6-v2")
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
        st.caption("800-token chunks with 120-token overlap; pages are split independently.")
        if st.button("Generate text chunks"):
            st.session_state["text_chunks"] = chunk_pages(extracted_pages)
            st.session_state.pop("vector_store", None)

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
                "The embedding model converts each chunk into a numerical "
                "vector. FAISS stores those vectors for similarity search."
            )

            if st.button("Generate embeddings and build search index", type="primary"):
                try:
                    texts = [chunk["text"] for chunk in text_chunks]
                    with st.spinner(
                        "Loading the embedding model and generating vectors. "
                        "The first run may download model files..."
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
                        "the FAISS index locally."
                    )
                except Exception as exc:
                    st.error(f"Could not build the vector index: {exc}")

            store = st.session_state.get("vector_store")
            if store is not None:
                st.subheader("Test semantic search")
                query = st.text_input(
                    "Ask a search question about your uploaded papers",
                    placeholder="e.g. How does the paper detect network intrusions?",
                )
                top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)

                if st.button("Search relevant chunks"):
                    if not query.strip():
                        st.warning("Enter a search question first.")
                    else:
                        try:
                            with st.spinner("Searching the vector index..."):
                                query_vector = generate_embeddings(
                                    [query],
                                    model_name=DEFAULT_MODEL_NAME,
                                )
                                results = store.search(query_vector, top_k=top_k)

                            for rank, result in enumerate(results, start=1):
                                metadata = result["metadata"]
                                with st.expander(
                                    f'#{rank} · {metadata.get("source")} · '
                                    f'Page {metadata.get("page")} · '
                                    f'Similarity {result["similarity"]:.3f}',
                                    expanded=(rank == 1),
                                ):
                                    st.write(result["text"])
                        except Exception as exc:
                            st.error(f"Semantic search failed: {exc}")


if __name__ == "__main__":
    main()
