"""Streamlit entry point for the AI Research Paper Assistant."""

import streamlit as st

from modules.pdf_processor import PDFProcessingError, extract_pdf_pages
from modules.text_chunker import chunk_pages


def main() -> None:
    st.set_page_config(
        page_title="AI Research Paper Assistant",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 AI Research Paper Assistant")
    st.caption("Understand research papers with Retrieval-Augmented Generation.")

    st.info(
        "PDF extraction and page-aware text chunking are available. "
        "Embedding generation and retrieval are the next milestones."
    )

    with st.sidebar:
        st.header("About")
        st.write(
            "Upload research papers, extract their text, and prepare "
            "source-aware chunks for semantic search."
        )
        st.divider()
        st.caption("Development build")

    st.subheader("Research library")
    uploaded_files = st.file_uploader(
        "Select research papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more text-based research papers.",
    )

    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} PDF file(s).")
        if st.button("Process PDFs", type="primary"):
            all_pages = []
            errors = []

            with st.spinner("Extracting text from your PDFs..."):
                for uploaded_file in uploaded_files:
                    try:
                        pages = extract_pdf_pages(
                            uploaded_file.getvalue(),
                            uploaded_file.name,
                        )
                        all_pages.extend(pages)
                    except PDFProcessingError as exc:
                        errors.append(str(exc))

            st.session_state["extracted_pages"] = all_pages
            st.session_state.pop("text_chunks", None)

            if all_pages:
                st.success(
                    f"Successfully extracted {len(all_pages)} page(s) "
                    "from the uploaded document(s)."
                )
            for error in errors:
                st.error(error)
    else:
        st.write("Upload one or more research papers to get started.")

    extracted_pages = st.session_state.get("extracted_pages", [])
    if extracted_pages:
        st.subheader("Extraction results")
        st.write(f"Extracted text from {len(extracted_pages)} page(s).")

        with st.expander("Preview extracted pages"):
            for page_record in extracted_pages:
                metadata = page_record["metadata"]
                st.markdown(
                    f'**{metadata["source"]} — Page {metadata["page"]}**'
                )
                st.text_area(
                    "Extracted text",
                    value=page_record["text"],
                    height=180,
                    key=f'page_{metadata["source"]}_{metadata["page"]}',
                    disabled=True,
                )

        st.subheader("Text chunking")
        st.caption(
            "Chunks are created separately for each page to preserve "
            "accurate source-page citations."
        )
        if st.button("Generate text chunks"):
            with st.spinner("Splitting extracted text into chunks..."):
                st.session_state["text_chunks"] = chunk_pages(extracted_pages)

        text_chunks = st.session_state.get("text_chunks", [])
        if text_chunks:
            st.success(f"Generated {len(text_chunks)} text chunk(s).")
            st.caption("Configuration: 800 tokens per chunk, 120-token overlap.")
            for index, chunk in enumerate(text_chunks[:10], start=1):
                metadata = chunk["metadata"]
                label = (
                    f'Chunk {index} · {metadata.get("source", "Unknown source")} '
                    f'· Page {metadata.get("page", "?")}'
                )
                with st.expander(label):
                    st.write(chunk["text"])
            if len(text_chunks) > 10:
                st.caption(
                    f"Showing the first 10 of {len(text_chunks)} chunks."
                )


if __name__ == "__main__":
    main()
