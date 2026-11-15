"""Streamlit entry point for the AI Research Paper Assistant."""

import streamlit as st

from modules.pdf_processor import PDFProcessingError, extract_pdf_pages


def main() -> None:
    st.set_page_config(
        page_title="AI Research Paper Assistant",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 AI Research Paper Assistant")
    st.caption("Understand research papers with Retrieval-Augmented Generation.")

    st.info(
        "PDF text extraction is now available. Text chunking, embeddings, "
        "retrieval, and question answering are the next milestones."
    )

    with st.sidebar:
        st.header("About")
        st.write(
            "Upload research papers, extract their text, and then ask "
            "questions grounded in the source documents."
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

    if not uploaded_files:
        st.write("Upload one or more research papers to get started.")
        return

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

        if all_pages:
            st.success(
                f"Successfully extracted {len(all_pages)} page(s) "
                f"from the uploaded document(s)."
            )
        for error in errors:
            st.error(error)

    extracted_pages = st.session_state.get("extracted_pages", [])
    if extracted_pages:
        st.subheader("Extraction results")
        st.write(f"Extracted text from {len(extracted_pages)} page(s).")

        for page_record in extracted_pages:
            metadata = page_record["metadata"]
            with st.expander(
                f'{metadata["source"]} — Page {metadata["page"]}'
            ):
                st.text_area(
                    "Extracted text",
                    value=page_record["text"],
                    height=220,
                    key=f'preview_{metadata["source"]}_{metadata["page"]}',
                    disabled=True,
                )


if __name__ == "__main__":
    main()
