"""Streamlit entry point for the AI Research Paper Assistant.

This is the initial UI scaffold. PDF ingestion, retrieval, and LLM
answer generation will be added in subsequent implementation steps.
"""

import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="AI Research Paper Assistant",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 AI Research Paper Assistant")
    st.caption("Understand research papers with Retrieval-Augmented Generation.")

    st.info(
        "Project scaffold is ready. Document processing and RAG functionality "
        "are the next implementation milestones."
    )

    with st.sidebar:
        st.header("About")
        st.write(
            "Upload research papers, explore their contents, and ask "
            "questions grounded in the source documents."
        )
        st.divider()
        st.caption("Development build")

    st.subheader("Research library")
    uploaded_files = st.file_uploader(
        "Select research papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="PDF processing will be connected in the document-ingestion milestone.",
    )

    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} PDF file(s):")
        for uploaded_file in uploaded_files:
            st.write(f"- {uploaded_file.name} ({uploaded_file.size:,} bytes)")
        st.warning(
            "Files are currently selected in this session only. "
            "They are not processed or stored yet."
        )
    else:
        st.write("Upload one or more research papers to get started.")


if __name__ == "__main__":
    main()
