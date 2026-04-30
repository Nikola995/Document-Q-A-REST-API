import streamlit as st
import requests
import os

# Covers running via docker-compose and locally as fallback
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="centered")
st.title("📄 Document Q&A")
st.caption("Upload a PDF or image, then ask questions about its content.")

# --- Session state ---
st.session_state.setdefault("document_id", None)
st.session_state.setdefault("filename", None)

# --- Upload section ---
st.subheader("1. Upload a document")
uploaded_file = st.file_uploader(
    "Choose a PDF or image", type=["pdf", "png", "jpg", "jpeg", "tiff", "webp"]
)

if uploaded_file and st.button("Upload & Index"):
    with st.spinner("Extracting and indexing..."):
        resp = requests.post(
            f"{API_BASE}/upload",
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            },
        )
    if resp.ok:
        data = resp.json()
        st.session_state.document_id = data["document_id"]
        st.session_state.filename = data["filename"]
        if data.get("already_exists"):
            st.warning(
                f"⚠️ **{data['filename']}** was already indexed. Using existing document."
            )
        else:
            st.success(f"✅ **{data['filename']}** indexed successfully.")
    else:
        st.error(f"Upload failed: {resp.json().get('detail', resp.text)}")

# --- Q&A section ---
if st.session_state.document_id:
    st.divider()
    col1, col2 = st.columns([6, 1])
    with col1:
        st.subheader("2. Ask a question")
        st.caption(
            f"Document: `{st.session_state.filename}` · ID: `{st.session_state.document_id}`"
        )

        with st.form("qa_form", clear_on_submit=False):
            question = st.text_input(
                "Your question", placeholder="What is the main topic of this document?"
            )
            submitted = st.form_submit_button("Ask")

        if submitted and question:
            with st.spinner("Thinking..."):
                resp = requests.post(
                    f"{API_BASE}/ask",
                    json={
                        "document_id": st.session_state.document_id,
                        "question": question,
                    },
                )
            if resp.ok:
                data = resp.json()
                context = data["context"]

                # Context first — framed as "retrieved passage"
                st.markdown("**📄 Retrieved context**")
                st.info(
                    context["text"]
                )  # info box gives it a distinct visual treatment
                if context.get("score") is not None:
                    st.caption(f"Relevance score: {context['score']:.3f}")

                st.divider()

                # Answer below
                st.markdown("**💬 Answer**")
                st.success(data["answer"])  # success box distinguishes it from context
            else:
                st.error(f"Request failed: {resp.json().get('detail', resp.text)}")
    with col2:
        if st.button("🔄 New", help="Clear current document and start over"):
            st.session_state.clear()
            st.rerun()
else:
    st.info("Upload a document above to get started.")
