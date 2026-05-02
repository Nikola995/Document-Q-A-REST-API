import streamlit as st
import requests
import os

HIGHLIGHT_COLORS = [
    "#AED6F1",
    "#A9DFBF",
    "#F9E79F",
    "#FAD7A0",
    "#D7BDE2",
    "#F1948A",
    "#A8D8EA",
    "#F0B27A",
    "#A3E4D7",
    "#F8C471",
    "#C39BD3",
    "#76D7C4",
    "#F1948A",
    "#85C1E9",
    "#82E0AA",
    "#F7DC6F",
    "#E59866",
    "#AF7AC5",
    "#48C9B0",
    "#EC7063",
]


def get_label_colors(entities: list[dict]) -> dict[str, str]:
    unique_labels = list(dict.fromkeys(e["label"] for e in entities))
    return {
        label: HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
        for i, label in enumerate(unique_labels)
    }


def highlight_entities(text: str, entities: list[dict], label_colors: dict) -> str:
    sorted_entities = sorted(entities, key=lambda e: len(e["text"]), reverse=True)
    for entity in sorted_entities:
        color = label_colors.get(entity["label"], "#D5D8DC")
        highlighted = (
            f'<mark style="background-color: {color}; padding: 2px 4px; '
            f'border-radius: 3px;" title="{entity["label"]}">{entity["text"]}</mark>'
        )
        text = text.replace(entity["text"], highlighted)
    return text


# Covers running via docker-compose and locally as fallback
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="centered")
st.title("📄 Document Q&A")
st.caption("Upload a PDF or image, then ask questions about its content.")

# --- Session state ---
st.session_state.setdefault("session_id", None)
st.session_state.setdefault("filenames", [])

# --- Upload section ---
st.subheader("1. Upload a document")
uploaded_files = st.file_uploader(
    "Choose a PDF or image",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "webp"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Upload & Index"):
    with st.spinner("Extracting and indexing..."):
        resp = requests.post(
            f"{API_BASE}/upload",
            data={"session_id": st.session_state.session_id or None},
            files=[("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files],
        )
    if resp.ok:
        data = resp.json()
        st.session_state.session_id = data["session_id"]
        for file_result in data["files"]:
            if file_result["filename"] not in st.session_state.filenames:
                st.session_state.filenames.append(file_result["filename"])

        new_files = [f for f in data["files"] if not f["already_exists"]]
        duplicates = [f for f in data["files"] if f["already_exists"]]

        if new_files:
            st.success(f"✅ Indexed: {', '.join(f['filename'] for f in new_files)}")
        if duplicates:
            st.warning(
                f"⚠️ Already indexed: {', '.join(f['filename'] for f in duplicates)}"
            )
    else:
        detail = resp.json().get("detail", resp.text)
        if isinstance(detail, list):
            for error in detail:
                st.error(error)
        else:
            st.error(f"Upload failed: {detail}")

# --- Q&A section ---
if st.session_state.session_id:
    st.divider()
    col1, col2 = st.columns([6, 1])
    with col1:
        st.caption(f"Session: `{st.session_state.session_id}`")
        if st.session_state.filenames:
            st.caption(f"Documents: {', '.join(st.session_state.filenames)}")
    with col2:
        if st.button("🔄 New", help="Clear current session and start over"):
            st.session_state.clear()
            st.rerun()
    st.divider()
    with st.form("qa_form", clear_on_submit=False):
        question = st.text_input(
            "Your question", placeholder="What is the main topic of this document?"
        )
        submitted = st.form_submit_button("Ask")

    if submitted and question:
        with st.spinner("Thinking..."):
            resp = requests.post(
                f"{API_BASE}/ask",
                json={"session_id": st.session_state.session_id, "question": question},
            )
        if resp.ok:
            data = resp.json()
            entities = data.get("answer_entities", [])
            label_colors = get_label_colors(entities)

            st.markdown("**📄 Retrieved context**")
            st.caption(f"Source: `{data['context_chunk']['filename']}`")
            st.markdown(
                highlight_entities(
                    data["context_chunk"]["text"], entities, label_colors
                ),
                unsafe_allow_html=True,
            )
            if data["context_chunk"].get("score") is not None:
                st.caption(f"Relevance score: {data['context_chunk']['score']:.3f}")

            st.divider()

            st.markdown("**💬 Answer**")
            st.markdown(
                highlight_entities(data["answer"], entities, label_colors),
                unsafe_allow_html=True,
            )

            if entities:
                with st.expander("Detected entities"):
                    for entity in entities:
                        color = label_colors.get(entity["label"], "#D5D8DC")
                        st.markdown(
                            f'<span style="background-color: {color}; padding: 2px 6px; '
                            f'border-radius: 3px;">{entity["text"]}</span> — {entity["label"]}',
                            unsafe_allow_html=True,
                        )
        else:
            st.error(f"Request failed: {resp.json().get('detail', resp.text)}")
else:
    st.info("Upload a document above to get started.")
