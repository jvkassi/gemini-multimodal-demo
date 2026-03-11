import streamlit as st
import os
import tempfile
from typing import List, Dict, Any

from google import genai

from config import EMBEDDING_MODEL, SUPPORTED_EXTENSIONS
from utils import (
    upload_and_embed_file,
    compute_similarities,
    get_file_type_category,
    logger
)

st.set_page_config(
    page_title="Gemini Multimodal Search",
    page_icon="🔍",
    layout="wide"
)


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "library" not in st.session_state:
        st.session_state["library"] = []
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = None
    if "client" not in st.session_state:
        st.session_state["client"] = None


def get_api_key() -> bool:
    """Get API key from environment or user input."""
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        st.sidebar.header("🔑 API Configuration")
        api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")
        if not api_key:
            st.info("👆 Please enter your Gemini API Key to get started.")
            st.markdown("""
            ### How to get an API key:
            1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
            2. Create a new API key
            3. Paste it here
            """)
            return False
    
    st.session_state["api_key"] = api_key
    return True


def initialize_client() -> bool:
    """Initialize the Gemini client."""
    try:
        if not st.session_state["api_key"]:
            return False
        
        if st.session_state["client"] is None:
            st.session_state["client"] = genai.Client(api_key=st.session_state["api_key"])
        
        return True
    except Exception as e:
        st.error(f"Failed to initialize Gemini client: {e}")
        return False


def process_files(uploaded_files: List[Any]) -> None:
    """Process uploaded files and add them to the library."""
    if not uploaded_files:
        return

    client = st.session_state["client"]
    if not client:
        st.error("Gemini client not initialized")
        return

    existing_names = {item["name"] for item in st.session_state["library"]}
    files_to_process = [f for f in uploaded_files if f.name not in existing_names]
    
    if not files_to_process:
        st.sidebar.success("All files already in library!")
        return

    st.sidebar.markdown("---")
    st.sidebar.subheader("📤 Processing Files")
    
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    success_count = 0
    error_count = 0
    
    for i, uploaded_file in enumerate(files_to_process):
        progress = (i + 1) / len(files_to_process)
        progress_bar.progress(progress)
        
        suffix = os.path.splitext(uploaded_file.name)[1] or ""
        
        data_dir = "/data"
        os.makedirs(data_dir, exist_ok=True)
        tmp_file_path = os.path.join(data_dir, f"{int(time.time())}_{uploaded_file.name}")
        with open(tmp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            with st.status(f"Processing {uploaded_file.name}...", expanded=True) as status:
                embedding, file_uri = upload_and_embed_file(
                    client=client,
                    file_path=tmp_file_path,
                    display_name=uploaded_file.name,
                    model_name=EMBEDDING_MODEL,
                    progress_bar=progress_bar,
                    status_text=status_text
                )
                
                if embedding is not None:
                    st.session_state["library"].append({
                        "name": uploaded_file.name,
                        "type": uploaded_file.type,
                        "embedding": embedding,
                        "file_uri": file_uri
                    })
                    status.update(label=f"✅ {uploaded_file.name} added!", state="complete")
                    success_count += 1
                else:
                    status.update(label=f"❌ Failed: {uploaded_file.name}", state="error")
                    error_count += 1
                    
        except Exception as e:
            st.sidebar.error(f"Error processing {uploaded_file.name}: {e}")
            error_count += 1
            
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    progress_bar.empty()
    status_text.empty()
    
    if success_count > 0:
        st.sidebar.success(f"✅ Added {success_count} file(s) to library")
    if error_count > 0:
        st.sidebar.error(f"❌ {error_count} file(s) failed")


def render_library() -> None:
    """Render the current library in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Library")
    
    library_size = len(st.session_state["library"])
    st.sidebar.write(f"**{library_size} file(s)** in library")
    
    if library_size > 0:
        with st.sidebar.expander("View Library Contents", expanded=False):
            for i, item in enumerate(st.session_state["library"]):
                category = get_file_type_category(item["type"])
                st.write(f"{i+1}. {category}: **{item['name']}**")
        
        if st.sidebar.button("🗑️ Clear Library", use_container_width=True):
            st.session_state["library"] = []
            st.rerun()


def render_search_section() -> None:
    """Render the search interface."""
    st.markdown("---")
    st.subheader("🔍 Semantic Search")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., 'A picture of a dog', 'A video about coding', 'A document about finance'",
            label_visibility="collapsed"
        )
    with col2:
        search_button = st.button("Search", type="primary", use_container_width=True)
    
    if search_button and query:
        if not st.session_state["library"]:
            st.warning("⚠️ Your library is empty! Add some files first.")
            return
        
        if not st.session_state["client"]:
            st.error("Gemini client not initialized")
            return
        
        with st.spinner("Embedding query and finding matches..."):
            try:
                result = st.session_state["client"].models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=query
                )
                
                query_embedding = result.embeddings[0].values
                results = compute_similarities(query_embedding, st.session_state["library"])
                
                st.markdown("### 🎯 Top Matches")
                
                for i, (sim, item) in enumerate(results):
                    category = get_file_type_category(item["type"])
                    
                    col_score, col_info = st.columns([1, 3])
                    with col_score:
                        st.metric("Similarity", f"{sim:.3f}")
                    with col_info:
                        st.info(f"**#{i+1}: {item['name']}**  \nType: {category}")
                    
                    st.progress(sim, text=f"Match confidence: {sim*100:.1f}%")
                    st.markdown("---")
                    
            except Exception as e:
                st.error(f"Search failed: {e}")
                logger.error(f"Search error: {e}")


def main():
    """Main application entry point."""
    st.title("🔍 Gemini Multimodal Semantic Search")
    st.markdown("""
    Upload **images, PDFs, audio, or video** files to your library and perform 
    cross-modal semantic search using **Gemini Embedding 2.0**.
    """)
    
    init_session_state()
    
    if not get_api_key():
        return
    
    if not initialize_client():
        return
    
    with st.sidebar:
        st.header("📁 Add Files")
        uploaded_files = st.file_uploader(
            "Choose files to upload:",
            type=SUPPORTED_EXTENSIONS,
            accept_multiple_files=True,
            help=f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
        
        if uploaded_files and st.button("Process & Add to Library", type="primary"):
            process_files(uploaded_files)
    
    render_library()
    render_search_section()
    
    st.markdown("---")
    st.caption("Powered by Gemini Embedding 2.0 | Files are automatically deleted after embedding")


if __name__ == "__main__":
    main()
