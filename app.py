import streamlit as st
import google.generativeai as genai
import os
import tempfile
import time
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Gemini Embedding 2.0 Search", page_icon="🔍")

st.title("🔍 Semantic Search with Gemini Embedding 2.0")
st.write("Upload a document, video, or audio file, and perform a semantic search over its content using `text-embedding-004` (Gemini's latest embedding model).")

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.warning("Please provide your Gemini API Key:")
    api_key = st.text_input("API Key:", type="password")
    if not api_key:
        st.stop()

genai.configure(api_key=api_key)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def extract_text(file_path, file_type, file_name):
    # For text/pdf, we could parse locally, but let's use Gemini 1.5 Flash to robustly extract/transcribe ALL modalities
    st.info("Extracting and transcribing content with Gemini 1.5 Flash...")
    gemini_file = genai.upload_file(path=file_path, display_name=file_name)
    
    # Wait for processing (for video/audio)
    if file_type.startswith("video") or file_type.startswith("audio"):
        st.info("Waiting for Google servers to process the media file...")
        while gemini_file.state.name == 'PROCESSING':
            time.sleep(3)
            gemini_file = genai.get_file(gemini_file.name)
            
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    prompt = "Please provide a highly detailed, comprehensive transcript and description of this file. Include all text, speech, and key visual details if it's a video."
    response = model.generate_content([gemini_file, prompt])
    
    genai.delete_file(gemini_file.name)
    return response.text

@st.cache_data(show_spinner=False)
def get_embeddings(chunks):
    st.info(f"Generating embeddings using models/text-embedding-004 for {len(chunks)} chunks...")
    embeddings = []
    # Batch embedding
    for chunk in chunks:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=chunk,
            task_type="RETRIEVAL_DOCUMENT"
        )
        embeddings.append(result['embedding'])
    return embeddings

# File uploader
uploaded_file = st.file_uploader(
    "Upload a file (PDF, TXT, MP3, MP4, WAV, JPG, PNG)", 
    type=["pdf", "txt", "mp3", "mp4", "wav", "jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    suffix = "." + uploaded_file.name.split(".")[-1] if "." in uploaded_file.name else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    if "extracted_text" not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
        with st.spinner("Processing file..."):
            try:
                text = extract_text(tmp_file_path, uploaded_file.type, uploaded_file.name)
                st.session_state["extracted_text"] = text
                
                # Chunking the text (simple overlap chunking)
                words = text.split()
                chunk_size = 150
                overlap = 30
                chunks = []
                for i in range(0, len(words), chunk_size - overlap):
                    chunk = " ".join(words[i:i + chunk_size])
                    if chunk:
                        chunks.append(chunk)
                
                st.session_state["chunks"] = chunks
                st.session_state["embeddings"] = get_embeddings(chunks)
                st.session_state["last_file"] = uploaded_file.name
                
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.stop()
            finally:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

    st.success("File processed and embedded successfully! ✅")
    
    with st.expander("Show Extracted/Transcribed Text"):
        st.write(st.session_state["extracted_text"])

    st.markdown("---")
    st.subheader("Semantic Search")
    query = st.text_input("Search the document/video/audio:")
    
    if st.button("Search") and query:
        with st.spinner("Embedding query and calculating similarity..."):
            query_embedding = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="RETRIEVAL_QUERY"
            )['embedding']
            
            # Calculate similarities
            similarities = []
            for i, doc_emb in enumerate(st.session_state["embeddings"]):
                sim = cosine_similarity(query_embedding, doc_emb)
                similarities.append((sim, st.session_state["chunks"][i]))
                
            # Sort by similarity
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            st.write("### Top Results:")
            for i, (sim, chunk) in enumerate(similarities[:3]):
                st.info(f"**Result {i+1} (Confidence: {sim:.2f})**\n\n{chunk}")
