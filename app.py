import streamlit as st
import os
import tempfile
import time
import numpy as np

from google import genai
from google.genai import types

st.set_page_config(page_title="Gemini Embedding 2.0 Search", page_icon="🔍")

st.title("🔍 Multimodal Semantic Search")
st.write("Upload images, documents, audio, or video files to your library, and perform cross-modal semantic search using the new **Gemini Embedding 2.0** (`gemini-embedding-2-preview`) model.")

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.warning("Please provide your Gemini API Key:")
    api_key = st.text_input("API Key:", type="password")
    if not api_key:
        st.stop()

# Initialize the new SDK client
client = genai.Client(api_key=api_key)

def cosine_similarity(a, b):
    # Ensure a and b are 1-D numpy arrays of floats
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if "library" not in st.session_state:
    st.session_state["library"] = []

st.sidebar.header("📁 Add to Library")
uploaded_files = st.sidebar.file_uploader(
    "Upload files (PDF, TXT, MP3, MP4, WAV, JPG, PNG)", 
    type=["pdf", "txt", "mp3", "mp4", "wav", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if st.sidebar.button("Process & Embed Files") and uploaded_files:
    with st.spinner("Processing files and generating multimodal embeddings..."):
        for uploaded_file in uploaded_files:
            if any(item['name'] == uploaded_file.name for item in st.session_state["library"]):
                continue
                
            suffix = "." + uploaded_file.name.split(".")[-1] if "." in uploaded_file.name else ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                st.toast(f"Uploading {uploaded_file.name} to Gemini...")
                g_file = client.files.upload(file=tmp_file_path, config={"display_name": uploaded_file.name})
                
                # Wait for processing if video/audio
                if uploaded_file.type.startswith("video") or uploaded_file.type.startswith("audio") or uploaded_file.type == "application/pdf":
                    st.toast(f"Waiting for {uploaded_file.name} to finish processing...")
                    while True:
                        state_val = getattr(g_file, 'state', None)
                        if state_val:
                            state_str = str(state_val)
                            if 'PROCESSING' not in state_str:
                                break
                        else:
                            break
                        time.sleep(3)
                        g_file = client.files.get(name=g_file.name)
                        
                st.toast(f"Embedding {uploaded_file.name}...")
                result = client.models.embed_content(
                    model='gemini-embedding-2-preview',
                    contents=g_file
                )
                
                # Access embeddings array
                embedding_vector = result.embeddings[0].values
                
                # Store in library
                st.session_state["library"].append({
                    "name": uploaded_file.name,
                    "type": uploaded_file.type,
                    "embedding": embedding_vector,
                    "file_uri": g_file.uri
                })
                
            except Exception as e:
                st.sidebar.error(f"Error processing {uploaded_file.name}: {e}")
            finally:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

        st.sidebar.success("Library updated!")

st.write(f"📚 **Current Library Size:** {len(st.session_state['library'])} files")

st.markdown("---")
st.subheader("🔍 Search Your Library")
query = st.text_input("Describe what you're looking for (e.g., 'A picture of a dog', 'A video about coding', 'A document about finance'):")

if st.button("Search") and query:
    if not st.session_state["library"]:
        st.warning("Please add some files to your library first!")
    else:
        with st.spinner("Embedding query and finding matches..."):
            query_result = client.models.embed_content(
                model='gemini-embedding-2-preview',
                contents=query
            )
            query_embedding = query_result.embeddings[0].values
            
            similarities = []
            for item in st.session_state["library"]:
                sim = cosine_similarity(query_embedding, item["embedding"])
                similarities.append((sim, item))
                
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            st.write("### Top Matches:")
            for i, (sim, item) in enumerate(similarities[:5]):
                st.info(f"**#{i+1}: {item['name']}** (Similarity: {sim:.3f} | Type: {item['type']})")

