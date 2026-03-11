import streamlit as st
import google.generativeai as genai
import os
import tempfile
import time
from io import BytesIO

st.set_page_config(page_title="Gemini Multimodal Search", page_icon="✨")

st.title("✨ Gemini Multimodal Search")
st.write("Upload a document, video, or audio file, and ask questions about its content!")

# Check for API Key
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.warning("GEMINI_API_KEY environment variable is not set. Please set it below or in your deployment variables.")
    api_key = st.text_input("Enter your Gemini API Key:", type="password")
    if not api_key:
        st.stop()

genai.configure(api_key=api_key)

# File uploader
uploaded_file = st.file_uploader(
    "Upload a file (PDF, TXT, MP3, MP4, WAV, JPG, PNG)", 
    type=["pdf", "txt", "mp3", "mp4", "wav", "jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Save the uploaded file temporarily
    suffix = "." + uploaded_file.name.split(".")[-1] if "." in uploaded_file.name else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    st.success(f"File '{uploaded_file.name}' loaded temporarily!")
    
    prompt = st.text_input("What do you want to know about this file?", "Summarize the key points.")
    
    if st.button("Search & Analyze") and prompt:
        with st.spinner("Uploading and processing with Gemini..."):
            gemini_file = None
            try:
                # Upload to Gemini File API
                gemini_file = genai.upload_file(path=tmp_file_path, display_name=uploaded_file.name)
                
                # Check processing state for videos
                if uploaded_file.type.startswith("video"):
                    st.info("Video detected. Waiting for Gemini to finish processing... (this might take a minute)")
                    while gemini_file.state.name == 'PROCESSING':
                        time.sleep(3)
                        gemini_file = genai.get_file(gemini_file.name)
                    if gemini_file.state.name == 'FAILED':
                        st.error("Video processing failed.")
                        st.stop()
                
                st.success("File processed! Generating answer...")
                
                # Use Gemini 1.5 Flash for multimodal context
                model = genai.GenerativeModel("models/gemini-1.5-flash")
                response = model.generate_content([gemini_file, prompt])
                
                st.write("### Analysis Results")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"Error during processing: {e}")
            finally:
                # Cleanup
                if gemini_file:
                    try:
                        genai.delete_file(gemini_file.name)
                    except Exception:
                        pass
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
