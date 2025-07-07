import streamlit as st
from openai import OpenAI
import os
import tempfile
import time
from pytubefix import YouTube
from pytubefix.cli import on_progress
import re
import librosa
import soundfile as sf
import nemo.collections.asr as nemo_asr
import glob
from tqdm import tqdm
from uuid import uuid4 
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
import numpy as np
from sentence_transformers import SentenceTransformer

@st.cache_resource()
def load_model():
    transcriber = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    return transcriber

@st.cache_resource() 
def load_embedder(): 
    embedder = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
    return embedder

# Initialize cached models
transcriber = load_model()
embedder = load_embedder()

qdrant_client = QdrantClient(host="localhost", port=6333)
# Configure Streamlit page
st.set_page_config(
    page_title="YouTube Audio Analysis",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = ""
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'current_step' not in st.session_state:
    st.session_state.current_step = "" 
if "video_title" not in st.session_state:
    st.session_state.video_title = ""
    


def create_collection(collection_name: str):
    if not qdrant_client.collection_exists(collection_name):
        print(f"Creating collection: {collection_name}")
        qdrant_client.create_collection(
            collection_name=collection_name,
             vectors_config=models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE
                ),
            hnsw_config=models.HnswConfigDiff(
                m=64,
                ef_construct=200
            ),
        )
        print(f"Collection {collection_name} created successfully")
    else:
        print(f"Collection {collection_name} already exists")

def store_transcript(collection_name: str, transcript: str, video_title: str, index: int): 
    dense_embedding = embedder.encode(
        sentences=[transcript],
        task="retrieval.query",
    ).tolist()[0]
    
    points = [
        models.PointStruct(
            id = str(uuid4()),
            vector = dense_embedding,
            payload = {
                "transcript": transcript,
                "index": index,
                "video_title": video_title
            }
        )
    ]
    
    res = qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    if res.status.value == 'completed':
        print(f"Transcript {index} stored successfully in collection {collection_name}")
    else:
        print(f"Failed to store transcript {index} in collection {collection_name}")
        
def chunk_audio(temp_dir, audio_file_path: str, chunk_duration: int = 120, overlap_duration: int = 20):
    chunks_dir = os.path.join(temp_dir, "audio_chunks")
    audio_file_name = os.path.basename(audio_file_path).split(".")[0] 
    os.makedirs(chunks_dir, exist_ok=True)
    
    audio, sr = librosa.load(audio_file_path, sr=None, mono=True)
    samples_per_chunk = int(chunk_duration * sr)
    overlap_samples = int(overlap_duration * sr)
    step_size = samples_per_chunk - overlap_samples
    
    total_chunks = max(1, (len(audio) - overlap_samples) // step_size + 
                      (1 if (len(audio) - overlap_samples) % step_size else 0))
    
    print(f"Total chunks: {total_chunks}") 
    
    for i in range(total_chunks):
        start_idx = i * step_size
        end_idx = min(start_idx + samples_per_chunk, len(audio))
        if end_idx - start_idx < samples_per_chunk * 0.2: 
            break 
        
        chunk = audio[start_idx:end_idx]
        output_file = os.path.join(chunks_dir, f"{audio_file_name}_chunk_{i+1:03d}.wav")
        sf.write(output_file, chunk, sr)
        print(f"Chunk {i+1} saved to {output_file}") 
    
    return chunks_dir

def transcribe_audio(chunks_dir: str):
    all_transcripts = []
    wav_files = sorted(glob.glob(os.path.join(chunks_dir, "*.wav")))        
    for i, wav_file in enumerate(wav_files):
        output = transcriber.transcribe([wav_file])[0].text
        all_transcripts.append(output)
    
    return all_transcripts
    
def get_sources(question: str, video_title: str):
    dense_embedding = embedder.encode(
        sentences=[question],
        task="retrieval.query",
    ).tolist()[0]
    
    
    qdrant_collection_name = f"yt_transcripts"
    
    results = qdrant_client.query_points(
        collection_name=qdrant_collection_name,
        query=dense_embedding,
        with_payload=True,
        with_vectors=False,
        score_threshold=0.2, 
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="video_title",
                    match=models.MatchValue(value=video_title)
                )
            ]
        ),
        limit=5
    ) 
    
    processed_results = []
    for i, result in enumerate(results.points): 
        payload = result.payload 
        text = f"""Information {i+1}: 
                Transcript: {payload['transcript']}
                Video Title: {payload['video_title']}
                """
        processed_results.append(text)
    
    return "\n\n".join(processed_results)
    
    
    
    
# Main title
st.title("🎬 YouTube Audio Analysis and Chat")
st.markdown("---")

# Create two columns (30:70 ratio)
col1, col2 = st.columns([3, 7])

with col1:
    st.header("📹 YT Audio Processing")
    
    youtube_url = st.text_input(
        "Enter YouTube video URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )
     
    process_button = st.button("🚀 Process Video", type="primary")
    
    # Progress section
    st.subheader("📊 Progress")
    
    if st.session_state.current_step:
        st.info(f"Current Step: {st.session_state.current_step}")
    
    if st.session_state.progress > 0: 
        progress_bar = st.progress(st.session_state.progress / 100)
        st.text(f"Progress: {st.session_state.progress}%")

    if st.session_state.processing_status:
        st.write(st.session_state.processing_status) 
    
with col2:
    st.header("💬 Chat")
    
    if st.session_state.transcript:
        with st.expander("📄 View Transcript", expanded=False):
            st.text_area(
                "Transcript",
                value=st.session_state.transcript,
                height=200,
                disabled=False
            )
    
        st.subheader("Ask questions about the content:")
        
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"): 
                st.markdown(f"{question}")
            with st.chat_message("assistant"):
                st.markdown(f"{answer}")
            
        new_question = st.text_input("Ask a question:", key="question_input") 
        if st.button("Send Question") and new_question:
            llm = OpenAI(
                base_url="http://192.168.0.153:7001/v1",
                api_key="token-abc123",
            )
            def response_generator():
                video_title = st.session_state.get("video_title", "")
                print(f"Sources Video title: {video_title}")
                informations = get_sources(new_question, video_title) 
                
                prompt = f"""You are an expert AI assistant specialized in analyzing YouTube video transcripts and providing accurate, contextual answers.

                **VIDEO TRANSCRIPT SOURCES:**
                {informations}

                **USER QUESTION:** {new_question}

                **RESPONSE GUIDELINES:**
                1. **Source Fidelity**: Base your answer EXCLUSIVELY on the provided transcript sources. Never add external knowledge or assumptions.

                2. **Response Adaptation**: 
                - **Simple questions**: Provide direct, concise answers (1-2 sentences)
                - **Complex questions**: Deliver comprehensive, well-structured explanations with supporting details
                - **Clarification requests**: Break down concepts step-by-step with examples from the transcript

                3. **Information Boundaries**: If the transcript lacks sufficient information, respond with: "The provided transcript doesn't contain enough information to fully answer this question."

                4. **Formatting Standards**:
                - Use **bold** for key concepts and emphasis
                - Apply bullet points (•) for lists and multiple items
                - Use `code blocks` for specific quotes or technical terms
                - Structure complex answers with ## headers for different sections
                - Include > blockquotes for direct transcript citations

                5. **Accuracy Requirements**:
                - Quote directly from transcript when relevant
                - Maintain context and meaning from original source
                - Avoid paraphrasing that could change meaning
                - Reference specific parts of the video when applicable

                6. **Response Quality**:
                - Match depth to question complexity
                - Provide actionable insights when possible
                - Connect related concepts from different parts of the transcript
                - Maintain conversational yet informative tone

                Provide your response in well-formatted markdown:"""
                
                messages = [
                    {"role": "user", "content": prompt}
                ]
                
                model_kwargs = {
                    "temperature": 0.6,
                    "max_tokens": 8096,
                    "top_p": 0.8
                }
                
                completion = llm.chat.completions.create(
                    model="unsloth/Qwen3-8B-unsloth-bnb-4bit",
                    messages=messages,
                    **model_kwargs,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False}
                    }
                )
                answer = completion.choices[0].message.content
                st.session_state.chat_history.append((new_question, answer))
                st.rerun()

            with st.chat_message("assistant"):
                response = st.write_stream(response_generator())
                print(f"Response: {response}")
    else:
        st.info("👈 Process a YouTube video to start chatting with its transcript!")


def download_and_process_video(url):
    progress_bar = st.progress(0, width=220)
    status_text = st.empty()

    video = YouTube(url, on_progress_callback=on_progress)
    video_title = video.title
    print(f"Extracting audio: {video_title}")
    status_text.info(f"🔊 Extracting audio from: {video_title}", width=220)
    progress_bar.progress(10, width=220)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_file_name = f"{re.sub(r'[ -]+', '_', video_title)}.m4a"
        audio_file_path = os.path.join(temp_dir, audio_file_name)
        ys = video.streams.get_audio_only()
        ys.download(output_path=temp_dir, filename=audio_file_name)
        print(f"Audio extracted successfully: {video_title} and saved to {audio_file_path} list of files: {os.listdir(temp_dir)}")
    
        print(f"Converting audio: {video_title}")
        status_text.info("🎧 Converting audio to chunks...", width=220)
        progress_bar.progress(30, width=220)
        
        chunks_dir = chunk_audio(temp_dir, audio_file_path)
        progress_bar.progress(50, width=220)
        
        status_text.info("✍️ Transcribing audio...", width=220)
        transcripts = transcribe_audio(chunks_dir)
        progress_bar.progress(85, width=220)        
        
        status_text.info("🔍 Building database...", width=220)
        collection_name = f"yt_transcripts"
        create_collection(collection_name) 
        
        for i, transcript in enumerate(transcripts):
            store_transcript(collection_name, transcript, video_title, i)
        
        st.session_state.progress = 100
        st.session_state.transcript = "\n".join(transcripts)
        status_text.success("✅ Processing complete!", width=220)
        progress_bar.progress(100, width=220)
        st.session_state.video_title = video_title
        
if process_button and youtube_url:
    if not youtube_url.startswith(('https://www.youtube.com', 'https://youtu.be')):
        st.error("Please enter a valid YouTube URL")
    else:
        st.session_state.transcript = ""
        st.session_state.chat_history = []
        
        st.session_state.processing_status = ""
        st.session_state.progress = 0
        st.session_state.current_step = ""
        
        # Process video in a separate thread
        with st.spinner("Processing the video..."):
            download_and_process_video(youtube_url)
        
        time.sleep(1)
        st.rerun()

    
st.sidebar.title("ℹ️ How to Use")
st.sidebar.markdown("""
1. **Enter YouTube URL** in the left panel
2. **Click Process Video** to start
3. **Wait for processing** (download → audio → transcript)
4. **Chat with the transcript** in the right panel
5. **Ask questions** about the video content
""")

st.sidebar.title("⚠️ Important Notes")
st.sidebar.markdown("""
- Processing may take several minutes
- Large videos require more time
- Keep the page open during processing
""")


