# Chat with YouTube Videos

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about YouTube videos with timestamped citation links. Paste a video URL, index it once, then chat with its content instead of watching the full video.

## Features

- Accepts watch, youtu.be, shorts, embed and live URLs
- Fetches English and Arabic transcripts
- Time-window chunking (60s window with 15s overlap)
- Local embeddings with Chroma vector store, one collection per video
- Grounded answers: the model answers only from retrieved excerpts
- Streaming Gradio chat interface with a Sources table per answer
- Bidirectional text handling for mixed Arabic and English output

## Architecture

```
URL -> extract video_id -> fetch transcript -> clean
-> chunk by time (60s / 15s overlap) -> context header per chunk
-> embeddings (all-MiniLM-L6-v2) -> Chroma (collection per video)
```

Query flow:

```
question -> retriever (top-6) -> format docs with timestamps
-> prompt -> Groq LLM -> streamed answer + clickable timestamp links
```

## Project Structure

```
chat_with_youtube_videos/
├── app.py
├── requirements.txt
├── .env
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── transcript.py
│   ├── indexing.py
│   └── rag.py
└── chroma_db/
```

- `src/transcript.py`: URL parsing, transcript fetching, timestamp URL builder
- `src/indexing.py`: time chunking, embeddings, Chroma build and load
- `src/rag.py`: prompt template, RAG chain, streaming answer generator
- `app.py`: Gradio ChatInterface with index and chat flow

## Setup

Requirements: Python 3.13, a Groq API key.

```powershell
cd chat_with_youtube_videos
.\venv\Scripts\python.exe -m venv venv
.\venv\Scripts\pip.exe install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

Run the app:

```powershell
.\venv\Scripts\python.exe app.py
```

Open the local URL printed in the terminal, paste a YouTube link, press Index Video, then ask questions.

## Usage Notes

- Indexing runs once per video. Repeated indexing of the same video reuses the existing collection.
- Each answer contains a Sources table with timestamps linking back to the exact video position with a 5 second backup.
- If the answer is not in the transcript, the model replies that it could not find it in the video.
- Long videos over one hour need timestamp format attention.

## Tech Stack

| Stage | Tool |
|---|---|
| Transcript | youtube-transcript-api |
| Chunking | time-window chunking (60s / 15s overlap) |
| Vector store | Chroma via langchain-chroma |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 |
| LLM | Groq via langchain-groq |
| Orchestration | LangChain LCEL |
| UI | Gradio ChatInterface with streaming |

## Design Decisions

1. Time-window chunking instead of character chunking to preserve timestamps for citations.
2. Context header per chunk with video id, segment index and time range to improve retrieval.
3. One Chroma collection per video to avoid mixing sources.
4. Timestamp backup of 5 seconds so links land slightly before the spoken sentence.
5. Local embeddings to keep indexing free and offline-capable.
6. Separate build and load paths so questions do not re-embed the transcript.
