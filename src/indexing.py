from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.transcript import create_timestamp_url


PERSIST_DIRECTORY = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_collection_name(video_id):
    safe_video_id = "".join(
        char if char.isalnum() else "_"
        for char in video_id
    )

    return f"video_{safe_video_id}"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

def chunk_by_time(transcript, window=60, overlap=15):
    if not transcript:
        return []

    if overlap >= window:
        raise ValueError(
            "Overlap must be less than the window size."
        )

    chunks = []
    chunk_index = 0

    max_time = max(
        item["start_seconds"]
        for item in transcript
    )

    start_time = 0
    step = window - overlap

    while start_time <= max_time:
        end_time = start_time + window

        chunk_items = [
            item
            for item in transcript
            if start_time
            <= item["start_seconds"]
            < end_time
        ]

        if chunk_items:
            chunk_text = " ".join(
                item.get("text", "")
                for item in chunk_items
            ).strip()

            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "start_sec": start_time,
                    "end_sec": end_time,
                    "text": chunk_text,
                    "items": chunk_items,
                }
            )

            chunk_index += 1

        start_time += step

    return chunks

def load_vectorstore(video_id):
    return Chroma(
        collection_name=get_collection_name(video_id),
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=get_embeddings(),
    )

def build_vectorstore(chunks, video_id):
    vectorstore = load_vectorstore(video_id)

    if vectorstore._collection.count() > 0:
        return vectorstore

    documents = []
    document_ids = []

    for chunk in chunks:
        chunk_idx = chunk["chunk_index"]
        start_time = chunk["start_sec"]
        end_time = chunk["end_sec"]
        text = chunk["text"]

        timestamp_url = create_timestamp_url(
            video_id,
            start_time
        )

        context_header = (
            f"[Video ID: {video_id} | "
            f"Segment: {chunk_idx} | "
            f"Time: {start_time}s - {end_time}s]\n"
        )

        metadata = {
            "video_id": video_id,
            "chunk_index": chunk_idx,
            "start_seconds": start_time,
            "end_seconds": end_time,
            "timestamp_url": timestamp_url,
            "context_header": context_header.strip(),
        }

        documents.append(
            Document(
                page_content=context_header + text,
                metadata=metadata,
            )
        )

        document_ids.append(
            f"{video_id}_chunk_{chunk_idx}"
        )

    if documents:
        vectorstore.add_documents(
            documents=documents,
            ids=document_ids,
        )

    return vectorstore