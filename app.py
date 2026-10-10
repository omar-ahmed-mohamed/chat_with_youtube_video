import gradio as gr
from src.transcript import extract_video_id, fetch_transcript
from src.indexing import load_vectorstore
from src.rag import index_transcript, RAG_stream, format_timestamp

SESSION = {"video_id": None}

BIDI_CSS = ".gradio-container .message-wrap p { unicode-bidi: plaintext; text-align: start; } .gradio-container .message p { unicode-bidi: plaintext; text-align: start; } .gradio-container table { unicode-bidi: plaintext; }"


def isolate_bidi(text):
    import re
    return re.sub(r"(\[\d{2}:\d{2}(?::\d{2})?\])", r"<bdi>\1</bdi>", text)


def index_video(url):
    if not url or not url.strip():
        return "## Status\n\nPlease enter a YouTube URL."
    try:
        video_id = extract_video_id(url)
        transcript = fetch_transcript(video_id)
        if not transcript:
            return "## Status\n\nNo transcript lines were found for this video."
        index_transcript(transcript, video_id)
        SESSION["video_id"] = video_id
        watch_url = f"https://www.youtube.com/watch?v={video_id}"
        return (
            "## Indexed\n\n"
            f"- Video ID: `{video_id}`\n"
            f"- Transcript lines: `{len(transcript)}`\n"
            f"- Watch: [Open video]({watch_url})\n\n"
            "You can now ask questions below."
        )
    except ValueError as error:
        return f"## Indexing failed\n\n{error}"
    except Exception as error:
        import traceback
        traceback.print_exc()
        return f"## Indexing failed\n\n{type(error).__name__}: {error}"


def get_sources(video_id, question, k=6):
    try:
        vectorstore = load_vectorstore(video_id)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(question)
        sources = []
        seen = set()
        for doc in docs:
            start = int(doc.metadata.get("start_seconds", 0))
            url = doc.metadata.get("timestamp_url", "")
            if not url or url in seen:
                continue
            seen.add(url)
            excerpt = " ".join(doc.page_content.split())
            excerpt = excerpt[-160:] if len(excerpt) > 160 else excerpt
            sources.append({
                "timestamp": format_timestamp(start),
                "url": url,
                "excerpt": excerpt,
            })
        sources.sort(key=lambda item: item["timestamp"])
        return sources
    except Exception:
        return []


def build_sources_table(sources):
    header = "| # | Timestamp | Excerpt |\n| --- | --- | --- |"
    rows = []
    for position, source in enumerate(sources, start=1):
        excerpt = source["excerpt"].replace("|", "/")
        rows.append(f"| {position} | <bdi>[{source['timestamp']}]({source['url']})</bdi> | {excerpt} |")
    return header + "\n" + "\n".join(rows)


def chat_fn(message, history):
    if not message or not message.strip():
        yield "Please enter a question."
        return
    video_id = SESSION.get("video_id")
    if not video_id:
        yield "## No video indexed\n\nPlease index a video first."
        return
    try:
        question = message.strip()
        yield "## Answer\n\nSearching relevant segments..."
        partial = ""
        for partial in RAG_stream(video_id, question):
            yield f"## Answer\n\n{isolate_bidi(partial.strip())}"
        answer = isolate_bidi(partial.strip())
        sources = get_sources(video_id, question)
        output = f"## Answer\n\n{answer}"
        if sources:
            output = output + "\n\n---\n\n### Sources\n\n" + build_sources_table(sources)
        output = output + f"\n\n---\n\nVideo: `{video_id}`"
        yield output
    except ValueError as error:
        yield f"## Request failed\n\n{error}"
    except Exception as error:
        import traceback
        traceback.print_exc()
        yield f"## Request failed\n\n{type(error).__name__}: {error}"


with gr.Blocks(title="Chat with YouTube Videos") as demo:
    gr.Markdown("## Chat with YouTube Videos\n\nIndex a video, then ask questions with timestamped sources.")
    with gr.Row():
        url_input = gr.Textbox(label="YouTube URL", placeholder="Paste a YouTube link here", scale=4)
        index_button = gr.Button("Index Video", scale=1)
    status_output = gr.Markdown()
    chat = gr.ChatInterface(
        fn=chat_fn,
        chatbot=gr.Chatbot(),
        textbox=gr.Textbox(placeholder="Ask about the video content"),
        description="Answers come only from the indexed transcript.",
    )
    index_button.click(fn=index_video, inputs=url_input, outputs=status_output)

if __name__ == "__main__":
    demo.launch(css=BIDI_CSS)
