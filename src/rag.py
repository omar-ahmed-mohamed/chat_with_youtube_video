import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.indexing import chunk_by_time, load_vectorstore ,build_vectorstore 


load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"

PROMPT_TEMPLATE = """
You are a helpful assistant answering questions about a YouTube video.

Use ONLY the transcript excerpts below to answer.

If the answer is not in the excerpts, say:
"I couldn't find that in the video."

Write the answer in short paragraphs with one blank line between paragraphs.
Use bullet points for lists.
Cite the timestamp after each claim like [mm:ss].

Transcript excerpts:
{context}

Question: {question}

Answer:
"""


def index_transcript(transcript, video_id):
    chunks = chunk_by_time(transcript)
    return build_vectorstore(chunks, video_id)


def format_timestamp(seconds):
    seconds = int(seconds)
    minutes = seconds // 60
    remaining_seconds = seconds % 60

    return f"{minutes:02d}:{remaining_seconds:02d}"


def format_docs(docs):
    formatted_docs = []

    for doc in docs:
        start_seconds = doc.metadata["start_seconds"]
        timestamp = format_timestamp(start_seconds)

        formatted_docs.append(
            f"[{timestamp}]\n{doc.page_content}"
        )

    return "\n\n".join(formatted_docs)


def RAG_pipeline(video_id, question):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    llm = ChatGroq(
        model=MODEL_NAME,
        groq_api_key=api_key
    )

    vectorstore = load_vectorstore(video_id)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6}
    )

    prompt = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)


def RAG_stream(video_id, question):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    llm = ChatGroq(
        model=MODEL_NAME,
        groq_api_key=api_key
    )

    vectorstore = load_vectorstore(video_id)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6}
    )

    docs = retriever.invoke(question)
    context = format_docs(docs)

    prompt = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    )

    prompt_value = prompt.format(
        context=context,
        question=question
    )

    accumulated = ""

    for chunk in llm.stream(prompt_value):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        accumulated = accumulated + token
        yield accumulated
