"""Generation: answer questions grounded in retrieved chunks, with page citations."""

import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-8")
client = anthropic.Anthropic()

SYSTEM_PROMPT = (
    "You are a document Q&A assistant. Answer using ONLY the provided document "
    "excerpts. Cite the page for every claim in the form [p. N]. If the "
    "excerpts do not contain the answer, say so plainly — never invent "
    "information that is not in the document."
)


def answer_question(question: str, chunks: list[dict], history: list[dict]):
    """Stream an answer grounded in the retrieved chunks (generator of text).

    history is a list of {"role": "user"|"assistant", "content": str} from
    prior turns, so follow-up questions keep their context.
    """
    context = "\n\n".join(
        f"[Excerpt from p. {c['page']}]\n{c['text']}" for c in chunks
    )
    user_message = (
        f"Document excerpts:\n\n{context}\n\n---\n\nQuestion: {question}"
    )

    messages = [
        {"role": m["role"], "content": m["content"]} for m in history
    ] + [{"role": "user", "content": user_message}]

    with client.messages.stream(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        yield from stream.text_stream
