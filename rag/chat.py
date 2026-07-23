"""Generation: answer questions grounded in retrieved chunks, with page citations.

Uses Llama 3.3 70B via Groq's free tier — no paid API key needed.
"""

import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
client = Groq()

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

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + [{"role": m["role"], "content": m["content"]} for m in history]
        + [{"role": "user", "content": user_message}]
    )

    stream = client.chat.completions.create(
        model=MODEL,
        max_tokens=4000,
        stream=True,
        messages=messages,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
