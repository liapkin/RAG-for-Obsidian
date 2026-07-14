from .indexing import load_index
from .llm import rewrite_query, stream_deepseek
from .retrieval import retrieve

SYSTEM = {
    "role": "system",
    "content": "Answer from the retrieved notes and the conversation history — nothing else. "
    "Your own previous answers are a fully valid source: for follow-ups, ignore the freshly "
    "retrieved notes if the conversation already holds the answer. Cite the [file#heading] "
    "you used. If nothing covers it, say so.",
}


def build_query(question: str, history: list[dict] | None) -> tuple[list[dict], list[str]]:
    """Retrieve context and build the message list. Returns (messages, source labels)."""
    vecs, chunks = load_index()
    search = rewrite_query(question, history) if history else question
    top = retrieve(search, vecs, chunks)
    context = "\n\n---\n\n".join(
        f"[{chunks[i]['file']}#{chunks[i]['heading']}]\n{chunks[i]['text']}" for i in top
    )
    prompt = f"Notes:\n{context}\n\nQuestion: {question}"
    messages = [SYSTEM] + (history or []) + [{"role": "user", "content": prompt}]
    return messages, [f"{chunks[i]['file']}#{chunks[i]['heading']}" for i in top]


def query(question: str, history: list[dict] | None = None) -> list[dict]:
    messages, sources = build_query(question, history)
    parts = []
    for delta in stream_deepseek(messages):
        parts.append(delta)
        print(delta, end="", flush=True)
    print("\n\nSources:")
    for s in sources:
        print(f"  {s}")
    # keep history lean: store the bare question, not the injected notes
    return (history or []) + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": "".join(parts)},
    ]


def chat():
    history: list[dict] = []
    print("obsidian rag — ask away (ctrl-d to quit)")
    while True:
        try:
            question = input("\n> ").strip()
        except EOFError:
            break
        if question:
            history = query(question, history)
