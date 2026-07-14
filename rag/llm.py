import json
import os
import urllib.request


def stream_deepseek(messages: list[dict]):
    """Yield answer deltas; API errors are yielded as text so every caller shows them."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        yield "**DEEPSEEK_API_KEY not set** — put it in .env"
        return
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps({"model": "deepseek-chat", "messages": messages, "stream": True}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                yield json.loads(line[6:])["choices"][0]["delta"].get("content", "")
    except urllib.error.HTTPError as e:
        yield f"**DeepSeek API error {e.code}**: {e.read().decode(errors='replace')}"


def rewrite_query(question: str, history: list[dict]) -> str:
    """Turn a follow-up ("how do I drop it?") into a standalone question for retrieval."""
    msgs = history[-6:] + [{
        "role": "user",
        "content": "Rewrite my next question as one standalone search query, resolving any "
        f"references to the conversation above. Reply with only the query.\n\n{question}",
    }]
    rewritten = "".join(stream_deepseek(msgs)).strip()
    # stream_deepseek yields errors as **bold** text — fall back to the raw question
    return question if not rewritten or rewritten.startswith("**") else rewritten
