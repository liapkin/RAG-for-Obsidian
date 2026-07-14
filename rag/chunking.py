from pathlib import Path

from .config import MAX_CHUNK


def chunk_file(path: Path) -> list[dict]:
    """Split a markdown file on ## headings; split oversized chunks on blank lines."""
    text = path.read_text(errors="replace")
    chunks = []
    heading = path.stem
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if not body:
            return
        # split oversized sections on blank lines
        piece = ""
        for para in body.split("\n\n"):
            if piece and len(piece) + len(para) > MAX_CHUNK:
                chunks.append({"text": piece.strip(), "file": path.name, "heading": heading})
                piece = para
            else:
                piece = f"{piece}\n\n{para}" if piece else para
        if piece.strip():
            chunks.append({"text": piece.strip(), "file": path.name, "heading": heading})

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            buf = []
            heading = line[3:].strip()
        buf.append(line)
    flush()
    return chunks
