"""Layout-aware, section-preserving PDF chunker.

Why not fixed-size splitting: an insurance policy wording is a hierarchy of
numbered clauses (`4.2 Waiting Periods` -> `4.2.a ...`). A clause split in half
loses the condition that changes the decision, and a chunk without its heading
cannot be cited. So we:

  1. read the PDF with per-span font metadata (PyMuPDF),
  2. classify each line as heading / body / table-ish using font size relative to
     the document body size, boldness, numbering pattern and line length,
  3. maintain a heading stack so every chunk knows its full section path,
  4. pack paragraphs into chunks that never cross a heading boundary, splitting
     over-long clauses at sentence boundaries with a small overlap,
  5. prepend the section path to the chunk text so both BM25 and the embedder
     see the clause title.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.models import Chunk

NUMBERED = re.compile(r"^\s*(\d+(\.\d+)*|[IVXLC]+\.|[a-z]\))\s+\S")
ALLCAPS = re.compile(r"^[A-Z0-9 ,\-&/()\.']{4,80}$")
SENT_SPLIT = re.compile(r"(?<=[.;:])\s+(?=[A-Z0-9(])")
DEFINITION_HINT = re.compile(r"\bmeans\b|\bshall mean\b|\bis defined as\b", re.I)


@dataclass
class Line:
    text: str
    page: int
    size: float
    bold: bool
    y: float


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return (s[:48] or "section")


def read_lines(pdf_path: Path) -> list[Line]:
    import fitz  # PyMuPDF

    out: list[Line] = []
    with fitz.open(pdf_path) as doc:
        for pno, page in enumerate(doc, start=1):
            data = page.get_text("dict")
            for block in data.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = "".join(s.get("text", "") for s in spans).strip()
                    if not text:
                        continue
                    size = max((s.get("size", 0.0) for s in spans), default=0.0)
                    bold = any("bold" in str(s.get("font", "")).lower() for s in spans)
                    y = line.get("bbox", [0, 0, 0, 0])[1]
                    out.append(Line(text=text, page=pno, size=round(size, 1), bold=bold, y=y))
    return out


def _body_size(lines: list[Line]) -> float:
    """Body font size = the size carrying the most *characters*.

    Counting lines instead would let a document with many short headings and few
    body lines elect a heading size as the body size, after which no heading is
    detected at all and the section hierarchy collapses.
    """
    weight: dict[float, int] = {}
    for l in lines:
        if l.size > 0:
            weight[l.size] = weight.get(l.size, 0) + len(l.text)
    if not weight:
        return 10.0
    return max(weight.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def _is_heading(line: Line, body: float) -> bool:
    t = line.text.strip()
    if len(t) > 110 or len(t) < 3:
        return False
    if t.endswith("."):  # full sentences are rarely headings
        if not NUMBERED.match(t):
            return False
    big = line.size >= body * 1.08
    if big:
        return True
    if line.bold and len(t) <= 90:
        return True
    if NUMBERED.match(t) and len(t) <= 90 and (line.bold or big):
        return True
    if ALLCAPS.match(t) and len(t.split()) <= 10:
        return True
    return False


def _heading_level(size: float, ranks: list[float]) -> int:
    for i, s in enumerate(ranks):
        if abs(size - s) < 0.25:
            return i
    return len(ranks)


def _is_noise(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if re.fullmatch(r"(page\s*)?\d{1,3}(\s*(of|/)\s*\d{1,3})?", t, re.I):
        return True
    if re.fullmatch(r"UIN[: ]*[A-Z0-9]+", t, re.I):
        return False
    return False


def _pack(
    blocks: list[tuple[str, int]],
    section_path: list[str],
    source: str,
    settings,
    seq: dict[str, int],
) -> list[Chunk]:
    """Pack (paragraph, page) tuples belonging to one section into chunks."""
    chunks: list[Chunk] = []
    if not blocks:
        return chunks

    section = section_path[-1] if section_path else "Preamble"
    header = " > ".join(section_path) if section_path else "Preamble"

    buf: list[str] = []
    buf_pages: list[int] = []

    def flush() -> None:
        if not buf:
            return
        body = "\n".join(buf).strip()
        if len(body) < 40:  # too small to be useful evidence on its own
            return
        page = min(buf_pages)
        page_end = max(buf_pages)
        key = _slug(section)
        seq[key] = seq.get(key, 0) + 1
        raw = f"{source}|{page}|{header}|{seq[key]}|{body[:64]}"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:6]
        kind = "definition" if DEFINITION_HINT.search(body[:300]) else "prose"
        chunks.append(
            Chunk(
                chunk_id=f"p{page:03d}-{key}-{seq[key]:02d}-{digest}",
                text=f"[{header}]\n{body}",
                page=page,
                page_end=page_end,
                section=section,
                section_path=list(section_path),
                source=source,
                char_len=len(body),
                kind=kind,
            )
        )
        buf.clear()
        buf_pages.clear()

    for para, page in blocks:
        para = para.strip()
        if not para:
            continue
        # over-long single paragraph -> sentence-level split with overlap
        if len(para) > settings.chunk_max_chars:
            flush()
            sents = SENT_SPLIT.split(para)
            cur = ""
            for s in sents:
                if cur and len(cur) + len(s) + 1 > settings.chunk_target_chars:
                    buf.append(cur.strip())
                    buf_pages.append(page)
                    flush()
                    cur = cur[-settings.chunk_overlap_chars :] + " " + s
                else:
                    cur = f"{cur} {s}".strip()
            if cur:
                buf.append(cur.strip())
                buf_pages.append(page)
            continue

        cur_len = sum(len(b) for b in buf)
        if cur_len + len(para) > settings.chunk_target_chars and buf:
            tail = buf[-1][-settings.chunk_overlap_chars :]
            tail_page = buf_pages[-1]
            flush()
            buf.append(tail)
            buf_pages.append(tail_page)
        buf.append(para)
        buf_pages.append(page)

    flush()
    return chunks


def chunk_pdf(pdf_path: Path, settings=None) -> list[Chunk]:
    settings = settings or get_settings()
    lines = read_lines(Path(pdf_path))
    if not lines:
        raise ValueError(f"No extractable text in {pdf_path}. Is it a scanned PDF?")

    body = _body_size(lines)
    heading_sizes = sorted({l.size for l in lines if l.size >= body * 1.08}, reverse=True)[:4]

    source = Path(pdf_path).name
    chunks: list[Chunk] = []
    seq: dict[str, int] = {}
    stack: list[str] = []
    blocks: list[tuple[str, int]] = []
    para: list[str] = []
    para_page = 1

    def close_para() -> None:
        nonlocal para
        if para:
            blocks.append((" ".join(para), para_page))
            para = []

    for line in lines:
        if _is_noise(line.text):
            continue
        if _is_heading(line, body):
            close_para()
            chunks.extend(_pack(blocks, stack, source, settings, seq))
            blocks = []
            level = _heading_level(line.size, heading_sizes)
            stack = stack[:level]
            stack.append(line.text.strip())
            continue
        if not para:
            para_page = line.page
        para.append(line.text)
        # a line that ends a sentence and is short = end of paragraph
        if line.text.rstrip().endswith((".", ";", ":")) and len(line.text) < 90:
            close_para()

    close_para()
    chunks.extend(_pack(blocks, stack, source, settings, seq))
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [Chunk(**json.loads(line)) for line in fh if line.strip()]
