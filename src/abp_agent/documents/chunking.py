"""Chunks of the docs: split at headings, packed by whole blocks, so a code sample stays whole.

Each chunk gets a `search_text` that starts with a context header (the page's place in the docs).
Only search_text is embedded and BM25-indexed; `text` stays the original, for display and citations.
"""

import hashlib
import json
from dataclasses import asdict, dataclass

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.utils import get_tokenizer

from abp_agent.config import get_settings


@dataclass
class TextChunk:
    section: str
    section_title: str
    chunk_index: int
    text: str          # original, shown to users and cited
    search_text: str   # what gets embedded and BM25-indexed
    token_count: int
    meta: dict | None = None   # per-chunk extras, e.g. the url with the heading anchor


@dataclass(frozen=True)
class MarkdownChunkingConfig:
    chunk_size: int = 512     # tokens; small chunks give precise retrieval and citations
    embed_model: str = ""
    # bump when parsing/cleaning (markdown_docs.py) or packing (below) changes. v2: strips
    # doc-nav/doc-params blocks, images, {{if UI}} tags, {%{ }%} markers. v3: no stub chunks
    parser: str = "md-docs-v3"
    context_header: bool = True
    # A section or tail piece smaller than this joins its neighbour instead of becoming a
    # chunk on its own: a 5-token chunk ("This module doesn't seed any data.") has too little
    # context to embed well and crowds real answers out of the top k.
    min_tokens: int = 50
    skip_headings: tuple[str, ...] = ("see also",)    # link lists, not content

    def version(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return f"md1-{hashlib.sha256(payload).hexdigest()[:10]}"


def markdown_config() -> MarkdownChunkingConfig:
    return MarkdownChunkingConfig(embed_model=get_settings().embed_model)


def markdown_blocks(text: str) -> list[str]:
    """Paragraphs, lists, tables and whole fenced code blocks: the units a chunk boundary may
    fall between. A blank line ends a block, except inside a code fence."""
    blocks, buf, fence = [], [], None
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[: len(stripped) - len(stripped.lstrip(stripped[0]))]
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
        if fence is None and not line.strip():
            if buf:
                blocks.append("\n".join(buf))
                buf = []
        else:
            buf.append(line)
    if buf:
        blocks.append("\n".join(buf))
    return blocks


def pack_blocks(blocks: list[str], chunk_size: int, min_tokens: int = 0) -> list[str]:
    """Greedily fill chunks with whole blocks. A block larger than a chunk on its own is split
    with SentenceSplitter (long code listings, very long tables). A last piece smaller than
    min_tokens is appended to the one before (slightly over chunk_size beats a stub)."""
    tokenize = get_tokenizer()
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=0)
    out, cur, cur_tokens = [], [], 0
    for b in blocks:
        n = len(tokenize(b))
        if n > chunk_size:
            if cur and cur_tokens >= min_tokens:
                out.append("\n\n".join(cur))
                cur, cur_tokens = [], 0
            # a short lead-in ("replace the content as below:") stays with its code block
            pieces = splitter.split_text("\n\n".join([*cur, b]))
            out.extend(pieces[:-1])
            # the split's remainder keeps packing with the next blocks instead of standing alone
            cur, cur_tokens = [pieces[-1]], len(tokenize(pieces[-1]))
            continue
        if cur and cur_tokens + n > chunk_size:
            out.append("\n\n".join(cur))
            cur, cur_tokens = [], 0
        cur.append(b)
        cur_tokens += n
    if cur:
        out.append("\n\n".join(cur))
    if len(out) > 1 and len(tokenize(out[-1])) < min_tokens:
        tail = out.pop()
        out[-1] = f"{out[-1]}\n\n{tail}"
    return out


def merge_small_sections(sections: list, min_tokens: int) -> list:
    """Join a section under min_tokens with its neighbour: it goes into the section before it
    (or absorbs the next one when it comes first), its heading kept as a ### line so the text
    still says what it is about. The merged chunk cites the first section's anchor."""
    from abp_agent.sources.markdown_docs import Section

    tokenize = get_tokenizer()
    out: list[Section] = []
    for s in sections:
        if out and (len(tokenize(out[-1].text)) < min_tokens
                    or len(tokenize(s.text)) < min_tokens):
            prev = out[-1]
            out[-1] = Section(prev.heading, prev.anchor,
                              f"{prev.text}\n\n### {s.heading}\n\n{s.text}")
        else:
            out.append(s)
    return out


def markdown_header(corpus_name: str, breadcrumb: list[str], heading: str) -> str:
    path = " > ".join(breadcrumb)
    return f"{corpus_name} | {path} | {heading}" if heading != breadcrumb[-1] else \
        f"{corpus_name} | {path}"


def chunk_markdown(page, corpus_name: str, cfg: MarkdownChunkingConfig) -> list[TextChunk]:
    """Chunks of a ParsedPage (sources.markdown_docs). One section = one `##` heading; chunks
    never cross sections, so each cites one heading anchor."""
    tokenize = get_tokenizer()
    out: list[TextChunk] = []
    seen: dict[str, int] = {}
    sections = [s for s in page.sections if s.heading.lower() not in cfg.skip_headings]
    for sec in merge_small_sections(sections, cfg.min_tokens):
        n = seen[sec.heading] = seen.get(sec.heading, 0) + 1
        key = sec.heading[:240] + (f" ({n})" if n > 1 else "")   # unique per document
        header = markdown_header(corpus_name, page.breadcrumb, sec.heading) \
            if cfg.context_header else ""
        url = f"{page.url}#{sec.anchor}" if sec.anchor else page.url
        pieces = pack_blocks(markdown_blocks(sec.text), cfg.chunk_size, cfg.min_tokens)
        for i, piece in enumerate(pieces):
            search_text = f"{header}\n\n{piece}" if header else piece
            out.append(TextChunk(key, " > ".join([*page.breadcrumb, sec.heading]), i, piece,
                                 search_text, len(tokenize(piece)), {"url": url}))
    return out
