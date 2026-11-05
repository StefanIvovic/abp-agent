"""Connector for documentation kept as markdown in a git repository (ABP's docs/en).

Turns each page into a ParsedPage: title, breadcrumb, description, and sections split at `##`
headings. Repository conventions it understands:
- a navigation file (docs-nav.json) that gives each page its place in the docs: the breadcrumb
  "Framework > Infrastructure > Background Jobs > Hangfire" becomes the chunks' context header
- a leading ```json //[doc-seo] {...}``` block: page metadata, not content (its Description is
  kept); other //[doc-nav] / //[doc-params] blocks, images and {{if UI == ...}} template tags
  are removed (clean_body)
- stub pages ("This document has been moved to ...") are skipped
"""

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DocsRepoConfig:
    corpus: str
    name: str
    root: Path                          # the docs folder, e.g. <repo>/docs/en
    base_url: str                       # page URL = base_url + path without .md
    nav_file: str | None = "docs-nav.json"
    exclude_dirs: tuple[str, ...] = ()
    min_chars: int = 300                # shorter pages are stubs/redirects


@dataclass
class Section:
    heading: str        # "Installation"; the page title for the intro before the first ##
    anchor: str         # "installation"
    text: str


@dataclass
class ParsedPage:
    path: str                    # "framework/infrastructure/background-jobs/hangfire.md"
    category: str                # "framework"
    title: str
    breadcrumb: list[str]
    description: str
    url: str
    content_hash: str
    sections: list[Section] = field(default_factory=list)


_SEO_BLOCK = re.compile(r"\A\s*(`{3,})json\s*\n\s*//\[doc-seo\]\s*\n(.*?)\n\1\s*\n", re.DOTALL)
# Other docs-site metadata blocks anywhere in the page: //[doc-nav] (prev/next links),
# //[doc-params] (UI/DB selectors). Not content, but they'd be embedded and keyword-indexed.
_META_BLOCK = re.compile(r"^(`{3,})json[ \t]*\n\s*//\[doc-[\w-]+\][^\n]*\n.*?\n\1[ \t]*(?:\n|\Z)",
                         re.DOTALL | re.MULTILINE)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")           # screenshots: only a file name
# Docs-site template directives that pick content per UI/DB: {{if UI == "MVC"}} ... {{end}}.
# The tags go, every variant's text stays (search should find all of them).
_DIRECTIVE = re.compile(r"\{\{\s*(?:if|else\s*if|else|end)\b[^}]*\}\}")
_RAW_MARKERS = re.compile(r"^[ \t]*(?:\{%\{|\}%\})[ \t]*$", re.MULTILINE)  # {%{ raw }%} wrappers
_BLANKS = re.compile(r"\n{3,}")
_FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})")
_MOVED = re.compile(r"has been moved|has moved to", re.IGNORECASE)


def clean_body(body: str) -> str:
    """Remove docs-site machinery that isn't documentation text."""
    body = _META_BLOCK.sub("", body)
    body = _IMAGE.sub("", body)
    body = _DIRECTIVE.sub("", body)
    body = _RAW_MARKERS.sub("", body)
    return _BLANKS.sub("\n\n", body)


def slug(heading: str) -> str:
    """GitHub-style anchor: lowercase, punctuation removed, spaces to hyphens."""
    s = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return s.replace(" ", "-")      # each space, not collapsed: "a & b" -> "a--b" like GitHub


def nav_breadcrumbs(nav_path: Path) -> dict[str, list[str]]:
    """page path -> breadcrumb, from the docs navigation tree. Folder paths with isIndex map to
    their index.md."""
    out: dict[str, list[str]] = {}

    def walk(items, trail):
        for it in items if isinstance(items, list) else []:
            text = (it.get("text") or "").strip()
            crumb = [*trail, text] if text and text != "Overview" else trail
            p = (it.get("path") or "").strip().strip("/")
            if p:
                key = p if p.endswith(".md") else f"{p}/index.md"
                out.setdefault(key, crumb or [text])
            walk(it.get("items"), crumb)

    walk(json.loads(nav_path.read_text(encoding="utf-8")).get("items", []), [])
    return out


def split_sections(body: str, title: str) -> list[Section]:
    """Split at level-2 headings, ignoring '##' inside fenced code blocks."""
    sections: list[Section] = []
    heading, buf, fence = title, [], None
    for line in body.splitlines():
        m = _FENCE.match(line)
        if m:
            marker = m.group(2)
            if fence is None:
                fence = marker
            elif marker.startswith(fence[0]) and len(marker) >= len(fence):
                fence = None
        if fence is None and line.startswith("## "):
            if "".join(buf).strip():
                sections.append(Section(heading, slug(heading) if heading != title else "",
                                         "\n".join(buf).strip()))
            heading, buf = line[3:].strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append(Section(heading, slug(heading) if heading != title else "",
                                "\n".join(buf).strip()))
    return sections


def parse_page(cfg: DocsRepoConfig, file: Path, crumbs: dict[str, list[str]]) -> ParsedPage | None:
    raw = file.read_text(encoding="utf-8", errors="replace")
    rel = file.relative_to(cfg.root).as_posix()
    description = ""
    m = _SEO_BLOCK.match(raw)
    body = raw[m.end():] if m else raw
    if m:
        try:
            description = json.loads(m.group(2)).get("Description", "")
        except json.JSONDecodeError:
            pass
    body = clean_body(body)
    if(len(body.strip()) < cfg.min_chars and _MOVED.search(body)) or len(body.strip()) < 80:
        return None
    h1 = next((ln[2:].strip() for ln in body.splitlines() if ln.startswith("# ")), None)
    if h1:
        body = re.sub(r"^# .*\n?", "", body, count=1, flags=re.MULTILINE)
    breadcrumb = crumbs.get(rel) or [p.replace("-", " ").title()
                                     for p in rel.removesuffix(".md").split("/")]
    title = h1 or breadcrumb[-1]
    page = rel.removesuffix(".md").removesuffix("/index").removesuffix("index")
    return ParsedPage(
        path=rel, category=rel.split("/")[0] if "/" in rel else "general", title=title,
        breadcrumb=breadcrumb, description=description,
        url=f"{cfg.base_url.rstrip('/')}/{page}".rstrip("/"),
        content_hash=hashlib.sha256(raw.encode()).hexdigest(),
        sections=split_sections(body, title),
    )


def list_pages(cfg: DocsRepoConfig) -> list[Path]:
    files = []
    for f in sorted(cfg.root.rglob("*.md")):
        top = f.relative_to(cfg.root).parts[0]
        if top not in cfg.exclude_dirs:
            files.append(f)
    return files


def repo_commit(path: Path) -> str | None:
    """Short commit of the repo containing `path`. Reads .git directly (the image has no git
    binary); falls back to `git` when available."""
    for d in [path, *path.parents]:
        git = d / ".git"
        if git.is_dir():
            head = (git / "HEAD").read_text().strip()
            if not head.startswith("ref: "):
                return head[:10]
            ref = head[5:]
            if (git / ref).exists():
                return (git / ref).read_text().strip()[:10]
            packed = git / "packed-refs"
            if packed.exists():
                for line in packed.read_text().splitlines():
                    if line.endswith(" " + ref):
                        return line.split()[0][:10]
            break
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
