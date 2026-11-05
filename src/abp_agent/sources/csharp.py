"""Connector for ABP's C# source: files -> types -> chunks, cut at real code boundaries.

tree-sitter parses each file, so a chunk is a whole type, or whole members of a large type. Text
splitters cut code anywhere, which loses what a piece belongs to.

BM25 treats `AbpIdentityUserValidator` as one word, so a search for "user validator" would miss
it. search_text therefore also lists the split-up names of what a chunk declares.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter_c_sharp
from llama_index.core.utils import get_tokenizer
from tree_sitter import Language, Node, Parser

from abp_agent.documents.chunking import TextChunk, pack_blocks

GITHUB = "https://github.com/abpframework/abp/blob"
PARSER_VERSION = "cs-v1"        # bump when parsing or chunking changes
CHUNK_TOKENS = 512
MIN_TOKENS = 50
SCOPES = ("framework/src/*/**/*.cs", "modules/*/src/*/**/*.cs")
# Generated code, not what anyone asks about
SKIP_PARTS = {"bin", "obj", "Migrations"}
SKIP_SUFFIXES = (".Designer.cs", ".g.cs", "ModelSnapshot.cs")

TYPES = {"class_declaration", "interface_declaration", "struct_declaration",
         "record_declaration", "enum_declaration"}
NAMESPACES = {"namespace_declaration", "file_scoped_namespace_declaration"}
NAMED_MEMBERS = {"method_declaration", "constructor_declaration", "property_declaration",
                 "event_declaration", "delegate_declaration", *TYPES}
_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_PARSER = Parser(Language(tree_sitter_c_sharp.language()))


@dataclass
class CodeMember:
    name: str
    text: str
    line: int       # 1-based


@dataclass
class CodeType:
    name: str
    kind: str                   # class, interface, ...
    namespace: str
    signature: str              # "public class A : B"
    text: str                   # the whole declaration, with its doc comment
    start_line: int
    end_line: int
    members: list[CodeMember] = field(default_factory=list)


@dataclass
class ParsedFile:
    path: str
    category: str               # "framework", or the module folder
    title: str
    url: str
    content_hash: str
    types: list[CodeType] = field(default_factory=list)


def split_name(name: str) -> list[str]:
    """AbpIdentityUserValidator -> Abp Identity User Validator; IThing -> I Thing"""
    return _WORD.findall(name)


def _text(source: bytes, start: int, end: int) -> str:
    return source[start:end].decode("utf-8", errors="replace")


def _leading_start(node: Node) -> int:
    """Where a declaration really starts: doc comments are siblings before it."""
    start, prev = node.start_byte, node.prev_named_sibling
    while prev is not None and prev.type == "comment":
        start, prev = prev.start_byte, prev.prev_named_sibling
    return start


def _name(node: Node, source: bytes) -> str:
    name = node.child_by_field_name("name")
    return _text(source, name.start_byte, name.end_byte) if name else ""


def _members(body: Node, source: bytes) -> list[CodeMember]:
    return [CodeMember(_name(m, source) or m.type, _text(source, _leading_start(m), m.end_byte),
                       m.start_point[0] + 1)
            for m in body.named_children if m.type != "comment"]


def _walk(node: Node, source: bytes, namespace: str, found: list[CodeType]) -> None:
    for child in node.named_children:
        if child.type == "file_scoped_namespace_declaration":
            namespace = _name(child, source)        # applies to the siblings that follow
            _walk(child, source, namespace, found)
        elif child.type == "namespace_declaration":
            _walk(child.child_by_field_name("body"), source, _name(child, source), found)
        elif child.type in TYPES:
            body = child.child_by_field_name("body")
            start = body.start_byte if body else child.end_byte
            found.append(CodeType(
                _name(child, source), child.type.removesuffix("_declaration"), namespace,
                re.sub(r"\s+", " ", _text(source, child.start_byte, start)).strip(),
                _text(source, _leading_start(child), child.end_byte),
                child.start_point[0] + 1, child.end_point[0] + 1,
                _members(body, source) if body else []))


def parse_types(source: bytes) -> list[CodeType]:
    found: list[CodeType] = []
    _walk(_PARSER.parse(source).root_node, source, "", found)
    return found


def category_of(path: str) -> str:
    parts = path.split("/")
    return parts[1] if parts[0] == "modules" else "framework"


def parse_file(root: Path, file: Path, commit: str) -> ParsedFile:
    raw = file.read_bytes()
    path = file.relative_to(root).as_posix()
    return ParsedFile(path, category_of(path), file.stem, f"{GITHUB}/{commit}/{path}",
                      hashlib.sha256(raw).hexdigest(),
                      parse_types(raw.removeprefix(b"\xef\xbb\xbf")))


def list_files(root: Path) -> list[Path]:
    files = {f for scope in SCOPES for f in root.glob(scope)}
    return sorted(f for f in files if not SKIP_PARTS & set(f.relative_to(root).parts)
                  and not f.name.endswith(SKIP_SUFFIXES))


def search_header(t: CodeType, path: str, names: list[str]) -> str:
    words = sorted({w for n in names for w in split_name(n)})
    return f"ABP source | {t.namespace} | {t.signature} | {path}\n{' '.join(words)}"


def chunk_file(parsed: ParsedFile) -> list[TextChunk]:
    """A type that fits is one chunk. A larger one becomes runs of whole members."""
    tokenize = get_tokenizer()
    out: list[TextChunk] = []
    seen: dict[str, int] = {}
    for t in parsed.types:
        n = seen[t.name] = seen.get(t.name, 0) + 1
        key = t.name[:240] + (f" ({n})" if n > 1 else "")   # unique per document
        title = f"{t.namespace}.{t.name}" if t.namespace else t.name
        names = [t.name, *(m.name for m in t.members)]
        if len(tokenize(t.text)) <= CHUNK_TOKENS or not t.members:
            pieces = pack_blocks([t.text], CHUNK_TOKENS, MIN_TOKENS)
        else:
            pieces = pack_blocks([m.text for m in t.members], CHUNK_TOKENS, MIN_TOKENS)
        for i, piece in enumerate(pieces):
            inside = [m for m in t.members if m.text in piece]
            line = inside[0].line if inside and len(pieces) > 1 else t.start_line
            last = t.end_line if len(pieces) == 1 else None
            anchor = f"#L{line}-L{last}" if last else f"#L{line}"
            header = search_header(t, parsed.path, [t.name, *(m.name for m in inside)] if
                                   len(pieces) > 1 else names)
            out.append(TextChunk(key, title, i, piece, f"{header}\n\n{piece}",
                                 len(tokenize(piece)),
                                 {"url": parsed.url + anchor, "title": title}))
    return out
