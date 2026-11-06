"""Connector for ABP's own JavaScript: files -> statements -> chunks, grouped by namespace.

ABP's client code is mostly one wrapper function holding many `abp.x.y = ...` statements, so a chunk
can't be a function, as it can in C#. We step inside the wrapper and keep each namespace together,
because a detail such as `resource.texts[key]` sits beside the API that depends on it.

The folders are an allowlist: most scripts in the repo are third-party libraries.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter_javascript
from llama_index.core.utils import get_tokenizer
from tree_sitter import Language, Node, Parser

from abp_agent.documents.chunking import TextChunk, pack_blocks
from abp_agent.sources.csharp import GITHUB, category_of, split_name

PARSER_VERSION = "js-v1"        # bump when parsing or chunking changes
CHUNK_TOKENS = 512
MIN_TOKENS = 50
SCOPES = ("npm/packs/core/src/*.js", "npm/packs/*/src/abp.*.js",
          "framework/src/*/wwwroot/libs/abp/**/*.js", "framework/src/*/Pages/**/*.js",
          "modules/*/src/*/Pages/**/*.js", "modules/*/src/*/Components/**/*.js")
SKIP_SUFFIXES = (".min.js",)
FUNCTIONS = {"function_expression", "function", "arrow_function"}
_ABP_NAME = re.compile(r"\babp(?:\.\w+)+")
_PARSER = Parser(Language(tree_sitter_javascript.language()))


@dataclass
class Statement:
    text: str       # with the comments above it
    line: int       # 1-based


@dataclass
class Group:
    owner: str      # a namespace such as abp.localization, a function, or the file's name
    statements: list[Statement] = field(default_factory=list)


@dataclass
class ParsedScript:
    path: str
    category: str               # "framework", or the module folder
    title: str
    url: str
    content_hash: str
    groups: list[Group] = field(default_factory=list)


def _text(source: bytes, node: Node, start: int | None = None) -> str:
    return source[node.start_byte if start is None else start:node.end_byte].decode(
        "utf-8", errors="replace")


def _unwrapped(node: Node) -> Node:
    while node.type == "parenthesized_expression" and node.named_children:
        node = node.named_children[0]
    return node


def _wrapper_body(statement: Node, source: bytes) -> Node | None:
    """The body of `(function () {...})()` or `$(function () {...})`, else None."""
    if statement.type != "expression_statement" or not statement.named_children:
        return None
    call = _unwrapped(statement.named_children[0])
    if call.type != "call_expression":
        return None
    target = _unwrapped(call.child_by_field_name("function"))
    if target.type not in FUNCTIONS:
        args = call.child_by_field_name("arguments")
        name = _text(source, target)
        if name not in ("$", "jQuery") and not name.endswith(".ready"):
            return None
        target = next((a for a in (args.named_children if args else [])
                       if a.type in FUNCTIONS), None)
    body = target.child_by_field_name("body") if target else None
    return body if body is not None and body.type == "statement_block" else None


def _statements(parent: Node, source: bytes) -> list[tuple[Node, int]]:
    """Statements in order, stepping into wrappers, each with where its comments start."""
    found: list[tuple[Node, int]] = []
    comment_start = None
    for node in parent.named_children:
        if node.type == "comment":
            comment_start = node.start_byte if comment_start is None else comment_start
            continue
        body = _wrapper_body(node, source)
        if body is not None:
            found += _statements(body, source)
        else:
            found.append((node, node.start_byte if comment_start is None else comment_start))
        comment_start = None
    return found


def _owner(node: Node, source: bytes) -> str | None:
    """What a statement belongs to: abp.a.b = ... gives abp.a; function f() gives f."""
    if node.type == "function_declaration":
        return _text(source, node.child_by_field_name("name"))
    if node.type == "expression_statement" and node.named_children:
        expression = node.named_children[0]
        if expression.type == "assignment_expression":
            name = _text(source, expression.child_by_field_name("left"))
            if name.startswith("abp."):
                return ".".join(name.split(".")[:2])
    return None


def parse_groups(source: bytes, fallback: str) -> list[Group]:
    """Runs of statements with one owner. A statement without one stays with the run before it,
    so helpers and local variables stay beside the code that uses them. Statements before the
    first owner, such as `var abp = abp || {};`, join it: alone they'd be a chunk too small to
    find."""
    groups: list[Group] = []
    for node, start in _statements(_PARSER.parse(source).root_node, source):
        owner = _owner(node, source)
        if node.type == "function_declaration" and groups and groups[-1].owner == fallback:
            owner = fallback        # a helper in a page script is part of that page
        if not groups or (owner and owner != groups[-1].owner):
            groups.append(Group(owner or fallback))
        groups[-1].statements.append(Statement(_text(source, node, start),
                                               node.start_point[0] + 1))
    if len(groups) > 1 and groups[0].owner == fallback:
        first = groups.pop(0)
        groups[0].statements[:0] = first.statements
    return groups


def parse_file(root: Path, file: Path, commit: str) -> ParsedScript:
    raw = file.read_bytes()
    path = file.relative_to(root).as_posix()
    return ParsedScript(path, category_of(path), file.name, f"{GITHUB}/{commit}/{path}",
                        hashlib.sha256(raw).hexdigest(),
                        parse_groups(raw.removeprefix(b"\xef\xbb\xbf"), file.stem))


def list_files(root: Path) -> list[Path]:
    files = {f for scope in SCOPES for f in root.glob(scope)}
    return sorted(f for f in files if not f.name.endswith(SKIP_SUFFIXES)
                  and "client-proxies" not in f.parts)


def search_header(parsed: ParsedScript, owner: str, piece: str) -> str:
    """abp.localization.getResource stays whole, and its words are listed, so a search finds it
    by the exact name or by "get resource"."""
    names = list(dict.fromkeys(_ABP_NAME.findall(piece)))
    words = sorted({w for n in names for part in n.split(".") for w in split_name(part)})
    return (f"ABP JavaScript | {parsed.path} | {owner}\n"
            f"{' '.join(names)}\n{' '.join(words)}")


def chunk_file(parsed: ParsedScript) -> list[TextChunk]:
    tokenize = get_tokenizer()
    out: list[TextChunk] = []
    seen: dict[str, int] = {}
    for group in parsed.groups:
        n = seen[group.owner] = seen.get(group.owner, 0) + 1
        key = group.owner[:240] + (f" ({n})" if n > 1 else "")      # unique per document
        title = f"{group.owner} ({parsed.title})"
        pieces = pack_blocks([s.text for s in group.statements], CHUNK_TOKENS, MIN_TOKENS)
        for i, piece in enumerate(pieces):
            line = next((s.line for s in group.statements if s.text in piece),
                        group.statements[0].line)
            header = search_header(parsed, group.owner, piece)
            out.append(TextChunk(key, title, i, piece, f"{header}\n\n{piece}",
                                 len(tokenize(piece)),
                                 {"url": f"{parsed.url}#L{line}", "title": title}))
    return out
