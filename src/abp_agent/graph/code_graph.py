"""ABP's structure in Neo4j, parsed from its source, not extracted by an LLM. The code states these
facts exactly, so an LLM could only add errors, and the same commit always gives the same graph.

    (:Type)-[:INHERITS]->(:Type)          base class or interface
    (:Module:Type)-[:DEPENDS_ON]->(:Module)   [DependsOn(typeof(X))]
    (:Module)-[:DEFINES]->(:Type)         options classes in the module's package
    (:Doc)-[:MENTIONS {chunk_id}]->(:Type)    `TypeName` in a docs passage

Types are matched by simple name: the code doesn't always say which namespace a base comes from,
so same-named types share their edges. Each node keeps the id of the chunk that states it, so every
fact cites real code.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

from neo4j import Driver, GraphDatabase

from abp_agent.config import get_settings
from abp_agent.sources.csharp import ParsedFile, package_of

BATCH = 2000
_DOC_NAME = re.compile(r"`([A-Z][A-Za-z0-9]+)(?:<[^`]*>)?`")
RELATIONS = ("depends_on", "all_dependencies", "used_by", "all_used_by", "implementations",
             "bases", "options", "module_of_options", "docs")


@dataclass
class DocChunk:
    chunk_id: int
    path: str
    title: str
    url: str
    text: str


@lru_cache
def driver() -> Driver:
    s = get_settings()
    return GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))


def run(query: str, **params) -> list[dict]:
    records, _, _ = driver().execute_query(query, **params)
    return [r.data() for r in records]


def is_module(name: str, bases: list[str]) -> bool:
    return name.endswith("Module") and any(b.endswith("Module") for b in bases)


def type_rows(files: list[ParsedFile], chunk_ids: dict[tuple[str, str], int]) -> list[dict]:
    return [{"key": f"{f.path}#{t.name}", "name": t.name, "kind": t.kind,
             "namespace": t.namespace, "path": f.path, "package": package_of(f.path),
             "area": f.category, "abstract": t.is_abstract,
             "module": is_module(t.name, t.bases),
             "chunk_id": chunk_ids.get((f.path, t.name))}
            for f in files for t in f.types]


def edge_rows(files: list[ParsedFile]) -> tuple[list[dict], list[dict], list[dict]]:
    """(inherits, depends_on, defines). Options classes belong to the modules of their package."""
    modules: dict[str, list[str]] = defaultdict(list)
    options: dict[str, list[str]] = defaultdict(list)
    inherits, depends = [], []
    for f in files:
        package = package_of(f.path)
        for t in f.types:
            key = f"{f.path}#{t.name}"
            inherits += [{"key": key, "target": b} for b in t.bases if b != t.name]
            if is_module(t.name, t.bases):
                modules[package].append(key)
                depends += [{"key": key, "target": d} for d in t.depends_on]
            if t.kind == "class" and t.name.endswith("Options"):
                options[package].append(f"{f.path}#{t.name}")
    defines = [{"module": m, "options": o} for package, keys in modules.items()
               for m in keys for o in options.get(package, [])]
    return inherits, depends, defines


def doc_rows(chunks: list[DocChunk], known: set[str]) -> tuple[list[dict], list[dict]]:
    """Docs pages, and which known types each passage names in `backticks`."""
    docs = {c.path: {"path": c.path, "title": c.title, "url": c.url} for c in chunks}
    mentions = {(c.path, name): c.chunk_id for c in reversed(chunks)
                for name in _DOC_NAME.findall(c.text) if name in known}
    return list(docs.values()), [{"path": p, "name": n, "chunk_id": i}
                                 for (p, n), i in mentions.items()]


def _batches(query: str, rows: list[dict]) -> None:
    for i in range(0, len(rows), BATCH):
        run(query, rows=rows[i:i + BATCH])


def load(files: list[ParsedFile], chunk_ids: dict[tuple[str, str], int],
         chunks: list[DocChunk]) -> dict:
    """Replace the whole graph (seconds, no LLM)."""
    run("MATCH (n) DETACH DELETE n")
    run("CREATE CONSTRAINT type_key IF NOT EXISTS FOR (t:Type) REQUIRE t.key IS UNIQUE")
    run("CREATE INDEX type_name IF NOT EXISTS FOR (t:Type) ON (t.name)")
    run("CREATE CONSTRAINT doc_path IF NOT EXISTS FOR (d:Doc) REQUIRE d.path IS UNIQUE")
    types = type_rows(files, chunk_ids)
    _batches("UNWIND $rows AS r MERGE (t:Type {key: r.key}) SET t += r", types)
    run("MATCH (t:Type {module: true}) SET t:Module")
    inherits, depends, defines = edge_rows(files)
    _batches("""UNWIND $rows AS r MATCH (a:Type {key: r.key}), (b:Type {name: r.target})
                MERGE (a)-[:INHERITS]->(b)""", inherits)
    _batches("""UNWIND $rows AS r MATCH (a:Module {key: r.key}), (b:Module {name: r.target})
                MERGE (a)-[:DEPENDS_ON]->(b)""", depends)
    _batches("""UNWIND $rows AS r MATCH (a:Module {key: r.module}), (b:Type {key: r.options})
                MERGE (a)-[:DEFINES]->(b)""", defines)
    docs, mentions = doc_rows(chunks, {t["name"] for t in types})
    _batches("UNWIND $rows AS r MERGE (d:Doc {path: r.path}) SET d += r", docs)
    _batches("""UNWIND $rows AS r MATCH (d:Doc {path: r.path}), (t:Type {name: r.name})
                MERGE (d)-[m:MENTIONS]->(t) SET m.chunk_id = r.chunk_id""", mentions)
    return stats()


def stats() -> dict:
    return run("""MATCH (t:Type) WITH count(t) AS types
                  OPTIONAL MATCH (m:Module) WITH types, count(m) AS modules
                  OPTIONAL MATCH (d:Doc) WITH types, modules, count(d) AS docs
                  OPTIONAL MATCH ()-[r]->() RETURN types, modules, docs, count(r) AS edges""")[0]


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower().removeprefix("volo.abp."))


def match_name(name: str, rows: list[dict]) -> tuple[str | None, list[str]]:
    """The type a user means, and other candidates. Takes a class name, a package name or plain
    words ("hangfire"). An exact match wins, then the shortest name containing it."""
    key = _key(name).removesuffix("module")

    def short(row: dict) -> str:
        return _key(row["name"]).removesuffix("module").removeprefix("abp")

    exact = [r for r in rows if key in (_key(r["name"]), short(r), _key(r["package"]))]
    hits = exact or sorted((r for r in rows if key and key in _key(r["name"])),
                           key=lambda r: (len(r["name"]), r["name"]))
    names = list(dict.fromkeys(r["name"] for r in hits))
    return (names[0], names[1:6]) if names else (None, [])


def find(name: str, modules_only: bool = False) -> tuple[str | None, list[str]]:
    label = "Module" if modules_only else "Type"
    return match_name(name, run(f"MATCH (t:{label}) RETURN t.name AS name, t.package AS package"))


_ROW = "RETURN DISTINCT x.name AS name, x.package AS package, x.chunk_id AS chunk_id ORDER BY name"
QUERIES = {
    "depends_on": f"MATCH (:Module {{name: $name}})-[:DEPENDS_ON]->(x) {_ROW}",
    "all_dependencies": f"MATCH (:Module {{name: $name}})-[:DEPENDS_ON*1..]->(x) {_ROW}",
    "used_by": f"MATCH (x:Module)-[:DEPENDS_ON]->(:Module {{name: $name}}) {_ROW}",
    "all_used_by": f"MATCH (x:Module)-[:DEPENDS_ON*1..]->(:Module {{name: $name}}) {_ROW}",
    # What you can actually use: not interfaces or abstract classes, however deep the chain
    "implementations": f"""MATCH (x:Type)-[:INHERITS*1..]->(:Type {{name: $name}})
                           WHERE x.kind <> 'interface' AND NOT x.abstract {_ROW}""",
    "bases": f"MATCH (:Type {{name: $name}})-[:INHERITS*1..]->(x) {_ROW}",
    "options": f"MATCH (:Module {{name: $name}})-[:DEFINES]->(x) {_ROW}",
    "module_of_options": f"MATCH (x:Module)-[:DEFINES]->(:Type {{name: $name}}) {_ROW}",
    "docs": """MATCH (d:Doc)-[m:MENTIONS]->(:Type {name: $name})
               RETURN DISTINCT d.title AS name, d.path AS package, m.chunk_id AS chunk_id
               ORDER BY package""",
}
MODULE_RELATIONS = {"depends_on", "all_dependencies", "used_by", "all_used_by", "options"}


def type_chunk(name: str) -> int | None:
    rows = run("MATCH (t:Type {name: $name}) RETURN t.chunk_id AS id LIMIT 1", name=name)
    return rows[0]["id"] if rows else None


def relation(name: str, kind: str) -> dict:
    """One relation of one type: what the API, and later the agent's tool, read."""
    found, others = find(name, modules_only=kind in MODULE_RELATIONS)
    rows = run(QUERIES[kind], name=found) if found else []
    return {"name": found, "others": others, "relation": kind, "rows": rows}
