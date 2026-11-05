from abp_agent.graph import code_graph as g
from abp_agent.sources import csharp

MODULE = b"""
[DependsOn(
    typeof(AbpCoreModule),
    // typeof(AbpOldModule),
    typeof(Volo.Abp.AbpOtherModule)
)]
public class AbpThingModule : AbpModule { }
"""
TYPES = b"""
public abstract class SenderBase : ISender, ITransientDependency { }
public class SmtpSender : SenderBase, ISmtp<int> { }
public interface ISmtp<T> : ISender { }
"""


def parsed(source: bytes, path: str = "framework/src/Volo.Abp.Thing/Volo/A.cs"):
    return csharp.ParsedFile(path, "framework", "A", "https://x", "h", csharp.parse_types(source))


def test_depends_on_reads_real_code_not_comments():
    t = csharp.parse_types(MODULE)[0]
    assert t.depends_on == ["AbpCoreModule", "AbpOtherModule"]
    assert t.bases == ["AbpModule"]


def test_bases_are_simple_names_and_abstract_is_known():
    base, smtp, iface = csharp.parse_types(TYPES)
    assert base.bases == ["ISender", "ITransientDependency"] and base.is_abstract
    assert smtp.bases == ["SenderBase", "ISmtp"] and not smtp.is_abstract
    assert iface.kind == "interface" and iface.bases == ["ISender"]


def test_simple_names():
    assert csharp.simple_name("Volo.Abp.IFoo<Bar, Baz>") == "IFoo"
    assert csharp.simple_name("Base(arg)") == "Base"


def test_package_comes_from_the_project_folder():
    assert csharp.package_of("framework/src/Volo.Abp.Core/Volo/X.cs") == "Volo.Abp.Core"
    assert csharp.package_of("modules/identity/src/Volo.Abp.Identity.Domain/X.cs") == \
        "Volo.Abp.Identity.Domain"


def test_a_module_is_a_class_named_module_deriving_from_a_module():
    assert g.is_module("AbpThingModule", ["AbpModule"])
    assert not g.is_module("AbpThingModule", ["IThing"])
    assert not g.is_module("Thing", ["AbpModule"])


def test_options_belong_to_modules_of_the_same_package():
    module = parsed(MODULE)
    options = parsed(b"public class ThingOptions { }", "framework/src/Volo.Abp.Thing/Volo/O.cs")
    elsewhere = parsed(b"public class OtherOptions { }", "framework/src/Volo.Abp.Other/O.cs")
    inherits, depends, defines = g.edge_rows([module, options, elsewhere])
    assert {d["target"] for d in depends} == {"AbpCoreModule", "AbpOtherModule"}
    assert defines == [{"module": "framework/src/Volo.Abp.Thing/Volo/A.cs#AbpThingModule",
                        "options": "framework/src/Volo.Abp.Thing/Volo/O.cs#ThingOptions"}]


def test_docs_link_to_known_types_named_in_backticks():
    chunks = [g.DocChunk(7, "a.md", "A", "u", "`IThing`, `IThing<T>`, not `Unknown` or IOther"),
              g.DocChunk(8, "a.md", "A", "u", "Later `IThing` again")]
    docs, mentions = g.doc_rows(chunks, {"IThing", "IOther"})
    assert docs == [{"path": "a.md", "title": "A", "url": "u"}]
    assert mentions == [{"path": "a.md", "name": "IThing", "chunk_id": 7}]   # the first passage


def test_a_name_can_be_a_class_a_package_or_plain_words():
    rows = [{"name": "AbpCachingModule", "package": "Volo.Abp.Caching"},
            {"name": "AbpCachingStackExchangeRedisModule",
             "package": "Volo.Abp.Caching.StackExchangeRedis"}]
    assert g.match_name("AbpCachingModule", rows)[0] == "AbpCachingModule"
    assert g.match_name("Volo.Abp.Caching", rows)[0] == "AbpCachingModule"
    assert g.match_name("caching", rows)[0] == "AbpCachingModule"
    assert g.match_name("redis", rows)[0] == "AbpCachingStackExchangeRedisModule"
    assert g.match_name("nothing", rows) == (None, [])
