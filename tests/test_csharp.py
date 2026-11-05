from pathlib import Path

from abp_agent.sources import csharp

SOURCE = b"""\xef\xbb\xbfusing System;

namespace Volo.Abp.Identity;

/// <summary>Checks users.</summary>
[Dependency(ReplaceServices = true)]
public class AbpIdentityUserValidator : IUserValidator<IdentityUser>
{
    private readonly string _name;

    /// <summary>Runs the check.</summary>
    public virtual async Task<IdentityResult> ValidateAsync(IdentityUser user)
    {
        return IdentityResult.Success;
    }
}

public interface IThing
{
    void Run();
}
"""


def parsed(tmp_path: Path, source: bytes = SOURCE) -> csharp.ParsedFile:
    file = tmp_path / "framework" / "src" / "Volo.Abp.Thing" / "A.cs"
    file.parent.mkdir(parents=True)
    file.write_bytes(source)
    return csharp.parse_file(tmp_path, file, "abc123")


def test_names_are_split_for_keyword_search():
    assert csharp.split_name("AbpIdentityUserValidator") == ["Abp", "Identity", "User", "Validator"]
    assert csharp.split_name("IAzureBlobNameCalculator")[:2] == ["I", "Azure"]
    assert csharp.split_name("CMSKit") == ["CMS", "Kit"]


def test_types_keep_namespace_signature_and_doc_comment(tmp_path):
    f = parsed(tmp_path)
    assert [(t.kind, t.name) for t in f.types] == [("class", "AbpIdentityUserValidator"),
                                                   ("interface", "IThing")]
    t = f.types[0]
    assert t.namespace == "Volo.Abp.Identity"
    assert t.signature.endswith("public class AbpIdentityUserValidator : "
                                "IUserValidator<IdentityUser>")
    assert t.text.startswith("/// <summary>Checks users.")
    assert t.start_line == 6
    method = next(m for m in t.members if m.name == "ValidateAsync")
    assert method.text.startswith("/// <summary>Runs the check.")


def test_a_small_type_is_one_chunk_findable_by_split_words(tmp_path):
    chunks = csharp.chunk_file(parsed(tmp_path))
    assert [c.section for c in chunks] == ["AbpIdentityUserValidator", "IThing"]
    first = chunks[0]
    assert "Validator" in first.search_text.splitlines()[1]
    assert "User" in first.search_text.splitlines()[1]
    assert "ValidateAsync" in first.text
    assert first.meta["url"].endswith("/abc123/framework/src/Volo.Abp.Thing/A.cs#L6-L16")


def test_a_large_type_is_cut_between_members(tmp_path):
    body = "\n".join(
        f"    public int Method{i}() {{ return {' + '.join(['1'] * 60)}; }}" for i in range(20))
    source = f"namespace N {{ public class Big\n{{\n{body}\n}}\n}}".encode()
    chunks = csharp.chunk_file(parsed(tmp_path, source))
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.lstrip().startswith("public int Method")     # never mid-method
        assert c.search_text.startswith("ABP source | N | public class Big")
    assert sum(c.text.count("public int Method") for c in chunks) == 20
    assert len({c.chunk_index for c in chunks}) == len(chunks)


def test_block_scoped_namespaces_work(tmp_path):
    f = parsed(tmp_path, b"namespace A.B { public class C { } }")
    assert [(t.namespace, t.name) for t in f.types] == [("A.B", "C")]


def test_generated_code_and_tests_are_left_out(tmp_path):
    keep = tmp_path / "framework/src/Volo.Abp.X/Volo/X.cs"
    skipped = [tmp_path / "framework/src/Volo.Abp.X/obj/Y.cs",
               tmp_path / "modules/a/src/Volo.A/Migrations/1_Init.cs",
               tmp_path / "modules/a/src/Volo.A/Foo.Designer.cs",
               tmp_path / "framework/test/Volo.Abp.X.Tests/T.cs"]
    for f in [keep, *skipped]:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("class A {}")
    assert csharp.list_files(tmp_path) == [keep]


def test_category_is_the_module_or_framework():
    assert csharp.category_of("modules/identity/src/X.cs") == "identity"
    assert csharp.category_of("framework/src/Volo.Abp.Core/X.cs") == "framework"
