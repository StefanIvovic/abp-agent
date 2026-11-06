from pathlib import Path

from abp_agent.sources import javascript

ABP_JS = b"""\xef\xbb\xbfvar abp = abp || {};
(function () {

    /* LOCALIZATION */
    abp.localization = abp.localization || {};
    abp.localization.resources = abp.localization.resources || {};

    // Finds a resource by name.
    abp.localization.internal = function (resourceName) {
        var resource = abp.localization.resources[resourceName];
        return resource.texts[resourceName];
    };

    abp.localization.getResource = function (name) {
        return name;
    };

    var helper = 1;

    abp.auth = abp.auth || {};
    abp.auth.isGranted = function (policyName) {
        return true;
    };

})();
"""

PAGE_JS = b"""$(function () {
    var l = abp.localization.getResource('Identity');
    function refresh() {
        return l('Refreshed');
    }
    refresh();
});
"""


def parsed(tmp_path: Path, source: bytes = ABP_JS,
           path: str = "npm/packs/core/src/abp.js") -> javascript.ParsedScript:
    file = tmp_path / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(source)
    return javascript.parse_file(tmp_path, file, "abc123")


def test_statements_are_found_inside_the_outer_function(tmp_path):
    chunks = javascript.chunk_file(parsed(tmp_path))
    assert all("(function () {" not in c.text.splitlines()[0] for c in chunks)
    assert any("abp.auth.isGranted = function" in c.text for c in chunks)


def test_a_namespace_stays_together(tmp_path):
    chunks = javascript.chunk_file(parsed(tmp_path))
    assert [c.section for c in chunks] == ["abp.localization", "abp.auth"]
    assert "abp.localization.getResource" in chunks[0].text
    assert "resource.texts[resourceName]" in chunks[0].text
    assert "abp.auth.isGranted" not in chunks[0].text


def test_a_statement_with_no_owner_stays_with_the_one_before(tmp_path):
    chunks = javascript.chunk_file(parsed(tmp_path))
    assert "var helper = 1" in chunks[0].text


def test_a_comment_stays_with_its_statement(tmp_path):
    assert "// Finds a resource by name." in javascript.chunk_file(parsed(tmp_path))[0].text


def test_the_exact_name_and_its_words_can_be_searched(tmp_path):
    first = javascript.chunk_file(parsed(tmp_path))[0]
    head = first.search_text.split("\n\n", 1)[0]
    assert head.startswith("ABP JavaScript | npm/packs/core/src/abp.js | abp.localization")
    assert "abp.localization.getResource" in head
    assert "Resource" in head.split()


def test_a_chunk_links_to_its_line_at_the_pinned_commit(tmp_path):
    url = javascript.chunk_file(parsed(tmp_path))[1].meta["url"]
    assert url.startswith("https://github.com/abpframework/abp/blob/abc123/npm/packs/core/src/abp.js#L")
    assert int(url.rsplit("#L", 1)[1]) == 20        # abp.auth = ...


def test_a_page_script_is_read_through_the_jquery_ready_wrapper(tmp_path):
    path = "modules/identity/src/Volo.Abp.Identity.Web/Pages/Identity/Users/index.js"
    chunks = javascript.chunk_file(parsed(tmp_path, PAGE_JS, path))
    assert [c.section for c in chunks] == ["index"]
    assert "function refresh()" in chunks[0].text
    assert parsed(tmp_path, PAGE_JS, path).category == "identity"


def test_a_large_namespace_is_cut_between_statements(tmp_path):
    lines = "\n".join(f"abp.big.m{i} = function () {{ return {' + '.join(['1'] * 60)}; }};"
                      for i in range(20))
    chunks = javascript.chunk_file(parsed(tmp_path, lines.encode()))
    assert len(chunks) > 1
    assert all(c.text.startswith("abp.big.m") for c in chunks)
    assert len({c.chunk_index for c in chunks}) == len(chunks)


def test_a_namespace_that_comes_back_gets_its_own_key(tmp_path):
    source = b"abp.a.x = 1;\nabp.b.y = 2;\nabp.a.z = 3;\n"
    assert [c.section for c in javascript.chunk_file(parsed(tmp_path, source))] == [
        "abp.a", "abp.b", "abp.a (2)"]


def test_only_abps_own_scripts_are_listed(tmp_path):
    keep = ["npm/packs/core/src/abp.js",
            "npm/packs/jquery/src/abp.jquery.js",
            "framework/src/Volo.Abp.AspNetCore.Mvc.UI.Theme.Shared/wwwroot/libs/abp/x/toast/t.js",
            "modules/identity/src/Volo.Abp.Identity.Web/Pages/Identity/Users/index.js"]
    leave = ["npm/packs/codemirror/src/codemirror.js",
             "npm/packs/jquery/abp.resourcemapping.js",
             "modules/account/src/Volo.Abp.Account.Web/wwwroot/client-proxies/account-proxy.js",
             "modules/identity/src/Volo.Abp.Identity.Web/Pages/Identity/Users/index.min.js",
             "framework/src/Volo.Abp.AspNetCore.Mvc.UI.Theme.Shared/wwwroot/libs/jquery/jquery.js"]
    for path in keep + leave:
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("abp.a = 1;")
    found = {f.relative_to(tmp_path).as_posix() for f in javascript.list_files(tmp_path)}
    assert found == set(keep)


def test_search_treats_javascript_as_code(monkeypatch):
    from types import SimpleNamespace

    from abp_agent.retrieval import hybrid
    asked = []

    def search(db, query, limit, filters, query_vector):
        asked.append(filters.sources)
        return SimpleNamespace(hits=[])

    monkeypatch.setattr(hybrid, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(hybrid, "search", search)
    hybrid.search_both(None, "abp.ajax")
    assert asked == [["docs"], ["code", "js"]]
