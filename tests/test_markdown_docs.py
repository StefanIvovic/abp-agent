import json
from pathlib import Path

from abp_agent.documents.chunking import (
    MarkdownChunkingConfig,
    chunk_markdown,
    markdown_blocks,
    pack_blocks,
)
from abp_agent.sources.markdown_docs import (
    DocsRepoConfig,
    list_pages,
    nav_breadcrumbs,
    parse_page,
    slug,
)

PAGE = """```json
//[doc-seo]
{
    "Description": "Integrate Hangfire with ABP."
}
```

# Hangfire Background Job Manager

Hangfire is an advanced background job manager. """ + "Intro sentence. " * 30 + """

## Installation

Run this:

````bash
abp add-package Volo.Abp.BackgroundJobs.HangFire
## not a heading, inside a code block
````

## Configuration

Configure it in your module.

## Configuration

A second section with the same heading.
"""


def make_repo(tmp_path: Path) -> DocsRepoConfig:
    root = tmp_path / "docs"
    (root / "framework" / "background-jobs").mkdir(parents=True)
    (root / "framework" / "background-jobs" / "hangfire.md").write_text(PAGE, encoding="utf-8")
    (root / "framework" / "background-jobs" / "index.md").write_text(
        "# Background Jobs\n\n" + "Jobs run in the background. " * 20, encoding="utf-8")
    (root / "basic.md").write_text("This document has been moved to [here](./x.md).",
                                   encoding="utf-8")
    (root / "Blog-Posts").mkdir()
    (root / "Blog-Posts" / "post.md").write_text("# Post\n\n" + "x " * 300, encoding="utf-8")
    (root / "docs-nav.json").write_text(json.dumps({"items": [{"text": "Framework", "items": [
        {"text": "Background Jobs", "items": [
            {"text": "Overview", "path": "framework/background-jobs", "isIndex": True},
            {"text": "Hangfire", "path": "framework/background-jobs/hangfire.md"}]}]}]}),
        encoding="utf-8")
    return DocsRepoConfig(corpus="abp", name="ABP docs", root=root,
                          base_url="https://abp.io/docs/latest", exclude_dirs=("Blog-Posts",))


def test_nav_gives_breadcrumbs_including_index_pages(tmp_path):
    cfg = make_repo(tmp_path)
    crumbs = nav_breadcrumbs(cfg.root / "docs-nav.json")
    assert crumbs["framework/background-jobs/hangfire.md"] == [
        "Framework", "Background Jobs", "Hangfire"]
    assert crumbs["framework/background-jobs/index.md"] == ["Framework", "Background Jobs"]


def test_parse_strips_seo_block_and_splits_at_h2_outside_code(tmp_path):
    cfg = make_repo(tmp_path)
    crumbs = nav_breadcrumbs(cfg.root / "docs-nav.json")
    page = parse_page(cfg, cfg.root / "framework/background-jobs/hangfire.md", crumbs)
    assert page.title == "Hangfire Background Job Manager"
    assert page.description == "Integrate Hangfire with ABP."
    assert page.category == "framework"
    assert page.url == "https://abp.io/docs/latest/framework/background-jobs/hangfire"
    assert [s.heading for s in page.sections] == [
        "Hangfire Background Job Manager", "Installation", "Configuration", "Configuration"]
    assert "doc-seo" not in page.sections[0].text
    assert "## not a heading" in page.sections[1].text      # stayed inside its code block
    assert page.sections[1].anchor == "installation"


def test_index_page_url_and_stub_and_excluded_pages(tmp_path):
    cfg = make_repo(tmp_path)
    crumbs = nav_breadcrumbs(cfg.root / "docs-nav.json")
    index = parse_page(cfg, cfg.root / "framework/background-jobs/index.md", crumbs)
    assert index.url == "https://abp.io/docs/latest/framework/background-jobs"
    assert parse_page(cfg, cfg.root / "basic.md", crumbs) is None          # moved stub
    assert all("Blog-Posts" not in str(p) for p in list_pages(cfg))


def test_code_blocks_stay_whole():
    text = "Para one.\n\n```csharp\nvar a = 1;\n\nvar b = 2;\n```\n\nPara two."
    blocks = markdown_blocks(text)
    assert blocks == ["Para one.", "```csharp\nvar a = 1;\n\nvar b = 2;\n```", "Para two."]
    assert pack_blocks(blocks, 512) == [text]


def test_pack_splits_only_between_blocks_and_oversized_blocks_get_split():
    blocks = [("word " * 300).strip(), ("word " * 300).strip()]
    assert len(pack_blocks(blocks, 512)) == 2
    assert len(pack_blocks([("word " * 1500).strip()], 512)) >= 3


def test_chunks_have_unique_sections_breadcrumb_header_and_anchor_url(tmp_path):
    cfg = make_repo(tmp_path)
    page = parse_page(cfg, cfg.root / "framework/background-jobs/hangfire.md",
                      nav_breadcrumbs(cfg.root / "docs-nav.json"))
    # min_tokens=0: the fixture sections are tiny; merging is tested separately
    chunks = chunk_markdown(page, "ABP docs", MarkdownChunkingConfig(min_tokens=0))
    keys = [(c.section, c.chunk_index) for c in chunks]
    assert len(keys) == len(set(keys))
    assert [c.section for c in chunks][-2:] == ["Configuration", "Configuration (2)"]
    install = next(c for c in chunks if c.section == "Installation")
    assert install.search_text.startswith(
        "ABP docs | Framework > Background Jobs > Hangfire | Installation")
    assert install.meta["url"].endswith("/hangfire#installation")


def test_slug():
    assert slug("Using the ABP CLI") == "using-the-abp-cli"
    assert slug("CSRF/XSRF & Anti Forgery") == "csrfxsrf--anti-forgery"


def test_clean_body_removes_site_machinery_but_keeps_text_and_code():
    from abp_agent.sources.markdown_docs import clean_body
    body = """````json
//[doc-nav]
{ "Next": { "Name": "MCP" } }
````

Intro text. ![monitoring](./images/monitoring.png)

{{if UI == "MVC"}}Use Razor Pages.{{else}}Use Angular.{{end}}

```html
<span>{{ user.name }}</span>
```
"""
    out = clean_body(body)
    assert "doc-nav" not in out and "Next" not in out
    assert "images/" not in out and "Intro text." in out
    assert "Use Razor Pages.Use Angular." in out and "{{if" not in out
    assert "{{ user.name }}" in out          # Angular interpolation in code is content


def test_small_sections_join_a_neighbour_and_see_also_is_skipped():
    from abp_agent.documents.chunking import MarkdownChunkingConfig, chunk_markdown
    from abp_agent.sources.markdown_docs import ParsedPage, Section
    long = "The module stores menus in the database and renders them. " * 20
    page = ParsedPage(
        path="modules/x.md", category="modules", title="X", breadcrumb=["Modules", "X"],
        description="", url="https://d/modules/x", content_hash="h",
        sections=[Section("X", "", "Short intro."),
                  Section("Installation", "installation", long),
                  Section("Data Seed", "data-seed", "This module doesn't seed any data."),
                  Section("See Also", "see-also", "* [Other](other.md)")])
    chunks = chunk_markdown(page, "ABP", MarkdownChunkingConfig(chunk_size=512))
    assert len(chunks) == 1                       # intro absorbs Installation, Data Seed joins
    c = chunks[0]
    assert c.text.startswith("Short intro.") and "### Installation" in c.text
    assert "### Data Seed" in c.text and "seed any data" in c.text
    assert "Other" not in c.text                  # See Also dropped
    assert c.meta["url"] == "https://d/modules/x"


def test_raw_markers_are_removed():
    from abp_agent.sources.markdown_docs import clean_body
    assert clean_body("{%{\n```yaml\na: 1\n```\n}%}\n") == "\n```yaml\na: 1\n```\n\n"


def test_lead_in_stays_with_an_oversized_block():
    code = "```csharp\n" + "var x = 1;\n" * 400 + "```"
    pieces = pack_blocks(["Replace the content as below:", code, "Done."], 512, min_tokens=50)
    assert pieces[0].startswith("Replace the content as below:")
    assert all(not p.strip().startswith("Done.") for p in pieces)   # joined the remainder
