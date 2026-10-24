from pepbot.ingest import anchor, clean_rst, pack, parse_header, parse_pep, split_sections

from .conftest import SAMPLE_PEP


def test_parse_header_handles_continuation_lines():
    fields, body_start = parse_header(SAMPLE_PEP.splitlines())
    assert fields["PEP"] == "9999"
    assert fields["Title"] == "Walrus Assignment Expressions"
    assert "John Roe" in fields["Author"]
    assert SAMPLE_PEP.splitlines()[body_start - 1] == ""


def test_split_sections_detects_rst_headings():
    _, start = parse_header(SAMPLE_PEP.splitlines())
    titles = [t for t, _ in split_sections(SAMPLE_PEP.splitlines()[start:])]
    assert titles == ["Abstract", "Rationale", "Examples", "Copyright"]


def test_clean_rst_removes_markup():
    text = "Use ``x := 1``, see :pep:`572` and `the docs <https://x.org>`_ [#ref]_."
    assert clean_rst(text) == "Use `x := 1`, see PEP 572 and the docs ."


def test_clean_rst_unescapes_characters():
    text = r"Preserving \*\*kwargs and ``name``\ , ok"
    assert clean_rst(text) == "Preserving **kwargs and `name`, ok"


def test_parse_pep_builds_chunks_with_provenance():
    chunks = parse_pep(SAMPLE_PEP)
    assert {c.section for c in chunks} == {"Abstract", "Rationale", "Examples"}  # no Copyright
    abstract = next(c for c in chunks if c.section == "Abstract")
    assert abstract.pep == 9999 and abstract.python_version == "3.8"
    assert abstract.url == "https://peps.python.org/pep-9999/#abstract"
    assert "walrus operator" in abstract.text and "``" not in abstract.text
    assert all("@example.com" not in c.text for c in chunks)


def test_pack_respects_limit_and_overlaps():
    paragraphs = [" ".join(["word"] * 40) + f" p{i}" for i in range(6)]
    chunks = pack(paragraphs, max_words=100)
    assert all(len(c.split()) <= 100 for c in chunks)
    assert chunks[0].split("\n\n")[-1] == chunks[1].split("\n\n")[0]  # one paragraph overlap
    assert "p5" in chunks[-1]


def test_long_paragraph_is_split_on_sentences():
    paragraph = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = pack([paragraph], max_words=50)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 50 for c in chunks)


def test_anchor():
    assert anchor("Backwards Compatibility") == "backwards-compatibility"
    assert anchor("The `with` Statement!") == "the-with-statement"
