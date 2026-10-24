"""Download the PEP sources, parse the reStructuredText and split it into chunks.

Each chunk keeps its PEP number, title, section and URL for citations and for the eval.
"""

from __future__ import annotations

import io
import logging
import re
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import requests

from pepbot.config import CHUNK_MAX_WORDS, PEP_URL, PEPS_TARBALL_URL, RAW_DIR

logger = logging.getLogger(__name__)

UNDERLINE = re.compile(r"^([=\-~^\"'`#*+:.])\1{2,}\s*$")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    pep: int
    title: str
    status: str
    pep_type: str
    created: str
    python_version: str
    section: str
    text: str
    url: str


def download_peps(dest: Path = RAW_DIR, url: str = PEPS_TARBALL_URL) -> int:
    """Fetch the python/peps repository tarball and extract only ``peps/pep-*.rst``."""
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as archive:
        for member in archive.getmembers():
            name = Path(member.name).name
            if member.isfile() and re.fullmatch(r"pep-\d{4}\.rst", name):
                handle = archive.extractfile(member)
                if handle is not None:  # copy the bytes, never extract paths from the archive
                    (dest / name).write_bytes(handle.read())
                    count += 1
    logger.info("Downloaded %d PEP sources to %s", count, dest)
    return count


def parse_header(lines: list[str]) -> tuple[dict[str, str], int]:
    """Parse the RFC 822 style header; returns the fields and the index of the body start."""
    fields: dict[str, str] = {}
    key = None
    for i, line in enumerate(lines):
        if not line.strip():
            return fields, i + 1
        match = re.match(r"^([A-Za-z-]+):\s*(.*)$", line)
        if match and not line.startswith((" ", "\t")):
            key = match.group(1)
            fields[key] = match.group(2).strip()
        elif key:  # continuation line
            fields[key] += " " + line.strip()
    return fields, len(lines)


def clean_rst(text: str) -> str:
    """Remove the most common reStructuredText markup."""
    text = re.sub(r":pep:`(?:[^`<]*<)?(\d+)(?:#[^`>]*)?>?`", r"PEP \1", text)
    text = re.sub(r":[a-z]+(?::[a-z]+)?:`~?([^`<]+?)(?:\s*<[^>]+>)?`", r"\1", text)
    text = re.sub(r"`([^`<]+?)\s*<[^>]+>`_{1,2}", r"\1", text)
    text = re.sub(r"``([^`]+)``", r"`\1`", text)
    text = re.sub(r"`([^`]+)`_{1,2}", r"\1", text)
    text = re.sub(r"\[#?[\w-]*\]_", "", text)
    text = text.replace("\\ ", "")  # RST escaped space
    return re.sub(r"\\([*_`|\\])", r"\1", text)  # escaped markup characters


def split_sections(body: list[str]) -> list[tuple[str, list[str]]]:
    """Split the body on RST headings (a text line followed by an underline)."""
    sections: list[tuple[str, list[str]]] = []
    current_title, current = "Preamble", []
    i = 0
    while i < len(body):
        line = body[i]
        nxt = body[i + 1] if i + 1 < len(body) else ""
        if (
            line.strip()
            and not line.startswith(" ")
            and UNDERLINE.match(nxt)
            and len(nxt.strip()) >= len(line.strip())
        ):
            if any(x.strip() for x in current):
                sections.append((current_title, current))
            current_title, current = line.strip(), []
            i += 2
            continue
        current.append(line)
        i += 1
    if any(x.strip() for x in current):
        sections.append((current_title, current))
    return sections


def paragraphs(lines: list[str]) -> list[str]:
    blocks, block = [], []
    for line in lines:
        if line.strip().startswith(".. ") and not block:
            continue  # directives, comments, link targets
        if line.strip():
            block.append(line.rstrip())
        elif block:
            blocks.append("\n".join(block))
            block = []
    if block:
        blocks.append("\n".join(block))
    return [clean_rst(b).strip() for b in blocks if b.strip()]


def pack(paras: list[str], max_words: int) -> list[str]:
    """Pack paragraphs into chunks of at most `max_words` words.

    Paragraphs that are too long are split on sentences. Consecutive chunks share one paragraph
    so context isn't lost at the boundary.
    """
    units: list[str] = []
    for para in paras:
        if len(para.split()) <= max_words:
            units.append(para)
            continue
        sentences, piece = re.split(r"(?<=[.!?])\s+", para), []
        for sentence in sentences:
            if piece and len(" ".join(piece + [sentence]).split()) > max_words:
                units.append(" ".join(piece))
                piece = []
            piece.append(sentence)
        if piece:
            units.append(" ".join(piece))

    chunks: list[str] = []
    current: list[str] = []
    has_new = False  # True once current has more than the overlap paragraph
    for unit in units:
        if has_new and len("\n\n".join(current + [unit]).split()) > max_words:
            chunks.append("\n\n".join(current))
            short_tail = len(current[-1].split()) < max_words // 2
            current = [current[-1]] if short_tail else []
        current.append(unit)
        has_new = True
    if has_new:
        chunks.append("\n\n".join(current))
    return chunks


def anchor(section: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", section.lower()).strip("-")


def parse_pep(source: str, max_words: int = CHUNK_MAX_WORDS) -> list[Chunk]:
    lines = source.splitlines()
    fields, start = parse_header(lines)
    if "PEP" not in fields or not fields["PEP"].strip().isdigit():
        return []
    number = int(fields["PEP"])
    base = {
        "pep": number,
        "title": clean_rst(fields.get("Title", "")),
        "status": fields.get("Status", ""),
        "pep_type": fields.get("Type", ""),
        "created": fields.get("Created", ""),
        "python_version": fields.get("Python-Version", ""),
    }
    chunks = []
    for section, section_lines in split_sections(lines[start:]):
        if section.lower() in {"copyright", "references", "footnotes"}:
            continue
        for text in pack(paragraphs(section_lines), max_words):
            if len(text.split()) < 8:
                continue
            url = PEP_URL.format(number=number)
            if section != "Preamble":
                url += f"#{anchor(section)}"
            chunks.append(
                Chunk(
                    chunk_id=f"pep{number:04d}-{len(chunks):03d}",
                    section=section,
                    text=text,
                    url=url,
                    **base,
                )
            )
    return chunks


def build_chunks(raw_dir: Path = RAW_DIR, max_words: int = CHUNK_MAX_WORDS) -> pd.DataFrame:
    rows = []
    for path in sorted(raw_dir.glob("pep-*.rst")):
        rows.extend(asdict(c) for c in parse_pep(path.read_text(encoding="utf-8"), max_words))
    frame = pd.DataFrame(rows)
    logger.info("Parsed %d chunks from %d PEPs", len(frame), frame["pep"].nunique())
    return frame


def heading_for(row) -> str:
    return f"PEP {row['pep']} – {row['title']} › {row['section']}"
