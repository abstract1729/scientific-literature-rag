from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ParsedPage:
    page_number: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedSection:
    title: str
    level: int
    text: str
    page_start: int | None = None
    page_end: int | None = None


@dataclass
class ParsedDocument:
    paper_id: str
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    abstract: str | None = None
    publication_date: str | None = None
    pages: list[ParsedPage] = field(default_factory=list)
    sections: list[ParsedSection] = field(default_factory=list)
    full_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class PDFParser(ABC):
    name: str = "base"

    @abstractmethod
    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        raise NotImplementedError
