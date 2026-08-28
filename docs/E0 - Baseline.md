# E0 — ColPali Baseline

## Objective

Establish a working multimodal RAG baseline for scientific literature using ColPali for page-level visual retrieval and Qdrant for persistent vector storage.

The primary design decision in E0 is to treat each PDF page as a visual retrieval unit rather than converting the scientific PDFs into conventional text chunks. This allows the retrieval system to operate directly over pages containing text, mathematical equations, tables, figures, diagrams, and other visual information.

---

## Version History

| Version | Status | Description |
|---|---|---|
| v0.1.1 | Completed | End-to-end ColPali visual RAG baseline |
| v0.1.2 | Planned | Retrieval evaluation infrastructure and validated evaluation set |

---

# v0.1.1 — ColPali Visual RAG Baseline

## Overview

The first milestone establishes the complete retrieval pipeline from raw scientific PDFs to retrieved PDF pages.

The pipeline is:

PDF
→ Page Rendering
→ ColPali Multi-Vector Encoding
→ Qdrant
→ Query Encoding
→ Late-Interaction Retrieval
→ Top-K Page Retrieval

A downstream generation smoke test was also performed using the retrieved page images as multimodal context.

---

## Corpus

The initial corpus contains:

- 125 scientific research papers
- 2,884 rendered PDF pages
- Page-level visual representations stored in Qdrant
- Approximately 3.5 GB of local Qdrant storage

Each page is treated as an independent retrieval unit.

Example page identifier:

`2106.09685_page_0004`

The individual ColPali vectors within a page are internal components of the page representation and are not treated as independent retrieval units.

---

## 1. PDF Page Rendering

The PDF corpus is first converted into page-level images.

The renderer maintains a persistent representation of the corpus:

```text
data/processed/colpali/pages/
├── <paper_id>/
│   ├── page_0001.png
│   ├── page_0002.png
│   └── ...
```