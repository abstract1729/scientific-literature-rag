## 0. Objective

Evaluate different PDF parsing approaches for extracting technically meaningful and structurally faithful content from the 125-paper corpus. The goal is to determine which parser best preserves the paper's textual content, section structure, metadata, equations, tables, and other elements required for constructing the downstream canonical PaperDocument representation.

## 1. Experimental Setup

Run multiple PDF parsing variants over the same 125-paper corpus. Each parser should produce a standardized intermediate representation containing:

- Paper metadata
- Extracted text
- Section headings and hierarchy
- Page boundaries
- Tables/equations where supported
- Parsing errors or missing elements

The experiment should compare both extraction quality and structural fidelity, with particular attention to technical papers containing equations, multi-column layouts, tables, figures, and references.

## 2. Parser Variants

Three PDF Parsers have been considered for evaluation based on the quality of a common structured output produced as a result of parsing:

- **PyMuPDF** — fast, reliable baseline.
- **Docling** — strongest candidate for structural extraction.
- **Marker** — strong PDF → Markdown reconstruction.


## 3. Experimental Results

### 3.1 PyMuPDF

PyMuPDF provides fast and reliable basic text extraction, but loses substantial document structure. Section and subsection hierarchy is not explicitly preserved, tables are largely flattened, mathematical notation is only partially retained, and figures are not represented as visual elements.

**Overall assessment:** Strong lightweight baseline, but limited suitability for structured scientific-document ingestion.

### 3.2 Docling

Docling provides substantially better document structure. It correctly represents section and subsection hierarchy, preserves table structure in Markdown, and maintains figure locations and captions. Displayed equations are a major weakness, with several formulas represented as `formula-not-decoded` placeholders. Page-level text is also not currently exposed correctly through our adapter.

**Overall assessment:** Best overall parser for the current text-centric ingestion pipeline.

### 3.3 Marker

Marker provides strong section hierarchy and better recovery of mathematical expressions than Docling. However, mathematical notation is not always reconstructed correctly, and table extraction is substantially less reliable than Docling. Figures are identified, but their semantic content is not extracted.

**Overall assessment:** Strong alternative, particularly for mathematical content, but less reliable for structured tabular information.

---

## 4. Parser Comparison

| Capability | PyMuPDF | Docling | Marker |
|---|---|---|---|
| Basic text extraction | Good | Very Good | Very Good |
| Section hierarchy | Poor | Excellent | Excellent |
| Reading order | Moderate | Good | Good |
| Table preservation | Poor | Excellent | Weak |
| Equation extraction | Weak | Weak | Better |
| Mathematical formatting | Weak | Weak | Better, but imperfect |
| Figure detection | Poor | Partial | Partial |
| Bibliography structure | Good | Good | Good |
| Processing efficiency | Excellent | Moderate | Moderate |
| Overall suitability | Baseline | **Best** | Strong alternative |

---

## 5. Key Findings

1. Plain-text extraction alone is insufficient for scientific PDFs because important information is contained in document structure, tables, equations, and figures.
2. Docling provides the strongest structural representation among the three tested parsers.
3. Marker provides better equation recovery but introduces errors in mathematical notation and table reconstruction.
4. PyMuPDF is useful as a lightweight baseline but does not preserve enough scientific-document structure for the final pipeline.
5. No tested parser reliably handles every document element.
6. Formula and figure extraction should therefore be treated as specialized downstream ingestion tasks rather than relying entirely on the primary PDF parser.

---

## 6. Selected Parser

Based on the comparison on the initial technical-paper benchmark, **Docling is selected as the primary parser for the text/document ingestion stage**.

The selection is based primarily on:

- reliable section hierarchy,
- strong reading-order preservation,
- substantially better table reconstruction,
- coherent bibliography extraction,
- explicit identification of unresolved visual/formula elements.

Its higher processing cost is acceptable because PDF ingestion is an offline operation.

### 6.1 Performance across datasets for Docling

| Aspect            | Attention Is All You Need | U-Net                 | MobileNets                        |
| ----------------- | ------------------------- | --------------------- | --------------------------------- |
| Text extraction   | Very good                 | Very good             | Very good                         |
| Section hierarchy | Excellent                 | Excellent             | Excellent                         |
| Reading order     | Good                      | Good                  | Good                         |
| Tables            | Excellent                 | Good                  | Good–Very good              |
| Equations         | Weak                      | Weak                  | Weak                         |
| Figure handling   | Placeholder + caption     | Placeholder + caption | Placeholder + caption             |
| Complex layout    | Moderate                  | Moderate              | Much harder, still manageable |


---

## 7. Limitations

The current experiment is based on a limited initial benchmark and therefore does not establish universal parser superiority.

The following limitations remain:

- Displayed mathematical equations are not consistently extracted.
- Subscripts and superscripts may be degraded.
- Figures are identified but their visual content is not represented in the text output.
- Page-wise text extraction through the current Docling adapter requires improvement.
- Table and equation quality should be evaluated across a larger and more diverse corpus.

## 8. Conclusion

**Docling** provides the best overall representation among the three evaluated PDF parsers for the current scientific-paper ingestion task. **PyMuPDF** is fast but structurally limited, while **Marker** provides stronger mathematical extraction but less reliable table reconstruction. Since scientific RAG requires preservation of relationships between text, sections, and structured information, Docling's structural advantages outweigh its higher processing cost and equation-extraction limitations. 

The weaknesses are remarkably consistent: formulas and actual figure content. That consistency is important—it suggests these are capability gaps of the text/document parser rather than failures caused by a particular PDF layout.

**Selected parser: Docling**