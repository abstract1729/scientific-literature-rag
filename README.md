# Scientific Literature RAG

A multimodal Retrieval-Augmented Generation (RAG) system designed for technical and scientific research papers. Instead of converting PDFs into conventional text chunks, the system treats each PDF page as a visual document and uses ColPali to build multi-vector visual representations that preserve information contained in text, equations, tables, figures, and diagrams.

The project currently establishes a ColPali-based baseline using page-level visual retrieval with Qdrant as the vector store. The system covers PDF ingestion, page rendering, ColPali indexing, late-interaction retrieval, and downstream multimodal generation. The project is organized into incremental experimental stages to evaluate retrieval quality, retrieval efficiency, and multimodal generation performance.

## Experimental Roadmap

### E0 — ColPali Baseline

Establish a working end-to-end multimodal RAG baseline using ColPali and Qdrant. PDF pages are rendered as images, represented using ColPali multi-vectors, and retrieved using late-interaction MaxSim scoring.

The baseline serves as the reference system against which all subsequent retrieval and generation experiments are compared.

### E1 — Metadata Filtering

Evaluate whether query-dependent metadata filtering can reduce the retrieval search space while maintaining retrieval quality.

The experiment compares filtered retrieval against the full-corpus ColPali baseline using retrieval accuracy and latency metrics.

### E2 — Metadata Filtering + Re-ranking

Introduce a two-stage retrieval strategy using reduced page representations for efficient candidate retrieval, followed by re-ranking using the original ColPali multi-vector representations and late-interaction scoring.

The experiment evaluates the trade-off between retrieval latency and retrieval quality.

### E3 — Multimodal Generation with Qwen2.5-VL

Use the retrieved PDF page images as visual context for a locally deployed Qwen2.5-VL model.

The experiment evaluates whether an open-source vision-language model can generate grounded answers from the visually retrieved scientific evidence while enabling local inference without dependence on a hosted generation API.