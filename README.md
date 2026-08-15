# scientific-literature-rag
Scientific Literature RAG is an experimental NLP and information retrieval system for building and systematically evaluating Retrieval-Augmented Generation (RAG) over scientific research papers. The project investigates how document chunking, embeddings, sparse and dense retrieval, hybrid search, reranking, query processing, context construction, and grounding strategies affect retrieval quality and end-to-end answer generation.

The primary goal is not simply to build a RAG application, but to diagnose failures, quantify improvements, study engineering trade-offs, and develop a technically justified final configuration.

## Project Status

Early development.

The project will be developed incrementally:

**Baseline → Evaluate → Experiment → Analyze → Improve → Repeat**

## Initial Scope

- Scientific research paper corpus
- Text-based document processing
- Retrieval-Augmented Generation
- Sparse and dense information retrieval
- Embedding and reranking experiments
- RAG evaluation
- Grounding and hallucination analysis
- Controlled ML experiments and ablation studies

## Repository

```text
scientific-literature-rag/
├── README.md
├── .gitignore
├── requirements.txt
├── src/
├── configs/
├── experiments/
├── benchmarks/
├── results/
├── scripts/
└── docs/