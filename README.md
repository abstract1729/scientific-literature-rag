# scientific-literature-rag
Scientific Literature RAG is an experimental NLP and information retrieval system for building and systematically evaluating Retrieval-Augmented Generation (RAG) over scientific research papers. The project investigates how document chunking, embeddings, sparse and dense retrieval, hybrid search, reranking, query processing, context construction, and grounding strategies affect retrieval quality and end-to-end answer generation.

The primary goal is not simply to build a RAG application, but to diagnose failures, quantify improvements, study engineering trade-offs, and develop a technically justified final configuration.

## Project Status

### Stage 0 - Information Parsing

Any research paper will contain elements across different modalities. Also, it will contain information arranged in a hierarchical fashion. We want this project to be a multi-modal RAG built for scientific paper analysis. For this reason, we primarily divide the parsing part based on the modalities and utilise different parsers based on their strengths to get the best of both worlds.

                    Scientific PDF
                         │
                ┌────────┴────────┐
                │                 │
          Document Parser     Visual Elements
                │                 │
        ┌───────┼───────┐     ┌───┴────┐
        │       │       │     │        │
       Text   Tables  Structure  Figures  Equations
        │       │       │     │        │
        └───────┴───────┴─────┴────────┘
                         │
                  Canonical Document
                         │
                    Chunking/RAG

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