# E0 — Baseline Retrieval Evaluation

## 1. Experimental setup

The baseline uses **vanilla ColPali + Qdrant visual page retrieval** over the entire corpus, without metadata filtering or a second-stage reranker.

The evaluation benchmark contains **91 questions**:

| Category       | Count |
| -------------- | ----: |
| Answerable     |    80 |
| Unanswerable   |    11 |
| Direct factual |    14 |
| Formula        |    15 |
| Table          |    15 |
| Figure         |    12 |
| Conceptual     |     9 |
| Multi-hop      |     7 |
| Cross-paper    |     8 |

The benchmark is deliberately heterogeneous, with **58 hard, 26 medium, and 7 easy questions**. Each answerable question is evaluated against its manually validated **gold PDF page(s)**; supporting pages are not treated as required retrieval targets.

The retrieval system returns the top-10 pages for every question, allowing evaluation at multiple retrieval depths.

---

# 2. Overall retrieval performance

The central result for the 80 answerable questions is:

| Metric             | E0 Baseline |
| ------------------ | ----------: |
| Recall@1           |  **27.08%** |
| Recall@3           |  **45.56%** |
| Recall@5           |  **50.19%** |
| Recall@10          |  **68.15%** |
| Hit@5              |  **62.50%** |
| NDCG@5             |   **0.431** |
| MRR                |   **0.478** |
| Gold-page coverage |  **68.15%** |
| Paper coverage     |  **90.63%** |

The most important observation is the difference between **paper-level retrieval and page-level retrieval**.

The system identifies the correct paper for approximately **90.6%** of answerable questions, while complete gold-page coverage is only **68.2%**. This indicates that the baseline generally understands **which paper is relevant**, but has substantially more difficulty locating **the exact page(s) containing the required evidence**.

That distinction is important for the subsequent experiments.

---

# 3. Retrieval-depth behaviour

There is a clear improvement as the retrieval budget increases:

```text
Recall@1   27.08%
Recall@3   45.56%
Recall@5   50.19%
Recall@10  68.15%
```

The jump from **27.1% at rank 1 to 68.2% at rank 10** indicates that relevant pages are frequently present in the retrieved candidate set but are not ranked highly enough.

This is reinforced by:

* Hit@5 = **62.5%**
* MRR = **0.478**
* NDCG@5 = **0.431**

Therefore, the primary baseline weakness is **ranking quality rather than complete inability to retrieve relevant information**.

In practical terms, the baseline often retrieves the correct evidence somewhere in its candidate set, but competing pages are ranked above it.

This is exactly the type of behaviour where a **metadata-constrained first stage + reranking stage** is worth investigating.

---

# 4. Performance by question type

The benchmark reveals substantial differences between retrieval scenarios.

| Question type  |   Recall@5 |  Recall@10 | Paper coverage |
| -------------- | ---------: | ---------: | -------------: |
| Direct factual |     46.43% |     64.29% |           100% |
| Formula        |     40.00% |     73.33% |         93.33% |
| Table          | **66.67%** |     73.33% |         93.33% |
| Figure         |     62.50% | **83.33%** |         91.67% |
| Conceptual     |     57.41% | **66.67%** |           100% |
| Multi-hop      |     41.67% |     64.29% |         85.71% |
| Cross-paper    | **25.83%** | **37.71%** |     **56.25%** |

### Stronger areas

**Figure and table questions** perform relatively well at higher retrieval depths.

Figure questions achieve:

* Recall@5: **62.5%**
* Recall@10: **83.3%**

Table questions achieve:

* Recall@5: **66.7%**
* Recall@10: **73.3%**

This is encouraging for a visual retrieval architecture because these are precisely the kinds of elements where conventional text chunking/parsing can lose information.

### Weakest area: cross-paper retrieval

Cross-paper questions are clearly the largest weakness:

* Recall@5: **25.8%**
* Recall@10: **37.7%**
* Gold-page coverage: **37.7%**
* Paper coverage: **56.3%**

This is substantially below every other category.

>This suggests that when the query requires evidence from **multiple papers**, the global similarity search tends to concentrate on only part of the required evidence rather than ensuring balanced coverage across the relevant papers.
This is an important finding because it gives a concrete **motivation for metadata-aware retrieval.**

---

# 5. Multi-hop retrieval

Multi-hop questions show an interesting result.

The system has:

* Hit@1: **57.1%**
* Hit@3: **85.7%**
* Hit@5: **85.7%**
* Hit@10: **85.7%**

Yet:

* Recall@5 = **41.7%**
* Gold-page coverage = **64.3%**

This means the system frequently retrieves **at least one required page**, but does not necessarily retrieve **all pages required to answer the question**.

For a multi-hop question, finding one relevant page is insufficient. The retrieval system needs to cover the complete evidence chain.

>Therefore, the baseline appears reasonably capable of finding *a relevant starting point*, but considerably less reliable at assembling the complete evidence set.

---

# 6. Unanswerable questions

The 11 unanswerable questions produce:

* Mean maximum retrieval score: **25.89**
* Mean retrieval score: **23.07**

The important finding is that the baseline still produces **high-confidence-looking retrieval scores for questions whose evidence does not exist in the corpus**.

This demonstrates a limitation of vanilla top-k retrieval:

> **The retriever is forced to return pages even when no page actually contains the answer.**

Therefore, the raw ColPali similarity score cannot currently be interpreted as an answerability signal.

This becomes particularly relevant for a later retrieval improvement: introducing **score calibration / thresholding or candidate filtering** could allow the system to distinguish between "relevant evidence exists" and "the nearest available pages are merely the least irrelevant pages."

---

# 7. Difficulty behaviour

The difficulty breakdown needs to be interpreted carefully because the dataset is highly imbalanced:

* Easy: **7**
* Medium: **26**
* Hard: **58 total**, of which **47 are answerable**

The answerable hard questions achieve:

* Recall@5: **47.84%**
* Recall@10: **71.31%**
* Paper coverage: **86.17%**

Medium questions actually achieve slightly higher Recall@5 (**56.41%**) than hard questions.

However, the **7 easy questions are too few** to draw meaningful conclusions from their 42.86% Recall@10. In addition, their latency statistics contain very large outliers.

---

# 8. Latency

For the 80 answerable questions:

| Metric           |     Mean |       Median |      P95 |
| ---------------- | -------: | -----------: | -------: |
| Query encoding   |   396 ms |   **140 ms** |   861 ms |
| Qdrant retrieval |   820 ms |   **917 ms** | 1,065 ms |
| Total            | 1,216 ms | **1,058 ms** | 1,632 ms |

The encoding mean is inflated by a few extreme outliers: the maximum encoding latency is approximately **15 seconds**, while the median is only **140 ms**.

Therefore, for reporting the baseline latency, I would emphasize:

> **~1.06 s median end-to-end retrieval latency and ~1.63 s P95**, while noting that occasional local-system outliers affected the mean.

The Qdrant component is relatively stable compared with the encoding component, whereas query encoding exhibits the largest variance.

Given that these experiments were conducted on a local development machine, I would treat latency primarily as a **baseline engineering measurement**, rather than making strong claims about production throughput. The latency might substantially improve once shifted to a dedicated compute as there will not be other processes competing for resources.

---

# 9. What the baseline tells us

The E0 experiment establishes a useful baseline rather than demonstrating that the retriever is already optimal.

The overall picture is:

### What works

1. **Strong paper-level identification**

   * 90.6% paper coverage.

2. **Useful candidate retrieval**

   * Recall increases from 27.1% → 68.2% between top-1 and top-10.
   * This means relevant pages are often present in the retrieved candidate pool.

3. **Good visual-element retrieval**

   * Tables and figures are among the strongest categories.
   * This supports the motivation for using a vision-based page retriever rather than relying entirely on text parsing.

### What does not work well

1. **Exact page ranking**

   * Only 27.1% Recall@1.
   * Relevant pages often exist lower in the ranking.

2. **Complete evidence retrieval**

   * Gold-page coverage is only 68.2%.
   * This is particularly problematic for multi-hop questions.

3. **Cross-paper retrieval**

   * Only 56.3% paper coverage and 37.7% Recall@10.
   * The baseline does not reliably distribute retrieval across multiple required papers.

4. **Unanswerable detection**

   * High retrieval scores are still produced for unanswerable questions.
   * Vanilla top-k retrieval has no explicit abstention mechanism.

---

# 10. Implication for E1

These results give a clear experimental motivation for **E1: Metadata Filtering**.

The hypothesis should **not** simply be:

> "Metadata filtering will make retrieval faster."

The stronger hypothesis is:

> **Constraining the retrieval search space using query-relevant metadata should reduce irrelevant candidates and improve the probability that the limited top-k budget is allocated to relevant pages.**

For example, when the query clearly refers to a particular paper, author, dataset, method, or other indexed metadata, we can restrict the candidate pool before performing visual similarity retrieval.

The metrics to compare against E0 should remain identical:

```text
Recall@1
Recall@3
Recall@5
Recall@10
Hit@5
NDCG@5
MRR
Gold-page coverage
Paper coverage
Latency
```

For cross-paper questions, metadata filtering will require particular care: **the filter must preserve all relevant papers rather than accidentally improving precision by eliminating one of the required evidence sources.**

---

# 11. Baseline verdict

### Overall verdict

**The vanilla ColPali + Qdrant baseline is functional but has substantial ranking and evidence-coverage limitations.**

It demonstrates that ColPali can retrieve relevant scientific-document pages without conventional text chunking, parsing, or OCR-based preprocessing, and the **90.6% paper coverage** indicates strong high-level document identification. However, the **68.2% Recall@10 / gold-page coverage** and especially the **37.7% Recall@10 for cross-paper questions** show that the baseline does not reliably retrieve the complete evidence required for complex scientific QA.

The large gap between paper coverage and gold-page coverage is arguably the most useful E0 finding: **finding the right document is considerably easier than finding all the precise evidence pages within it.**

That gives a strong empirical justification for the next retrieval experiment.

---

# Resume Points — E0

* **Built and benchmarked** a multimodal scientific-literature retrieval pipeline using ColPali + Qdrant over a 91-question benchmark, achieving **90.6% paper coverage and 68.2% gold-page coverage at top-10 retrieval**.
* **Diagnosed retrieval bottlenecks** across factual, formula, table, figure, multi-hop and cross-paper queries, identifying cross-paper retrieval as the primary weakness with **37.7% Recall@10**.
* **Established an evaluation and latency framework** reporting ranking metrics and **~1.06 s median end-to-end retrieval latency**, providing a reproducible baseline for subsequent retrieval optimizations.

---

# Interview Speech

> “For the baseline, I implemented vanilla ColPali with Qdrant for visual page-level retrieval and evaluated it on 91 manually curated scientific QA queries. The retriever achieved about 90.6% paper coverage and 68.2% gold-page coverage at top-10, with Recall increasing from 27.1% at top-1 to 68.2% at top-10, showing that relevant pages were often retrieved but not ranked highly enough. The major weakness was cross-paper retrieval, where Recall@10 dropped to about 37.7%, indicating difficulty in covering evidence across multiple documents. This motivated the next experiment of using metadata-aware candidate filtering before visual retrieval.”

> “I also evaluated latency separately for query encoding and Qdrant search, obtaining roughly 1.06 seconds median end-to-end retrieval latency. The unanswerable subset showed that vanilla similarity retrieval still returned high scores even when the corpus contained no answer, highlighting the need for better candidate filtering or score-based abstention.” 
