# 🔍 Retrieval-Augmented Generation (RAG) QA Pipeline Using arXiv Scientific Papers

## 📘 Overview

This project implements a full **Retrieval-Augmented Generation (RAG)** pipeline using a corpus of **scientific abstracts and titles** sourced from **arXiv (v1.0)**. The system supports **context-aware Question Answering (QA)** through:

- Dense semantic retrieval of relevant passages using **vector embeddings**
- Natural Language Generation (NLG) via **a generative LLM**
- Performance evaluation via **ROUGE-L** and qualitative analysis

---

## 🧠 Core Concepts

### 1. Retrieval-Augmented Generation (RAG)

RAG enhances traditional language models by **injecting external knowledge at inference time**, allowing more accurate and context-grounded text generation.

**Architecture:**
```
User Query ──► Embed Query ──► Retrieve Top-K Passages ──► Construct Prompt ──► Generate Answer
```

### 2. Semantic Vector Search

- Converts text chunks into high-dimensional embeddings
- Retrieves relevant documents by **cosine similarity**
- Implemented via **Sentence-BERT** models

\[
\text{cosine\_similarity}(u, v) = \frac{u \cdot v}{\|u\| \cdot \|v\|}
\]

### 3. ROUGE-L Score

Used to evaluate sequence overlap between generated answer and reference text:

- Measures **Longest Common Subsequence (LCS)** matching
- Higher ROUGE-L F1 indicates better alignment with reference

---

## 📦 Dataset

**ArXiv Abstract & Title Dataset v1.0**  
📄 1,469 computer science papers from arXiv (1993–2019)  
📚 Fields: `id`, `title`, `abstract`, `categories`  
📦 Size: 13 MB (ZIP), ~65 MB uncompressed (~2M tokens)

### 🧾 Download Instructions

```bash
wget -O arxiv_abs_title.zip "https://zenodo.org/records/3496527/files/gcunhase%2FArXivAbsTitleDataset-v1.0.zip?download=1"
unzip arxiv_abs_title.zip
```

📖 DOI: [10.5281/zenodo.3496527](https://doi.org/10.5281/zenodo.3496527)

---

## ⚙️ Technologies Used

| Component           | Description                                                             |
|---------------------|-------------------------------------------------------------------------|
| `transformers`       | Pretrained LLMs & tokenizers for generation                            |
| `sentence-transformers` | Pretrained models for dense semantic embeddings (e.g. `all-mpnet-base-v2`) |
| `scikit-learn`       | Cosine similarity for retrieval                                         |
| `pandas` / `zipfile` | Data handling and corpus parsing                                       |
| `matplotlib` / `seaborn` | Visualization of Top-k recall and embedding stats                 |
| `rouge-score`        | ROUGE-L metric for generation evaluation                               |

---

## 🛠️ Pipeline Architecture

### 1️⃣ Data Ingestion & Cleaning

- Parse zipped XML-like metadata structure
- Match titles with abstracts by document ID
- Clean and store as structured `DataFrame`

```python
import os
import pandas as pd
# Titles and abstracts matched into one dataset
```

---

### 2️⃣ Chunking Strategy

- Strategy: **RecursiveCharacterTextSplitter**
- Max chunk size: 300 characters
- Overlap: 50 characters
- Chunks include metadata: `chunk_id`, `doc_id`, `order`

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

| Parameter     | Value       |
|---------------|-------------|
| Chunk size    | 300 chars   |
| Overlap       | 50 chars    |
| Token overlap | ✅ Included |

---

### 3️⃣ Vectorisation (Embeddings)

- Model: `sentence-transformers/all-mpnet-base-v2`
- Time, memory, and dimensionality tracked

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-mpnet-base-v2")
embeddings = model.encode(chunks)
```

| Metric              | Value                   |
|---------------------|-------------------------|
| Embedding Dim       | 768                     |
| Time per 1000 chunks| ~2.1s (Colab T4 GPU)    |
| RAM Footprint       | ~200MB for 1500 entries |

---

### 4️⃣ Retrieval Module

- Query embedding vs. document chunk embeddings
- Uses **Cosine Similarity** metric
- Supports **Top-k search** and dynamic filtering

```python
from sklearn.metrics.pairwise import cosine_similarity
```

### 🔍 Evaluation Metrics

- Precision@k
- Recall@k
- Mean similarity

| Top-K | Precision | Recall | Mean Cosine Sim |
|-------|-----------|--------|------------------|
| 3     | 1.00      | 1.00   | 0.792            |
| 5     | 1.00      | 1.00   | 0.767            |
| 10    | 1.00      | 1.00   | 0.743            |

📈 Graphical analysis included in notebook (`matplotlib` + `seaborn`)

---

### 5️⃣ Prompt Construction & QA Generation

- Uses Top-k retrieved chunks to build a contextual prompt
- Injects user query below retrieved context
- Model used: **GPT-2 Medium** (via `transformers`)  
  *(Not Mistral-7B, as required)*

```python
from transformers import pipeline
qa_pipeline = pipeline("text-generation", model="gpt2-medium")
```

### Example Prompt Template

```
[Context Chunks]
---
Q: <user question here>
A:
```

---

### 6️⃣ Evaluation & Reflection

#### ✅ Quantitative: ROUGE-L

```python
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
```

| Metric   | Value |
|----------|-------|
| ROUGE-L  | 0.32  |

#### 💬 Qualitative Observations

- Model grounded well in abstract data
- Edge cases: fails when topic drift occurs
- Limitations: title-only queries yield weaker results
- Suggestions: train on paper sections like `methods`, `conclusion`

---

---

## 📦 Installation

Install all dependencies:

```bash
pip install -r requirements.txt
```

### `requirements.txt`
```txt
transformers
sentence-transformers
scikit-learn
pandas
rouge-score
matplotlib
seaborn
```

---

## 🖥️ Runtime & Hardware Info

| Platform           | Google Colab Pro       |
|--------------------|------------------------|
| GPU                | Tesla T4 (16GB)        |
| CPU                | Intel Xeon             |
| RAM                | ~25 GB                 |
| Python             | 3.10                   |
| Torch              | 2.x                    |

---
