# Semantic Search with Fuzzy Clustering and Semantic Cache

This project implements a semantic search system using the 20 Newsgroups dataset.
## Dataset

This project uses the 20 Newsgroups dataset.

Download it from:
https://archive.ics.uci.edu/dataset/113/twenty+newsgroups

Extract it into:

data/20_newsgroups

## Components

1. Embedding generation using Sentence Transformers
2. Vector database using FAISS
3. Fuzzy clustering using Gaussian Mixture Models
4. Semantic cache for query reuse
5. FastAPI service
6. Docker containerization

## Running with Docker

Build and run:

```bash
docker compose up --build