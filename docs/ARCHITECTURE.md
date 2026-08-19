# Architecture and design decisions

## Two execution modes

`local` is the default. It uses deterministic fixtures, costs nothing, needs no AWS account, and
makes tests and interviews reproducible. `aws` replaces the fixture runtime with Amazon Bedrock,
S3, and FAISS while preserving the same API contract.

## Index lifecycle

The original prototype proved joint indexing of 59 text chunks and 129 images with Amazon Nova
Multimodal Embeddings. This repository treats indexing as an offline pipeline and serving as a
separate runtime concern. The serving container downloads a versioned FAISS index and JSON metadata
from S3, validates their item counts, and reuses them across requests.

## Why FAISS and S3

FAISS is inexpensive and sufficient for a focused corpus. S3 provides durable, private storage for
the source assets and generated index. For higher update frequency or multi-tenant filtering,
replace FAISS with OpenSearch Serverless, Aurora PostgreSQL with pgvector, or a managed vector store.

## Grounding contract

The answer response contains citations independently of model prose. The backend selects the image
URL from retrieved metadata instead of asking the model to reproduce a URL. This prevents invented
links and makes the returned evidence machine-readable.

## Observability

Every completed request emits a structured event containing request ID, runtime mode, item count,
citation count, tokens, estimated cost, and total latency. The API returns the same operational
measurements to make the portfolio demo inspectable.
