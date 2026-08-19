# Portfolio case study

## Problem

Traditional text-only search cannot connect a literary question with both the relevant passage and
its visual interpretation. Multilingual collections add another challenge: questions and sources
may use Georgian, English, or Arabic.

## Prototype evidence

The original RAG-Knight notebook built a shared 1,024-dimensional index containing 188 items: 59
text chunks and 129 images. Saved executions demonstrate answers in Georgian, English, Arabic, and
Georgian transliterated into Latin characters, as well as stanza and character queries.

## Engineering evolution

This repository converts that notebook into a backend product pattern: a versioned API, clean AWS
adapter, deterministic demo runtime, explicit citations, pre-signed media access, token/cost and
latency attribution, evaluation fixtures, tests, CI, and container packaging.

## What this demonstrates

- Multimodal retrieval rather than text retrieval with decorative media
- AWS Bedrock and serverless/container architecture knowledge
- Separation of offline indexing from online inference
- Reproducible evaluation and operational guardrails
- Translation of a successful experiment into a reusable engineering asset
