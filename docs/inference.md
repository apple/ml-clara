---
layout: default
title: Inference Guide
permalink: /inference/
---

# Inference Guide

This guide shows how to use CLaRa models for inference at different stages.

## Loading Models

CLaRa models can be loaded using the standard `AutoModel` interface:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "path/to/model",
    trust_remote_code=True
).to('cuda')
```

## Stage 1: Compression Pretraining Model

Generate paraphrases from compressed document representations.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "path/to/stage1/model",
    trust_remote_code=True
).to('cuda')

# Example documents
documents = [
    [
        "Document 1 content...",
        "Document 2 content...",
        "Document 3 content..."
    ]
]

questions = ["" for _ in range(len(documents))]

# Generate paraphrase from compressed representations
output = model.generate_from_paraphrase(
    questions=questions, 
    documents=documents, 
    max_new_tokens=64
)

print('Generated paraphrase:', output[0])
```

## Stage 2: Compression Instruction Tuning Model

Generate answers from compressed representations for QA tasks.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "path/to/stage2/model",
    trust_remote_code=True
).to('cuda')

# Example documents and question
documents = [
    [
        "Document 1 content...",
        "Document 2 content...",
        "Document 3 content..."
    ]
]

questions = ["Your question here"]

# Generate answer from compressed representations
output = model.generate_from_text(
    questions=questions, 
    documents=documents, 
    max_new_tokens=64
)

print('Generated answer:', output[0])
```

## Stage 3: End-to-End (CLaRa) Model

Generate answers with retrieval and reranking using joint optimization.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "path/to/stage3/model",
    trust_remote_code=True
).to('cuda')

# Example documents and question
# Note: Stage 3 supports retrieval with multiple candidate documents
documents = [
    ["Document 1 content..." for _ in range(20)]  # 20 candidate documents
]

questions = ["Your question here"]

# Generate answer with retrieval and reranking
# The top-k is decided by generation_top_k in config.json
output, topk_indices = model.generate_from_questions(
    questions=questions, 
    documents=documents, 
    max_new_tokens=64
)

print('Generated answer:', output[0])
print('Top-k selected document indices:', topk_indices)
```

## Key Parameters

- `max_new_tokens`: Maximum number of tokens to generate (default: 128)
- `generation_top_k`: Number of top documents to select (configured in model config)

## Model Methods

- `generate_from_paraphrase()` - Stage 1: Generate paraphrases
- `generate_from_text()` - Stage 2: Generate answers from compressed docs
- `generate_from_questions()` - Stage 3: Generate with retrieval and reranking



