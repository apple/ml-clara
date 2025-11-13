---
layout: default
title: Training Guide
permalink: /training/
---

# Training Guide

This guide covers the three-stage training process in CLaRa.

## Overview

CLaRa uses a three-stage training approach:

1. **Stage 1**: Compression Pretraining
2. **Stage 2**: Compression Instruction Tuning
3. **Stage 3**: End-to-End Fine-tuning (CLaRa)

## Stage 1: Compression Pretraining

Train the compressor to learn effective document compression.

### Key Parameters

- `--stage stage1`: Training stage identifier
- `--compress_rate`: Compression rate (default: 32)
- `--doc_max_length`: Maximum document length (default: 256)
- `--mse_loss`: Use MSE loss for compression alignment
- `--qa_loss`: Use QA loss for semantic preservation

### Example Command

```bash
bash scripts/train_pretraining.sh
```

### Data Format

**Stage 1 Pretraining Data:**
```json
{
    "data_type": "qa",
    "question": ["Question 1", "Question 2", ...],
    "answers": ["Answer 1", "Answer 2", ...],
    "docs": ["Document 1", "Document 2", ...]
}
```

## Stage 2: Compression Instruction Tuning

Fine-tune the compressor on instruction-following tasks.

### Key Parameters

- `--stage stage1_2`: Training stage identifier
- `--pretrain_checkpoint`: Path to Stage 1 checkpoint
- `--generation_top_k`: Top-k sampling (default: 5)
- `--mse_loss`: Continue using MSE loss
- `--do_eval_gen`: Enable generation evaluation

### Example Command

```bash
bash scripts/train_instruction_tuning.sh
```

### Data Format

**Stage 2 Instruction Tuning Data:**
```json
{
    "question": "Single question text",
    "docs": ["Document 1", "Document 2", ...],
    "gold_answer": "Reference answer",
    "answer": "Generated answer"
}
```

## Stage 3: End-to-End Training

Jointly train reranker and generator with retrieval.

### Key Parameters

- `--stage stage2`: Training stage identifier
- `--pretrain_checkpoint`: Path to Stage 2 checkpoint
- `--generation_top_k`: Top-k sampling for generation
- `--do_eval_gen`: Enable generation evaluation

### Example Command

```bash
bash scripts/train_stage_end_to_end.sh
```

### Data Format

**Stage 3 End-to-End Data:**
```json
{
    "question": "Single question text",
    "docs": ["Document 1", "Document 2", ...],
    "gold_answer": "Reference answer"
}
```

## Distributed Training

All training stages support distributed training across multiple nodes and GPUs.

### Key Parameters

- `--max_len`: Maximum sequence length (2048 for stage1/stage2, 1024 for stage3)
- `--train_batch_size`: Training batch size
- `--micro_train_batch_size`: Micro batch size for gradient accumulation
- `--learning_rate`: Learning rate (1e-4 for stage1/stage2, 5e-6 for stage3)
- `--max_epochs`: Maximum training epochs
- `--zero_stage`: ZeRO optimization stage (default: 2)
- `--bf16`: Use bfloat16 precision
- `--flash_attn`: Use Flash Attention 2

## Monitoring Training

Training progress is logged via:
- Console output
- Wandb (if configured)
- Checkpoint files

Checkpoints are saved at the path specified by `--save_path`.
