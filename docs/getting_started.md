---
layout: default
title: Getting Started
permalink: /getting_started/
---

# Getting Started with CLaRa

This guide will help you get started with CLaRa, from installation to running your first training.

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- PyTorch 2.0+
- CUDA 11.8 or 12.x

### Step 1: Create Conda Environment

```bash
env=clara
conda create -n $env python=3.10 -y
conda activate $env
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `torch>=2.0`
- `transformers>=4.20`
- `deepspeed>=0.18`
- `flash-attn>=2.8.0`
- `accelerate>=1.10.1`
- `peft>=0.17.1`

### Step 3: Set Environment Variables

```bash
export PYTHONPATH=/path/to/clara:$PYTHONPATH
```

## Quick Start

### 1. Prepare Your Data

CLaRa uses JSONL format for training data. See the [Training Guide](./training.md) for data format details.

### 2. Train Stage 1: Compression Pretraining

```bash
bash scripts/train_pretraining.sh
```

### 3. Train Stage 2: Instruction Tuning

```bash
bash scripts/train_instruction_tuning.sh
```

### 4. Train Stage 3: End-to-End Training

```bash
bash scripts/train_stage_end_to_end.sh
```

### 5. Run Inference

See the [Inference Guide](./inference.md) for examples of using all three model stages.

## Next Steps

- [Training Guide](./training.md) - Detailed training instructions and data formats
- [Inference Guide](./inference.md) - Inference examples for all model stages

