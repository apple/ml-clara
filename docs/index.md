---
layout: default
title: CLaRa Documentation
---

# CLaRa Documentation

Welcome to the CLaRa documentation! This site provides comprehensive guides and references for using CLaRa.

## What is CLaRa?

**CLaRa** (Continuous Latent Reasoning) is a unified framework for retrieval-augmented generation that performs embedding-based compression and joint optimization in a shared continuous space.

[![Paper](https://img.shields.io/badge/Paper-Arxiv%20Link-green)](https://arxiv.org/abs/XXXX.XXXXX) [![License](https://img.shields.io/badge/License-Apple-blue)](../LICENSE) [![deploy](https://img.shields.io/badge/Hugging%20Face-CLaRa_Base-FFEB3B)](https://huggingface.co/your-org/clara-base) [![deploy](https://img.shields.io/badge/Hugging%20Face-CLaRa_Instruct-FFEB3B)](https://huggingface.co/your-org/clara-instruct) [![deploy](https://img.shields.io/badge/Hugging%20Face-CLaRa_End_to_end-FFEB3B)](https://huggingface.co/your-org/clara-e)

## Documentation

- **[Getting Started](./getting_started.md)** - Installation and quick start guide
- **[Training Guide](./training.md)** - Detailed instructions for all three training stages including data formats
- **[Inference Guide](./inference.md)** - How to use CLaRa models for inference

## Quick Links

- **GitHub Repository**: [github.com/apple/ml-CLaRa](https://github.com/apple/ml-CLaRa)
- **Main README**: [../README.md](../README.md)
- **Model Checkpoints**: [Hugging Face](https://huggingface.co/your-org/clara-base) (Coming Soon)

## Overview

CLaRa uses a three-stage training approach:

1. **Stage 1: Compression Pretraining** - Learn effective document compression
2. **Stage 2: Compression Instruction Tuning** - Adapt for downstream QA tasks  
3. **Stage 3: End-to-End Fine-tuning (CLaRa)** - Joint retrieval and generation optimization

For more details, see the [Training Guide](./training.md).

## Citation

If you use CLaRa in your research, please cite:

```bibtex
@article{clara2024,
  title={CLaRa: Unified Retrieval-Augmented Generation with Compression},
  author={[Authors]},
  journal={[Journal]},
  year={2024},
  eprint={XXXX.XXXXX},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/XXXX.XXXXX}
}
```
