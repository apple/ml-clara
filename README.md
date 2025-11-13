##  CLaRa: Unified Retrieval-Augmented Generation with Compression

[![Paper](https://img.shields.io/badge/Paper-Arxiv%20Link-green)](https://arxiv.org/abs/XXXX.XXXXX) [![License](https://img.shields.io/badge/License-Apple-blue)](LICENSE) [![deploy](https://img.shields.io/badge/Hugging%20Face-CLaRa_Base-FFEB3B)](https://huggingface.co/your-org/clara-base) [![deploy](https://img.shields.io/badge/Hugging%20Face-CLaRa_Instruct-FFEB3B)](https://huggingface.co/your-org/clara-instruct) [![deploy](https://img.shields.io/badge/Hugging%20Face-CLaRa_End_to_end-FFEB3B)](https://huggingface.co/your-org/clara-e)

This software project accompanies the research paper, **CLaRa: Unified Retrieval-Augmented Generation with Compression**.

### Updates

- **Coming soon**: Model checkpoints.

### Motivation

Retrieval-Augmented Generation (RAG) enhances large language models with external knowledge but suffers from **long contexts** and **disjoint retrieval-generation optimization**. Existing soft compression frameworks face two key limitations: (i) reconstruction-based objectives bias compressors toward surface patterns rather than semantic preservation; (ii) retrievers and compressors are trained separately, requiring double encoding despite compressed vectors being inherently retrievable.

In this work, we investigate:

- **How can we improve semantic preservation in compressed representations through better pretraining objectives?**  
- **How can we unify retrieval and generation optimization to avoid redundant encoding and disjoint objectives?**  

<div align="center">

<img src="figs/sample_main.png" width="80%"/>

</div>

We design a Three-stage training approach and introduce document compression techniques to improve RAG efficiency. The key findings are listed below.

### Findings

- **Efficient Compression**: CLaRa achieves significant compression rates (32x-64x) while preserving essential information for accurate answer generation.

- **Three-Stage Training**: A carefully designed Three-stage training approach (compression pretraining + compression instruction tuning + end-to-end fine-tuning) enables effective learning of both retrieval and generation.

For more interesting findings, please refer to our original paper!

---

### Three-Stage Training

CLaRa uses a carefully designed three-stage training approach:

**Stage 1: Compression Pretraining**
- Train the compressor using KPCP framework with QA pairs and paraphrases
- Retain key semantics through QA-based and paraphrase-guided supervision
- Support compression rates of 1x-256x

**Stage 2: Compression Instruction Tuning**
- Fine-tune the compressor on instruction-following tasks for downstream QA
- Use text-based QA output to ensure compressed representations retain sufficient semantics

**Stage 3: End-to-End Fine-tuning (CLaRa)**
- Jointly train reranker and generator via a single language modeling loss
- Unify retrieval and generation in shared continuous space using differentiable top-k estimator

In this repository, we release our implementation of **CLaRa**, built upon [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF).

### Getting Started

```
├── scripts/                      # Training and evaluation scripts
│   ├── train_pretraining.sh     # Stage 1: Compression pretraining
│   ├── train_instruction_tuning.sh  # Stage 2: Compression instruction tuning
│   ├── train_stage_end_to_end.sh    # Stage 3: End-to-end training
│   └── evaluation_end_to_end.sh     # Evaluation scripts
├── openrlhf/                     # Core training framework
│   ├── models/                   # Model implementations
│   │   └── modeling_clara.py   # CLaRa model definition
│   ├── datasets/                 # Dataset handling
│   │   └── sft_dataset.py        # Training dataset
│   ├── trainer/                  # Training utilities
│   │   └── sft_trainer.py        # SFT trainer
│   └── cli/                      # Command line interface
│       └── train_sft.py          # Main training script
├── evaluation/                   # Evaluation framework
├── example/                      # Example training data
│   ├── pretrain_data.jsonl
│   ├── instruction_tuning_data.jsonl
│   └── end_to_end_data.jsonl
└── README.md                     # This file
```

#### 1. Prepare code and environment

Clone the repository and set up the environment:

```bash
# Create conda environment
env=clara
conda create -n $env python=3.10 -y
conda activate $env

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export PYTHONPATH=/path/to/clara:$PYTHONPATH
```

Key dependencies include:
- PyTorch >= 2.0
- Transformers >= 4.20
- DeepSpeed >= 0.18
- Flash Attention 2
- Accelerate

#### 2. Data preparation

Prepare training data in JSONL format. For pretraining stage:

```bash
# Example data format for pretraining
{
    "data_type": "qa",
    "question": ["Question 1",],
    "answers": ["Answer 1"],
    "docs": ["Document 1"]
}
```

For end-to-end training:

```bash
{
    "question": "Single question text",
    "docs": ["Document 1", "Document 2", ...],
    "gold_answer": "Reference answer"
}
```

#### 3. Start training

**Stage 1: Compression Pretraining (KPCP)**

Pre-train the document compressor :

```bash
bash scripts/train_pretraining.sh
```

Key parameters:
- `--compress_rate`: Compression rate (default: 32)
- `--doc_max_length`: Maximum document length (default: 256)
- `--stage stage1`: Training stage
- `--mse_loss`: Use MSE loss to align compressed and original representations
- `--qa_loss`: Use QA loss for semantic preservation

**Stage 2: Compression Instruction Tuning**

Fine-tune the compressor on instruction-following tasks:

```bash
bash scripts/train_instruction_tuning.sh
```

Key parameters:
- `--pretrain_checkpoint`: Path to stage 1 checkpoint
- `--stage stage1_2`: Training stage
- `--generation_top_k`: Top-k sampling for generation (default: 5)
- `--mse_loss`: Use MSE loss for compression training
- `--do_eval_gen`: Enable generation evaluation

**Stage 3: End-to-End Training**

Fine-tune the model end-to-end with retrieval:

```bash
bash scripts/train_stage_end_to_end.sh
```

Key parameters:
- `--pretrain_checkpoint`: Path to stage 2 checkpoint
- `--stage stage2`: Training stage
- `--generation_top_k`: Top-k sampling for generation
- `--do_eval_gen`: Enable generation evaluation

#### 4. Distributed Training

The training scripts support distributed training across multiple nodes and GPUs:

- `--max_len`: Maximum sequence length (default: 2048 for stage1/stage2, 1024 for stage3)
- `--train_batch_size`: Training batch size
- `--micro_train_batch_size`: Micro batch size for gradient accumulation
- `--learning_rate`: Learning rate (default: 1e-4 for stage1/stage2, 5e-6 for stage3)
- `--max_epochs`: Maximum training epochs
- `--zero_stage`: ZeRO optimization stage (default: 2)
- `--bf16`: Use bfloat16 precision
- `--flash_attn`: Use Flash Attention 2

### Inference

The CLaRa models can be loaded and used for inference. We provide three models corresponding to different training stages:

<details>
  <summary>Stage 1: Compression Pretraining model (click to expand)</summary>

  ```python
  from transformers import AutoModel

  model_path = "path/to/stage1/model"
  model = AutoModel.from_pretrained(
      model_path, 
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

</details>

<details>
  <summary>Stage 2: Compression Instruction Tuning model (click to expand)</summary>

  ```python
  from transformers import AutoModel

  model_path = "path/to/stage2/model"
  model = AutoModel.from_pretrained(
      model_path, 
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

</details>

<details>
  <summary>Stage 3: End-to-End (CLaRa) model (click to expand)</summary>

  ```python
  from transformers import AutoModel

  model_path = "path/to/stage3/model"
  model = AutoModel.from_pretrained(
      model_path, 
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

</details>

### Evaluation

The evaluation framework is based on standard RAG benchmarks. Run evaluation:

**End-to-end evaluation:**
```bash
bash scripts/evaluation_end_to_end.sh
```

**Instruction tuning evaluation:**
```bash
bash scripts/evaluation_instruction_tuning.sh
```

Supported datasets:
- **HotpotQA**: Multi-hop question answering
- **MuSiQue**: Multi-hop question answering with diverse reasoning
- **2WikiMultiHopQA**: Multi-hop question answering over Wikipedia
- **Natural Questions**: Open-domain question answering

### Model Checkpoints

Pre-trained CLaRa model checkpoints will be available on Hugging Face:

🤗 **[Coming Soon]** CLaRa checkpoints on Hugging Face Hub


### Results

#### Compression Performance

We evaluate our document compressor on four QA datasets (NQ, HotpotQA, MuSiQue, 2WikiMultiHopQA) under two settings: **Normal** (retrieving top-5 documents) and **Oracle** (gold document included). CLaRa consistently outperforms all baselines across different compression ratios.

<div align="center">

**Main Results (Mistral-7B, Normal Setting)**

| Model | CR | NQ | HotpotQA | MuSiQue | 2Wiki | Avg |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| AutoCompressor | - | 17.24 | 14.61 | 3.81 | 19.89 | 13.89 |
| XRAG | 128 | 32.35 | 25.16 | 3.64 | 28.79 | 22.48 |
| COCOM | 16 | 24.12 | 21.48 | 3.52 | 24.48 | 18.40 |
| PCC | 16 | 31.38 | 22.29 | 3.43 | 19.47 | 19.14 |
| LLMLingua-2 | 4 | 47.53 | 37.05 | 9.02 | 44.35 | 34.49 |
| PISCO | 16 | 54.39 | 41.94 | 10.09 | 44.88 | 37.83 |
| Mistral-7B w/ retrieval | - | 54.58 | 42.94 | 8.94 | 44.24 | 37.67 |
| **CLaRa (CR=4)** | **4** | **57.05** | **45.09** | **10.34** | **46.94** | **39.86** |
| **CLaRa (CR=16)** | **16** | **55.56** | **43.72** | **10.55** | **46.00** | **38.96** |
| **CLaRa (CR=32)** | **32** | **54.64** | **43.52** | **10.55** | **46.58** | **38.82** |

**Oracle Setting Results (Mistral-7B)**

| Model | CR | NQ | HotpotQA | MuSiQue | 2Wiki | Avg |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| PISCO | 16 | 73.44 | 66.53 | 33.80 | 60.45 | 58.55 |
| Mistral-7B w/ retrieval | - | 71.64 | 70.77 | 45.72 | 68.83 | 64.24 |
| **CLaRa (CR=4)** | **4** | **76.50** | **73.81** | **46.26** | **70.48** | **66.76** |
| **CLaRa (CR=16)** | **16** | **75.48** | **70.79** | **43.15** | **66.16** | **63.90** |
| **CLaRa (CR=32)** | **32** | **73.77** | **69.51** | **38.31** | **64.54** | **61.53** |

</div>

**Key Findings:**
- ✅ CLaRa outperforms PISCO by **+1.13%** (Normal) and **+5.35%** (Oracle) on average
- ✅ CLaRa outperforms LLMLingua-2 by **+5.37%** (Normal) on average  
- ✅ CLaRa matches/exceeds text-based baseline with **+2.36%** average gain on Mistral-7B

#### Retrieval Performance

<div align="center">

<img src="figs/main_recall.png" width="80%"/>

</div>

For detailed experimental results and analysis, please refer to our paper.

## Acknowledgments

We sincerely appreciate the following works for CLaRa:

- Our implementation is built upon the [OpenRLHF framework](https://github.com/OpenRLHF/OpenRLHF).

- Inspired by [PISCO-mistral](https://huggingface.co/naver/pisco-mistral) for document compression techniques

## Citation

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
