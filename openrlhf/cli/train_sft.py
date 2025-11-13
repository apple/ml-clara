#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
CLaRa Training Script

This script handles training of the CLaRa model for both stage1 and stage2 training.
"""

import argparse
import math
import os
from datetime import datetime
from typing import Optional

from transformers.trainer import get_scheduler

from openrlhf.datasets import SFTDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.models import Actor
from openrlhf.trainer.sft_trainer import SFTTrainer
from openrlhf.utils import get_strategy, get_tokenizer
from openrlhf.models.modeling_clara import CLaRaConfig, CLaRa
from openrlhf.datasets.sft_dataset import make_collate_fn


def create_clara_config(args: argparse.Namespace) -> CLaRaConfig:
    """Create CLaRa configuration from command line arguments."""
    return CLaRaConfig(
        decoder_model_name=args.pretrain,
        compr_rate=args.compress_rate,
        doc_max_length=args.doc_max_length,
        compr_n_layers=5,
        compr_use_mlp=False,
        compr_model_name=None,
        lora=True,  # LoRA on decoder and compressor
        lora_compressor=False,  # For BERT-style compressors only
        load_adapters=True,
        kbtc_training=False,
        optimize_mem_tokens=True,
        different_mem_tokens=True,
        generation_top_k=args.generation_top_k,
        device_map=None,
        lora_r=16,
        training_form="both_separately",
        training_stage=args.stage,
        sep=True,
        attn_implementation='flash_attention_2',
        stage2_retrieval_top_n=args.stage2_retrieval_top_n,
        pure_inference=args.pure_inference
    )


def setup_model(args: argparse.Namespace) -> CLaRa:
    """Setup CLaRa model from arguments."""
    cfg = create_clara_config(args)
    
    if args.pretrain_checkpoint is not None:
        print(f"Loading model from checkpoint: {args.pretrain_checkpoint}")
        model = CLaRa.from_pretrained(
            args.pretrain_checkpoint,
            training_stage=args.stage,
            generation_top_k=args.generation_top_k,
            doc_max_length=args.doc_max_length,
            compress_rate=args.compress_rate
        )
    else:
        print("Initializing new model")
        model = CLaRa(cfg)
    
    return model


def setup_datasets(args: argparse.Namespace, tokenizer, strategy, model: CLaRa):
    """Setup training and evaluation datasets."""
    # Training dataset
    train_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        dataset_split=args.dataset_split,
    )
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    
    train_dataset = SFTDataset(
        train_data,
        tokenizer,
        args.max_len,
        strategy,
    )
    
    # Training dataloader
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        collate_fn=make_collate_fn(
            model, 
            qa_loss=args.qa_loss, 
            dec_max_len=args.max_len, 
            enc_max_len=args.doc_max_length
        ),
    )

    # Evaluation dataset (optional)
    eval_dataloader = None
    if getattr(args, "eval_dataset", None):
        eval_data = blending_datasets(
            args.eval_dataset,
            None,
            strategy,
            dataset_split=args.eval_split,
        )
        eval_dataset = SFTDataset(
            eval_data,
            tokenizer,
            args.max_len,
            strategy,
        )
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            args.micro_train_batch_size,
            True,
            False,
            collate_fn=make_collate_fn(
                model, 
                qa_loss=args.qa_loss, 
                dec_max_len=args.max_len, 
                enc_max_len=args.doc_max_length
            ),
        )
    
    return train_dataset, train_dataloader, eval_dataloader


def setup_training_components(args: argparse.Namespace, model: CLaRa, train_dataset, strategy):
    """Setup optimizer, scheduler and other training components."""
    # Configure optimizer
    optimizer = strategy.create_optimizer(
        model, 
        lr=args.learning_rate, 
        betas=args.adam_betas, 
        weight_decay=args.l2
    )

    # Configure scheduler
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    # Prepare models with strategy
    model, optimizer, scheduler = strategy.prepare((model, optimizer, scheduler))
    
    return model, optimizer, scheduler, num_update_steps_per_epoch


def load_checkpoint_if_exists(args: argparse.Namespace, strategy, model: CLaRa) -> int:
    """Load checkpoint if it exists and return consumed samples."""
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model, args.ckpt_path)
        consumed_samples = states.get("consumed_samples", 0)
        strategy.print(f"Loaded checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")
    
    return consumed_samples


def train(args: argparse.Namespace):
    """Main training function."""
    # Configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()
    
    # Setup model
    model = setup_model(args)
    
    # Configure tokenizer
    tokenizer = get_tokenizer(
        args.pretrain, 
        model, 
        "right", 
        strategy, 
        use_fast=not args.disable_fast_tokenizer
    )
    strategy.print(model)

    # Setup datasets
    train_dataset, train_dataloader, eval_dataloader = setup_datasets(
        args, tokenizer, strategy, model
    )

    # Setup training components
    model, optimizer, scheduler, num_update_steps_per_epoch = setup_training_components(
        args, model, train_dataset, strategy
    )

    # Load checkpoint if exists
    consumed_samples = load_checkpoint_if_exists(args, strategy, model)

    # Ensure save directory exists
    os.makedirs(args.save_path, exist_ok=True)

    # Configure trainer
    trainer = SFTTrainer(
        model=model,
        strategy=strategy,
        optim=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        pretrain_mode=args.pretrain_mode,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        tokenizer=tokenizer,
        save_hf_ckpt=args.save_hf_ckpt,
        disable_ds_ckpt=args.disable_ds_ckpt,
    )

    # Start training
    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    # Save final model
    strategy.save_model(model, tokenizer, args.save_path)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="CLaRa Training Script")

    # Model and checkpoint arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--pretrain", type=str, required=True, help="Base model path")
    model_group.add_argument("--pretrain_checkpoint", type=str, default=None, 
                           help="CLaRa checkpoint to continue training from")
    model_group.add_argument("--stage", type=str, default="stage1", choices=["stage1", "stage1_2", "stage2", "stage2_reasoning"],
                           help="Training stage")
    model_group.add_argument("--generation_top_k", type=int, default=1, help="Top-k for generation")
    model_group.add_argument("--pure_inference", action="store_true", default=False, 
                           help="Pure inference mode")

    # CLaRa specific arguments
    clara_group = parser.add_argument_group("CLaRa Configuration")
    clara_group.add_argument("--doc_max_length", type=int, default=256, help="Max document length")
    clara_group.add_argument("--compress_rate", type=int, default=32, help="Document compression rate")
    clara_group.add_argument("--qa_loss", action="store_true", default=True, 
                            help="Use QA loss for joint training")
    clara_group.add_argument("--stage2_mips", action="store_true", default=False, 
                            help="Use MIPS for stage2 retrieval")
    clara_group.add_argument("--stage2_retrieval_top_n", type=int, default=1, 
                            help="Top-n documents for stage2 retrieval")
    clara_group.add_argument("--mse_loss", action="store_true", default=False, 
                            help="Add MSE loss during compression training")
    clara_group.add_argument("--do_eval_gen", action="store_true", default=False, 
                            help="Evaluate generation during eval")

    # Checkpoint and saving
    checkpoint_group = parser.add_argument_group("Checkpointing")
    checkpoint_group.add_argument("--save_path", type=str, default="./ckpt", help="Model save path")
    checkpoint_group.add_argument("--save_steps", type=int, default=-1, help="Save every N steps")
    checkpoint_group.add_argument("--save_hf_ckpt", action="store_true", default=False, 
                                help="Save HuggingFace checkpoint")
    checkpoint_group.add_argument("--disable_ds_ckpt", action="store_true", default=False, 
                                help="Disable DeepSpeed checkpoint")
    checkpoint_group.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_sft", 
                                help="Checkpoint path to load")
    checkpoint_group.add_argument("--load_checkpoint", action="store_true", default=False, 
                                help="Load from checkpoint")
    checkpoint_group.add_argument("--max_ckpt_num", type=int, default=3, help="Max checkpoint number")
    checkpoint_group.add_argument("--max_ckpt_mem", type=int, default=1e8, help="Max checkpoint memory")

    # Training configuration
    training_group = parser.add_argument_group("Training Configuration")
    training_group.add_argument("--max_epochs", type=int, default=2, help="Maximum training epochs")
    training_group.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    training_group.add_argument("--lr_warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    training_group.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr", 
                              help="Learning rate scheduler")
    training_group.add_argument("--l2", type=float, default=0, help="Weight decay")
    training_group.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), 
                              help="Adam optimizer betas")
    training_group.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    training_group.add_argument("--pretrain_mode", action="store_true", default=False, 
                              help="Use pretrain loss")

    # DeepSpeed and distributed training
    distributed_group = parser.add_argument_group("Distributed Training")
    distributed_group.add_argument("--micro_train_batch_size", type=int, default=8, 
                                 help="Batch size per GPU")
    distributed_group.add_argument("--train_batch_size", type=int, default=128, 
                                 help="Global training batch size")
    distributed_group.add_argument("--local_rank", type=int, default=-1, 
                                 help="Local rank for DeepSpeed")
    distributed_group.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    distributed_group.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    distributed_group.add_argument("--gradient_checkpointing", action="store_true", default=False, 
                                 help="Enable gradient checkpointing")
    distributed_group.add_argument("--flash_attn", action="store_true", default=False, 
                                 help="Enable FlashAttention2")
    distributed_group.add_argument("--ds_tensor_parallel_size", type=int, default=1, help="DeepSpeed Tensor parallel size")
    # Dataset configuration
    dataset_group = parser.add_argument_group("Dataset Configuration")
    dataset_group.add_argument("--dataset", type=str, required=True, help="Training dataset path")
    dataset_group.add_argument("--dataset_probs", type=str, default=None, 
                             help="Dataset sampling probabilities")
    dataset_group.add_argument("--eval_dataset", type=str, default=None, help="Evaluation dataset path")
    dataset_group.add_argument("--dataset_split", type=str, default="train", help="Dataset split")
    dataset_group.add_argument("--eval_split", type=str, default="train", help="Evaluation split")
    dataset_group.add_argument("--max_samples", type=int, default=1000000, 
                             help="Maximum samples to use")
    dataset_group.add_argument("--max_len", type=int, default=2048, help="Maximum sequence length")

    # Logging and monitoring
    logging_group = parser.add_argument_group("Logging and Monitoring")
    logging_group.add_argument("--logging_steps", type=int, default=1, help="Log every N steps")
    logging_group.add_argument("--eval_steps", type=int, default=-1, help="Evaluate every N steps")
    logging_group.add_argument("--use_wandb", type=str, default=None, help="Wandb project name")
    logging_group.add_argument("--wandb_org", type=str, default=None, help="Wandb organization")
    logging_group.add_argument("--wandb_group", type=str, default=None, help="Wandb group")
    logging_group.add_argument("--wandb_project", type=str, default="CLaRa", help="Wandb project")
    logging_group.add_argument("--wandb_run_name", type=str, 
                             default="clara_%s" % datetime.now().strftime("%m%dT%H:%M"),
                             help="Wandb run name")
    logging_group.add_argument("--use_tensorboard", type=str, default=None, 
                             help="TensorBoard logging path")

    # Additional arguments
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument("--seed", type=int, default=42, help="Random seed")
    misc_group.add_argument("--disable_fast_tokenizer", action="store_true", default=False,
                          help="Disable fast tokenizer")
    misc_group.add_argument("--use_ms", action="store_true", default=False, 
                          help="Use ModelScope")

    return parser


def validate_arguments(args: argparse.Namespace):
    """Validate command line arguments."""
    # Validate training stage
    if args.stage not in ["stage1", "stage1_2", "stage2", "stage2_reasoning"]:
        raise ValueError(f"Invalid stage: {args.stage}")
    
    # Validate compression parameters
    if args.compress_rate <= 0:
        raise ValueError("Compression rate must be positive")
    
    if args.doc_max_length <= 0:
        raise ValueError("Document max length must be positive")
    
    if args.doc_max_length % args.compress_rate != 0:
        print(f"Warning: doc_max_length ({args.doc_max_length}) is not divisible by "
              f"compress_rate ({args.compress_rate})")
    
    # Validate paths
    if args.dataset and not os.path.exists(args.dataset):
        print(f"Warning: Dataset path does not exist: {args.dataset}")
    
    if args.pretrain_checkpoint and not os.path.exists(args.pretrain_checkpoint):
        raise ValueError(f"Pretrain checkpoint path does not exist: {args.pretrain_checkpoint}")


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    validate_arguments(args)
    
    # Handle ModelScope patch
    if args.use_ms:
        try:
            from modelscope.utils.hf_util import patch_hub
            patch_hub()
            print("ModelScope hub patched successfully")
        except ImportError:
            print("Warning: ModelScope not available, skipping hub patch")
    
    # Print configuration
    print("=" * 60)
    print("CLaRa Training Configuration")
    print("=" * 60)
    print(f"Training stage: {args.stage}")
    print(f"Base model: {args.pretrain}")
    print(f"Document max length: {args.doc_max_length}")
    print(f"Compression rate: {args.compress_rate}")
    print(f"Generation top-k: {args.generation_top_k}")
    print(f"Dataset: {args.dataset}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size (micro/global): {args.micro_train_batch_size}/{args.train_batch_size}")
    print("=" * 60)
    
    # Start training
    train(args)
    print("Training completed successfully!")


if __name__ == "__main__":
    main()