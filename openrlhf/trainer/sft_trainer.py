#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
CLaRa SFT Trainer

This module provides the supervised fine-tuning trainer for CLaRa models.
"""

import os
import re
import string
from abc import ABC
from collections import Counter
from typing import Dict, Any, Optional, List

import torch
from torch.optim import Optimizer
from tqdm import tqdm

# from openrlhf.models import SFTLoss
from openrlhf.utils.distributed_sampler import DistributedSampler

# Set torch print options for better debugging
torch.set_printoptions(profile="default")


class EvaluationMetrics:
    """Utility class for evaluation metrics."""
    
    @staticmethod
    def bool_mapping(text: str) -> str:
        """Map boolean values to yes/no format."""
        mapping = {"True": "yes", "False": "no"}
        return mapping.get(text, text)
    
    @staticmethod
    def normalize_answer(text: str) -> str:
        """Normalize answer text for comparison."""
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)
        
        def white_space_fix(text):
            return " ".join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(text))))
    
    @classmethod
    def exact_match_score(cls, prediction: str, ground_truth: str) -> bool:
        """Calculate exact match score."""
        pred_norm = cls.normalize_answer(cls.bool_mapping(prediction))
        gt_norm = cls.normalize_answer(cls.bool_mapping(ground_truth))
        return pred_norm == gt_norm
    
    @classmethod
    def cover_exact_match_score(cls, prediction: str, ground_truth: str) -> bool:
        """Calculate coverage exact match score (all ground truth tokens in prediction)."""
        pred_tokens = cls.normalize_answer(cls.bool_mapping(prediction)).split()
        gt_tokens = cls.normalize_answer(cls.bool_mapping(ground_truth)).split()
        return all(token in pred_tokens for token in gt_tokens)
    
    @classmethod
    def f1_score(cls, prediction: str, ground_truth: str) -> float:
        """Calculate F1 score between prediction and ground truth."""
        pred_norm = cls.normalize_answer(cls.bool_mapping(prediction))
        gt_norm = cls.normalize_answer(cls.bool_mapping(ground_truth))
        
        # Handle special cases for yes/no/noanswer
        if pred_norm in ["yes", "no", "noanswer"] and pred_norm != gt_norm:
            return 0.0
        if gt_norm in ["yes", "no", "noanswer"] and pred_norm != gt_norm:
            return 0.0
        
        pred_tokens = pred_norm.split()
        gt_tokens = gt_norm.split()
        
        # Calculate common tokens
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        # Calculate precision and recall
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        
        # Calculate F1
        return (2 * precision * recall) / (precision + recall)
    
    @staticmethod
    def extract_answers(text: str) -> List[str]:
        """Extract answers from text using regex."""
        return [ans.strip() for ans in re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)]


class RetrievalMetrics:
    """Utility class for retrieval evaluation metrics."""
    
    @staticmethod
    def calculate_recall_and_precision(pos_indices: List[List[int]], 
                                     pred_indices: List[List[int]], 
                                     k_values: List[int] = [1, 3, 5]) -> Dict[str, float]:
        """Calculate recall and precision at different k values."""
        metrics = {f"recall_{k}": 0.0 for k in k_values}
        metrics.update({f"precision_{k}": 0.0 for k in k_values})
        
        valid_samples = 0
        
        for gold_pos, pred_pos in zip(pos_indices, pred_indices):
            if not gold_pos:  # Skip samples with no positive indices
                continue
                
            valid_samples += 1
            gold_set = set(gold_pos)
            
            for k in k_values:
                pred_k = set(pred_pos[:k])
                hits = len(gold_set & pred_k)
                
                # Recall: hits / total_positives
                metrics[f"recall_{k}"] += hits / len(gold_set)
                
                # Precision: hits / k
                metrics[f"precision_{k}"] += hits / k
        
        # Average across valid samples
        if valid_samples > 0:
            for key in metrics:
                metrics[key] /= valid_samples
        
        return metrics, valid_samples


class SFTTrainer(ABC):
    """
    Trainer for CLaRa supervised fine-tuning (SFT).

    This trainer handles multi-stage training for CLaRa models including:
    - Stage 1: Document compression training
    - Stage 2: Retrieval and generation training
    - Stage 2 Reasoning: Multi-step reasoning training
    """

    def __init__(self,
                 model,
                 strategy,
                 optim: Optimizer,
                 train_dataloader,
                 eval_dataloader,
                 scheduler,
                 max_norm: float = 1.0,
                 pretrain_mode: bool = False,
                 batch_size: int = 1,
                 max_epochs: int = 2,
                 tokenizer=None,
                 save_hf_ckpt: bool = False,
                 disable_ds_ckpt: bool = False) -> None:
        """
        Initialize the SFT trainer.
        
        Args:
            model: CLaRa model to train
            strategy: Training strategy (distributed, etc.)
            optim: Optimizer for training
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            scheduler: Learning rate scheduler
            max_norm: Maximum gradient norm for clipping
            pretrain_mode: Whether in pretraining mode
            batch_size: Training batch size
            max_epochs: Maximum number of training epochs
            tokenizer: Tokenizer for the model
            save_hf_ckpt: Whether to save HuggingFace format checkpoints
            disable_ds_ckpt: Whether to disable DeepSpeed checkpoints
        """
        super().__init__()
        
        # Core components
        self.model = model
        self.strategy = strategy
        self.optimizer = optim
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        
        # Training configuration
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.pretrain_mode = pretrain_mode
        
        # Data loaders
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Checkpointing
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt
        
        # Training arguments
        self.args = strategy.args
        
        # Model-specific settings
        # self.aux_loss = self.args.aux_loss_coef > 1e-8  # For Mixtral 8x7b
       # self.packing_samples = strategy.args.packing_samples
        
        # Initialize logging
        self._wandb = None
        self._tensorboard = None
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup Wandb and TensorBoard logging."""
        # Setup Wandb
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb
            
            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=self.strategy.args.use_wandb)
            
            wandb.init(
                entity=self.strategy.args.wandb_org,
                project=self.strategy.args.wandb_project,
                group=self.strategy.args.wandb_group,
                name=self.strategy.args.wandb_run_name,
                config=self.strategy.args.__dict__,
                reinit=True,
            )
            
            # Define metrics
            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)
        
        # Setup TensorBoard if Wandb is not available
        if (self.strategy.args.use_tensorboard and self._wandb is None and 
            self.strategy.is_rank_0()):
            from torch.utils.tensorboard import SummaryWriter
            
            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(
                self.strategy.args.use_tensorboard, 
                self.strategy.args.wandb_run_name
            )
            self._tensorboard = SummaryWriter(log_dir=log_dir)
    
    def fit(self, args, consumed_samples: int = 0, num_update_steps_per_epoch: Optional[int] = None):
        """
        Main training loop.
        
        Args:
            args: Training arguments
            consumed_samples: Number of samples already consumed
            num_update_steps_per_epoch: Number of update steps per epoch
        """
        # Configure evaluation and saving steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")
        elif args.save_steps == -2:
            args.save_steps = num_update_steps_per_epoch
        
        # Calculate starting point
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)
        
        # Setup MSE loss weight
        mse_loss_weight = 0.1 if args.mse_loss else 0.0
        
        # Initialize tracking variables
        training_metrics = self._init_training_metrics()
        
        # Main training loop
        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        
        for epoch in range(start_epoch, self.epochs):
            # Set epoch for distributed sampler
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )
            
            step_bar = tqdm(
                range(len(self.train_dataloader)),
                desc=f"Train step of epoch {epoch}",
                disable=not self.strategy.is_rank_0(),
            )
            
            # Training loop for this epoch
            self.model.train()
            
            for batch in self.train_dataloader:
                # Forward pass
                loss, outputs = self.model(
                    batch=batch, 
                    stage2_mips=self.args.stage2_mips, 
                    stage2_retrieval_top_n=self.args.stage2_retrieval_top_n
                )
                
                # Calculate total loss
                total_loss = loss + mse_loss_weight * outputs["mse_loss"]
                
                # Backward pass
                self.strategy.backward(total_loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                
                # Update metrics
                step_metrics = self._calculate_step_metrics(batch, outputs)
                training_metrics = self._update_training_metrics(training_metrics, step_metrics, loss, outputs)
                
                # Log and save
                if step % self.strategy.accumulated_gradient == 0:
                    self._process_accumulated_step(
                        args, step, training_metrics, step_bar, num_update_steps_per_epoch
                    )
                    training_metrics = self._reset_training_metrics()
                
                step += 1
                step_bar.update()
            
            epoch_bar.update()
        
        # Cleanup
        self._cleanup_logging()
    
    def _init_training_metrics(self) -> Dict[str, float]:
        """Initialize training metrics tracking."""
        return {
            "loss_sum": 0.0,
            "mse_loss_sum": 0.0,
            "retrieval_recall_1": 0.0,
            "retrieval_recall_3": 0.0,
            "retrieval_recall_5": 0.0,
            "retrieval_precision_1": 0.0,
            "retrieval_precision_3": 0.0,
            "retrieval_precision_5": 0.0,
            "retrieval_samples": 0.0,
        }
    
    def _reset_training_metrics(self) -> Dict[str, float]:
        """Reset training metrics for next accumulation cycle."""
        return self._init_training_metrics()
    
    def _calculate_step_metrics(self, batch: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, float]:
        """Calculate metrics for a single training step."""
        step_metrics = {
            "retrieval_recall_1": 0.0,
            "retrieval_recall_3": 0.0,
            "retrieval_recall_5": 0.0,
            "retrieval_precision_1": 0.0,
            "retrieval_precision_3": 0.0,
            "retrieval_precision_5": 0.0,
            "retrieval_samples": 0.0,
        }
        
        # Calculate retrieval metrics for stage2
        if 'stage2' in self.args.stage and 'topk_idx' in outputs:
            pos_indices = batch["pos_index"]
            pred_indices = outputs["topk_idx"]
            
            metrics, valid_samples = RetrievalMetrics.calculate_recall_and_precision(
                pos_indices, pred_indices.tolist()
            )
            
            step_metrics.update(metrics)
            step_metrics["retrieval_samples"] = valid_samples
        else:
            step_metrics["retrieval_samples"] = len(batch["questions"])
        
        return step_metrics
    
    def _update_training_metrics(self, 
                                training_metrics: Dict[str, float], 
                                step_metrics: Dict[str, float],
                                loss: torch.Tensor,
                                outputs: Dict[str, Any]) -> Dict[str, float]:
        """Update accumulated training metrics."""
        training_metrics["loss_sum"] += loss.item()
        training_metrics["mse_loss_sum"] += outputs["mse_loss"].item()
        
        for key in step_metrics:
            if key in training_metrics:
                training_metrics[key] += step_metrics[key]
        
        return training_metrics
    
    def _process_accumulated_step(self, 
                                 args, 
                                 step: int, 
                                 training_metrics: Dict[str, float],
                                 step_bar: tqdm,
                                 num_update_steps_per_epoch: int):
        """Process accumulated gradients and log/save."""
        # Calculate averaged metrics
        logs_dict = {
            "loss": training_metrics["loss_sum"] / self.strategy.accumulated_gradient,
            "mse_loss": training_metrics["mse_loss_sum"] / self.strategy.accumulated_gradient,
            "lr": self.scheduler.get_last_lr()[0],
        }
        
        # Add retrieval metrics
        if training_metrics["retrieval_samples"] > 0:
            for metric in ["retrieval_recall_1", "retrieval_recall_3", "retrieval_recall_5",
                          "retrieval_precision_1", "retrieval_precision_3", "retrieval_precision_5"]:
                logs_dict[metric] = training_metrics[metric] / training_metrics["retrieval_samples"]
        
        # Aggregate across processes
        logs_dict = self.strategy.all_reduce(logs_dict, op="mean")
        
        # Update progress bar
        step_bar.set_postfix(logs_dict)
        
        # Global step for logging
        global_step = step // self.strategy.accumulated_gradient
        client_states = {"consumed_samples": global_step * args.train_batch_size}
        
        # Save logs and checkpoints
        self.save_logs_and_checkpoints(
            args, global_step, step_bar, logs_dict, client_states, num_update_steps_per_epoch
        )
    
    def save_logs_and_checkpoints(self, 
                                 args, 
                                 global_step: int, 
                                 step_bar: tqdm, 
                                 logs_dict: Dict[str, float] = None, 
                                 client_states: Dict[str, Any] = None,
                                 num_update_steps_per_epoch: Optional[int] = None):
        """Save logs and checkpoints."""
        logs_dict = logs_dict or {}
        client_states = client_states or {}
        
        # Logging
        if global_step % args.logging_steps == 0:
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)
        
        # Checkpointing
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
            self.strategy.save_model(self.model, self.tokenizer, save_path)
        
        # Evaluation
        if global_step % args.eval_steps == 0:
            self._run_evaluation(args, global_step, num_update_steps_per_epoch)
    
    def _run_evaluation(self, args, global_step: int, num_update_steps_per_epoch: Optional[int]):
        """Run evaluation based on schedule."""
        if self.eval_dataloader is None or len(self.eval_dataloader) == 0:
            return
        
        print("Starting evaluation")
        
        # Determine if we should do generation evaluation
        eval_gen = False
        if global_step % (args.eval_steps * 5) == 0 and args.do_eval_gen:
            eval_gen = True
        elif (num_update_steps_per_epoch and 
              global_step % num_update_steps_per_epoch == 0 and 
              args.do_eval_gen):
            eval_gen = False  # Loss-only eval at epoch end
        
        self.evaluate(self.eval_dataloader, global_step, eval_gen=eval_gen)
    
    def evaluate(self, eval_dataloader, steps: int = 0, eval_gen: bool = False):
        """
        Evaluate the model on the evaluation dataset.
        
        Args:
            eval_dataloader: Evaluation data loader
            steps: Current training step
            eval_gen: Whether to perform generation evaluation
        """
        print(f"Starting evaluation at step {steps}")
        self.model.eval()
        
        # Initialize evaluation metrics
        eval_metrics = {
            "loss_sum": 0.0,
            "mse_loss_sum": 0.0,
            "samples": 0,
            "correct": 0,
            "retrieval_recall_1": 0.0,
            "retrieval_recall_3": 0.0,
            "retrieval_recall_5": 0.0,
            "retrieval_precision_1": 0.0,
            "retrieval_precision_3": 0.0,
            "retrieval_precision_5": 0.0,
            "retrieval_samples": 0.0,
        }
        
        with torch.no_grad():
            step_bar = tqdm(
                range(len(eval_dataloader)),
                desc=f"Eval stage of steps {steps}",
                disable=not self.strategy.is_rank_0(),
            )
            
            for batch in eval_dataloader:
                # Forward pass
                loss, outputs = self.model(batch=batch)
                
                # Basic metrics
                batch_size = len(batch["answers"])
                eval_metrics["loss_sum"] += loss.item() * batch_size
                eval_metrics["mse_loss_sum"] += outputs["mse_loss"].item() * batch_size
                eval_metrics["samples"] += batch_size
                
                # Retrieval metrics
                if 'stage2' in self.args.stage and 'topk_idx' in outputs:
                    retrieval_metrics, valid_samples = RetrievalMetrics.calculate_recall_and_precision(
                        batch["pos_index"], outputs["topk_idx"].tolist()
                    )
                    
                    for key, value in retrieval_metrics.items():
                        eval_metrics[key] += value * valid_samples
                    eval_metrics["retrieval_samples"] += valid_samples
                else:
                    eval_metrics["retrieval_samples"] += batch_size
                
                # Generation evaluation
                if eval_gen:
                    predictions = self._generate_predictions(batch)
                    correct = self._calculate_accuracy(predictions, batch["answers"])
                    eval_metrics["correct"] += correct
                
                step_bar.update()
        
        # Aggregate metrics across processes
        eval_metrics = self.strategy.all_reduce(eval_metrics, op="sum")
        
        # Calculate final metrics
        final_metrics = self._calculate_final_eval_metrics(eval_metrics, eval_gen)
        
        # Log evaluation results
        self._log_evaluation_results(final_metrics, steps)
        
        # Update progress bar
        step_bar.set_postfix(final_metrics)
        
        self.model.train()  # Reset to training mode
    
    def _generate_predictions(self, batch: Dict[str, Any]) -> List[str]:
        """Generate predictions for evaluation."""
        questions = batch["questions"]
        docs = batch["docs"]
        answers = batch["answers"]
        
        if self.args.stage in ["stage1", "stage1_2"]:
            return self.model.generate_from_text(
                questions=questions,
                documents=docs,
                max_new_tokens=64
            )
        elif self.args.stage == "stage2":
            predictions, _ = self.model.generate_from_questions(
                questions=questions,
                documents=docs,
                max_new_tokens=64,
                stage2_mips=self.args.stage2_mips,
            )
            return predictions
        elif self.args.stage == "stage2_reasoning":
            predictions = []
            for question, answer in zip(questions, answers):
                prediction, _ = self.model.generate_from_reasoning(
                    questions=[question],
                    max_new_tokens=1024,
                    answers=[answer],
                    save_dir=self.args.save_path,
                )
                predictions.extend(prediction)
            return predictions
        else:
            return [""] * len(questions)
    
    def _calculate_accuracy(self, predictions: List[str], answers: List[str]) -> int:
        """Calculate accuracy for predictions."""
        correct = 0
        for pred, ans in zip(predictions, answers):
            if EvaluationMetrics.cover_exact_match_score(pred, ans):
                correct += 1
        return correct
    
    def _calculate_final_eval_metrics(self, eval_metrics: Dict[str, float], eval_gen: bool) -> Dict[str, float]:
        """Calculate final evaluation metrics."""
        final_metrics = {}
        
        # Basic metrics
        if eval_metrics["samples"] > 0:
            final_metrics["eval_loss"] = eval_metrics["loss_sum"] / eval_metrics["samples"]
            final_metrics["eval_mse_loss"] = eval_metrics["mse_loss_sum"] / eval_metrics["samples"]
            
            if eval_gen:
                final_metrics["eval_acc"] = eval_metrics["correct"] / eval_metrics["samples"]
        
        # Retrieval metrics
        if eval_metrics["retrieval_samples"] > 0:
            for metric in ["retrieval_recall_1", "retrieval_recall_3", "retrieval_recall_5",
                          "retrieval_precision_1", "retrieval_precision_3", "retrieval_precision_5"]:
                final_metrics[f"eval_{metric}"] = eval_metrics[metric] / eval_metrics["retrieval_samples"]
        
        return final_metrics
    
    def _log_evaluation_results(self, metrics: Dict[str, float], steps: int):
        """Log evaluation results to wandb/tensorboard."""
        if self.strategy.is_rank_0():
            if self._wandb is not None:
                logs = {"eval/%s" % k: v for k, v in {**metrics, "global_step": steps}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None:
                for k, v in metrics.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, steps)
    
    def _cleanup_logging(self):
        """Cleanup logging resources."""
        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()
