#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import os
import json
import argparse
import gc
from datetime import timedelta
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Tuple

import torch
import numpy as np
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import AutoModel
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import spacy
import evaluate
import re
import string

from openrlhf.models.modeling_clara import CLaRa

# Environment setup
os.environ["NCCL_TIMEOUT"] = "5400"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Global constants
TARGET_ENTITY_CATEGORIES = {"PERSON", "GPE", "DATE", "CARDINAL", "ORG"}


class EvaluationMetrics:
    """Handles all evaluation metrics and scoring functions."""
    
    def __init__(self):
        self.bertscore = evaluate.load("bertscore")
        self.rouge = evaluate.load("rouge")
        self.nlp = spacy.load("en_core_web_sm")
    
    @staticmethod
    def normalize_answer(text: str) -> str:
        """Normalize text for comparison."""
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)
        
        def white_space_fix(text):
            return " ".join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
        
        return white_space_fix(remove_articles(remove_punc(text.lower())))
    
    @staticmethod
    def bool_mapping(text: str) -> str:
        """Map boolean values to yes/no."""
        mapping = {"True": "yes", "False": "no"}
        return mapping.get(text, text)
    
    def exact_match_score(self, prediction: str, ground_truth: str) -> bool:
        """Calculate exact match score."""
        pred_norm = self.normalize_answer(self.bool_mapping(prediction))
        gt_norm = self.normalize_answer(self.bool_mapping(ground_truth))
        return pred_norm == gt_norm
    
    def cover_exact_match_score(self, prediction: str, ground_truth: str) -> bool:
        """Calculate coverage exact match score."""
        pred_tokens = self.normalize_answer(self.bool_mapping(prediction)).split()
        gt_tokens = self.normalize_answer(self.bool_mapping(ground_truth)).split()
        return all(token in pred_tokens for token in gt_tokens)
    
    def f1_score(self, prediction: str, ground_truth: str) -> float:
        """Calculate F1 score."""
        pred_norm = self.normalize_answer(self.bool_mapping(prediction))
        gt_norm = self.normalize_answer(self.bool_mapping(ground_truth))
        
        # Handle yes/no/noanswer cases
        if pred_norm in ["yes", "no", "noanswer"] and pred_norm != gt_norm:
            return 0.0
        if gt_norm in ["yes", "no", "noanswer"] and pred_norm != gt_norm:
            return 0.0
        
        pred_tokens = pred_norm.split()
        gt_tokens = gt_norm.split()
        
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        
        return (2 * precision * recall) / (precision + recall)
    
    def extract_entities(self, text: str) -> set:
        """Extract entities from text."""
        doc = self.nlp(text)
        return set(ent.text.lower().strip() for ent in doc.ents)
    
    def extract_entities_by_category(self, text: str) -> Dict[str, set]:
        """Extract entities by category."""
        doc = self.nlp(text)
        entities_by_category = defaultdict(set)
        
        for ent in doc.ents:
            if ent.label_ in TARGET_ENTITY_CATEGORIES:
                entities_by_category[ent.label_].add(ent.text.lower().strip())
        
        return entities_by_category
    
    def entity_preserve_metric(self, prediction: str, reference: str) -> float:
        """Calculate entity preservation rate."""
        ref_entities = self.extract_entities(reference)
        pred_entities = self.extract_entities(prediction)
        
        if not ref_entities:
            return 1.0
        
        preserved = ref_entities.intersection(pred_entities)
        return len(preserved) / len(ref_entities)
    
    def entity_preserve_metric_by_category(self, prediction_tokens: List[List[str]], 
                                         reference_docs: List[str]) -> Dict[str, float]:
        """Calculate entity preservation by category."""
        # Merge prediction tokens
        all_prediction_tokens = []
        for tokens in prediction_tokens:
            all_prediction_tokens.extend(tokens)
        prediction_text = " ".join(all_prediction_tokens)
        
        # Merge reference documents
        reference_text = " ".join(reference_docs)
        
        # Extract entities
        pred_entities = self.extract_entities_by_category(prediction_text)
        ref_entities = self.extract_entities_by_category(reference_text)
        
        # Calculate preservation rates
        preservation_rates = {}
        
        for category in TARGET_ENTITY_CATEGORIES:
            ref_ents = ref_entities.get(category, set())
            pred_ents = pred_entities.get(category, set())
            
            if not ref_ents:
                preservation_rates[category] = 1.0
            else:
                preserved = ref_ents.intersection(pred_ents)
                preservation_rates[category] = len(preserved) / len(ref_ents)
        
        # Calculate overall preservation
        all_ref_entities = set()
        all_pred_entities = set()
        
        for entities_set in ref_entities.values():
            all_ref_entities.update(entities_set)
        for entities_set in pred_entities.values():
            all_pred_entities.update(entities_set)
        
        if not all_ref_entities:
            preservation_rates["overall"] = 1.0
        else:
            preserved_overall = all_ref_entities.intersection(all_pred_entities)
            preservation_rates["overall"] = len(preserved_overall) / len(all_ref_entities)
        
        return preservation_rates


class ResultCalculator:
    """Handles result calculation and visualization."""
    
    def __init__(self):
        self.metrics = EvaluationMetrics()
    
    def calculate_basic_metrics(self, result_list: List[Dict]) -> Dict[str, float]:
        """Calculate basic metrics (F1, accuracy, exact match)."""
        f1_total = 0
        acc_total = 0
        em_total = 0
        avg_output_length = 0
        
        answer_key = "golden_answers" if "golden_answers" in result_list[0] else "answer"
        
        for result in result_list:
            prediction = result['CLaRa_normal_output']
            ground_truth = result[answer_key][0] if answer_key == "golden_answers" else result[answer_key]
            
            acc_total += self.metrics.cover_exact_match_score(prediction, ground_truth)
            f1_total += self.metrics.f1_score(prediction, ground_truth)
            em_total += self.metrics.exact_match_score(prediction, ground_truth)
            avg_output_length += len(prediction.split())
        
        n = len(result_list)
        return {
            "f1": f1_total / n,
            "acc": acc_total / n,
            "em": em_total / n,
            "avg_output_length": avg_output_length / n
        }
    
    def calculate_stage2_metrics(self, result_list: List[Dict], k_values: List[int] = [1, 3, 5]) -> Dict[str, float]:
        """Calculate stage2 metrics with recall and precision."""
        basic_metrics = self.calculate_basic_metrics(result_list)
        
        recall = {k: 0 for k in k_values}
        precision = {k: 0 for k in k_values}
        
        for result in result_list:
            scores = result['topk_idx']
            pos_index = set(result['pos_index'])
            
            for k in k_values:
                top_k = set(scores[:k])
                hit = len(top_k & pos_index)
                
                recall[k] += hit / len(pos_index) if len(pos_index) > 0 else 0
                precision[k] += hit / k
        
        n = len(result_list)
        recall_metrics = {f"recall@{k}": v / n for k, v in recall.items()}
        precision_metrics = {f"precision@{k}": v / n for k, v in precision.items()}
        
        return {**basic_metrics, **recall_metrics, **precision_metrics}
    
    def calculate_paraphrase_metrics(self, result_list: List[Dict]) -> Dict[str, float]:
        """Calculate paraphrase metrics."""
        seen_metrics = {'bert-score': 0, 'rouge-1': 0, 'rouge-L': 0, 'entity_preserve': 0}
        unseen_metrics = {'bert-score': 0, 'rouge-1': 0, 'rouge-L': 0, 'entity_preserve': 0}
        
        # Process seen data (first 2000)
        for result in result_list[:2000]:
            prediction = result['CLaRa_normal_output']
            ground_truth = result['doc']
            
            bs = self.metrics.bertscore.compute(predictions=[prediction], references=[ground_truth], lang="en")
            seen_metrics['bert-score'] += bs['f1'][0]
            
            rouge_scores = self.metrics.rouge.compute(predictions=[prediction], references=[ground_truth])
            seen_metrics['rouge-1'] += rouge_scores['rouge1']
            seen_metrics['rouge-L'] += rouge_scores['rougeL']
            
            seen_metrics['entity_preserve'] += self.metrics.entity_preserve_metric(prediction, ground_truth)
        
        # Process unseen data (after 2000)
        for result in result_list[2000:]:
            prediction = result['CLaRa_normal_output']
            ground_truth = result['doc']
            
            bs = self.metrics.bertscore.compute(predictions=[prediction], references=[ground_truth], lang="en")
            unseen_metrics['bert-score'] += bs['f1'][0]
            
            rouge_scores = self.metrics.rouge.compute(predictions=[prediction], references=[ground_truth])
            unseen_metrics['rouge-1'] += rouge_scores['rouge1']
            unseen_metrics['rouge-L'] += rouge_scores['rougeL']
            
            unseen_metrics['entity_preserve'] += self.metrics.entity_preserve_metric(prediction, ground_truth)
        
        # Normalize
        n_seen = min(len(result_list[:2000]), 2000)
        n_unseen = max(len(result_list) - 2000, 0)
        
        final_metrics = {}
        if n_seen > 0:
            for key, value in seen_metrics.items():
                final_metrics[f'seen_{key}'] = float(value / n_seen)
        
        if n_unseen > 0:
            for key, value in unseen_metrics.items():
                final_metrics[f'unseen_{key}'] = float(value / n_unseen)
        
        return final_metrics
    
    def visualize_mse(self, result_list: List[Dict], save_path: str) -> Dict[str, Any]:
        """Create t-SNE visualization for MSE analysis."""
        # Set scientific style
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'figure.titlesize': 18,
            'axes.linewidth': 1.2,
            'grid.alpha': 0.3,
        })
        
        # Collect representations
        mem_reps = []
        non_mem_reps = []
        
        for result in result_list:
            mem_rep = result['CLaRa_compressed_output']
            non_mem_rep = result['CLaRa_normal_output']
            
            if isinstance(mem_rep, torch.Tensor):
                mem_rep = mem_rep.float().cpu().numpy()
            if isinstance(non_mem_rep, torch.Tensor):
                non_mem_rep = non_mem_rep.float().cpu().numpy()
            
            mem_reps.append(mem_rep)
            non_mem_reps.append(non_mem_rep)
        
        mem_reps = np.array(mem_reps)
        non_mem_reps = np.array(non_mem_reps)
        
        print(f"Memory representations shape: {mem_reps.shape}")
        print(f"Document representations shape: {non_mem_reps.shape}")
        
        # Combine data for t-SNE
        all_data = np.vstack([mem_reps, non_mem_reps])
        original_dim = all_data.shape[1]
        
        # PCA preprocessing if needed
        if all_data.shape[1] > 50:
            print(f"Applying PCA preprocessing from {all_data.shape[1]} to 50 dimensions...")
            pca = PCA(n_components=50)
            all_data = pca.fit_transform(all_data)
            print(f"PCA explained variance ratio: {pca.explained_variance_ratio_[:5].sum():.3f}")
        
        # Apply t-SNE
        print("Applying t-SNE...")
        perplexity = min(30, max(5, len(all_data) // 3))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                   max_iter=1000, learning_rate=200, verbose=1)
        tsne_results = tsne.fit_transform(all_data)
        
        # Separate results
        mem_tsne = tsne_results[:len(mem_reps)]
        doc_tsne = tsne_results[len(mem_reps):]
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Add jitter to separate overlapping points
        np.random.seed(42)
        jitter_strength = 1.0
        
        mem_jitter = mem_tsne.copy()
        doc_jitter = doc_tsne.copy()
        
        mem_jitter[:, 0] += np.random.normal(0.5, jitter_strength, len(mem_tsne))
        mem_jitter[:, 1] += np.random.normal(0.5, jitter_strength, len(mem_tsne))
        
        doc_jitter[:, 0] += np.random.normal(-0.5, jitter_strength, len(doc_tsne))
        doc_jitter[:, 1] += np.random.normal(-0.5, jitter_strength, len(doc_tsne))
        
        # Plot scatter points
        ax.scatter(doc_jitter[:, 0], doc_jitter[:, 1], c='#0066CC', alpha=0.7, s=25,
                  marker='o', edgecolors='white', linewidth=0.5,
                  label='Document Representations', zorder=2)
        
        ax.scatter(mem_jitter[:, 0], mem_jitter[:, 1], c='#FF3333', alpha=0.7, s=25,
                  marker='o', edgecolors='white', linewidth=0.5,
                  label='Memory Tokens Representations', zorder=3)
        
        # Configure plot
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        
        legend = ax.legend(frameon=True, fancybox=True, shadow=True,
                          loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=14)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Save visualization
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'tsne_visualization_scientific.png'),
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Calculate statistics
        distances = np.array([
            np.linalg.norm(mem_reps[i] - non_mem_reps[i])
            for i in range(len(mem_reps))
        ])
        
        statistics = {
            'mean_distance': float(np.mean(distances)),
            'std_distance': float(np.std(distances)),
            'median_distance': float(np.median(distances)),
            'min_distance': float(np.min(distances)),
            'max_distance': float(np.max(distances))
        }
        
        print("\n" + "="*60)
        print("VISUALIZATION ANALYSIS REPORT")
        print("="*60)
        print(f"Dataset Statistics:")
        print(f"  • Total samples: {len(mem_reps)}")
        print(f"  • Original dimension: {original_dim}")
        print(f"  • t-SNE perplexity: {perplexity}")
        print(f"\nDistance Analysis:")
        for key, value in statistics.items():
            print(f"  • {key.replace('_', ' ').title()}: {value:.4f}")
        print("="*60)
        
        return {
            'mem_tsne': mem_tsne,
            'doc_tsne': doc_tsne,
            'original_distances': distances,
            'statistics': statistics
        }


class DataLoader:
    """Handles data loading for different datasets and stages."""
    
    @staticmethod
    def load_stage1_data(dataset: str, gold_retrieval: bool) -> List[Dict]:
        """Load stage1 evaluation data."""
        retrieval_type = "with_pos" if gold_retrieval else "no_pos"
        file_path = f"/mnt/conductor_data/data/compression_rag_data/generator_training_val_data/stage1_eval/{dataset}/eval_processed_{retrieval_type}.jsonl"
        
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        processed_data = []
        for index, item in enumerate(data):
            docs = item['docs'][:5]  # Take top 5 documents
            processed_item = {
                'original_data': item,
                'documents': docs,
                'question': item['question'],
                'global_index': index
            }
            processed_data.append(processed_item)
        
        return processed_data
    
    @staticmethod
    def load_stage2_data(dataset: str, gold_retrieval: bool) -> List[Dict]:
        """Load stage2 evaluation data."""
        retrieval_type = "with_pos" if gold_retrieval else "no_pos"
        file_path = f"/mnt/conductor_data/data/compression_rag_data/generator_training_val_data/stage2_eval/{dataset}/eval_processed_{retrieval_type}.jsonl"
        
        processed_data = []
        with open(file_path, 'r') as f:
            for index, line in enumerate(f):
                item = json.loads(line)
                processed_item = {
                    'original_data': item,
                    'documents': item['docs'],
                    'question': item['question'],
                    'global_index': index,
                    'pos_index': item['pos_index']
                }
                processed_data.append(processed_item)
        
        return processed_data
    
    @staticmethod
    def load_paraphrase_data(file_path: str) -> List[Dict]:
        """Load paraphrase data."""
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        processed_data = []
        for index, item in enumerate(data):
            processed_item = {
                'original_data': item,
                'documents': [item['doc']],
                'question': "",
                'global_index': index
            }
            processed_data.append(processed_item)
        
        return processed_data


class AcceleratedCLaRaInference:
    """Main inference engine using Accelerate for distributed processing."""
    
    def __init__(self, model_path: str, training_stage: str = None, 
                 generation_top_k: int = None, args = None):
        self.args = args
        
        # Initialize Accelerator
        process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
        self.accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])
        
        if self.accelerator.is_main_process:
            print(f"Using {self.accelerator.num_processes} GPUs for distributed inference")
            print(f"Current process: {self.accelerator.process_index}")
            print("Loading CLaRa model...")
        
        # Load model
        self.model = CLaRa.from_pretrained(
            model_path, 
            training_stage=training_stage, 
            generation_top_k=generation_top_k, 
            pure_inference=True
        )
        
        # Prepare model with Accelerator
        self.model = self.accelerator.prepare(self.model)
        self.model.eval()
        
        if self.accelerator.is_main_process:
            print("Model preparation completed")
    
    def _get_model(self):
        """Get the actual model (handles distributed vs single GPU)."""
        return self.model.module if hasattr(self.model, 'module') else self.model
    
    def process_batch(self, batch_questions: List[str], batch_documents: List[List[str]] = None,
                     stage2_mips: bool = False, training_stage: str = None,
                     batch_answers: List[str] = None, time_count: bool = False) -> Tuple:
        """Process a batch of questions and documents."""
        model = self._get_model()
        
        with torch.no_grad():
            try:
                if training_stage == 'stage2':
                    return self._process_stage2(model, batch_questions, batch_documents, 
                                              stage2_mips, time_count)
                elif training_stage in ['stage1', 'stage1_2']:
                    return self._process_stage1(model, batch_questions, batch_documents)
                elif training_stage == 'stage2_reasoning':
                    return self._process_reasoning(model, batch_questions, batch_answers)
                elif training_stage == 'stage1_paraphrase':
                    return self._process_paraphrase(model, batch_questions, batch_documents)
                elif training_stage == 'stage1_mse_visulize':
                    return self._process_mse_visualize(model, batch_documents)
                else:
                    raise ValueError(f"Unknown training stage: {training_stage}")
                    
            except torch.cuda.OutOfMemoryError as e:
                self.accelerator.print(f"CUDA OOM error: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                return self._create_empty_results(batch_questions, training_stage)
    
    def _process_stage2(self, model, batch_questions, batch_documents, stage2_mips, time_count):
        """Process stage2 inference."""
        if time_count:
            if stage2_mips:
                results = model.generate_from_questions(
                    questions=batch_questions, 
                    max_new_tokens=64,
                    stage2_mips=stage2_mips,
                    time_count=True
                )
            else:
                results = model.generate_from_questions(
                    questions=batch_questions, 
                    max_new_tokens=64,
                    stage2_mips=stage2_mips,
                    documents=batch_documents,
                    time_count=True
                )
            return results
        else:
            if stage2_mips:
                batch_out_normal, topk_idx = model.generate_from_questions(
                    questions=batch_questions, 
                    max_new_tokens=64,
                    stage2_mips=stage2_mips
                )
            else:
                batch_out_normal, topk_idx = model.generate_from_questions(
                    questions=batch_questions, 
                    max_new_tokens=64,
                    stage2_mips=stage2_mips,
                    documents=batch_documents
                )
            return batch_out_normal, batch_out_normal, topk_idx
    
    def _process_stage1(self, model, batch_questions, batch_documents):
        """Process stage1 inference."""
        batch_out_compressed = []
        
        for docs, question in zip(batch_documents, batch_questions):
            embeddings, _ = model.compress_documents(documents=docs)
            out_compressed = model.generate_from_compressed_documents_and_questions(
                questions=[question], 
                compressed_documents=embeddings
            )
            batch_out_compressed.extend(out_compressed)
            
            del embeddings
            torch.cuda.empty_cache()
        
        return batch_out_compressed, batch_out_compressed, None
    
    def _process_reasoning(self, model, batch_questions, batch_answers):
        """Process reasoning inference."""
        batch_out_normal = []
        batch_out_reasoning_list = []
        
        for question, answer in zip(batch_questions, batch_answers):
            temp_out, temp_out_reasoning = model.generate_from_reasoning(
                questions=[question],
                max_new_tokens=1024,
                answers=[answer],
                save_dir=self.args.model_path
            )
            batch_out_normal.append(temp_out[0])
            batch_out_reasoning_list.extend(temp_out_reasoning)
        
        return batch_out_normal, batch_out_normal, None, batch_out_reasoning_list
    
    def _process_paraphrase(self, model, batch_questions, batch_documents):
        """Process paraphrase inference."""
        batch_out_compressed = []
        
        for docs, question in zip(batch_documents, batch_questions):
            out_compressed = model.generate_from_paraphrase(
                questions=["" for _ in range(len(docs))], 
                documents=[docs]
            )
            batch_out_compressed.extend(out_compressed)
            torch.cuda.empty_cache()
        
        return batch_out_compressed, batch_out_compressed, None
    
    def _process_mse_visualize(self, model, batch_documents):
        """Process MSE visualization."""
        batch_out_normal = []
        batch_out_compressed = []
        
        for docs in batch_documents:
            mem_rep, non_mem_rep = model.compress_documents_mse_visulize(documents=docs)
            batch_out_compressed.append(mem_rep[0])
            batch_out_normal.append(non_mem_rep[0])
        
        return batch_out_normal, batch_out_compressed
    
    def _create_empty_results(self, batch_questions, training_stage):
        """Create empty results for error cases."""
        empty_results = [""] * len(batch_questions)
        if training_stage == 'stage2_reasoning':
            return empty_results, empty_results, None, empty_results
        elif training_stage == 'stage1_mse_visulize':
            return empty_results, empty_results
        else:
            return empty_results, empty_results, None


def convert_embeddings_to_list(data):
    """Convert tensor embeddings to lists for JSON serialization."""
    if isinstance(data, dict):
        return {k: convert_embeddings_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_embeddings_to_list(item) for item in data]
    elif isinstance(data, torch.Tensor):
        return data.cpu().to(torch.float32).numpy().tolist()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data


def main():
    parser = argparse.ArgumentParser(description="CLaRa Model Inference")
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--stage', type=str, default='stage1', 
                       choices=['stage1', 'stage1_2', 'stage2', 'stage2_reasoning', 
                               'stage1_paraphrase', 'stage1_mse_visulize'], 
                       help='Training stage')
    parser.add_argument('--stage2_mips', action='store_true', help='Use MIPS for stage2')
    parser.add_argument('--dataset', type=str, default='musique', 
                       help='Comma-separated list of datasets')
    parser.add_argument('--gold_retrieval', action='store_true', 
                       help='Use gold retrieval context')
    parser.add_argument('--generation_top_k', type=int, default=5, help='Top-k for generation')
    parser.add_argument('--paraphrase_path', type=str, help='Path to paraphrase data')
    parser.add_argument('--mse_visulize_path', type=str, help='Path to save MSE visualization')
    parser.add_argument('--efficient_count', action='store_true', help='Count efficiency metrics')
    
    args = parser.parse_args()
    
    # Process datasets
    all_results_metrics = {}
    datasets_list = args.dataset.split(',')
    
    for dataset in datasets_list:
        print(f"Processing dataset: {dataset}")
        
        # Load data based on stage
        if args.stage in ['stage1', 'stage1_2']:
            processed_data = DataLoader.load_stage1_data(dataset, args.gold_retrieval)
        elif args.stage == 'stage2':
            processed_data = DataLoader.load_stage2_data(dataset, args.gold_retrieval)
        elif args.stage in ['stage1_paraphrase', 'stage1_mse_visulize']:
            if not args.paraphrase_path:
                raise ValueError(f"--paraphrase_path required for stage {args.stage}")
            processed_data = DataLoader.load_paraphrase_data(args.paraphrase_path)
        else:
            raise ValueError(f"Unsupported stage: {args.stage}")
        
        print(f"Loaded {len(processed_data)} samples for {dataset}")
        
        # Initialize inference engine
        model_path = os.path.join('/mnt/task_wrapper/user_output/artifacts/data/train_checkpoint', args.model_path)
        args.model_path = model_path
        
        inference_engine = AcceleratedCLaRaInference(
            model_path=model_path,
            training_stage=args.stage,
            generation_top_k=args.generation_top_k,
            args=args
        )
        
        # Wait for all processes to be ready
        inference_engine.accelerator.wait_for_everyone()
        
        # Store results
        all_results = []
        time_count_dic = {"compress_time": 0, "query_time": 0, "generate_time": 0, "total_time": 0, "count": 0}
        
        # Process data in batches using accelerator
        with inference_engine.accelerator.split_between_processes(processed_data, apply_padding=False) as local_data:
            print(f"Process {inference_engine.accelerator.process_index}: processing {len(local_data)} samples")
            
            batch_size = args.batch_size
            num_batches = (len(local_data) + batch_size - 1) // batch_size
            
            for batch_idx in tqdm(range(num_batches), 
                                desc=f"GPU {inference_engine.accelerator.process_index}",
                                disable=not inference_engine.accelerator.is_local_main_process):
                
                # Get current batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(local_data))
                batch = local_data[start_idx:end_idx]
                
                # Prepare batch data
                batch_questions = [item['question'] for item in batch]
                batch_documents = [item['documents'] for item in batch] if 'documents' in batch[0] else None
                batch_answers = [item.get('answer') for item in batch] if args.stage == 'stage2_reasoning' else None
                
                # Process batch
                if args.efficient_count and args.stage == 'stage2':
                    results = inference_engine.process_batch(
                        batch_questions=batch_questions,
                        batch_documents=batch_documents,
                        stage2_mips=args.stage2_mips,
                        training_stage=args.stage,
                        time_count=True
                    )
                    batch_out_normal, batch_out_compressed, batch_topk_idx, compress_time, query_time, generate_time, total_time = results
                    
                    time_count_dic["compress_time"] += compress_time
                    time_count_dic["query_time"] += query_time
                    time_count_dic["generate_time"] += generate_time
                    time_count_dic["total_time"] += total_time
                    time_count_dic["count"] += 1
                else:
                    results = inference_engine.process_batch(
                        batch_questions=batch_questions,
                        batch_documents=batch_documents,
                        stage2_mips=args.stage2_mips,
                        training_stage=args.stage,
                        batch_answers=batch_answers
                    )
                    
                    if args.stage == 'stage2_reasoning':
                        batch_out_normal, batch_out_compressed, batch_topk_idx, batch_out_reasoning = results
                    elif args.stage == 'stage1_mse_visulize':
                        batch_out_normal, batch_out_compressed = results
                        batch_topk_idx = None
                    else:
                        batch_out_normal, batch_out_compressed, batch_topk_idx = results
                
                # Prepare results
                batch_results = []
                for i, (item, normal_out, compressed_out) in enumerate(zip(batch, batch_out_normal, batch_out_compressed)):
                    result_item = item['original_data'].copy()
                    result_item['CLaRa_normal_output'] = normal_out
                    result_item['CLaRa_compressed_output'] = compressed_out
                    result_item['global_index'] = item['global_index']
                    
                    if args.stage == 'stage2' and batch_topk_idx is not None:
                        result_item['topk_idx'] = batch_topk_idx[i].tolist()
                    elif args.stage == 'stage2_reasoning':
                        result_item['reasoning_output'] = batch_out_reasoning[i]
                    
                    batch_results.append(result_item)
                
                all_results.extend(batch_results)
            
                
                # Clean up memory
                torch.cuda.empty_cache()
                if batch_idx % 10 == 0:
                    gc.collect()
        
        # Save efficiency metrics if requested
        if args.efficient_count and inference_engine.accelerator.is_main_process:
            eff_dic = {
                "compress_time_ms": round((time_count_dic['compress_time'] / time_count_dic['count']) * 1000, 2),
                "query_time_ms": round((time_count_dic['query_time'] / time_count_dic['count']) * 1000, 2),
                "generate_time_ms": round((time_count_dic['generate_time'] / time_count_dic['count']) * 1000, 2),
                "total_time_ms": round((time_count_dic['total_time'] / time_count_dic['count']) * 1000, 2),
                "sample_count": time_count_dic['count']
            }
            eff_output_path = os.path.join(model_path, f"efficiency_{dataset}_{args.stage}_{args.gold_retrieval}_{args.generation_top_k}.json")
            with open(eff_output_path, 'w') as f:
                json.dump(eff_dic, f, indent=2)
        
        # Wait for all processes to complete
        inference_engine.accelerator.wait_for_everyone()
        
        # Gather results from all processes
        if inference_engine.accelerator.is_main_process:
            print("Collecting results from all processes...")
        
        all_results_gathered = inference_engine.accelerator.gather_for_metrics(all_results)
        
        # Process and save results (main process only)
        if inference_engine.accelerator.is_main_process:
            print("Processing and saving results...")
            
            # Flatten results
            final_results = []
            if isinstance(all_results_gathered, list):
                for result_batch in all_results_gathered:
                    if isinstance(result_batch, list):
                        final_results.extend(result_batch)
                    else:
                        final_results.append(result_batch)
            
            print(f"Collected {len(final_results)} results")
            
            # Sort by global index to maintain order
            final_results.sort(key=lambda x: x.get('global_index', 0))
            
            # Verify data integrity
            processed_indices = set(item.get('global_index', -1) for item in final_results)
            expected_indices = set(range(len(processed_data)))
            missing_indices = expected_indices - processed_indices
            
            if missing_indices:
                print(f"Warning: Missing indices: {sorted(list(missing_indices))}")
            else:
                print("✓ Data integrity verification passed")
            
            # Remove global index for clean output
            for item in final_results:
                item.pop('global_index', None)
            
            # Save results
            output_path = os.path.join(model_path, f"{dataset}_{args.stage}_{args.gold_retrieval}_{args.generation_top_k}.jsonl")
            with open(output_path, 'w') as f:
                if args.stage == 'stage1_mse_visulize':
                    converted_results = convert_embeddings_to_list(final_results)
                    for item in converted_results:
                        f.write(json.dumps(item) + '\n')
                else:
                    for item in final_results:
                        f.write(json.dumps(item) + '\n')
            
            print(f"Results saved to: {output_path}")
            
            # Calculate metrics
            calculator = ResultCalculator()
            
            if args.stage == 'stage2':
                metrics = calculator.calculate_stage2_metrics(final_results)
            elif args.stage == 'stage1_paraphrase':
                metrics = calculator.calculate_paraphrase_metrics(final_results)
            elif args.stage == 'stage1_mse_visulize':
                if args.mse_visulize_path:
                    metrics = calculator.visualize_mse(final_results, args.mse_visulize_path)
                else:
                    metrics = {"visualization": "completed"}
            else:
                metrics = calculator.calculate_basic_metrics(final_results)
            
            print(f"Metrics for {dataset}: {metrics}")
            all_results_metrics[dataset] = metrics
        
        # Clean up
        del inference_engine
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save final metrics
    if len(all_results_metrics) > 0:
        metrics_path = os.path.join(model_path, f"results_metrics_{args.stage}_{args.gold_retrieval}_{args.generation_top_k}.json")
        with open(metrics_path, 'w') as f:
            json.dump(all_results_metrics, f, indent=2)
        print(f"Final metrics saved to: {metrics_path}")


if __name__ == '__main__':
    main()