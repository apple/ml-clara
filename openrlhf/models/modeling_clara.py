#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import warnings
import os
import torch
import gc
import time
import json
import copy
import random
import requests
import re

from torch import nn
from torch.nn import functional as F
from torch.nn.functional import gelu
from jinja2.exceptions import TemplateError
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    PreTrainedModel, 
    PretrainedConfig, 
    StoppingCriteria, 
    StoppingCriteriaList
)
from huggingface_hub import hf_hub_download
from typing import List, Dict, Any, Optional, Tuple

# Environment setup
torch.set_printoptions(threshold=float("inf"))
os.environ["NCCL_TIMEOUT"] = "5400"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Constants
IGNORE_INDEX = -100
PARAPHRASE_INSTRUCTIONS = [
    'Background: {docs} means the same as',
    "Background: {docs} Can you put the above sentences in your own terms?",
    "Background: {docs} Please provide a reinterpretation of the preceding background text.",
    "These two expressions are equivalent in essence:\n(1) {docs}\n(2)",
    "Background: {docs} is a paraphrase of what?",
    "Background: {docs} Could you give me a different version of the background sentences above?",
    "In other words, background: {docs} is just another way of saying:",
    "You're getting across the same point whether you say background: {docs} or",
    "Background: {docs} After unpacking the ideas in the background information above, we got:",
    "Background: {docs} Please offer a restatement of the background sentences I've just read.",
    "Background: {docs}, which also means:",
    "Strip away the mystery, and you'll find background: {docs} is simply another rendition of:",
    "The essence of background: {docs} is captured again in the following statement:",
]


class StopOnCriteria(StoppingCriteria):
    """Custom stopping criteria for generation."""
    
    def __init__(self, tokenizer, stop_strings: List[str] = None, stop_token_ids: List[int] = None):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings or []
        self.stop_token_ids = stop_token_ids or []
        self.reason = None

    def __call__(self, input_ids, scores, **kwargs):
        # Check if last token is in stop_token_ids
        last_token = input_ids[0, -1].item()
        if last_token in self.stop_token_ids:
            self.reason = f"stop_token_{last_token}"
            return True

        # Check if any stop_strings appear in generated text
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        for stop_str in self.stop_strings:
            if stop_str in text:
                self.reason = f"stop_string_{stop_str}"
                return True

        return False


class LlamaRMSNorm(nn.Module):
    """Llama-style RMS normalization layer."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Converter(nn.Module):
    """Converter module for dimension transformation."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.rms_norm = LlamaRMSNorm(input_dim)
        self.dense_in = nn.Linear(input_dim, output_dim)
        self.dense_out = nn.Linear(output_dim, output_dim)
        
        self._print_trainable_parameters()
    
    def _print_trainable_parameters(self):
        """Print parameter statistics."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Converter trainable parameters: {trainable_params}, Total parameters: {total_params}")
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = self.rms_norm(embeddings)
        x = self.dense_in(embeddings)
        x = self.dense_out(gelu(x))
        return x.to(torch.float32)


class CLaRaConfig(PretrainedConfig):
    """Configuration class for CLaRa model."""
    
    model_type = "CLaRa"

    def __init__(self,
                 decoder_model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                 doc_max_length: int = 128,
                 quantization: str = 'no',
                 sep: bool = False,
                 compr_model_name: str = "google-bert/bert-base-uncased",
                 compr_rate: int = 64,
                 compr_n_layers: int = None,
                 compr_every_n_layer: int = None,
                 compr_base_model_name: str = 'mistralai/Mistral-7B-Instruct-v0.2',
                 compr_rms_norm: bool = False,
                 compr_mlp_hidden_dim: int = 8096,
                 compr_use_mlp: bool = True,
                 compr_linear_type: str = "concat",
                 lora: bool = False,
                 lora_compressor: bool = False,
                 training_form: str = "both",
                 training_stage: str = "stage1",
                 generation_top_k: int = 1,
                 lora_r: int = 16,
                 lora_r_compressor: int = None,
                 load_adapters: bool = True,
                 kbtc_training: bool = False,
                 optimize_mem_tokens: bool = False,
                 different_mem_tokens: bool = False,
                 attn_implementation: str = None,
                 _attn_implementation_autoset: bool = True,
                 ae_mode: str = "token",
                 max_new_tokens: int = 128,
                 stage2_retrieval_top_n: int = 1,
                 load_pretrained_checkpoint: bool = False,
                 device_map=None,
                 auto_map: dict = {
                     "AutoConfig": "modeling_clara.CLaRaConfig",
                     "AutoModel": "modeling_clara.CLaRa"
                 },
                 **kwargs):
        super().__init__(**kwargs)

        self.decoder_model_name = decoder_model_name
        self.doc_max_length = doc_max_length
        self.quantization = quantization
        self.sep = sep

        self.compr_model_name = compr_model_name
        self.compr_rate = compr_rate
        self.compr_use_mlp = compr_use_mlp
        self.compr_mlp_hidden_dim = compr_mlp_hidden_dim
        self.compr_n_layers = compr_n_layers
        self.compr_every_n_layer = compr_every_n_layer
        self.compr_base_model_name = compr_base_model_name
        self.compr_rms_norm = compr_rms_norm
        self.compr_linear_type = compr_linear_type

        self.lora = lora
        self.lora_compressor = lora_compressor
        self.training_form = training_form
        self.lora_r = lora_r
        self.lora_r_compressor = lora_r_compressor or lora_r
        self.load_adapters = load_adapters
        self.optimize_mem_tokens = optimize_mem_tokens
        self.different_mem_tokens = different_mem_tokens
        self.kbtc_training = kbtc_training
        self.training_stage = training_stage
        self.device_map = device_map
        self.attn_implementation = attn_implementation
        self._attn_implementation_autoset = _attn_implementation_autoset
        self.ae_mode = ae_mode
        self.max_new_tokens = max_new_tokens
        self.auto_map = auto_map
        self.load_pretrained_checkpoint = load_pretrained_checkpoint

        self.generation_top_k = generation_top_k
        self.stage2_retrieval_top_n = stage2_retrieval_top_n
        
        if training_form == 'compressor':
            assert compr_model_name is not None and not self.lora


# Utility functions
def remote_generate(docs: List[str], questions: List[str], api_url: str) -> List[str]:
    """Generate responses using remote API."""
    response = requests.post(
        f"{api_url}/generate",
        json={"docs": docs, "questions": questions}
    )
    return response.json()["texts"]


def add_memory_tokens_to_inputs(input_ids: torch.Tensor, 
                               attention_mask: torch.Tensor, 
                               n_mem_tokens: int, 
                               tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add memory tokens to input sequences."""
    assert len(tokenizer.mem_tokens) == n_mem_tokens
    
    mem_tokens = torch.stack([tokenizer.mem_token_ids_pt] * input_ids.size(0), 0)
    assert len(mem_tokens) == input_ids.size(0)
    assert len(mem_tokens[0]) == n_mem_tokens
    
    input_ids = torch.cat([input_ids, mem_tokens], dim=1)
    attention_mask = torch.cat([attention_mask, torch.ones(input_ids.size(0), n_mem_tokens)], dim=1)
    
    return input_ids, attention_mask


def build_pos_mask(pos_index: List[List[int]], N: int, device: torch.device) -> torch.Tensor:
    """Build positive mask for retrieval training."""
    if isinstance(pos_index, (list, tuple)):
        B = len(pos_index)
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for b, idxs in enumerate(pos_index):
            if len(idxs) > 0:
                mask[b, torch.as_tensor(idxs, device=device, dtype=torch.long)] = True
        return mask
    else:  # tensor [B, M]
        B, M = pos_index.shape
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for m in range(M):
            col = pos_index[:, m]
            v = col >= 0
            if v.any():
                mask[v, col[v]] = True
        return mask


def differentiable_topk_top_1(logits: torch.Tensor, k: int, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Implements differentiable top-1 selection using Gumbel-Softmax."""
    y = logits / temperature
    y_soft = F.softmax(y, dim=-1).float()
    
    # Hard one-hot version
    index = y_soft.argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
    
    # Straight-through estimator
    z = y_hard + y_soft - y_soft.detach()
    z = z.unsqueeze(1).to(logits.dtype)
    
    return z, index


def differentiable_topk(logits: torch.Tensor, k: int, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable top-k selection."""
    B, N = logits.shape
    perturbed = logits / max(temperature, 1e-6)
    
    # Hard top-k indices
    topk_vals, topk_idx = perturbed.topk(k, dim=-1)
    K_hard = torch.zeros(B, k, N, device=logits.device, dtype=logits.dtype)
    K_hard.scatter_(2, topk_idx.unsqueeze(-1), 1.0)
    
    # Soft distributions for each slot
    K_soft = torch.zeros_like(K_hard)
    taken = torch.zeros(B, N, device=logits.device, dtype=logits.dtype)
    
    for j in range(k):
        mask = (1.0 - taken.detach())
        masked = perturbed + (mask + 1e-8).log()
        pj = F.softmax(masked, dim=-1).float()
        K_soft[:, j, :] = pj
        taken = torch.clamp(taken + K_hard[:, j, :], max=1.0)
    
    # Straight-through estimator
    W = K_hard + (K_soft - K_soft.detach())
    return W, topk_idx


class CLaRa(PreTrainedModel):
    """CLaRa: Unified Retrieval-Augmented Generation Model."""
    
    config_class = CLaRaConfig
    
    def __init__(self, cfg: CLaRaConfig):
        super().__init__(cfg)
        self.decoder_model_name = cfg.decoder_model_name
        self.decoder = self._create_decoder(cfg)
        self.doc_max_length = cfg.doc_max_length
        
        print(f'Base decoder parameters: {self.decoder.num_parameters()}')
        
        # Model configuration
        self.compr_model_name = cfg.compr_model_name
        self.training_form = cfg.training_form
        self.lora = cfg.lora
        self.adapter_keys = []
        self.compr = None
        
        # Initialize LoRA adapters if needed
        if cfg.lora and not getattr(cfg, 'pure_inference', False):
            self._setup_lora_adapters(cfg)
        
        print(f'Model adapter keys: {self.adapter_keys}')
        
        # Initialize tokenizer and resize embeddings
        self.decoder_tokenizer = self._create_decoder_tokenizer(cfg)
        self.decoder.resize_token_embeddings(len(self.decoder_tokenizer))
        self._configure_generation_config()
        
        # Model parameters
        self.generation_top_k = cfg.generation_top_k
        self.training_stage = cfg.training_stage
        self.stage2_retrieval_top_n = cfg.stage2_retrieval_top_n
        self.sep = cfg.sep
        self.compr_rate = cfg.compr_rate
        self.local_rank = os.getenv('LOCAL_RANK', '0')
        
        self.n_mem_tokens = self.doc_max_length // self.compr_rate
        self.hidden_size = self.decoder.config.hidden_size
        
        # Setup adapters and memory token optimization
        if self.lora:
            self._setup_adapter_training()
        else:
            print(f'Total trainable parameters: {self.num_parameters(only_trainable=True)}')
        
        self._prepare_mem_tokens_optimization()
        
        # Retrieval configuration
        self.url_retrieval = "http://127.0.0.1:5004/queries"
    
    def _create_decoder(self, cfg: CLaRaConfig) -> AutoModelForCausalLM:
        """Create and configure the decoder model."""
        if not torch.cuda.is_available():
            return AutoModelForCausalLM.from_pretrained(
                cfg.decoder_model_name,
                torch_dtype=torch.bfloat16,
                resume_download=True,
                trust_remote_code=True,
                device_map=cfg.device_map
            )
        
        if cfg.quantization == "no":
            return AutoModelForCausalLM.from_pretrained(
                cfg.decoder_model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation=cfg.attn_implementation,
                device_map=cfg.device_map
            )
        elif cfg.quantization == "int4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype='bfloat16',
            )
            return AutoModelForCausalLM.from_pretrained(
                cfg.decoder_model_name,
                quantization_config=quant_config,
                attn_implementation=cfg.attn_implementation,
                torch_dtype=torch.bfloat16,
                resume_download=True,
                trust_remote_code=True,
                device_map=cfg.device_map
            )
        elif cfg.quantization == "int8":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                bnb_4bit_compute_dtype='bfloat16',
            )
            return AutoModelForCausalLM.from_pretrained(
                cfg.decoder_model_name,
                quantization_config=quant_config,
                attn_implementation=cfg.attn_implementation,
                torch_dtype=torch.bfloat16,
                resume_download=True,
                trust_remote_code=True,
                device_map=cfg.device_map
            )
        else:
            raise NotImplementedError(f"Quantization {cfg.quantization} not supported")
    
    def _setup_lora_adapters(self, cfg: CLaRaConfig):
        """Setup LoRA adapters based on training stage."""
        peft_config = self._get_peft_config(lora_r=cfg.lora_r)
        
        if cfg.training_stage == "stage1" and cfg.load_adapters:
            print('Loading encoder and decoder adapter for stage1')
            self.decoder.add_adapter(peft_config, 'decoder_adapter')
            self.adapter_keys.append('decoder_adapter')
            self.decoder.add_adapter(peft_config, 'encoder_adapter')
            self.adapter_keys.append('encoder_adapter')
        elif cfg.training_stage == "stage2" and cfg.load_adapters:
            if 'decoder_adapter' not in self.adapter_keys:
                self.decoder.add_adapter(peft_config, 'decoder_adapter')
                self.adapter_keys.append('decoder_adapter')
            if 'query_reasoner_adapter' not in self.adapter_keys:
                self.decoder.add_adapter(peft_config, 'query_reasoner_adapter')
                self.adapter_keys.append('query_reasoner_adapter')
        elif cfg.training_stage == 'stage1_2':
            if not cfg.load_adapters:
                print('Loading decoder adapter for stage1_2')
                self.decoder.add_adapter(peft_config, 'decoder_adapter')
                self.adapter_keys.append('decoder_adapter')
            elif cfg.load_adapters:
                print('Loading encoder and decoder adapter for stage1_2')
                self.decoder.add_adapter(peft_config, 'encoder_adapter')
                self.adapter_keys.append('encoder_adapter')
                self.decoder.add_adapter(peft_config, 'decoder_adapter')
                self.adapter_keys.append('decoder_adapter')
        elif cfg.training_stage == 'stage2_reasoning':
            if not cfg.load_adapters:
                print('Loading decoder adapter for stage2_reasoning')
                self.decoder.add_adapter(peft_config, 'decoder_adapter')
                self.adapter_keys.append('decoder_adapter')
    
    def _setup_adapter_training(self):
        """Setup adapters for training."""
        for adapter_key in self.adapter_keys:
            self.decoder.set_adapter(adapter_key)
            print(f'Adapter {adapter_key} trainable parameters: {self.num_parameters(only_trainable=True)}')
        self._set_all_adapters()
    
    def _configure_generation_config(self):
        """Configure generation parameters."""
        self.decoder.generation_config.top_p = None
        self.decoder.generation_config.temperature = None
        self.decoder.generation_config.pad_token_id = self.decoder_tokenizer.pad_token_id
    
    @staticmethod
    def _create_decoder_tokenizer(cfg: CLaRaConfig) -> AutoTokenizer:
        """Create and configure the decoder tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.decoder_model_name, 
            use_fast=True, 
            padding_side='left'
        )

        # Define special tokens
        n_mem_tokens = cfg.doc_max_length // cfg.compr_rate
        existing_special_tokens = tokenizer.special_tokens_map.get("additional_special_tokens", [])

        if cfg.different_mem_tokens:
            mem_tokens = [f'<MEM{i}>' for i in range(n_mem_tokens)]
            tokenizer.add_special_tokens({
                'additional_special_tokens': existing_special_tokens + mem_tokens + ['<AE>', '<ENC>', '<SEP>']
            })
            tokenizer.mem_tokens = mem_tokens
        else:
            tokenizer.add_special_tokens({
                'additional_special_tokens': existing_special_tokens + ['<MEM>', '<AE>', '<ENC>', '<SEP>']
            })
            tokenizer.mem_tokens = ['<MEM>'] * n_mem_tokens
        
        tokenizer.mem_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokenizer.mem_tokens]
        tokenizer.mem_token_ids_pt = torch.LongTensor(tokenizer.mem_token_ids)
        
        # Additional special tokens
        tokenizer.ae_token = '<AE>'
        tokenizer.ae_token_id = tokenizer.convert_tokens_to_ids('<AE>')
        tokenizer.enc_token = '<ENC>'
        tokenizer.sep_token = '<SEP>'
        tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids('<SEP>')
        
        # Handle model-specific tokens
        if tokenizer.bos_token is None and 'qwen' in cfg.decoder_model_name.lower():
            tokenizer.bos_token = tokenizer.special_tokens_map['additional_special_tokens'][0]
            tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
        
        if tokenizer.eos_token is None and "qwen" in cfg.decoder_model_name.lower():
            tokenizer.eos_token = tokenizer.special_tokens_map['additional_special_tokens'][1]
            tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

        # KBTC training tokens
        if cfg.kbtc_training:
            tokenizer.add_special_tokens({'additional_special_tokens': ['<KBTC>']})
            tokenizer.kbtc_token = '<KBTC>'
            tokenizer.kbtc_token_id = tokenizer.convert_tokens_to_ids('<KBTC>')

        # Set pad token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.bos_token_id
        
        print(f'Memory token count: {n_mem_tokens}')
        return tokenizer

    def _get_peft_config(self, lora_r: int) -> LoraConfig:
        """Build the PEFT configuration."""
        return LoraConfig(
            task_type="CAUSAL_LM", 
            r=lora_r, 
            lora_alpha=2*lora_r, 
            target_modules='all-linear', 
            lora_dropout=0.1
        )

    def _prepare_mem_tokens_optimization(self):
        """Setup memory token optimization if enabled."""
        if self.config.optimize_mem_tokens and self.compr is None:
            # Enable gradients for input embeddings
            self.decoder.get_input_embeddings().weight.requires_grad = True
            
            # Apply hook to zero gradients except for memory tokens
            def hook(grad):
                mask = torch.zeros_like(grad)
                mask[self.decoder_tokenizer.mem_token_ids] = 1.0
                return grad * mask
            
            self.decoder.get_input_embeddings().weight.register_hook(hook)
    
    def _set_all_adapters(self):
        """Activate all adapters for training."""
        if len(self.adapter_keys) > 0:
            self.decoder.set_adapter(self.adapter_keys)

    # Core compression and generation methods
    def compress(self, enc_input_ids: torch.Tensor, enc_attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress input documents."""
        if self.compr:
            return self.compr(enc_input_ids, enc_attention_mask)
        else:
            return self._compr_decoder(enc_input_ids, enc_attention_mask)
    
    def _compr_decoder(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use decoder as compressor."""
        assert input_ids.size() == attention_mask.size()
        
        if 'encoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('encoder_adapter')
        else:
            raise ValueError(f"encoder_adapter not in adapter_keys: {self.adapter_keys}")

        # Get embeddings from decoder
        emb = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        ).hidden_states[-1]

        # Create mask for memory tokens
        mask = torch.isin(
            input_ids, 
            self.decoder_tokenizer.mem_token_ids_pt.to(input_ids.device)
        )

        # Calculate MSE loss between memory and non-memory regions
        attn = attention_mask.bool()
        mem_mask = mask & attn
        non_mem_mask = (~mask) & attn

        mem_len = mem_mask.sum(dim=1)
        non_mem_len = non_mem_mask.sum(dim=1)

        if (mem_len == 0).any():
            raise ValueError("Some samples have no memory tokens")
        if (non_mem_len == 0).any():
            raise ValueError("Some samples have no non-memory tokens")

        mem_sum = (emb * mem_mask.unsqueeze(-1)).sum(dim=1)
        non_mem_sum = (emb * non_mem_mask.unsqueeze(-1)).sum(dim=1)

        mem_mean = mem_sum / mem_len.unsqueeze(-1)
        non_mem_mean = non_mem_sum / non_mem_len.unsqueeze(-1)

        mse_loss = F.mse_loss(non_mem_mean, mem_mean, reduction='mean')

        return emb[mask].reshape(emb.size(0), -1, emb.size(-1)), mse_loss

    def _compr_query_reasoner_stage2(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Query reasoning compression for stage 2."""
        assert input_ids.size() == attention_mask.size()
        
        if 'query_reasoner_adapter' in self.adapter_keys:
            self.decoder.set_adapter('query_reasoner_adapter')
        else:
            raise ValueError(f"query_reasoner_adapter not in adapter_keys: {self.adapter_keys}")

        emb = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        ).hidden_states[-1]

        mask = torch.isin(
            input_ids, 
            self.decoder_tokenizer.mem_token_ids_pt.to(input_ids.device)
        )

        return emb[mask].reshape(emb.size(0), -1)

    # Generation methods
    def generate_from_questions(self, 
                               questions: List[str], 
                               max_new_tokens: int = 128, 
                               temperature: float = 0.5, 
                               documents: List[List[str]] = None, 
                               stage2_mips: bool = False,  
                               stage2_retrieval_top_n: int = None,
                               time_count: bool = False) -> Tuple[List[str], torch.Tensor]:
        """Generate answers from questions using query reasoning."""
        if "query_reasoner_adapter" not in self.adapter_keys:
            raise ValueError("Query reasoner adapter not found")
        
        self.eval()
        
        with torch.no_grad():
            # Encode questions
            self.decoder.set_adapter('query_reasoner_adapter')
            flat_questions = [q for q in questions]
            
            if time_count:
                start_time = time.time()
            
            q_tok = self._prepare_encoder_inputs(flat_questions, max_length=self.doc_max_length)
            query_reps = self._compr_query_reasoner_stage2(
                q_tok["input_ids"].to(self.decoder.device), 
                q_tok["attention_mask"].to(self.decoder.device)
            )
            
            # Document retrieval and selection
            if stage2_mips:
                retrieved_doc_embeddings = self._retrieve_embeddings(
                    query_reps, stage2_retrieval_top_n=stage2_retrieval_top_n
                )
                scores = torch.bmm(
                    query_reps.unsqueeze(1), 
                    retrieved_doc_embeddings.transpose(1, 2)
                ).squeeze(1)
                z, topk_idx = differentiable_topk(scores, self.generation_top_k, temperature=0.5)
                selected_doc_embeddings = torch.einsum('bkn,bnd->bkd', z, retrieved_doc_embeddings)
                selected_doc_embeddings = selected_doc_embeddings.view(
                    selected_doc_embeddings.size(0) * selected_doc_embeddings.size(1), 
                    -1, self.hidden_size
                )
            else:
                # Use provided documents
                flat_documents = sum(documents, [])
                
                if time_count:
                    start_time1 = time.time()
                
                input_encoder = self._prepare_encoder_inputs(flat_documents, max_length=self.doc_max_length)
                device = self.decoder.device
                enc_input_ids = input_encoder['input_ids'].to(device)
                enc_attention_mask = input_encoder['attention_mask'].to(device)
                retrieved_doc_embeddings, _ = self.compress(enc_input_ids, enc_attention_mask)
                
                if time_count:
                    start_time2 = time.time()
                    compress_time = start_time2 - start_time1
                
                B = len(questions)
                stage2_retrieval_top_n = retrieved_doc_embeddings.shape[0] // B
                retrieved_doc_embeddings = retrieved_doc_embeddings.reshape(B, stage2_retrieval_top_n, -1)
                query_reps = query_reps.to(retrieved_doc_embeddings.dtype)

                if time_count:
                    start_time3 = time.time()
                
                scores = torch.bmm(
                    F.normalize(query_reps, dim=-1, p=2).unsqueeze(1).float(),
                    F.normalize(retrieved_doc_embeddings, dim=-1, p=2).float().transpose(1, 2)
                ).squeeze(1)
                
                z, topk_idx = differentiable_topk(scores, self.generation_top_k, temperature=0.02)
                selected_doc_embeddings = torch.einsum('bkn,bnd->bkd', z.to(retrieved_doc_embeddings.dtype), retrieved_doc_embeddings)
                selected_doc_embeddings = selected_doc_embeddings.view(
                    selected_doc_embeddings.size(0) * selected_doc_embeddings.size(1), 
                    -1, self.hidden_size
                )
                
                if time_count:
                    start_time4 = time.time()
                    query_time = start_time4 - start_time3 + start_time1 - start_time

            # Generate instructions and decode
            if time_count:
                start_time5 = time.time()
            
            instructions = [
                self._blend_prompt_and_selected_memory_tokens(query=q)[1] 
                for q in questions
            ]
            
            decoder_inputs = self.decoder_tokenizer(
                instructions,
                return_tensors='pt',
                padding="longest",
                add_special_tokens=False,
                truncation=True,
                max_length=1024,
            )
            
            dec_input_ids = decoder_inputs['input_ids'].to(self.decoder.device)
            dec_attention_mask = decoder_inputs['attention_mask'].to(self.decoder.device)
            
            # Replace memory token embeddings
            inputs_embeds = self._replace_emb_stage2(selected_doc_embeddings, dec_input_ids)
            
            # Switch to decoder adapter for generation
            if 'decoder_adapter' in self.adapter_keys:
                self.decoder.set_adapter('decoder_adapter')
            
            # Generate answers
            output_ids = self.decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=dec_attention_mask,
                do_sample=False,
                top_p=None,
                temperature=None,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.decoder_tokenizer.pad_token_id
            )
            
            if time_count:
                start_time6 = time.time()
                generate_time = start_time6 - start_time5
            
            # Decode generated tokens
            decoded = self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        if time_count:
            return decoded, topk_idx, compress_time, query_time, generate_time, compress_time + query_time + generate_time
        else:
            return decoded, topk_idx
    def generate_from_paraphrase(self, questions: list[str], documents: list[list[str]], max_new_tokens: int = 128) -> list[str]:
        """
        Generates answers from documents (via compression then decoding)
        questions: list of string
        documents: list of list of strings (they should all be of equal length: the nb of doc for each question)
        """
        self.generation_top_k = len(documents[0])
        assert len(documents) == len(questions)
        assert all([len(context) == len(documents[0]) for context in documents])
        flat_documents = sum(documents, [])
        
        model_input = {}
        
        # Creating encoder inputs:
        input_encoder = self._prepare_encoder_inputs(flat_documents, max_length=self.doc_max_length)
        device = self.decoder.device
        model_input['enc_input_ids'], model_input['enc_attention_mask'] = input_encoder['input_ids'].to(device), input_encoder['attention_mask'].to(device)
        
        # Creating decoder inputs
        instr = [self._blend_prompt_and_memory_tokens(query="", stage = "stage1", paraphrase_loss = True) for q in questions]
        inp_dec = self.decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False, truncation=True,  max_length=1024)
        model_input['dec_input_ids'], model_input['dec_attention_mask'] = inp_dec['input_ids'].to(device), inp_dec['attention_mask'].to(device)
        
        # Generation
        return self._generate(model_input, max_new_tokens=max_new_tokens)


    def generate_from_text(self, 
                          questions: List[str], 
                          documents: List[List[str]], 
                          max_new_tokens: int = 128) -> List[str]:
        """Generate answers from documents via compression then decoding."""
        self.generation_top_k = len(documents[0])
        assert len(documents) == len(questions)
        assert all(len(context) == len(documents[0]) for context in documents)
        
        flat_documents = sum(documents, [])
        
        # Create encoder inputs
        input_encoder = self._prepare_encoder_inputs(flat_documents, max_length=self.doc_max_length)
        device = self.decoder.device
        enc_input_ids = input_encoder['input_ids'].to(device)
        enc_attention_mask = input_encoder['attention_mask'].to(device)
        
        # Create decoder inputs
        instructions = [self._blend_prompt_and_memory_tokens(query=q, stage="stage1_2") for q in questions]
        inp_dec = self.decoder_tokenizer(
            instructions, 
            return_tensors='pt', 
            padding="longest", 
            add_special_tokens=False, 
            truncation=True,  
            max_length=1024
        )
        dec_input_ids = inp_dec['input_ids'].to(device)
        dec_attention_mask = inp_dec['attention_mask'].to(device)
        
        # Generate
        return self._generate({
            'enc_input_ids': enc_input_ids,
            'enc_attention_mask': enc_attention_mask,
            'dec_input_ids': dec_input_ids,
            'dec_attention_mask': dec_attention_mask
        }, max_new_tokens=max_new_tokens)

    def generate_from_compressed_documents_and_questions(self, 
                                                        questions: List[str], 
                                                        compressed_documents: torch.Tensor, 
                                                        max_new_tokens: int = 128) -> List[str]:
        """Generate answers from compressed documents."""
        self.generation_top_k = compressed_documents.size(0) // len(questions)
        assert compressed_documents.size(0) % self.generation_top_k == 0
        
        # Create decoder inputs
        instructions = [self._blend_prompt_and_memory_tokens(query=q, stage="stage1_2") for q in questions]
        inp_dec = self.decoder_tokenizer(
            instructions, 
            return_tensors='pt', 
            padding="longest", 
            add_special_tokens=False, 
            truncation=True,  
            max_length=1024
        )
        device = self.decoder.device
        dec_input_ids = inp_dec['input_ids'].to(device)
        dec_attention_mask = inp_dec['attention_mask'].to(device)

        # Create input decoder embeddings from prompt + compressed documents
        inputs_embeds = self._replace_emb(compressed_documents, dec_input_ids)
        
        # Activate decoder generator
        if 'decoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('decoder_adapter')
            
        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=dec_attention_mask,
            max_new_tokens=max_new_tokens
        )
        
        return self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def compress_documents(self, documents: List[str]) -> torch.Tensor:
        """Compress a list of documents."""
        input_encoder = self._prepare_encoder_inputs(documents, max_length=self.doc_max_length)
        enc_input_ids = input_encoder['input_ids'].to(self.decoder.device)
        attention_mask = input_encoder['attention_mask'].to(self.decoder.device)
        return self.compress(enc_input_ids=enc_input_ids, enc_attention_mask=attention_mask)

    # Helper methods
    def _prepare_encoder_inputs(self, texts: List[str], max_length: int, q_texts: List[str] = None) -> Dict[str, torch.Tensor]:
        """Create inputs for the encoder."""
        if q_texts is not None:
            assert len(texts) == len(q_texts)

        if self.compr is None:
            return self._prepare_encoder_inputs_to_decoder(texts, max_length, q_texts)
        else:
            return self.compr.prepare_inputs(texts, max_length, q_texts)

    def _prepare_encoder_inputs_to_decoder(self, texts: List[str], max_length: int, q_texts: List[str] = None) -> Dict[str, torch.Tensor]:
        """Prepare encoder inputs when using decoder as compressor."""
        if q_texts is not None:
            texts_to_encode = [
                self.decoder_tokenizer.enc_token + 
                self.decoder_tokenizer.bos_token + 
                '\nQuery:\n' + query + 
                'Document:\n' + text + 
                self.decoder_tokenizer.eos_token 
                for text, query in zip(texts, q_texts)
            ]
            inp_enc = self.decoder_tokenizer(
                texts_to_encode, 
                return_tensors='pt', 
                padding='max_length', 
                max_length=max_length + 8,
                truncation=True, 
                add_special_tokens=False
            )
        else:
            inp_enc = [
                self.decoder_tokenizer.enc_token + 
                self.decoder_tokenizer.bos_token + 
                text + 
                self.decoder_tokenizer.eos_token 
                for text in texts
            ]
            inp_enc = self.decoder_tokenizer(
                inp_enc, 
                return_tensors='pt', 
                padding="max_length", 
                max_length=max_length + 3,
                truncation=True, 
                add_special_tokens=False
            )

        num_mem_tokens = self.doc_max_length // self.compr_rate
        assert num_mem_tokens == len(self.decoder_tokenizer.mem_tokens)

        inp_enc['input_ids'], inp_enc['attention_mask'] = add_memory_tokens_to_inputs(
            inp_enc['input_ids'], 
            inp_enc['attention_mask'], 
            num_mem_tokens, 
            tokenizer=self.decoder_tokenizer
        )

        return inp_enc

    def _replace_emb(self, compressed_embs: torch.Tensor, dec_input_ids: torch.Tensor) -> torch.Tensor:
        """Replace memory tokens in decoder input with compressed embeddings."""
        indices = range(0, compressed_embs.size(0) + 1, self.generation_top_k)            
        return self._replace_embeddings(compressed_embs, dec_input_ids, indices)

    def _replace_emb_stage2(self, compressed_embs: torch.Tensor, dec_input_ids: torch.Tensor) -> torch.Tensor:
        """Replace memory tokens for stage 2."""
        indices = range(0, compressed_embs.size(0) + 1, self.generation_top_k)            
        return self._replace_embeddings(compressed_embs, dec_input_ids, indices)

    def _replace_embeddings(self, compressed_embs: torch.Tensor, dec_input_ids: torch.Tensor, indices: range) -> torch.Tensor:
        """Replace memory tokens with compressed embeddings."""
        inputs_embeds = self.decoder.get_input_embeddings()(dec_input_ids)
        num_embs = compressed_embs.size(1)
        slot_len = num_embs + (1 if self.sep else 0)
        
        # Get first memory token indices
        first_mem_token_indices = torch.argmax(
            (dec_input_ids == self.decoder_tokenizer.mem_token_ids[0]).int(), dim=1
        )
        batch_size = inputs_embeds.size(0)
        
        # Replace with compressed embeddings
        for i in range(batch_size):
            for j in range(indices[i], indices[i + 1]):
                start_idx = first_mem_token_indices[i].item() + (j - indices[i]) * slot_len
                assert inputs_embeds[i, start_idx:start_idx + num_embs, :].size() == compressed_embs[j].size()
                inputs_embeds[i, start_idx:start_idx + num_embs, :] = compressed_embs[j]
        
        return inputs_embeds

    def _retrieve_embeddings(self, questions: torch.Tensor, stage2_retrieval_top_n: int = 1) -> torch.Tensor:
        """Retrieve embeddings of documents."""
        response = requests.post(
            self.url_retrieval, 
            json={
                "queries": questions.detach().cpu().float().numpy().tolist(), 
                'k': self.generation_top_k
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")
        
        results = response.json()
        retrieval_embeddings = results['retrieved_embeddings']
        retrieval_embeddings = torch.tensor(
            retrieval_embeddings, 
            dtype=torch.bfloat16, 
            device=questions.device
        )
        
        if len(retrieval_embeddings.shape) == 4:
            retrieval_embeddings = retrieval_embeddings.reshape(
                retrieval_embeddings.shape[0] * retrieval_embeddings.shape[1], 
                retrieval_embeddings.shape[2], -1
            )
        
        return retrieval_embeddings

    def _blend_prompt_and_memory_tokens(self, query: str, answer: str = None, qa_loss: bool = False, 
                                       paraphrase_loss: bool = False, stage: str = "stage1") -> Tuple[int, str]:
        """Blend prompt with memory tokens for different training stages."""
        mem_tokens_str = ''.join(self.decoder_tokenizer.mem_tokens) + self.decoder_tokenizer.sep_token
        docs = mem_tokens_str * self.generation_top_k
        
        if stage == "stage1":
            if qa_loss:
                return self._blend_qa_prompt(docs, query, answer)
            elif paraphrase_loss:
                return self._blend_paraphrase_prompt(docs, answer)
        elif stage == "stage1_2":
            return self._blend_standard_prompt(docs, query, answer)
        
        raise ValueError(f"Unknown stage: {stage}")

    def _blend_qa_prompt(self, docs: str, query: List[str], answer: List[str]) -> Tuple[int, str]:
        """Create QA prompt for stage 1."""
        prompt_system = 'You are a helpful assistant. Given a document, your task is to generate some single questions to cover all key information of the document and answer them sequentially.'
        prompt_user = f"Background:\n{docs}"
        
        sys_prompt = [{"role": "system", "content": prompt_system}]
        user_prompt = [{"role": "user", "content": prompt_user.replace(':\ ', ': ')}]

        qa_lines = [f"Question: {q}\nAnswer: {a}" for q, a in zip(query, answer)]
        query_answer = "\n".join(qa_lines)
        assistant_prompt = [{"role": "assistant", "content": query_answer}]
        
        try:
            prompt = self.decoder_tokenizer.apply_chat_template(
                sys_prompt + user_prompt, 
                tokenize=False, 
                add_generation_prompt=True, 
                enable_thinking=False
            )
            response = self.decoder_tokenizer.apply_chat_template(
                sys_prompt + user_prompt + assistant_prompt, 
                tokenize=False, 
                add_generation_prompt=False, 
                enable_thinking=False
            )
            prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))
        except TemplateError as e:
            if "System role not supported" in str(e):
                messages = [{"role": "user", "content": sys_prompt[0]['content'] + '\n' + user_prompt[0]['content']}]
                prompt = self.decoder_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))
                # Handle response for unsupported system role
                messages_with_answer = messages + assistant_prompt
                response = self.decoder_tokenizer.apply_chat_template(
                    messages_with_answer, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
            else:
                raise e
        
        return prompt_len, response

    def _blend_paraphrase_prompt(self, docs: str, answer: str) -> Tuple[int, str]:
        """Create paraphrase prompt for stage 1."""
        prompt_system = 'You are a helpful assistant. Your task is follow the instructions to paraphrase the background information.'
        prompt_user = random.choice(PARAPHRASE_INSTRUCTIONS).format(docs=docs)

        sys_prompt = [{"role": "system", "content": prompt_system}]
        user_prompt = [{"role": "user", "content": prompt_user.replace(':\ ', ': ')}]
        
        try:
            prompt = self.decoder_tokenizer.apply_chat_template(
                sys_prompt + user_prompt, 
                tokenize=False, 
                add_generation_prompt=True, 
                enable_thinking=False
            )
            if answer is None:
                return prompt
            
            assistant_prompt = [{"role": "assistant", "content": answer}]
            response = self.decoder_tokenizer.apply_chat_template(
                sys_prompt + user_prompt + assistant_prompt, 
                tokenize=False, 
                add_generation_prompt=False, 
                enable_thinking=False
            )
            prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))
        except TemplateError as e:
            if "System role not supported" in str(e):
                combined_content = prompt_system + '\n' + prompt_user.replace(':\ ', ': ')
                messages = [{"role": "user", "content": combined_content}]
                prompt = self.decoder_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                if answer is None:
                    return prompt
                prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))
                messages_with_answer = messages + [{"role": "assistant", "content": answer}]
                response = self.decoder_tokenizer.apply_chat_template(
                    messages_with_answer, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
            else:
                raise e
        
        return prompt_len, response

    def _blend_standard_prompt(self, docs: str, query: str, answer: str) -> Tuple[int, str]:
        """Create standard prompt for stage 1_2."""
        prompt_system = 'You are a helpful assistant. Your task is to extract relevant information from provided documents and to answer to questions as briefly as possible.'
        prompt_user = f"Background:\n{docs}\n\nQuestion:{query}"
        
        sys_prompt = [{"role": "system", "content": prompt_system}]
        user_prompt = [{"role": "user", "content": prompt_user.replace(':\ ', ': ')}]
        
        try:
            prompt = self.decoder_tokenizer.apply_chat_template(
                sys_prompt + user_prompt, 
                tokenize=False, 
                add_generation_prompt=True, 
                enable_thinking=False
            )
            if answer is None:
                return prompt
            
            assistant_prompt = [{"role": "assistant", "content": answer}]
            response = self.decoder_tokenizer.apply_chat_template(
                sys_prompt + user_prompt + assistant_prompt, 
                tokenize=False, 
                add_generation_prompt=False, 
                enable_thinking=False
            )
            prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))
        except TemplateError as e:
            if "System role not supported" in str(e):
                combined_content = prompt_system + '\n' + prompt_user.replace(':\ ', ': ')
                messages = [{"role": "user", "content": combined_content}]
                prompt = self.decoder_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                if answer is None:
                    return prompt
                prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))
                messages_with_answer = messages + [{"role": "assistant", "content": answer}]
                response = self.decoder_tokenizer.apply_chat_template(
                    messages_with_answer, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
            else:
                raise e
        
        return prompt_len, response

    def _blend_prompt_and_selected_memory_tokens(self, query: str, answer: str = None) -> Tuple[int, str]:
        """Create prompt for stage 2 with selected memory tokens."""
        mem_tokens_str = ''.join(self.decoder_tokenizer.mem_tokens) + self.decoder_tokenizer.sep_token
        docs = mem_tokens_str * self.generation_top_k
        
        prompt_system = 'You are a helpful assistant. Your task is to extract relevant information from provided documents and to answer to questions as briefly as possible.'
        prompt_user = f"Background:\n{docs}\n\nQuestion:{query}"
        
        sys_prompt = [{"role": "system", "content": prompt_system}]
        user_prompt = [{"role": "user", "content": prompt_user.replace(':\ ', ': ')}]
        
        try:
            prompt = self.decoder_tokenizer.apply_chat_template(
                sys_prompt + user_prompt, 
                tokenize=False, 
                add_generation_prompt=True, 
                enable_thinking=False
            )
            prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))
            
            if answer is not None:
                assistant_prompt = [{"role": "assistant", "content": answer}]
                response = self.decoder_tokenizer.apply_chat_template(
                    sys_prompt + user_prompt + assistant_prompt, 
                    tokenize=False, 
                    add_generation_prompt=False,
                    enable_thinking=False
                )
            else:
                response = prompt
                
        except TemplateError as e:
            if "System role not supported" in str(e):
                combined_content = prompt_system + '\n' + prompt_user.replace(':\ ', ': ')
                messages = [{"role": "user", "content": combined_content}]
                
                prompt = self.decoder_tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True, 
                    enable_thinking=False
                )
                prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))
                
                if answer is not None:
                    messages_with_answer = messages + [{"role": "assistant", "content": answer}]
                    response = self.decoder_tokenizer.apply_chat_template(
                        messages_with_answer, 
                        tokenize=False, 
                        add_generation_prompt=False, 
                        enable_thinking=False
                    )
                else:
                    response = prompt
            else:
                raise e
        
        return prompt_len, response

    # Model saving and loading methods
    def save_pretrained(self, save_directory: str, **kwargs):
        """Save only the LoRA adapters and their configurations."""
        if self.lora:
            if not os.path.exists(save_directory):
                os.makedirs(save_directory) 

            # Save LoRA adapter weights
            torch.save(
                self._get_all_adapters_state_dict(), 
                os.path.join(save_directory, "adapters.pth")
            )
            
            # Save first and last layers of decoder
            torch.save(
                self._get_decoder_first_and_last_layer_state_dict(), 
                os.path.join(save_directory, "decoder_first_last_layers.pth")
            )
            
            # Save configuration
            self.config.save_pretrained(save_directory)
        else:
            super().save_pretrained(save_directory, **kwargs)

    def _get_all_adapters_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return the state dicts of all adapters."""
        return {
            key: {k: v.cpu() for k, v in self.decoder.get_adapter_state_dict(key).items()} 
            for key in self.adapter_keys
        }

    def _get_decoder_first_and_last_layer_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get first and last layers that change when adding tokens."""
        out = {}
        for k, v in self.decoder.named_parameters():
            if 'lm_head.weight' in k or 'embed_tokens.weight' in k:
                out[k] = v.cpu()
        return out

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        """Load model from pretrained checkpoint."""
        # Load configuration
        config = CLaRaConfig.from_pretrained(pretrained_model_name_or_path)
        
        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        map_location = torch.device("cpu") if not torch.cuda.is_available() else None

        if config.lora:
            # Delay adapter construction
            config.load_adapters = False
            if 'device_map' in kwargs:
                config.device_map = kwargs['device_map']

            # Initialize model
            print(f"Initializing model from trained checkpoint: {config}")
            model = cls(config)

            # Load first and last layers
            try:
                first_and_last_layers_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path, 
                    filename="decoder_first_last_layers.pth"
                )
            except Exception:
                first_and_last_layers_path = os.path.join(
                    pretrained_model_name_or_path, "decoder_first_last_layers.pth"
                )

            if os.path.exists(first_and_last_layers_path):
                first_and_last_decoder_state_dict = torch.load(
                    first_and_last_layers_path, map_location=map_location, weights_only=True
                )
                for key in first_and_last_decoder_state_dict:
                    assert key in model.decoder.state_dict()
                model.decoder.load_state_dict(first_and_last_decoder_state_dict, strict=False)
            else:
                print(f'First and last layer not found: {first_and_last_layers_path}')

            peft_config = model._get_peft_config(lora_r=config.lora_r)
            
            # Load LoRA adapters
            try:
                adapters_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path, 
                    filename="adapters.pth"
                )
            except Exception:
                adapters_path = os.path.join(pretrained_model_name_or_path, "adapters.pth")
    
            if os.path.exists(adapters_path):
                adapters_state_dict = torch.load(adapters_path, map_location=map_location, weights_only=True)
                model._load_adapters_from_state_dict(adapters_state_dict, peft_config, config)
            else:
                warnings.warn(f'Adapters not found at {adapters_path}')

            model._set_all_adapters()
            config.load_adapters = True
            return model
        else:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
    def _load_adapters_from_state_dict(self, adapters_state_dict: Dict, peft_config: LoraConfig, config: CLaRaConfig):
        """Load adapters from state dict based on training stage."""
        if not getattr(config, 'pure_inference', False):
            for key, val in adapters_state_dict.items():
                # Skip certain adapters based on training stage
                if config.training_stage == 'stage1' and key == 'query_reasoner_adapter':
                    continue
                elif config.training_stage == 'stage1_2' and key in ['query_reasoner_adapter', 'decoder_adapter']:
                    continue
                elif config.training_stage == 'stage2_reasoning' and key == 'decoder_adapter':
                    continue

                self._load_adapter_from_state_dict(
                    peft_config=peft_config, 
                    adapter_name=key, 
                    adapter_state_dict=val
                )
        else:
            # Load all adapters for pure inference
            for key, val in adapters_state_dict.items():
                self._load_adapter_from_state_dict(
                    peft_config=peft_config, 
                    adapter_name=key, 
                    adapter_state_dict=val
                )

        # Handle special cases for stage 2 training
        if config.training_stage == 'stage2' and 'query_reasoner_adapter' not in adapters_state_dict:
            self._handle_query_reasoner_adapter_loading(adapters_state_dict, peft_config)

    def _load_adapter_from_state_dict(self, peft_config: LoraConfig, adapter_name: str, adapter_state_dict: Dict):
        """Create adapter from state dict."""
        print(f'Loading checkpoint adapter: {adapter_name}')
        self.decoder.load_adapter(
            peft_config=peft_config, 
            adapter_name=adapter_name, 
            adapter_state_dict=adapter_state_dict
        )
        self.adapter_keys.append(adapter_name)

    def _handle_query_reasoner_adapter_loading(self, adapters_state_dict: Dict, peft_config: LoraConfig):
        """Handle special loading logic for query reasoner adapter."""
        if 'encoder_adapter' in adapters_state_dict and 'query_reasoner_adapter' not in adapters_state_dict:
            # Rename encoder adapter to query reasoner adapter
            renamed = {}
            for k, v in adapters_state_dict['encoder_adapter'].items():
                new_k = k.replace('encoder_adapter', 'query_reasoner_adapter')
                renamed[new_k] = v.detach().clone()
            
            self._load_adapter_from_state_dict(
                peft_config=peft_config,
                adapter_name='query_reasoner_adapter',
                adapter_state_dict=renamed
            )
            print('Loaded query_reasoner_adapter from stage 1 compressor checkpoint')
        else:
            # Create new adapter randomly
            self.decoder.add_adapter(peft_config, 'query_reasoner_adapter')
            self.adapter_keys.append('query_reasoner_adapter')
            print('Loaded query_reasoner_adapter randomly for stage 2 training')

    # Forward pass methods
    def forward(self, 
                batch: Dict = None,
                questions: List[str] = None,
                documents: List[List[str]] = None,
                answers: List[str] = None,
                original_answer_gen_api: str = None,
                stage2_mips: bool = False,
                stage2_retrieval_top_n: int = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with support for both batch and legacy interfaces.
        
        Args:
            batch: Preprocessed batch dict (new interface)
            questions: List of questions (legacy interface)  
            documents: List of document lists (legacy interface)
            answers: List of answers (legacy interface)
            original_answer_gen_api: API URL for generation (legacy interface)
            stage2_mips: Whether to use MIPS for stage2
            stage2_retrieval_top_n: Top-n for stage2 retrieval
            
        Returns:
            Tuple of (loss, additional_outputs_dict)
        """
        if batch is not None:
            return self._forward_batch(batch, stage2_mips, stage2_retrieval_top_n)
        else:
            return self._forward_legacy(questions, documents, answers, original_answer_gen_api)

    def _forward_batch(self, batch: Dict, stage2_mips: bool, stage2_retrieval_top_n: int) -> Tuple[torch.Tensor, Dict]:
        """Handle batch-based forward pass."""
        stage = batch.get("stage", None)
        
        if stage in ["stage1", "stage1_2"]:
            return self._forward_stage1_batch(batch)
        elif stage == "stage2":
            return self._forward_stage2_batch(batch, stage2_mips, stage2_retrieval_top_n)
        elif stage == "stage2_pretrain_retrieval":
            return self._forward_stage2_pretrain_batch(batch, stage2_mips, stage2_retrieval_top_n)
        elif stage == "stage2_reasoning":
            return self._forward_stage2_reasoning_batch(batch)
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def _forward_stage1_batch(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """Forward pass for stage 1 training."""
        # Move tensors to device
        enc_input_ids = batch["enc_input_ids"].to(self.decoder.device)
        enc_attention_mask = batch["enc_attention_mask"].to(self.decoder.device)
        dec_input_ids = batch["dec_input_ids"].to(self.decoder.device)
        dec_attention_mask = batch["dec_attention_mask"].to(self.decoder.device)
        labels = batch["labels"].to(self.decoder.device)
        
        out = self._forward_stage_1(
            enc_input_ids=enc_input_ids,
            enc_attention_mask=enc_attention_mask,
            dec_input_ids=dec_input_ids,
            dec_attention_mask=dec_attention_mask,
            labels=labels,
        )
        return out["loss"], {"logits": out["logits"], "mse_loss": out["mse_loss"]}

    def _forward_stage2_batch(self, batch: Dict, stage2_mips: bool, stage2_retrieval_top_n: int) -> Tuple[torch.Tensor, Dict]:
        """Forward pass for stage 2 training."""
        self.decoder.set_adapter('query_reasoner_adapter')
        
        B = batch["labels"].shape[0]
        query_reps = self._compr_query_reasoner_stage2(
            batch["query_input_ids"].to(self.decoder.device), 
            batch["query_attention_mask"].to(self.decoder.device)
        )

        enc_input_ids = batch["enc_input_ids"].to(self.decoder.device)
        enc_attention_mask = batch["enc_attention_mask"].to(self.decoder.device)
        dec_input_ids = batch["dec_input_ids"].to(self.decoder.device)
        dec_attention_mask = batch["dec_attention_mask"].to(self.decoder.device)
        labels = batch["labels"].to(self.decoder.device)

        # Document retrieval and selection
        if stage2_mips:
            retrieved_doc_embeddings = self._retrieve_embeddings(
                query_reps, stage2_retrieval_top_n=stage2_retrieval_top_n
            )
            scores = torch.bmm(
                query_reps.unsqueeze(1), 
                retrieved_doc_embeddings.transpose(1, 2)
            ).squeeze(1)
            z, topk_idx = differentiable_topk(scores, self.generation_top_k, temperature=1)
            selected = torch.einsum('bkn,bnd->bkd', z, retrieved_doc_embeddings)
            selected = selected.view(selected.size(0) * selected.size(1), -1, self.hidden_size)
        else:
            with torch.no_grad():
                retrieved_doc_embeddings, mse_loss = self.compress(enc_input_ids, enc_attention_mask)
            
            stage2_retrieval_top_n = retrieved_doc_embeddings.shape[0] // B
            retrieved_doc_embeddings = retrieved_doc_embeddings.reshape(B, stage2_retrieval_top_n, -1)
            query_reps = query_reps.to(retrieved_doc_embeddings.dtype)
            
            scores = torch.bmm(
                F.normalize(query_reps, dim=-1, p=2).unsqueeze(1).float(),
                F.normalize(retrieved_doc_embeddings, dim=-1, p=2).float().transpose(1, 2)
            ).squeeze(1)
            
            z, topk_idx = differentiable_topk(scores, self.generation_top_k, temperature=0.02)
            selected = torch.einsum('bkn,bnd->bkd', z.to(retrieved_doc_embeddings.dtype), retrieved_doc_embeddings)
            selected = selected.view(selected.size(0) * selected.size(1), -1, self.hidden_size)

        inputs_embeds = self._replace_emb_stage2(selected, dec_input_ids)
        
        if 'decoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('decoder_adapter')
        
        dec_out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=dec_attention_mask,
            labels=labels,
        )
        
        self.decoder.set_adapter(['decoder_adapter', 'query_reasoner_adapter'])
        return dec_out.loss, {"logits": dec_out.logits, "topk_idx": topk_idx, "mse_loss": mse_loss}

    def _forward_stage2_pretrain_batch(self, batch: Dict, stage2_mips: bool, stage2_retrieval_top_n: int) -> Tuple[torch.Tensor, Dict]:
        """Forward pass for stage 2 pretraining with retrieval."""
        self.decoder.set_adapter('query_reasoner_adapter')
        
        B = batch["labels"].shape[0]
        N = batch["enc_input_ids"].shape[0] // B
        device = self.decoder.device
        
        query_reps = self._compr_query_reasoner_stage2(
            batch["query_input_ids"].to(device), 
            batch["query_attention_mask"].to(device)
        )

        enc_input_ids = batch["enc_input_ids"].to(device)
        enc_attention_mask = batch["enc_attention_mask"].to(device)

        with torch.no_grad():
            retrieved_doc_embeddings, mse_loss = self.compress(enc_input_ids, enc_attention_mask)
        
        stage2_retrieval_top_n = retrieved_doc_embeddings.shape[0] // B
        retrieved_doc_embeddings = retrieved_doc_embeddings.reshape(B, stage2_retrieval_top_n, -1)
        query_reps = query_reps.to(retrieved_doc_embeddings.dtype)
        
        scores = torch.bmm(
            F.normalize(query_reps, dim=-1, p=2).unsqueeze(1).float(),
            F.normalize(retrieved_doc_embeddings, dim=-1, p=2).float().transpose(1, 2)
        ).squeeze(1)
        
        pos_index = batch["pos_index"]
        pos_mask = build_pos_mask(pos_index, N, device)
        tau = 0.02
        logits = scores / tau
        
        pos_logits = logits.masked_fill(~pos_mask, float('-inf'))
        num = torch.logsumexp(pos_logits, dim=-1)
        den = torch.logsumexp(logits, dim=-1)
        loss_vec = -(num - den)
        valid = pos_mask.any(dim=-1)
        loss = loss_vec[valid].mean()

        topk = self.generation_top_k
        topk_idx = logits.topk(k=min(topk, N), dim=-1).indices
        
        return loss, {"logits": [[]], "topk_idx": topk_idx, "mse_loss": mse_loss}

    def _forward_stage2_reasoning_batch(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """Forward pass for stage 2 reasoning training."""
        B = batch["labels"].shape[0]
        enc_input_ids = batch["enc_input_ids"].to(self.decoder.device)
        enc_attention_mask = batch["enc_attention_mask"].to(self.decoder.device)
        dec_input_ids = batch["dec_input_ids"].to(self.decoder.device)
        dec_attention_mask = batch["dec_attention_mask"].to(self.decoder.device)
        labels = batch["labels"].to(self.decoder.device)

        if sum(batch["docs_num"]) != 0:
            with torch.no_grad():
                selected, mse_loss = self.compress(enc_input_ids, enc_attention_mask)
                indices = batch["docs_num"]
                inputs_embeds = self._replace_reasoning_embeddings(selected, dec_input_ids, indices)
        else:
            inputs_embeds = self.decoder.get_input_embeddings()(dec_input_ids)
            mse_loss = 0

        if 'decoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('decoder_adapter')
        
        dec_out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=dec_attention_mask,
            labels=labels,
        )
        
        self.decoder.set_adapter(['decoder_adapter'])
        return dec_out.loss, {"logits": dec_out.logits, "mse_loss": mse_loss}

    def _forward_stage_1(self,
                        enc_input_ids: torch.LongTensor = None,
                        enc_attention_mask: torch.LongTensor = None,
                        dec_input_ids: torch.LongTensor = None,
                        dec_attention_mask: torch.LongTensor = None,
                        labels: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """Stage 1 forward pass for document compression and QA."""
        assert enc_input_ids.size() == enc_attention_mask.size()
        
        # Flatten 3D inputs to 2D if needed
        if len(enc_input_ids.size()) == 3:
            batch_size, top_k, seq_length = enc_input_ids.size()
            enc_input_ids = enc_input_ids.view(batch_size * top_k, seq_length)
            enc_attention_mask = enc_attention_mask.view(batch_size * top_k, seq_length)
        
        assert enc_input_ids.size(0) == dec_input_ids.size(0) * self.generation_top_k
        
        # Compress documents
        compressed_embs, mse_loss = self.compress(enc_input_ids, enc_attention_mask)
        
        # Replace memory tokens with compressed embeddings
        inputs_embeds = self._replace_emb(compressed_embs, dec_input_ids)

        # Detach if compressor-only training
        if (self.training_form == "compressor") and (self.compr is None):
            inputs_embeds = inputs_embeds.detach()

        # Set decoder adapter
        if 'decoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('decoder_adapter')

        # Forward through decoder
        decoder_outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=dec_attention_mask,
            labels=labels
        )

        # Reactivate all adapters
        self.decoder.set_adapter(['decoder_adapter', 'encoder_adapter'])
        
        return {
            "loss": decoder_outputs.loss, 
            "logits": decoder_outputs.logits, 
            "mse_loss": mse_loss
        }

    def _replace_reasoning_embeddings(self,
                                    compressed_embs: torch.Tensor,
                                    dec_input_ids: torch.LongTensor,
                                    docs_per_example: List[int]) -> torch.Tensor:
        """Replace memory slots with compressed embeddings for reasoning."""
        device = dec_input_ids.device
        inputs_embeds = self.decoder.get_input_embeddings()(dec_input_ids)

        num_embs = compressed_embs.size(1)
        slot_len = num_embs + (1 if getattr(self, "sep", False) else 0)

        if not isinstance(docs_per_example, torch.Tensor):
            docs_per_example = torch.tensor(docs_per_example, device=device, dtype=torch.long)
        else:
            docs_per_example = docs_per_example.to(device=device, dtype=torch.long)

        offsets = torch.zeros(docs_per_example.size(0) + 1, device=device, dtype=torch.long)
        offsets[1:] = torch.cumsum(docs_per_example, dim=0)
        total_docs = int(offsets[-1].item())
        assert total_docs == compressed_embs.size(0)

        mem_id = self.decoder_tokenizer.mem_token_ids[0]
        B, L, H = inputs_embeds.size()

        for i in range(B):
            # Find first memory token position
            mem_pos = (dec_input_ids[i] == mem_id).nonzero(as_tuple=True)[0]
            if mem_pos.numel() == 0:
                continue
            first_mem_idx = int(mem_pos[0].item())

            n_docs_i = int(docs_per_example[i].item())
            base = int(offsets[i].item())

            needed_len = first_mem_idx + n_docs_i * slot_len
            assert needed_len <= L

            for local_j in range(n_docs_i):
                global_j = base + local_j
                start_idx = first_mem_idx + local_j * slot_len
                target_slice = inputs_embeds[i, start_idx:start_idx + num_embs, :]
                src = compressed_embs[global_j]
                assert target_slice.size() == src.size()
                inputs_embeds[i, start_idx:start_idx + num_embs, :] = src

        return inputs_embeds

    def _generate(self, model_input: Dict[str, torch.Tensor], max_new_tokens: int = 128, 
                 return_doc_embeddings: bool = False) -> List[str]:
        """Generate text from model inputs."""
        enc_input_ids = model_input['enc_input_ids']
        enc_attention_mask = model_input['enc_attention_mask']
        dec_input_ids = model_input['dec_input_ids']
        dec_attention_mask = model_input['dec_attention_mask']
        
        assert enc_input_ids.size() == enc_attention_mask.size()
        
        if len(enc_input_ids.size()) == 3:
            batch_size, top_k, seq_length = enc_input_ids.size()
            enc_input_ids = enc_input_ids.view(batch_size * top_k, seq_length)
            enc_attention_mask = enc_attention_mask.view(batch_size * top_k, seq_length)
            
        assert enc_input_ids.size(0) == dec_input_ids.size(0) * self.generation_top_k
            
        compressed_embs, _ = self.compress(enc_input_ids.to('cuda'), enc_attention_mask.to('cuda'))
        inputs_embeds = self._replace_emb(compressed_embs, dec_input_ids.to('cuda'))
        
        if 'decoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('decoder_adapter') 

        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds.to("cuda"),
            attention_mask=dec_attention_mask.to("cuda"),
            do_sample=False,
            top_p=None,
            max_new_tokens=max_new_tokens
        )

        decoded = self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        if return_doc_embeddings:
            assert 'batch_size' in locals() and 'top_k' in locals()
            compressed_embs = compressed_embs.view(batch_size, top_k, compressed_embs.size(1), compressed_embs.size(2))
            return decoded, compressed_embs
        else:
            return decoded


# Example usage and testing
if __name__ == '__main__':
    # Example configuration
    cfg = CLaRaConfig(
        decoder_model_name='/mnt/ceph_rbd/model/Mistral-7B-Instruct-v0.2',
        compr_model_name="mistral_trimmed",
        compr_rate=64,
        compr_n_layers=5,
        compr_mlp_hidden_dim=8096,
        compr_use_mlp=False, 
        lora=True,
        lora_compressor=True,
        training_form="both",
        load_adapters=True,
        kbtc_training=False,
        optimize_mem_tokens=True,
        different_mem_tokens=True,
        attn_implementation='flash_attention_2'
    )
    
    # Initialize model
    clara = CLaRa(cfg)
    
    # Save and reload test
    clara.save_pretrained('test_ckpt')
    
    del clara
    torch.cuda.empty_cache()
    gc.collect()
    
    # Reload model
    clara = CLaRa.from_pretrained('test_ckpt')
    print("Model successfully loaded!")