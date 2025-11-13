#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


set -ex

DEBUG=${DEBUG:-0}

########################################
# Environment Variables
########################################
export PYTHONPATH=/mnt/ceph_rbd/comp_rag/clara:$PYTHONPATH
export WANDB_DIR=/mnt/task_wrapper/user_output/artifacts/data/wandb_logs
export NCCL_DEBUG=INFO

########################################
# Configuration
########################################
data_path=/mnt/ceph_rbd/comp_rag/clara/debug_data
SAVE_MODEL_NAME=clara_cluster1_2_2m_split_data_single_32_mistral
SAVE_PATH=/mnt/ceph_rbd/comp_rag/clara/debug_data/train_checkpoint/$SAVE_MODEL_NAME
WANDB_TOKEN=xx
MODEL_PATH=/mnt/ceph_rbd/model/Mistral-7B-Instruct-v0.2
PRETRAIN_CKPT=/mnt/ceph_rbd/comp_rag/clara/debug_data/train_checkpoint/clara_cluster2_2m_mix_stage1

mkdir -p $SAVE_PATH
# cp -r /mnt/ceph_rbd/comp_rag/clara $SAVE_PATH/

echo "Currently using $(which python)"

########################################
# Extract Distributed Parameters
########################################
NUM_NODES=1
MASTER=127.0.0.1
MASTER_PORT=29500
NODE_RANK=0
NUM_LOCAL_GPUS=4
WORLD_SIZE=$((NUM_LOCAL_GPUS * NUM_NODES))

echo "Number of nodes: ${NUM_NODES}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "Number of local GPUs: ${NUM_LOCAL_GPUS}"
echo "Master: ${MASTER}"
echo "Master port: ${MASTER_PORT}"
echo "Node rank: ${NODE_RANK}"

eval_dataset=xx
########################################
# Training Command
########################################
training_commands="openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset $data_path/instruction_tuning_data.jsonl \
   --pretrain $MODEL_PATH \
   --pretrain_checkpoint $PRETRAIN_CKPT \
   --train_batch_size 128 \
   --micro_train_batch_size 2 \
   --ckpt_path $SAVE_PATH \
   --max_samples 500 \
   --save_path $SAVE_PATH \
   --save_steps -2 \
   --logging_steps 1 \
   --eval_steps 30 \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 1e-4 \
   --gradient_checkpointing \
   --generation_top_k 5 \
   --stage stage1_2 \
   --doc_max_length 256 \
   --compress_rate 32 \
   --mse_loss \
   --do_eval_gen"

#--eval_dataset $eval_dataset \
#    --use_wandb $WANDB_TOKEN"
#    --wandb_run_name $SAVE_MODEL_NAME \
#    --wandb_project CLaRa"

########################################
# Distributed Arguments for torchrun
########################################
DISTRIBUTED_ARGS="--nproc_per_node ${NUM_LOCAL_GPUS} \
   --nnodes ${NUM_NODES} \
   --rdzv_id 101 \
   --rdzv_backend c10d \
   --rdzv_endpoint ${MASTER}:${MASTER_PORT} \
   --master_addr ${MASTER} \
   --master_port ${MASTER_PORT} \
   --node_rank ${NODE_RANK}"

########################################
# Multi-node Training
########################################
echo "Starting CLaRa stage1_2 training (multinode with torchrun)..."
if [ $DEBUG -eq 0 ]; then
    if [ "$NUM_NODES" -gt 1 ]; then
        # Check EFA for multi-node if available
        if command -v fi_info >/dev/null 2>&1; then
            fi_info -p efa -t FI_EP_RDM || true
        fi
        torchrun $DISTRIBUTED_ARGS -m $training_commands
    else
        torchrun $DISTRIBUTED_ARGS -m $training_commands
    fi
else
    # Debug mode
    WORLD_SIZE=1 LOCAL_RANK=0 \
    python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
    -m torch.distributed.launch --nproc_per_node=2 --master_port=20001 \
    -m $training_commands
fi

########################################
# Copy Model Files
########################################
cp ../openrlhf/models/modeling_clara.py $SAVE_PATH

########################################
# Final Inference (only on rank 0)
########################################
# echo "Running final inference..."
# cd /mnt/ceph_rbd/comp_rag/clara/evaluation

# # Clean and set PYTHONPATH to avoid conflicts
# unset PYTHONPATH
# export PYTHONPATH=/mnt/conductor_data/clara:$PYTHONPATH

# echo "Starting inference on node $NODE_RANK of $NUM_NODES nodes..."
# if [ "$NODE_RANK" -eq 0 ]; then
#     # Run inference with gold retrieval
#     accelerate launch \
#         --num_processes=8 \
#         --num_machines=1 \
#         evaluate.py \
#         --model_path $SAVE_MODEL_NAME \
#         --stage stage1 \
#         --dataset musique,hotpotqa,2wiki,nq \
#         --gold_retrieval
    
#     # Run inference without gold retrieval
#     accelerate launch \
#         --num_processes=8 \
#         --num_machines=1 \
#         evaluate.py \
#         --model_path $SAVE_MODEL_NAME \
#         --stage stage1 \
#         --dataset musique,hotpotqa,2wiki,nq
# else
#     echo "Node rank $NODE_RANK: skipping inference"
#     exit 0
# fi

# echo "CLaRa stage1_2 training and inference completed successfully!"