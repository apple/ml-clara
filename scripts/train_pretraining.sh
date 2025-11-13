#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


set -ex

DEBUG=${DEBUG:-0}

# Set environment variables
export PYTHONPATH=/mnt/ceph_rbd/comp_rag/clara:$PYTHONPATH
export WANDB_DIR=/mnt/ceph_rbd/comp_rag/clara/debug_data/wandb_logs

# Configuration
data_path=/mnt/ceph_rbd/comp_rag/clara/debug_data
SAVE_MODEL_NAME=clara_cluster2_2m_mix_stage1
SAVE_PATH=/mnt/ceph_rbd/comp_rag/clara/debug_data/train_checkpoint/$SAVE_MODEL_NAME
WANDB_TOKEN=xx
MODEL_PATH=/mnt/ceph_rbd/model/Mistral-7B-Instruct-v0.2

mkdir -p $SAVE_PATH
# cp -r /mnt/conductor_data/code/clara_project/clara_training $SAVE_PATH/

# Extract distributed parameters dynamically
NCCL_DEBUG=INFO
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

echo "Currently using $(which python)"

# Training command with torchrun
training_commands="openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset $data_path/pretrain_data.jsonl \
   --pretrain $MODEL_PATH \
   --train_batch_size 128 \
   --micro_train_batch_size 2 \
   --ckpt_path $SAVE_PATH \
   --max_samples 500 \
   --save_path $SAVE_PATH \
   --save_steps -2 \
   --logging_steps 1 \
   --eval_steps 20 \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 1e-4 \
   --stage stage1 \
   --generation_top_k 1 \
   --qa_loss \
   --doc_max_length 256 \
   --compress_rate 32 \
   --mse_loss \
   --gradient_checkpointing"

# --use_wandb $WANDB_TOKEN"
#    --wandb_run_name $SAVE_MODEL_NAME \
#    --wandb_project CLaRa"

# Build distributed arguments
DISTRIBUTED_ARGS="--nproc_per_node ${NUM_LOCAL_GPUS} --nnodes ${NUM_NODES} --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint ${MASTER}:${MASTER_PORT} --master_addr ${MASTER} --master_port ${MASTER_PORT} --node_rank ${NODE_RANK}"

# Run training with torchrun for multinode
echo "Starting CLaRa training on node $NODE_RANK of $NUM_NODES nodes..."
if [ $DEBUG -eq 0 ]; then
    if [ $NUM_NODES -gt 1 ]; then
        # For multinode, check if EFA is available
        if command -v fi_info >/dev/null 2>&1; then
            fi_info -p efa -t FI_EP_RDM; torchrun $DISTRIBUTED_ARGS -m $training_commands
        else
            torchrun $DISTRIBUTED_ARGS -m $training_commands
        fi
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

# Copy model file
cp ../openrlhf/models/modeling_clara.py $SAVE_PATH

##############################
# Optional: Final inference
##############################
# echo "Running final inference..."
# cd /mnt/conductor_data/code/clara_project/clara
# unset PYTHONPATH
# export PYTHONPATH=$SAVE_PATH:$PYTHONPATH
# 
# if [ "$NODE_RANK" -eq 0 ]; then
#     accelerate launch \
#         --num_processes=8 \
#         --num_machines=1 \
#         evaluate.py \
#         --model_path $SAVE_MODEL_NAME \
#         --stage stage1 \
#         --dataset hotpotqa,multihoprag,musique
# else
#     echo "Node rank $NODE_RANK: skipping inference"
#     exit 0
# fi

echo "CLaRa training completed successfully!"