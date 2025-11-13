#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

unset PYTHONPATH
cd /mnt/conductor_data/clean_code/evaluation

SAVE_PATH=/mnt/task_wrapper/user_output/artifacts/data/train_checkpoint/xx
SAVE_MODEL_NAME=${SAVE_PATH##*/}
export PYTHONPATH="$SAVE_PATH:$PYTHONPATH"


echo "Starting inference on node $NODE_RANK of $NUM_NODES nodes..."
accelerate launch \
    --num_processes=8 \
    --num_machines=1 \
    evaluate.py \
    --model_path $SAVE_MODEL_NAME \
    --stage stage1 \
    --dataset musique,hotpotqa,2wiki,nq \
    --gold_retrieval
