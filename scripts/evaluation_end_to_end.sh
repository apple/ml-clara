#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

cd /mnt/conductor_data/clara/evaluation
unset PYTHONPATH
SAVE_PATH=/mnt/task_wrapper/user_output/artifacts/data/train_checkpoint/xx
SAVE_MODEL_NAME=${SAVE_PATH##*/}
export PYTHONPATH="$SAVE_PATH:$PYTHONPATH"

export PYTHONPATH=$SAVE_PATH:$PYTHONPATH

# Run inference with torchrun for multinode
echo "Starting inference on node $NODE_RANK of $NUM_NODES nodes..."
for i in musique; do    
    accelerate launch \
        --num_processes=8 \
        --num_machines=1 \
        --mixed_precision=bf16 \
        evaluate.py \
        --model_path $SAVE_MODEL_NAME \
        --stage stage2 \
        --dataset $i \
        --gold_retrieval
done
echo "✅ All steps completed successfully with torchrun multinode!"