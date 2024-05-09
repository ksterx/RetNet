#!/bin/bash

#################################################
## Jobの設定
#SBATCH --nodes=1                # 確保するノード数
#SBATCH --gpus=8                 # 確保する全体のGPU数
#SBATCH --job-name=retnet        # ジョブの名前を設定
#SBATCH --output=logs/%x_%j.log  # 標準出力のファイル名指定 %x:ジョブ名 %j:ジョブID
#SBATCH --error=logs/%x_%j.log   # エラー出力のファイル名指定 (標準出力と同じファイルに出力している)
#SBATCH --ntasks-per-node=1      # ノードあたりのタスク数
#SBATCH --gpus-per-node=8        # ノードあたりのGPU数
#SBATCH --gpus-per-task=8        # 1つのタスクで利用するGPU数
#SBATCH --wait-all-nodes=1       # 要求したすべてのノードが利用可能になるまで待機
set -euxo pipefail

## 各タスクがマスターノードを見つけられるように環境変数を設定
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 1024-65535 -n 1)

## InfiniBandを使う場合
export MELLANOX_VISIBLE_DEVICES=all
export UCX_TLS=rc,cuda,sm,self

## TCP/IPでプロセス間通信を行う場合
# export UCX_TLS=tcp,cuda,sm,self

## IntelMPIを使ってInfiniBandでプロセス間通信を行う場合
export FI_LOG_LEVEL=1
export FI_PROVIDER=mlx

## NCCLライブラリを使う場合に切り分けのためにログを表示させる
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

export SPLLM_LIBRARY_PATH=/home/k_ishikawa/workspace/SpiralLLM
export SPCHAT_LIBRARY_PATH=/home/k_ishikawa/workspace/SpiralChat


#################################################
## 実行するイメージやマウント場所の指定
: "${APPS_PATH:=/home/k_ishikawa/workspace/RetNet}"
: "${DATA_PATH:=/nas/share}" # 読み込むデータのディレクトリ
: "${RESULT_PATH:=/nas/$USER/results}" # 結果を保存するディレクトリ
: "${IMAGE:=/home/k_ishikawa/sqshes/ksterx+pyxis+2.1.sqsh}"
: "${WORK_DIR:=/home/$USER/workspace}" # 作業ディレクトリの指定

: "${DATA_MOUNT:=$DATA_PATH:$DATA_PATH}" # データマウントポイントの指定
: "${APPS_MOUNT:=$APPS_PATH:$APPS_PATH}" # アプリケーションマウントポイントの指定
: "${RESULT_MOUNT:=$RESULT_PATH:$RESULT_PATH}" # 結果マウントポイントの指定
: "${WORK_MOUNT:=$WORK_DIR:$WORK_DIR}" # 作業ディレクトリマウントポイントの指定

declare -a CONTAINER_OPTIONS=(
    --container-image $IMAGE
    --container-mounts ${DATA_MOUNT},${APPS_MOUNT},${RESULT_MOUNT},${WORK_DIR}
    --container-workdir $APPS_PATH
)

#################################################
## W&Bの設定
export WANDB_API_KEY=$WANDB_API_KEY
export WANDB_PROJECT="RetNet"
export WANDB_NAME="open-calm_alpaca+ichikara"
export WANDB_NOTES="open-calm-3b, ctxLen2048, alpaca+ichikara"
export RESULT_DIR=/nas/$USER/results
export HF_HOME=/nas/share/huggingface
export SLACK_API_TOKEN=$SLACK_API_TOKEN

#################################################
## 実行するプログラムと引数
srun -l "${CONTAINER_OPTIONS[@]}" \
    torchrun \
        --nproc_per_node 8 \
        train_instruct_calm.py \
            --max_seq_length 2048 \
            --bf16 True \
            --optim adamw_bnb_8bit \
            --gradient_checkpointing False \
            --output_dir $RESULT_PATH/$WANDB_PROJECT/$WANDB_NAME \
            --do_train \
            --do_eval \
            --num_train_epochs 1 \
            --save_steps 10 \
            --eval_steps 10 \
            --evaluation_strategy steps \
            --remove_unused_columns True \
            --lr_scheduler_type cosine \
            --learning_rate 1e-5 \
            --weight_decay 0.01 \
            --logging_steps 1 \
            --ddp_find_unused_parameters False \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps 8
