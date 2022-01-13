#!/bin/bash

MODEL_CLASS=$1
MODEL_DIR=$2
DATASET=$3
DATA_FORMAT=$4
CONCAT_WAY=$5
GPU_IDX=$6

export CUDA_VISIBLE_DEVICES=${GPU_IDX}

# OPT_PARAM
if [ $MODEL_CLASS == "bert" ]
then
  OPT_PARAM="--do_lower_case --adam_epsilon 1e-6 --max_grad_norm 1. --weight_decay 0.01 --warmup_proportion 0.1"
elif [ $MODEL_CLASS == "roberta" ]
then
  OPT_PARAM="--adam_betas 0.9,0.98 --adam_epsilon 1e-6 --max_grad_norm 0. --weight_decay 0.01 --warmup_proportion 0.05"
else
  OPT_PARAM=""
fi


python run_absa_train.py --dataset $DATASET --do_train --do_eval --do_prediction --max_seq_length 64 \
  $OPT_PARAM \
  --num_train_epochs 10 \
  --eval_step 200 --train_batch_size 16 \
  --eval_batch_size 32 --logging_step 10 \
  --gradient_accumulation_steps 1 \
  --model_class $MODEL_CLASS --model_name_or_path $MODEL_DIR \
  --output_dir ./runtime/${DATASET}-${MODEL_DIR} \
  --seed 9 \
  --data_format $DATA_FORMAT \
  --concat_way $CONCAT_WAY \
  --disc_lr 2e-4 \
  --task_lr 1e-5 \
  --discriminator_ratio 0.3333 \
  --encoder_ratio 1 \
  --encoder_lr 1e-5 \
  --adv_loss_weight 0.05 \
  --use_dep_dis_features \
  --save_spans_info \
  --eval_interpretable \
  --use_gate --overwrite_output_dir  \


# Please use "--overwrite_output_dir" to re-create the output folder to save model.