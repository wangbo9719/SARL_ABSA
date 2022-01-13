# SARL_ABSA

This repo contains the source code of the paper accepted by Findings of EMNLP'2021. "[Eliminating Sentiment Bias for Aspect-Level Sentiment Classification with Unsupervised Opinion Extraction](https://aclanthology.org/2021.findings-emnlp.258.pdf)"

## 1. Thanks
The repository is partially based on [huggingface transformers](https://github.com/huggingface/transformers) and [Position-Aware Tagging for Aspect Sentiment Triplet Extraction (In EMNLP 2020)](https://github.com/xuuuluuu/SemEval-Triplet-data).

## 2. Installing requirement packages
- conda create -n absa python=3.6
- source activate absa
- pip install tqdm pandas torch==1.7.0 tensorboardX boto3 requests regex sacremoses sentencepiece lxml sklearn
- pip install nltk spacy==2.3.5 https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz networkx

## 3. Dataset
- The pre-processed datasets for SARL are in "./dataset/bias_xx". 
- Note, the pre-processed Twitter do not contained in this file due to the limitation of the file size. 
- The bias labels are derived from SentiWordNet and the opinion annotations are derived from [Position-Aware Tagging for Aspect Sentiment Triplet Extraction (In EMNLP 2020)](https://github.com/xuuuluuu/SemEval-Triplet-data). 


## 4. Training and Test
Run the following commands for getting the similar results reported in paper. 
Note, the results are not very stable, several times running with different seeds are needed for reproducing the reported results. . 

### 4.1 Laptop14
```
CUDA_VISIBLE_DEVICES=2 python run_absa_train.py --dataset lap14_wordnet_bias_distillation \
  --do_train --do_eval --do_prediction --max_seq_length 64 \
  --adam_betas 0.9,0.98 --adam_epsilon 1e-6 --max_grad_norm 0. --weight_decay 0.01 --warmup_proportion 0.05 \
  --num_train_epochs 10 \
  --eval_step 200 --train_batch_size 16 \
  --eval_batch_size 32 --logging_step 10 \
  --gradient_accumulation_steps 1 \
  --model_class roberta --model_name_or_path roberta-large\
  --output_dir ./runtime/lap14_roberta-large \
  --seed 54 \
  --data_format multi_aspects \
  --concat_way adv_distillation \
  --discriminator_ratio 0.3333 \
  --encoder_ratio 1 \
  --encoder_lr 1e-5 \
  --others_lr 2e-4 \
  --adv_loss_weight 0.1 \
  --use_dep_dis_features \
  --save_spans_info \
  --eval_interpretable \
  --use_gate
```
### 4.2 Restaurant14
```
CUDA_VISIBLE_DEVICES=3 python run_absa_train.py --dataset rest14_wordnet_bias_distillation \
  --do_train --do_eval --do_prediction --max_seq_length 64 \
  --adam_betas 0.9,0.98 --adam_epsilon 1e-6 --max_grad_norm 0. --weight_decay 0.01 --warmup_proportion 0.05 \
  --num_train_epochs 10 \
  --eval_step 200 --train_batch_size 16 \
  --eval_batch_size 32 --logging_step 10 \
  --gradient_accumulation_steps 1 \
  --model_class roberta --model_name_or_path roberta-large\
  --output_dir ./runtime/rest14_roberta-large \
  --seed 54 \
  --data_format multi_aspects \
  --concat_way adv_distillation \
  --discriminator_ratio 0.2 \
  --encoder_ratio 1 \
  --encoder_lr 1e-5 \
  --others_lr 1e-5 \
  --adv_loss_weight 0.1 \
  --use_dep_dis_features \
  --save_spans_info \
  --eval_interpretable \
  --use_gate 
```
### 4.3 Restaurant15
```
CUDA_VISIBLE_DEVICES=3 python run_absa_train.py --dataset rest15_wordnet_bias_distillation \
  --do_train --do_eval --do_prediction --max_seq_length 64 \
  --adam_betas 0.9,0.98 --adam_epsilon 1e-6 --max_grad_norm 0. --weight_decay 0.01 --warmup_proportion 0.05 \
  --num_train_epochs 10 \
  --eval_step 200 --train_batch_size 16 \
  --eval_batch_size 32 --logging_step 10 \
  --gradient_accumulation_steps 1 \
  --model_class roberta --model_name_or_path roberta-large\
  --output_dir ./runtime/rest15_roberta-large \
  --seed 54 \
  --data_format multi_aspects \
  --concat_way adv_distillation \
  --discriminator_ratio 0.2 \
  --encoder_ratio 1 \
  --encoder_lr 1e-5 \
  --others_lr 1e-5 \
  --adv_loss_weight 0.05 \
  --use_dep_dis_features \
  --save_spans_info \
  --eval_interpretable \
  --use_gate \
  --overwrite_output_dir 
```
### 4.4 Restaurant16
```
CUDA_VISIBLE_DEVICES=2 python run_absa_train.py --dataset rest16_wordnet_bias_distillation \
  --do_train --do_eval --do_prediction --max_seq_length 64 \
  --adam_betas 0.9,0.98 --adam_epsilon 1e-6 --max_grad_norm 0. --weight_decay 0.01 --warmup_proportion 0.05 \
  --num_train_epochs 10 \
  --eval_step 200 --train_batch_size 16 \
  --eval_batch_size 32 --logging_step 10 \
  --gradient_accumulation_steps 1 \
  --model_class roberta --model_name_or_path roberta-large\
  --output_dir ./runtime/rest16_roberta-large \
  --seed 54 \
  --data_format multi_aspects \
  --concat_way adv_distillation \
  --discriminator_ratio 0.1 \
  --encoder_ratio 1 \
  --encoder_lr 1e-5 \
  --others_lr 2e-4 \
  --adv_loss_weight 0.1 \
  --use_dep_dis_features \
  --save_spans_info \
  --eval_interpretable \
  --use_gate \
  --overwrite_output_dir 
``` 
### 4.5 Twitter
```
CUDA_VISIBLE_DEVICES=3 python run_absa_train.py --dataset twitter_wordnet_bias_distillation \
  --do_train --do_eval --do_prediction --max_seq_length 64 \
  --adam_betas 0.9,0.98 --adam_epsilon 1e-6 --max_grad_norm 0. --weight_decay 0.01 --warmup_proportion 0.05 \
  --num_train_epochs 7 \
  --eval_step 200 --train_batch_size 16 \
  --eval_batch_size 32 --logging_step 10 \
  --gradient_accumulation_steps 1 \
  --model_class roberta --model_name_or_path roberta-large\
  --output_dir ./runtime/twitter_roberta-large \
  --seed 9 \
  --data_format multi_aspects \
  --concat_way adv_distillation \
  --discriminator_ratio 0.3333 \
  --encoder_ratio 1 \
  --encoder_lr 2e-5 \
  --others_lr 1e-5 \
  --adv_loss_weight 0.05 \
  --use_dep_dis_features \
  --save_spans_info \
  --eval_interpretable \
  --use_gate \
  --overwrite_output_dir 
```
## 5. Dataset Processing
We have provided our used datasets in "./datasets". To get these datasets from scratch by yourself, please perform following steps.

### 5.1 Get the bias labels according to SentiWordNets.
- Convert the SentiWordNet file from text format to python dict and store it in json with "./absa/get_senti_dictionary.py".
- Add the bias label of each aspect in their original datasets and save as new files called "wordnet_bias_train.json" or "wordnet_bias_test.json". 
    ```
    python run_absa_train.py \
    --model_class roberta \ 
    --model_name_or_path roberta-large \
    --get_wordnet_bias \
    --output_dir ./runtime/get_bias_dataset
    ```
- For more details, please refer the code directly. 


### 5.2 Use phrase-level SST dataset to train a phrase-level sentiment classification models. 
```
CUDA_VISIBLE_DEVICES=0 python run_absa_train.py \
--dataset psst \
--do_train --do_eval \
--max_seq_length 64 \
--adam_betas 0.9,0.98 \
--learning_rate 1.3e-5 \
--num_train_epochs 7 \
--eval_step 5000 \
--train_batch_size 16 \
--eval_batch_size 32 \
--adam_epsilon 1e-6 \
--max_grad_norm 0. \
--weight_decay 0.01 \
--warmup_proportion 0.05 \
--logging_step 10 \
--model_class roberta \
--model_name_or_path roberta-large \
--output_dir ./runtime/psst_model \
--concat_way sent \
--data_format term_span \
--seed 9 --overwrite_output_dir
```

### 5.3 Use the trained phrase-level model to get the datasets for SARL. 
- Use restaurant14 dataset as example here. 
```
CUDA_VISIBLE_DEVICES=0 python run_absa_train.py \
--dataset rest14 \
--model_class roberta \ 
--model_name_or_path roberta-large \
--get_distillation_dataset \
--phrase_sentiment_model_path ./runtime/psst_model \
--output_dir ./runtime/get_final_dataset
```

### 5.4 Get the dataset with opinions for unsupervised opinion extraction evaluation. 
- run "./absa/get_opinion_dataset.py"
    
