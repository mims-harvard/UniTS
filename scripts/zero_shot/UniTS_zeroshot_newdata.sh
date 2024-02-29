model_name=UniTS_zeroshot
exp_name=UniTS_zeroshot_pretrain_x64
wandb_mode=online
ptune_name=zeroshot_newdata

d_model=64

random_port=$((RANDOM % 9000 + 1000))

# Pretrain of zero-shot version of UniTS
torchrun --nnodes 1 --nproc-per-node 2 --master_port $random_port run_pretrain.py \
  --is_training 1 \
  --model_id $exp_name \
  --model $model_name \
  --prompt_num 10 \
  --patch_len 16 \
  --stride 16 \
  --e_layers 3 \
  --d_model $d_model \
  --des 'Exp' \
  --acc_it 128 \
  --batch_size 32 \
  --learning_rate 5e-5 \
  --min_lr 1e-4 \
  --weight_decay 5e-6 \
  --train_epochs 10 \
  --warmup_epochs 0 \
  --min_keep_ratio 0.5 \
  --right_prob 0.5 \
  --min_mask_ratio 0.7 \
  --max_mask_ratio 0.8 \
  --debug $wandb_mode \
  --task_data_config_path data_provider/multi_task_pretrain.yaml

# Zero-shot test on new forecasting datasets
# Note: The inference in this code test all samples of the dataset, 
# which is not the same as the original paper that only test 1 sample for each dataset.
torchrun --nnodes 1 --master_port $random_port run.py \
  --is_training 0 \
  --model_id $exp_name \
  --model $model_name \
  --prompt_num 10 \
  --patch_len 16 \
  --stride 16 \
  --e_layers 3 \
  --d_model $d_model \
  --des 'Exp' \
  --debug $wandb_mode \
  --project_name $ptune_name \
  --pretrained_weight auto \
  --task_data_config_path  data_provider/zeroshot_task.yaml