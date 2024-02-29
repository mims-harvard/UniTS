

model_name=UniTS
exp_name=UniTS_pretrain_x128
wandb_mode=disabled
ptune_name=prompt_tuning

d_model=128

random_port=$((RANDOM % 9000 + 1000))

# Get the pretrained model
# cripts/pretrain_prompt_learning/UniTS_pretrain_x128.sh
ckpt_path=pretrain_ckpt.pth


offset=384
torchrun --nnodes 1 --master_port $random_port run.py \
  --is_training 0 \
  --zero_shot_forecasting_new_length unify \
  --max_offset 384 \
  --offset $offset \
  --model_id $exp_name \
  --model $model_name \
  --lradj prompt_tuning \
  --prompt_num 10 \
  --patch_len 16 \
  --stride 16 \
  --e_layers 3 \
  --d_model $d_model \
  --des 'Exp' \
  --itr 1 \
  --debug $wandb_mode \
  --project_name $ptune_name \
  --pretrained_weight $ckpt_path \
  --task_data_config_path  data_provider/multitask_zero_shot_new_length.yaml