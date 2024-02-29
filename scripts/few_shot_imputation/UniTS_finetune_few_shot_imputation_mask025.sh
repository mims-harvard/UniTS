model_name=UniTS
wandb_mode=online
project_name=fewshot_imputation
exp_name=fewshot_imputation_finetune_mask025
ckpt_path=newcheckpoints/units_x64_supervised_checkpoint.pth

random_port=$((RANDOM % 9000 + 1000))
torchrun --nnodes 1 --nproc-per-node=1  --master_port $random_port  run.py \
  --is_training 1 \
  --fix_seed 2021 \
  --model_id $exp_name \
  --subsample_pct 0.1 \
  --mask_rate 0.25 \
  --model $model_name \
  --prompt_num 10 \
  --patch_len 16 \
  --stride 16 \
  --e_layers 3 \
  --d_model 64 \
  --des 'Exp' \
  --itr 1 \
  --lradj finetune_imp \
  --learning_rate 3e-4 \
  --weight_decay 5e-6 \
  --train_epochs 20 \
  --batch_size 32 \
  --acc_it 32 \
  --clip_grad 1.0 \
  --debug $wandb_mode \
  --project_name $project_name \
  --pretrained_weight $ckpt_path \
  --task_data_config_path data_provider/imputation.yaml \