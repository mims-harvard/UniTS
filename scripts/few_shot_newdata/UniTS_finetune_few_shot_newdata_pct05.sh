model_name=UniTS
wandb_mode=online
project_name=fewshot_newdata
exp_name=fewshot_newdata_finetune_pct05
random_port=$((RANDOM % 9000 + 1000))
# Path to the supervised checkpoint
# get supervised checkpoint: scripts/supervised/UniTS_supervised_x64.sh
ckpt_path=newcheckpoints/units_x64_supervised_checkpoint.pth
torchrun --nnodes 1 --master_port $random_port run.py \
  --is_training 1 \
  --fix_seed 2021 \
  --model_id $exp_name \
  --subsample_pct 0.05 \
  --model $model_name \
  --prompt_num 10 \
  --patch_len 16 \
  --stride 16 \
  --e_layers 3 \
  --d_model 64 \
  --des 'Exp' \
  --train_epochs 5 \
  --learning_rate 1e-4 \
  --weight_decay 1e-5 \
  --lradj supervised \
  --dropout 0.1 \
  --acc_it 8 \
  --clip_grad 100 \
  --debug $wandb_mode \
  --project_name $project_name \
  --pretrained_weight $ckpt_path \
  --task_data_config_path data_provider/fewshot_new_task.yaml

