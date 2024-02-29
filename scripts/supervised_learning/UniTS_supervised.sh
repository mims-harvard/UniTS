model_name=UniTS
exp_name=UniTS_supervised_x64
wandb_mode=online
project_name=supervised_learning

random_port=$((RANDOM % 9000 + 1000))

# Supervised learning
torchrun --nnodes 1 --nproc-per-node=1  --master_port $random_port  run.py \
  --is_training 1 \
  --model_id $exp_name \
  --model $model_name \
  --lradj supervised \
  --prompt_num 10 \
  --patch_len 16 \
  --stride 16 \
  --e_layers 3 \
  --d_model 64 \
  --des 'Exp' \
  --learning_rate 1e-4 \
  --weight_decay 5e-6 \
  --train_epochs 5 \
  --batch_size 32 \
  --acc_it 32 \
  --debug $wandb_mode \
  --project_name $project_name \
  --clip_grad 100 \
  --task_data_config_path data_provider/multi_task.yaml