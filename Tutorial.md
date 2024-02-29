# Quick start for using UniTS on your own data.

## Classficiation with your own data.

We use a classification task as an example. The primary difference for other tasks lies in the data formats. You can follow the provided dataset as a guide to adapt your own data.

### 1. Prepare data

We support common data formats of time series datasets.

You can follow the [dataset format guide](https://www.aeon-toolkit.org/en/latest/examples/datasets/data_loading.html) to transfer your dataset into `.ts` format dataset.

The dataset should contain `newdata_TRAIN.ts` and `newdata_TEST.ts` files.

### 2. Define the dataset config file

To support multiple datasets, our code base uses the `data_set.yaml` to keep the dataset information.
Examples can be found in `data_provider` folder.

Here is an example for classification dataset. You can add multiple dataset config in one config file if you want to make UniTS support multiple datasets.
```yaml
task_dataset:
  CLS_ECG5000: # the dataset and task name
    task_name: classification # the type of task
    dataset: ECG5000 # the name of the dataset
    data: UEA # the data type of the dataset, use UEA if you use the '.ts' file
    embed: timeF # the embedding method used
    root_path: ../dataset/UCR/ECG5000 # the root path of the dataset
    seq_len: 140 # the length of the input sequence
    label_len: 0 # the length of the label sequence, 0 for classification
    pred_len: 0 # the length of the predicted sequence, 0 for classification
    enc_in: 1 # the number of variable numbers
    num_class: 5 # the number of classes
    c_out: None # the output variable numbers, 0 for classification
```

### 3. Finetune your UniTS model

#### Load Pretrained weights (Optional)
You can load the pretrained SSL/Supervised UniTS model.
Run [SSL Pretraining]() or [Supervised training]() scripts to get the pretrained checkpoints.
Normally, SSL pretrained model has better transfer learning abilities.

#### Setup finetuning script

**Note: Remove captions before using the following scripts!**

- Finetuning/Supervised training
```bash
model_name=UniTS # Model name, UniTS
exp_name=UniTS_supervised_x64 # Exp name
wandb_mode=online # Use wandb to log the training, change to disabled if you don't want to use it
project_name=supervised_learning # preject name in wandb

random_port=$((RANDOM % 9000 + 1000))

# Supervised learning
torchrun --nnodes 1 --nproc-per-node=1  --master_port $random_port  run.py \
  --is_training 1 \ # 1 for training, 0 for testing
  --model_id $exp_name \
  --model $model_name \
  --lradj supervised \ # You can define your own lr decay scheme in the adjust_learning_rate function of utils/tools.py
  --prompt_num 10 \ # The number of prompt tokens.
  --patch_len 16 \ # Patch size for each token in UniTS
  --stride 16 \ # Stride = patch size
  --e_layers 3 \
  --d_model 64 \
  --des 'Exp' \
  --learning_rate 1e-4 \ # Tune the following hp for your datasets. Due to the high deverse nature of time series data, you might need to tune the hp for your new data.
  --weight_decay 5e-6 \
  --train_epochs 5 \
  --batch_size 32 \ # Real batch size = batch_size * acc_it
  --acc_it 32 \
  --debug $wandb_mode \
  --project_name $project_name \
  --clip_grad 100 \ # Grad clip to avoid Nan.
  --pretrained_weight ckpt_path.pth \ # Path of pretrained ckpt if you want to finetune the model, otherwise just remove it
  --task_data_config_path data_provider/multi_task.yaml # Important: Change to your_own_data_config.yaml

```

- Prompt learning

For prompt learning, only tokens are finetuned and the model are fixed.
**You must load pretrained model weights.**
```bash
# Prompt tuning
torchrun --nnodes 1 --master_port $random_port run.py \
  --is_training 1 \
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
  --learning_rate 3e-3 \
  --weight_decay 0 \
  --prompt_tune_epoch 2 \ # Number of epochs for prompt tuning
  --train_epochs 0 \
  --acc_it 32 \
  --debug $wandb_mode \
  --project_name $ptune_name \
  --clip_grad 100 \
  --pretrained_weight auto \ # Path of pretrained ckpt, you must add it for prompt learning 
  --task_data_config_path  data_provider/multi_task.yaml # Important: Change to your_own_data_config.yaml
```

###
Feel free to open an issue if you have any problems in using our code.

This doc will be updated.