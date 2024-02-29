from data_provider.data_factory import data_provider
from utils.tools import cosine_scheduler
from utils.tools import NativeScalerWithGradNormCount as NativeScaler
from utils.losses import UnifiedMaskRecLoss
from utils.dataloader import BalancedDataLoaderIterator
from utils.ddp import is_main_process, get_world_size

import torch
import torch.nn as nn
from torch import optim
import torch.distributed as dist

import os
import time
import warnings
import numpy as np
import yaml
import wandb
import importlib
import sys

warnings.filterwarnings('ignore')

def custom_print_decorator(func):
    def wrapper(*args, **kwargs):
        text = ' '.join(map(str, args))
        if 'file' not in kwargs or kwargs['file'] is None:
            sys.stdout.write(text + '\n')
        else:
            kwargs['file'].write(text + '\n')

        if 'folder' in kwargs and kwargs['folder']:
            with open(f'{kwargs["folder"]}/finetune_output.log', 'a') as log_file:
                log_file.write(text + '\n')
        if 'folder' in kwargs:
            del kwargs['folder']
        if 'file' in kwargs:
            del kwargs['file']
    return wrapper


# replace print to save all print into log files
print = custom_print_decorator(print)


def read_task_data_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    task_dataset_config = config.get('task_dataset', {})
    return task_dataset_config


def get_task_data_config_list(task_data_config, default_batch_size=None):
    task_data_config_list = []

    for task_name, task_config in task_data_config.items():
        task_config['max_batch'] = default_batch_size
        task_data_config_list.append([task_name, task_config])

    return task_data_config_list


def init_and_merge_datasets(data_loader_list):
    dataloader = BalancedDataLoaderIterator(data_loader_list)
    train_steps = dataloader.__len__()

    return dataloader, train_steps


class Exp_All_Task(object):
    def __init__(self, args):
        super(Exp_All_Task, self).__init__()

        self.args = args
        self.task_data_config = read_task_data_config(
            self.args.task_data_config_path)
        self.task_data_config_list = get_task_data_config_list(
            self.task_data_config, default_batch_size=self.args.batch_size)
        device_id = dist.get_rank() % torch.cuda.device_count()
        print("this device_id:", device_id)
        self.device_id = device_id

    def _build_model(self, ddp=True):
        module = importlib.import_module("models."+self.args.model)
        model = module.Model(
            self.args, self.task_data_config_list, pretrain=True).to(self.device_id)
        if ddp:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[self.device_id], find_unused_parameters=True)
        return model.to(self.device_id)

    def _get_data(self, flag):
        data_set_list = []
        data_loader_list = []
        for task_data_name, task_config in self.task_data_config.items():
            print("loading dataset:", task_data_name, folder=self.path)
            if task_config['data'] == 'UEA' and flag == 'val':
                # TODO strange that no val set is used for classification. Set to test set for val
                flag = 'test'
            data_set, data_loader = data_provider(
                self.args, task_config, flag, ddp=True)
            data_set_list.append(data_set)
            data_loader_list.append(data_loader)
        return data_set_list, data_loader_list

    def _select_optimizer(self):
        eff_batch_size = self.args.batch_size * self.args.acc_it * get_world_size()
        real_learning_rate = self.args.learning_rate * eff_batch_size / 32
        print("base lr: %.2e" % (self.args.learning_rate * 32 / eff_batch_size))
        print("actual lr: %.2e" % real_learning_rate)
        self.real_learning_rate = real_learning_rate

        print("accumulate grad iterations: %d" % self.args.acc_it)
        print("effective batch size: %d" % eff_batch_size)
        model_optim = optim.Adam(self.model.parameters(
        ), lr=real_learning_rate, betas=(0.9, self.args.beta2), weight_decay=self.args.weight_decay, eps=self.args.eps)
        return model_optim

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and is_main_process():
            os.makedirs(path)
        self.path = path

        torch.cuda.synchronize()
        dist.barrier()

        # Data loader
        _, train_loader_list = self._get_data(flag='train')
        data_loader_cycle, train_steps = init_and_merge_datasets(
            train_loader_list)

        # Set up batch size for each task
        if self.args.memory_check:
            self.memory_check(data_loader_cycle)
            torch.cuda.empty_cache()

        torch.cuda.synchronize()
        dist.barrier()

        # Model
        self.model = self._build_model()

        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        print("Parameters number {} M".format(
            pytorch_total_params/1e6), folder=self.path)
        print("{} steps for each epoch".format(train_steps), folder=self.path)

        # Optimizer
        model_optim = self._select_optimizer()
        lr_schedule = cosine_scheduler(
            self.real_learning_rate,
            self.args.min_lr,
            self.args.train_epochs, train_steps,
            warmup_epochs=self.args.warmup_epochs,
        )

        # Loss
        criterion = UnifiedMaskRecLoss().to(self.device_id)
        scaler = NativeScaler()

        for epoch in range(self.args.train_epochs):
            train_loss = self.train_one_epoch(
                model_optim, data_loader_cycle, criterion, epoch, train_steps, scaler, lr_schedule)

            print("Epoch: {0}, Steps: {1} | Avg Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss), folder=self.path)
            if is_main_process():
                wandb.log({'train_loss_avg': train_loss})

            if is_main_process():
                save_dict = {
                    'student': self.model.state_dict(),
                    'optimizer': model_optim.state_dict(),
                    'epoch': epoch + 1,
                    'args': self.args,
                }

                torch.save(save_dict, path + '/' + 'pretrain_checkpoint.pth')

        return self.model

    def train_one_epoch(self, model_optim, data_loader_cycle, criterion, epoch, train_steps, scaler, lr_schedule):
        current_device = torch.cuda.current_device()
        train_loss_set = []

        acc_it = self.args.acc_it
        max_norm = self.args.clip_grad
        min_keep_ratio = self.args.min_keep_ratio

        self.model.train()
        epoch_time = time.time()
        self.model.zero_grad(set_to_none=True)
        loss_sum_display = 0

        for i, (sample_init, task_id) in enumerate(data_loader_cycle):
            it = train_steps * epoch + i
            for _, param_group in enumerate(model_optim.param_groups):
                param_group["lr"] = lr_schedule[it]

            # Get batch data based on the real batch size of each task: avoid OOM for large samples
            task_name = self.task_data_config_list[task_id][1]['task_name']
            small_batch_size = self.task_data_config_list[task_id][1]['max_batch']
            sample_list = self.get_multi_source_data(
                sample_init, task_name, small_batch_size, min_keep_ratio=min_keep_ratio)
            len_sample_list = len(sample_list)

            # Accumulate gradients of mulitple samples
            for sample_idx in range(len_sample_list):
                sample = sample_list[sample_idx]
                x_enc, x_mark_enc, pad_mask = sample
                with torch.cuda.amp.autocast():
                    model_output = self.model(
                        x_enc=x_enc, x_mark_enc=x_mark_enc, task_id=task_id, task_name=task_name, enable_mask=True)
                loss_dict = criterion(model_output, x_enc, pad_mask)
                loss = loss_dict['loss']
                loss /= acc_it
                loss /= len_sample_list
                if sample_idx < len_sample_list-1:
                    norm_value = scaler(loss, model_optim, clip_grad=max_norm,
                                        parameters=self.model.parameters(), create_graph=False, update_grad=False)

            loss_display = loss.item()*len_sample_list*acc_it
            train_loss_set.append(loss_display)

            norm_value = scaler(loss, model_optim, clip_grad=max_norm,
                                parameters=self.model.parameters(), create_graph=False, update_grad=((i + 1) % acc_it == 0))

            if (i+1) % acc_it == 0:
                model_optim.zero_grad()
            torch.cuda.synchronize()

            loss_sum_display += loss_display

            # release memory to avoid OOM
            del sample_init
            del sample_list
            if torch.cuda.memory_reserved(current_device) > 30*1e9:
                torch.cuda.empty_cache()

            if is_main_process():
                wandb_loss_dict = {
                    'norm': norm_value if norm_value is not None else 0,
                    'train_cls_loss_'+self.task_data_config_list[task_id][0]: loss_dict['cls_loss'].item(),
                    'train_mask_loss_'+self.task_data_config_list[task_id][0]: loss_dict['mask_loss'].item(),
                    'train_sum_loss_'+self.task_data_config_list[task_id][0]: loss_dict['loss'].item(),
                    "loss_avg": loss_sum_display/(i+1)
                }
                wandb.log(wandb_loss_dict)

            if (i + 1) % 50 == 0 and is_main_process():
                print("\titers: {0}, epoch: {1} | lr: {2:.5} | loss_avg: {3} | current_loss: {4} |current data: {5}".format(
                    i + 1, epoch + 1, lr_schedule[it], loss_sum_display/(i+1), loss.item() * acc_it, task_name), folder=self.path)

        if is_main_process():
            print("Epoch: {} cost time: {}".format(
                epoch + 1, time.time() - epoch_time), folder=self.path)
        train_loss = np.average(train_loss_set)

        return train_loss

    def get_multi_source_data(self, this_batch, task_name, small_batch_size, min_keep_ratio=None):
        """
        Splits the input batch into smaller batches based on the specified small_batch_size.

        Args:
            this_batch (tuple): The input batch containing all data of a task.
            task_name (str): The name of the task.
            small_batch_size (int): The size of the smaller batches to split the data into.
            min_keep_ratio (float, optional): The minimum ratio of data to keep in each smaller batch.

        Returns:
            list: A list of tuples, where each tuple contains a smaller batch of data, marks, and padding masks.
        """

        def split_tensor(tensor, size):
            return [tensor[i:min(i + size, tensor.size(0))] for i in range(0, tensor.size(0), size)]

        if "long_term_forecast" in task_name:
            batch_x, _, batch_x_mark, _ = this_batch
            batch_x = batch_x.float().to(self.device_id)
            batch_x_mark = batch_x_mark.float().to(self.device_id)
            batch_x_mark = batch_x_mark.max(dim=-1)[0]
            padding_mask = torch.ones(
                (batch_x.shape[0], batch_x.shape[1]), dtype=torch.bool).to(self.device_id)
        elif "classification" in task_name:
            batch_x, _, padding_mask = this_batch
            batch_x = batch_x.float().to(self.device_id)
            batch_x_mark = padding_mask.float().to(self.device_id)
            padding_mask = batch_x_mark.bool().to(self.device_id)

        if min_keep_ratio is not None:
            keep_ratios = torch.rand(
                1, device=batch_x.device) * (1.0 - min_keep_ratio) + min_keep_ratio
            L = batch_x.shape[1]
            len_keeps = (L * keep_ratios).long()
            len_keeps = (torch.ceil(len_keeps/self.args.patch_len)
                         )*self.args.patch_len
            len_keeps = len_keeps.int()

            batch_x = batch_x[:, :len_keeps]
            batch_x_mark = batch_x_mark[:, :len_keeps]
            padding_mask = padding_mask[:, :len_keeps]

        split_batch_x = split_tensor(batch_x, small_batch_size)
        split_batch_x_mark = split_tensor(batch_x_mark, small_batch_size)
        split_padding_mask = split_tensor(padding_mask, small_batch_size)

        return list(zip(split_batch_x, split_batch_x_mark, split_padding_mask))

    def memory_check(self, data_loader_cycle, holdout_memory=6):
        """
        Checks the memory usage of the model by gradually increasing the batch size until it reaches the maximum batch size that can be supported without running out of memory.

        Args:
            data_loader_cycle (DataLoaderCycle): The data loader cycle object.
            holdout_memory (int): The amount of memory (in GB) to hold out for other operations.

        Returns:
            None
        """
        num_elements = holdout_memory * 1024 * 1024 * 1024 // 4
        extra_mem = torch.empty(
            num_elements, dtype=torch.float32, device=self.device_id)

        model_tmp = self._build_model(ddp=False)
        criterion = UnifiedMaskRecLoss().to(self.device_id)
        model_tmp.train()
        model_tmp.zero_grad(set_to_none=True)

        for data_loader_id in range(data_loader_cycle.num_dataloaders):
            batch_size = 1
            max_batch_size = 0
            torch.cuda.synchronize()
            model_tmp.zero_grad(set_to_none=True)
            while True:
                try:
                    sample, task_id = data_loader_cycle.generate_fake_samples_for_batch(
                        data_loader_id, batch_size)
                    task_name = self.task_data_config_list[task_id][1]['task_name']
                    if "long_term_forecast" in task_name:
                        batch_x, _, batch_x_mark, _ = sample
                        batch_x = batch_x.float().to(self.device_id)
                        batch_x_mark = batch_x_mark.float().to(self.device_id)
                    elif "classification" in task_name:
                        batch_x, _, batch_x_mark = sample
                        batch_x = batch_x.float().to(self.device_id)
                        batch_x_mark = torch.ones(
                            (batch_x.shape[0], batch_x.shape[1]), dtype=torch.bool).to(self.device_id)

                    print(task_id, task_name,
                          sample[0].shape, "max batch size", max_batch_size)
                    with torch.cuda.amp.autocast():
                        model_output = model_tmp(
                            x_enc=batch_x, x_mark_enc=batch_x_mark, task_id=task_id, task_name=task_name, enable_mask=True)
                    loss = 0.0
                    for each in model_output:
                        if each is not None:
                            loss += each.sum()

                    loss.backward()
                    max_batch_size = batch_size
                    batch_size *= 2

                    if max_batch_size >= self.args.batch_size:
                        print("can support default batchsize:",
                              self.args.batch_size, max_batch_size)
                        self.task_data_config_list[task_id][1]['max_batch'] = max_batch_size
                        self.task_data_config_list[task_id][1]['checkpointing'] = False
                        break

                except Exception as e:
                    task_name = self.task_data_config_list[task_id][1]['task_name']
                    print(task_id,  "max batch size:", max_batch_size)
                    self.task_data_config_list[task_id][1]['max_batch'] = max_batch_size
                    print(f"An exception occurred: {e}")
                    del model_tmp
                    del criterion
                    torch.cuda.empty_cache()
                    model_tmp = self._build_model(ddp=False)
                    criterion = UnifiedMaskRecLoss().to(self.device_id)
                    break
        del extra_mem
        del model_tmp
        del criterion
        torch.cuda.empty_cache()
        print(self.task_data_config_list)
        return
