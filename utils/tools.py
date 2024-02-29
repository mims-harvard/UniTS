import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import inf

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, base_lr, args):
    assert args.prompt_tune_epoch >= 0, "args.prompt_tune_epoch >=0!"
    if args.lradj == 'prompt_tuning':
        if epoch < args.prompt_tune_epoch:
            lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch) // 1))}
        elif epoch == args.prompt_tune_epoch:
            lr_adjust = {epoch: base_lr}
        else:
            lr_adjust = {epoch: args.learning_rate *
                         (0.5 ** (((epoch-args.prompt_tune_epoch) - 1) // 1))}
    elif args.lradj == 'supervised':
        if epoch <= args.prompt_tune_epoch:
            lr_adjust = {epoch: base_lr}
        else:
            lr_adjust = {epoch: base_lr / 5 *
                         (0.5 ** (((epoch-args.prompt_tune_epoch)) // 1))}
    elif args.lradj == 'finetune_anl':
        k = 1
        lr_adjust = {epoch: base_lr / (2 ** ((epoch) // k))}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        print('Epoch {}: Updating learning rate to {}'.format(epoch+1, lr))


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(
            start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * \
        (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class NativeScalerWithGradNormCount:
    # https://github.com/facebookresearch/mae/blob/main/util/misc.py
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                # unscale the gradients of optimizer's assigned params in-place
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device)
                         for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(
            p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def check_cuda_memory():
    """
    Check and print the current GPU memory usage in PyTorch.
    """
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        total_memory = torch.cuda.get_device_properties(
            current_device).total_memory
        allocated_memory = torch.cuda.memory_allocated(current_device)
        cached_memory = torch.cuda.memory_reserved(current_device)

        print(f"GPU: {gpu_name}")
        print(f"Total Memory: {total_memory / 1e9:.5f} GB")
        print(f"Allocated Memory: {allocated_memory / 1e9:.5f} GB")
        print(f"Cached Memory: {cached_memory / 1e9:.5f} GB")
    else:
        print("CUDA is not available.")
