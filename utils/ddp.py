import torch.distributed as dist
import torch


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):

    dist.init_process_group(
        backend="nccl",
    )
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    print(f"Start running basic DDP on rank {rank}.")

    dist.barrier()
    setup_for_distributed(rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def gather_tensors_from_all_gpus(tensor_list, device_id, to_numpy=True):
    """
    Gather tensors from all GPUs in a DDP setup onto each GPU.

    Args:
    local_tensors (list of torch.Tensor): List of tensors on the local GPU.

    Returns:
    list of torch.Tensor: List of all tensors gathered from all GPUs, available on each GPU.
    """
    world_size = dist.get_world_size()
    tensor_list = [tensor.to(device_id).contiguous() for tensor in tensor_list]
    gathered_tensors = [[] for _ in range(len(tensor_list))]

    # Gathering tensors from all GPUs
    for tensor in tensor_list:
        # Each GPU will gather tensors from all other GPUs
        gathered_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_list, tensor)
        gathered_tensors.append(gathered_list)
    del tensor_list
    # Flattening the gathered list
    flattened_tensors = [
        tensor for sublist in gathered_tensors for tensor in sublist]
    del gathered_tensors
    if to_numpy:
        flattened_tensors_numpy = [tensor.cpu().numpy()
                                   for tensor in flattened_tensors]
        del flattened_tensors

        return flattened_tensors_numpy
    else:
        return flattened_tensors
