import torch
import torch.distributed as dist
import os


def init_comm(dist_backend='gloo'):
    world_size = -1

    world_size = int(os.environ.get("PMI_SIZE", -1))
    if world_size == -1:
        world_size = int(os.environ["WORLD_SIZE"])

    rank = int(os.environ.get("PMI_RANK", -1))
    if rank == -1:
        rank = int(os.environ["RANK"])

    dist.init_process_group(backend=dist_backend,
                            world_size=world_size, rank=rank)
    return rank, world_size


def collect_results(results_list):
    assert dist.is_initialized()
    combined_results = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(combined_results, results_list)
    combined_results = [torch.cat(x, dim=0) for x in list(zip(*combined_results))]

    print('combined results ', [x.size() for x in combined_results])
    return combined_results
