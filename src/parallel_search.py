import torch
import torch.multiprocessing as mp
import time
import torch.distributed as dist
from .comm import collect_results
from .sa_worker import SAWorker


class ParallelSearch:
    def __init__(self, n_workers,  initial_block_sizes_constraints, fixed_outline_params, hard_preplace, nets, terminals):
        self.n_workers = n_workers
        ctx = mp.get_context('spawn')
        self.output_q = ctx.Queue(n_workers)
        self.command_qs = [ctx.Queue(1) for _ in range(n_workers)]
        self.processes = [mp.Process(target=SAWorker, args=(idx, initial_block_sizes_constraints, fixed_outline_params, hard_preplace,
                                                            self.command_qs[idx], self.output_q, nets, terminals),
                                     daemon=True) for idx in range(n_workers)]
        [p.start() for p in self.processes]
        self.default_config = {}

    def run_one_stage(self, config_dict, block_sizes_constraints, n_steps=1000, btree=None):
        for q in self.command_qs:
            q.put(({**self.default_config, **config_dict},
                  block_sizes_constraints,
                  btree,
                  n_steps))
        sa_results = [self.output_q.get() for _ in range(self.n_workers)]
        sa_results = sorted(sa_results, key=lambda x: x[0])
        _, layouts, trees, costs = list(zip(*sa_results))
        layouts = torch.stack(layouts)
        trees = torch.stack(trees)
        costs = torch.Tensor(costs)
        if dist.is_initialized():
            layouts, trees, costs = collect_results([layouts, trees, costs])

        return layouts, trees, costs

    def terminate(self):
        for q in self.command_qs:
            q.put((None,
                  None,
                  None,
                  -1))
        [p.join() for p in self.processes]
