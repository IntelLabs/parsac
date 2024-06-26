import hydra
import torch
import omegaconf
import torch.distributed as dist
from omegaconf import DictConfig


from src.parallel_search import ParallelSearch
from src.comm import init_comm
from fp_problems.read_problem import read_problem


@ hydra.main(version_base='1.1', config_path='./configs', config_name='sa_config')
def main(cfg: DictConfig):
    if cfg.distributed:
        init_comm()

    (block_sizes, block_pos, pins_pos, fixed_block, ar_group,
     cluster_data, edge_constraints,
     preplaced_blocks, preplaced_pos, all_nets) = read_problem(cfg.problem_config)

    block_sizes = block_sizes.long()
    n_blocks = block_sizes.size(0)
    pins_pos = pins_pos.long().tolist()
    nets = [y[0] + [x+n_blocks for x in y[1]] for y in all_nets]

    rectilinear_data = torch.LongTensor(n_blocks).zero_()
    preplaced = torch.zeros(n_blocks, 3).long()
    preplaced[preplaced_blocks, 0] = 1
    preplaced[preplaced_blocks, 1:] = preplaced_pos  # Take the LL
    fixed_block[preplaced_blocks] = 1

    block_sizes_constraints = torch.cat((block_sizes.long(), edge_constraints.unsqueeze(-1),
                                         cluster_data.unsqueeze(
                                             -1), rectilinear_data.unsqueeze(-1),
                                         fixed_block.unsqueeze(-1), preplaced), dim=-1).tolist()
    optimal_tree = None
    ps = ParallelSearch(cfg.n_workers_per_machine, block_sizes_constraints,
                        cfg.fixed_outline_params, cfg.hard_preplace, nets, pins_pos)
    ps.default_config = dict(cfg.base_sa_config)
    schedule = dict(cfg.sa_schedule)
    for idx in range(cfg.n_stages):
        print(f'in iteration {idx}')
        for k, v in schedule.items():
            if idx == k:
                ps.default_config.update({'T0': v})
                print('setting temp to ', v)
                break

        layouts, trees, costs = ps.run_one_stage(
            {'ar_search': False}, None, n_steps=cfg.n_stage_steps, btree=optimal_tree)
        if cfg.ar_increment > 0.0:
            layouts, trees, costs = ps.run_one_stage(
                {'ar_search': True, 'ar_increment': cfg.ar_increment}, None, n_steps=cfg.n_stage_steps, btree=optimal_tree)
        costs = torch.Tensor(costs)
        print('costs2', costs)
    if not dist.is_initialized() or dist.get_rank() == 0:
        torch.save((block_sizes_constraints, nets, pins_pos,
                    layouts, trees, costs, cfg), cfg.results_path)

    ps.terminate()


if __name__ == "__main__":
    main()
