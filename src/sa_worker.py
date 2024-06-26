import torch
import time
from .sa_config import SAConfig
import numpy as np


def SAWorker(process_idx, initial_block_sizes_constraints, fixed_outline_params, hard_preplace, command_q, output_q, nets, terminals):
    from .c_src import ca_sa
    torch.manual_seed(torch.seed() + process_idx)
    ca_sa.set_seed((torch.seed() + process_idx) % 2**30)
    config_inst = SAConfig(ca_sa)

    config_inst.set_config(
        {'alpha': 0.0,
         'T0': 0.0001,
         'ws_ratio': 0.0,
         'chip_ar': 1.0,
         'inverse': 0,
         'ar_search': False,
         'ws_threshold': 10})

    if fixed_outline_params.fixed_outline_enable:
        floorplan_ar = fixed_outline_params.fixed_outline_ar
        extra_wspace = fixed_outline_params.fixed_outline_extra_wspace
        block_sizes = torch.LongTensor(initial_block_sizes_constraints)[:, :2]
        height = torch.sqrt(block_sizes.prod(-1).sum() *
                            (1+extra_wspace) / (floorplan_ar)).long().item()
        width = int(height * floorplan_ar)
        print('setting fixed outline to ', width, height)
        ca_sa.set_floorplan_boundaries(
            width, height, fixed_outline_params.fixed_outline_weight, True)
    else:
        ca_sa.set_floorplan_boundaries(0, 0, 0.0, False)

    ca_sa.read_nets(nets)
    ca_sa.read_terminals(terminals)
    ca_sa.read_blocks(initial_block_sizes_constraints, False)
    print(f'setting hard preplace to {hard_preplace}')
    ca_sa.hard_preplace_constraints(hard_preplace)

    ca_sa.initialize(1)
    ca_sa.clear_trajectory()
    while True:
        (config_dict,
         block_sizes_constraints,
         btree,
         n_steps) = command_q.get()
        if n_steps == -1:
            break

        config_inst.set_config(config_dict)
        if block_sizes_constraints is not None:
            ca_sa.read_blocks(block_sizes_constraints, False)

        if btree is not None:
            ca_sa.set_btree(btree)

        ca_sa.clear_trajectory()
        ca_sa.set_step_limit(n_steps)
        t1 = time.time()
        ca_sa.sa_refine()
        print(f'took {time.time() - t1} to refine for {n_steps} steps')
#        st.go_to_min_cost_sol()
        cost = ca_sa.compute_cost()
        pos_size = torch.LongTensor(ca_sa.get_bpos())
        btree = torch.LongTensor(ca_sa.get_btree())
        output_q.put((process_idx, pos_size, btree, cost))
