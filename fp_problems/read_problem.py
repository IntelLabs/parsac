import torch
import re
import os
from .rect_utils import is_overlap


def generate_preplaced_blocks(block_sizes, preplaced_blocks):
    all_rects = torch.LongTensor(0, 4)
    max_dim = torch.sqrt(block_sizes.prod(-1).sum()).long()
    for picked_block in preplaced_blocks:
        trial = 0
        while True:
            trial += 1
            picked_block_size = block_sizes[picked_block]
            urx_test = torch.randint(picked_block_size[0], max_dim, size=(1,))
            ury_test = torch.randint(picked_block_size[1], max_dim, size=(1,))
            test_rect = torch.LongTensor([urx_test - picked_block_size[0],
                                          ury_test - picked_block_size[1],
                                          urx_test,
                                          ury_test])
            if not torch.any(is_overlap(test_rect, all_rects)):
                all_rects = torch.cat((all_rects, test_rect.unsqueeze(0)), dim=0)
                break
            assert trial < 100, 'failed to pre-place blocks after 100 trials'
    return all_rects[:, :2]


def augment_problem(block_sizes, problem_config):
    n_blocks = block_sizes.size(0)
    n_edge_blocks = problem_config.n_edge_blocks
    n_clusters = problem_config.n_clusters
    cluster_size = problem_config.cluster_size
    n_preplaced = problem_config.n_preplaced
    torch.manual_seed(problem_config.seed)

    edge_constraints = [torch.LongTensor(n_edge_blocks).fill_(x) for x in [1, 2, 4, 8]] + \
        [torch.LongTensor(1).fill_(x) for x in [5, 6, 9, 10]]
    edge_constraints = torch.cat(edge_constraints)
    assert n_blocks > len(edge_constraints), 'too many edge constraints'
    edge_constraints = torch.cat(
        (edge_constraints, torch.zeros(n_blocks - len(edge_constraints)).long()))

    perm = torch.randperm(n_blocks)
    edge_constraints = edge_constraints[perm]

    non_edge_blocks = (edge_constraints == 0).nonzero().view(-1)
    cluster_assignment = torch.cat([torch.LongTensor(cluster_size).fill_(x+1)
                                   for x in range(n_clusters)])
    assert len(non_edge_blocks) > cluster_assignment.size(0), 'too many cluster assignments'
    perm = torch.randperm(len(non_edge_blocks))

    clusters = torch.zeros(n_blocks).long()
    clusters[non_edge_blocks[perm][:len(cluster_assignment)]] = cluster_assignment

    non_edge_or_cluster = torch.logical_and(edge_constraints == 0, clusters == 0).nonzero().view(-1)
    preplaced_blocks = non_edge_or_cluster[torch.randperm(len(non_edge_or_cluster))][:n_preplaced]
    preplaced_pos = generate_preplaced_blocks(block_sizes, preplaced_blocks)

    return edge_constraints, clusters, preplaced_blocks, preplaced_pos


def read_problem(problem_config):
    base_dir = problem_config.base_dir
    base_name = problem_config.base_name

    reg_ex = re.compile(
        '(\w+) [\s,\S]* \([\d.]+, *[\d.]+\) *\([\d.]+, *[\d.]+\) *\(([\d.]+, *[\d.]+)\) *\([\d.]+, *[\d.]+\)')
    block_file = base_name + '.blocks'
    net_file = base_name + '.nets'
    place_file = base_name + '.pl'

    with open(os.path.join(base_dir, block_file)) as f:
        all_lines = f.readlines()

    block_sizes = []
    block_names = []
    pin_names = []
    block_lines = [x for x in all_lines if 'hardrectilinear' in x]
    pin_lines = [x for x in all_lines if 'terminal' in x]
    for b_line in block_lines:
        m = reg_ex.search(b_line)
        block_names.append(m.groups()[0])
        block_sizes.append([int(z) for z in m.groups()[1].split(',')])
    for p_line in pin_lines:
        pin_names.append(p_line.strip().split()[0])
    block_sizes = torch.LongTensor(block_sizes)

    net_pins = []
    net_blocks = []
    all_nets = []
    n_blocks = len(block_names)
    n_pins = len(pin_names)
    b2b_connectivity = torch.zeros(n_blocks, n_blocks)
    p2b_connectivity = torch.zeros(n_blocks, n_pins)

    pins_pos = torch.zeros(n_pins, 2)
    blocks_pos = torch.zeros(n_blocks, 2)
    with open(os.path.join(base_dir, net_file)) as f:
        net_count = 0
        for line in f.readlines():
            if 'NetDegree' in line:
                net_count = int(line.split(':')[1])
            else:
                if net_count != 0:
                    net_target = line.split()[0]
                    if net_target in pin_names:
                        net_pins.append(pin_names.index(net_target))
                    elif net_target in block_names:
                        net_blocks.append(block_names.index(net_target))
                    net_count -= 1
                if net_count == 0 and (net_pins or net_blocks):
                    for block1 in net_blocks:
                        for pin in net_pins:
                            p2b_connectivity[block1, pin] += 1
                        for block2 in net_blocks:
                            b2b_connectivity[block1, block2] += 1
                    all_nets.append((net_blocks, net_pins))
                    net_pins = []
                    net_blocks = []
    b2b_connectivity[torch.arange(n_blocks), torch.arange(n_blocks)] = False
    with open(os.path.join(base_dir, place_file)) as f:
        for line in f.readlines():
            parts = line.split()
            if len(parts) == 3:
                if parts[0] in pin_names:
                    pins_pos[pin_names.index(parts[0])] = torch.Tensor(
                        [float(parts[1]), float(parts[2])])
                elif parts[0] in block_names:
                    blocks_pos[block_names.index(parts[0])] = torch.Tensor([
                        float(parts[1]), float(parts[2])])

    if problem_config.augment_constraints:
        edge_constraints, clusters, preplaced_blocks, preplaced_pos = augment_problem(block_sizes,
                                                                                      problem_config)
    else:
        edge_constraints = clusters = torch.zeros(len(block_sizes)).long()
        preplaced_blocks = torch.LongTensor([])
        preplaced_pos = torch.LongTensor(0, 2)

    fixed = torch.zeros(len(block_sizes)).long()
    ar_group = torch.zeros(len(block_sizes)).long()
    return (block_sizes, blocks_pos, pins_pos, fixed, ar_group, clusters, edge_constraints, preplaced_blocks,
            preplaced_pos, all_nets)


