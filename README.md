# PARSAC

This is the implementation of the Parallel Simulated Annealing with Constraints (PARSAC) algorithm described in the paper ['Design Planning of Modern SOCs with Complex Constraints: A Path Towards Human-Quality FloorPlanning'](https://arxiv.org/abs/2405.05495). The code can be used to reproduce the experiments in the paper. PARSAC achieves state-of-the-art performance on classical floorplanning problems. The Simulated Annealing (SA) engine in PARSAC is especially suited to problems with boundary constraints that require certain blocks to be placed at specific boundaries of the floorplan, and to problems with preplaced blocks where certain blocks have to be placed at specific x-y locations.

## Installing required packages
```shell
pip3 install -r requirements.txt
```
PARSAC was developed and tested on Python3.9. The SA C++ routines are compiled automatically on first use.   

## Usage

The floorplanning problem is described using three PyTorch tensors:

1. `block_specs` : This is a `n_blocks x 9` integer tensor with the following column format:
   -  `block_specs[:,:2]` : The initial width and height of the blocks
   -  `block_specs[:,2]` : The boundary constraints for each block. A block's boundary constraint is an integer from the enumeration below. `NC` indicates no boundary constraint

        ````cpp
        enum edge_loc{
        NC = 0,
        LEFT=1,
        RIGHT=2,
        TOP=4,
        BOTTOM=8,
        TOP_LEFT = 5,
        TOP_RIGHT=6,
        BOTTOM_LEFT=9,
        BOTTOM_RIGHT=10,
        };
        ````

   - `block_specs[:,3]` : Group IDs for grouping constraints. 0 indicates a block is not part of any grouping constraint. Blocks that are part of the same grouping constraint have the same non-zero ID.
   - `block_specs[:,4]` : Unused. Must be set to zero
   - `block_specs[:,5]` : 0/1 flag indicating whether a block has a fixed aspect ratio (1) or non-fixed aspect ratio (0). SA only modifies the aspect ratios of non-fixed blocks.
   - `block_specs[:,6]` : 0/1 flag indicating pre-placed block. Pre-placed blocks have to be placed at specific x-y positions
   - `block_specs[:,7:9]` : x-y locations of pre-placed blocks. This is only used for blocks with pre-placed flag set to 1

1.  `pin_loc` : An `n_pins x 2` tensor indicating the x-y locations of the floorplan pins. These are usually on the periphery
1. `nets` : A list of lists specifying the floorplan connectivity. `nets[i]` is a list of integers specifying the endpoints of net `i`. Integers in the range `[0,n_blocks)` denote a block's center as an endpoint, while integers in the range `[n_blocks,n_blocks + n_pins)` indicate a pin endpoint, i.e, pin indices have been shifted by `n\_blocks` to distinguish them from block indices.

The simulated annealing hyper-parameters are set using a Python dictionary which has the following fields:

```python

{'T0': 0.001, #Simulated Annealing temperature
 'beta_preplaced': 10, #Cost coefficient for preplaced blocks.
                     #Only applicable for soft pre-placement constraints
 'beta_cluster': 10, #Cost coefficient for grouping constraints violations
 'beta': 1,  #Cost coefficient for boundary constraints violations
 'alpha': 0.5, #Number between 0 and 1 governing the tradeoff between white space
             #and wirelength minimization. 1 gives full weight to wirelength minimization
             #while 0 gives full weight to white space minimization 
 'fixing_prob': 0.0005, #The probability of choosing a constraints-fixing move at
                      #each step vs. a traditional SA move
 'ar_increment': 0.1, #The aspect ratio adjustment step
 'ar_search': True, #Enable/Disable aspect ratio adjustment moves 
 }

```

The problem definition as well the SA hyper-parameters are configured using the Python API exposed by the C++ SA code. The following code snippet illustrates how to read a floorplanning problem, configure the SA hyper-parameters, run SA, and obtain the resulting floorplan:

```python

from src.c_src import ca_sa
from src.sa_config import SAConfig

##Read the problem 
ca_sa.read_nets(nets)
#The pin_loc and block_specs tensors have to be converted to lists first
ca_sa.read_terminals(pin_loc.tolist())
ca_sa.read_blocks(block_specs.tolist(), False)

##Set the SA configuration using the set_config method
#of an SAConfig instance
config_inst = SAConfig(ca_sa)
config_inst.set_config(
        {'alpha': 0.0,
         'T0': 0.0001,
         'ar_search': False
	 })

ca_sa.clear_trajectory()
ca_sa.set_step_limit(n_steps)

ca_sa.sa_refine() #Run for n_steps

#Get the block positions and sizes,  as well as the B*Tree at the end
pos_size = torch.LongTensor(ca_sa.get_bpos())
btree = torch.LongTensor(ca_sa.get_btree())

```
For more elaborate usage scenarios, check the `main.py` file for how to launch parallel SA workers, potentially across multiple machines. Some additional aspects of the SA search can be configured as follows: 

- Hard vs. soft pre-placement constraints: To enable hard pre-placement constraints that guarantee that the pre-placement constraints are exactly satisfied through the use of B*Trees with anchored blocks:
```python

ca_sa.hard_preplace_constraints(True)

```
- To solve floorplanning problems with a fixed outline:

```python

ca_sa.set_floorplan_boundaries(
            outline_width, outline_height, fixed_outline_cost_weight, True)

```

## Reproducing the experiments in the paper

`paper_experiments.yaml` contains a list of the commands needed to reproduce all experiments in the paper. `configs/sa_config.yaml` contains the configuration data for the main script: `main.py`. To launch distributed runs, use the `mpirun` command. For example, to launch on 2 machines:
```shell
mpirun -n 2 python3 main.py
```

