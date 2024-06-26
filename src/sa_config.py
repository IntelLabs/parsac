class SAConfig:
    def __init__(self, sa_module):
        self.sa_module = sa_module
        self.set_get_funcs = {
            'T0': (sa_module.set_T0, sa_module.get_T0),
            'beta_preplaced': (sa_module.set_beta_preplaced, sa_module.get_beta_preplaced),
            'beta_cluster': (sa_module.set_beta_cluster, sa_module.get_beta_cluster),
            'beta': (sa_module.set_beta, sa_module.get_beta),
            'alpha': (sa_module.set_alpha, sa_module.get_alpha),
            'fixing_prob': (sa_module.set_fixing_prob, sa_module.get_fixing_prob),
            'ar_increment': (sa_module.set_ar_increment, sa_module.get_ar_increment),
            'ar_search': (sa_module.ar_search, None),
            'btree': (sa_module.set_btree, sa_module.get_btree),
            'ws_ratio': (sa_module.set_ws_ratio, sa_module.get_ws_ratio),
            'inverse': (sa_module.set_inverse, sa_module.get_inverse),
            'chip_ar': (sa_module.set_chip_ar, sa_module.get_chip_ar),
            'ws_threshold': (sa_module.set_ws_thresh, sa_module.get_ws_thresh),

        }

    def get_blocks(self):
        return self.sa_module.get_bpos()

    def set_blocks(self, block_pos_constraints, init=False):
        self.sa_module.read_blocks(block_pos_constraints, init)

    def set_config(self, config_dict):
        for k, v in config_dict.items():
            self.__setitem__(k, v)

    def __setitem__(self, key, val):
        self.set_get_funcs[key][0](val)

    def __getitem__(self, key):
        return self.set_get_funcs[key][1]()
