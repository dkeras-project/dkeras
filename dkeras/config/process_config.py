#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function, division

import dkeras.config.config as config


type_dict = {
    'CPU_OVERLOAD_LIMIT': float,
    'N_CPUS_PER_SERVER': int,
    'N_CPUS_PER_WORKER': int
}


def verify_types():
    """
    
    :return: None
    """
    local_vars = list(locals().keys())
    config_vars = dir(config)

    for v_name in type_dict.keys():
        var = config.__dict__[v_name]
        if not isinstance(var, type_dict[v_name]):
            raise TypeError(
                "Config variable should be type: {}, not type: {}".format(
                    type_dict[v], type(var).__name__))


if __name__ == "__main__":
    verify_types()