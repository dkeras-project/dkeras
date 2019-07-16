#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function, division
import numpy as np
import time
import ray


@ray.remote(num_cpus=cpus_per_worker, num_gpus=gpus_per_worker)
def worker_task(weights, ds, make_model):
    """

    :param weights:
    :param ds:
    :return:
    """
    worker_model = make_model()
    worker_model.set_weights(weights)
    while True:
        packet_id, data = ray.get(ds.pull.remote())
        if packet_id == 'STOP':
            break
        if len(data) > 0:
            data = np.asarray(data)
            results = worker_model.predict(data)
            ds.push.remote(results, packet_id)
        else:
            time.sleep(1e-3)
