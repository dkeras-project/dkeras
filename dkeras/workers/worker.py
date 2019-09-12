#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function, division
import numpy as np
import time
import ray

from dkeras.config import config


@ray.remote(num_cpus=config.N_CPUS_PER_WORKER, num_gpus=config.N_GPUS_PER_WORKER)
def worker_task(worker_id, weights, ds, make_model):
    """

    :param weights:
    :param ds:
    :return:
    """
    worker_model = make_model()
    worker_model.set_weights(weights)
    batch_size = config.WORKER_INFERENCE_BATCH_SIZE
    wait_time = config.WORKER_WAIT_TIME
    ds.is_ready.remote(worker_id)
    while True:
        flag, data = ray.get(ds.pull.remote())
        packet_id, mode, datatype = flag.split('_')
        if mode == 'STOP':
            break
        if len(data) > 0:
            if mode == 'infer':
                if datatype == 'float':
                    data = np.asarray(data)
                    results = worker_model.predict(data, batch_size=batch_size)
                    ds.push.remote(results, packet_id)
                elif datatype == 'int8':
                    data = np.asarray(data)
                    data = np.float16(data/255)
                    results = worker_model.predict(data, batch_size=batch_size)
                    ds.push.remote(results, packet_id)
                else:
                    raise UserWarning("Invalid datatype flag {}".format(datatype))
            else:
                raise UserWarning("Invalid mode flag {}".format(mode))
        else:
            time.sleep(wait_time)
