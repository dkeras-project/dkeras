#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function, division

import time

import numpy as np
import ray

from dkeras.config import config


@ray.remote(num_cpus=config.N_CPUS_PER_WORKER, num_gpus=config.N_GPUS_PER_WORKER)
class WorkerTask(object):

    def __init__(self, worker_id, weights, ds, make_model):
        """

        :param worker_id:
        :param weights:
        :param ds:
        :param make_model:
        """
        self.worker_id = worker_id
        self.ds = ds
        self.model = make_model()
        self.model.set_weights(weights)
        self.batch_size = config.WORKER_INFERENCE_BATCH_SIZE
        self.wait_time = config.WORKER_WAIT_TIME
        self.ds.is_ready.remote(self.worker_id)

    def inference(self, data, datatype, packet_id):
        if datatype == 'float':
            data = np.asarray(data)
            results = self.model.predict(data, batch_size=self.batch_size)
            self.ds.push.remote(results, packet_id)
        elif datatype == 'int8':
            data = np.asarray(data)
            data = np.float16(data / 255)
            results = self.model.predict(data, batch_size=self.batch_size)
            self.ds.push.remote(results, packet_id)
        else:
            raise UserWarning("Invalid datatype flag {}".format(datatype))

    def main(self):
        while True:
            flag, data = ray.get(self.ds.pull.remote())
            packet_id, mode, datatype = flag.split('_')
            if mode == 'STOP':
                break
            if len(data) > 0:
                if mode == 'infer':
                    if datatype == 'float':
                        data = np.asarray(data)
                        results = self.model.predict(data, batch_size=self.batch_size)
                        self.ds.push.remote(results, packet_id)
                    elif datatype == 'int8':
                        data = np.asarray(data)
                        data = np.float16(data / 255)
                        results = self.model.predict(data, batch_size=self.batch_size)
                        self.ds.push.remote(results, packet_id)
                    else:
                        raise UserWarning("Invalid datatype flag {}".format(datatype))
                else:
                    raise UserWarning("Invalid mode flag {}".format(mode))
            else:
                time.sleep(self.wait_time)


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
        flag, data, job_config = ray.get(ds.pull.remote())
        packet_id, mode, datatype = flag.split('_')
        if mode == 'STOP':
            break
        if len(data) > 0:
            if mode == 'infer':
                verbose, steps, callbacks, max_queue_size, workers, use_multiprocessing = config
                if datatype == 'float':
                    data = np.asarray(data)
                elif datatype == 'int8':
                    data = np.asarray(data)
                    data = np.float16(data / 255)
                else:
                    raise UserWarning("Invalid datatype flag {}".format(datatype))
                results = worker_model.predict(data,
                                               batch_size=batch_size,
                                               verbose=verbose,
                                               steps=steps,
                                               callbacks=callbacks,
                                               max_queue_size=max_queue_size,
                                               workers=workers,
                                               use_multiprocessing=use_multiprocessing)
                ds.push.remote(results, packet_id)
            else:
                raise UserWarning("Invalid mode flag {}".format(mode))
        else:
            time.sleep(wait_time)
