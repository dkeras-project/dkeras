#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import division, print_function

import os
import time

import ray

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from dkeras.servers.data_server import DataServer
from dkeras.workers.worker import worker_task
from dkeras.config import config


class dKeras(object):

    def __init__(self,
                 model,
                 n_workers=None,
                 init_ray=True,
                 rm_existing_ray=False,
                 rm_local_model=True,
                 wait_for_workers=False,
                 redis_address=None):
        """

        :param model:
        :param n_workers:
        """
        if init_ray:
            if ray.is_initialized():
                if rm_existing_ray:
                    ray.shutdown()
                    ray.init()
            else:
                ray.init()

        if n_workers is None:
            # self.n_workers = max(1, psutil.cpu_count() - 2)
            self.n_workers = config.DEFAULT_N_WORKERS
        else:
            self.n_workers = n_workers
        worker_ids = []
        for i in range(self.n_workers):
            worker_ids.append('worker_{}'.format(i))
        self.worker_ids = worker_ids
        self.model = model()
        ds = DataServer.remote(self.n_workers, self.worker_ids)
        weights = self.model.get_weights()
        weights = ray.put(weights)
        if rm_local_model:
            del self.model
        else:
            self.__dict__.update(self.model.__dict__)
            for k in dir(self.model):
                try:
                    if not k in dir(self):
                        self.__dict__[k] = getattr(self.model, k)
                except AttributeError:
                    pass

        def make_model():
            return model()

        for i in range(self.n_workers):
            worker_id = self.worker_ids[i]
            worker_task.remote(worker_id, weights, ds, make_model)
        self.data_server = ds

        if wait_for_workers:
            while True:
                if self.is_ready():
                    break
                else:
                    time.sleep(1e-3)

    def predict(self, data, distributed=True, stop_ray=False):
        """

        :param data:
        :param distributed:
        :return:
        """
        if distributed:
            n_data = len(data)
            self.data_server.set_batch_size.remote(
                int(n_data / self.n_workers))
            self.data_server.push_data.remote(data)
            while not ray.get(self.data_server.is_complete.remote()):
                time.sleep(1e-4)
            return ray.get(self.data_server.pull_results.remote())
            print("Completed!")
        else:
            return self.model.predict(data)
        if stop_ray:
            ray.shutdown()

    def close(self, stop_ray=False):
        self.data_server.close.remote()
        if stop_ray:
            ray.shutdown()

    def is_ready(self):
        return ray.get(self.data_server.all_ready.remote())
