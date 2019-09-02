#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import division, print_function

import os
import time
import keras

import ray

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from dkeras.servers.data_server import DataServer
from dkeras.workers.worker import worker_task
from dkeras.config import config


class dKeras(object):
    """Distributed Keras Model Wrapper.

        It will automatically set up remote
        workers and data servers for data parallelism algorithms. Using
        the same notation as a regular Keras model, it makes distributing a
        Keras model simple.

        .. code-block:: python

            from tensorflow.keras.applications import ResNet50
            from dkeras import dKeras

            model = dKeras(ResNet50)
            preds = model.predict(data)

        Arguments:
            model: Un-initialized Keras model
            verbose: Verbose setting boolean variable. Default is False.
            weights: Weights arg for prebuilt models, example: ResNet50(
                weights='imagenet'). Default is None.
            n_workers: Integer number of worker processes. If left None,
                then it will automatically find the an estimate of the optimal
                number of workers. Default is None.
            init_ray: Boolean arg for whether to initialize Ray within
                the model initialization. Default is False.
            rm_existing_ray: Boolean arg for whether to remove any
                existing Ray clusters. Default is True.
            rm_local_model: Boolean arg for whether to remove the local
                copy of the Keras model for memory conservation. Default is
                False.
            wait_for_workers: Boolean arg for whether to wait for all of
                the worker processes to initialize and connect to the data
                server.
            redis_address: In the case of initializing Ray inside of
                model initialization, the redis address is required for
                connecting to existing Ray clusters.
            n_cpus_per_worker: The integer number of CPUs per worker
                processes. If left None, it will allocate automatically. The
                default is None.
            n_gpus_per_worker: The integer or float number of GPUs per
                worker processes. If left None, it will allocate
                automatically. The default is None.
            n_cpus_per_server: The integer number of CPUs per data
                server. If left None, it will allocate automatically. The
                default is None.
            """

    def __init__(self,
                 model,
                 verbose: bool = True,
                 weights: list = None,
                 n_workers: int = None,
                 init_ray: bool = True,
                 rm_existing_ray: bool = False,
                 rm_local_model: bool = True,
                 wait_for_workers: bool = False,
                 redis_address: str = None,
                 n_cpus_per_worker: int = None,
                 n_gpus_per_worker: int = None,
                 n_cpus_per_server: int = None):

        config.N_CPUS_PER_SERVER = n_cpus_per_server
        config.N_CPUS_PER_WORKER = n_cpus_per_worker
        config.N_CPUS_PER_SERVER = n_gpus_per_worker
        self.verbose = verbose
        if init_ray:
            if ray.is_initialized():
                if rm_existing_ray:
                    ray.shutdown()
                    ray.init()
                else:
                    if redis_address is None:
                        raise UserWarning(
                            "Ray already initialized, rm_existing_ray is "
                            "False, and redis_address is None")
                    else:
                        ray.init(redis_address=redis_address)
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
        self.model = model(weights=weights)
        self.input_shape = self.model.input_shape
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
            if n_data % self.n_workers > 0:
                self.data_server.set_batch_size.remote(
                    int(n_data / self.n_workers) + 1)
            else:
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
        time.sleep(5e-2)

    def is_ready(self):
        return ray.get(self.data_server.all_ready.remote())
