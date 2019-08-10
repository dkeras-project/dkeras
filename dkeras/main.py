#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function, division
import numpy as np
import psutil
import time
import pype
import ray

from dkeras.servers.data_server import DataServer
from dkeras.workers.worker import worker_task
from dkeras.config import config

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class dKeras(object):

    def __init__(self,
                 model,
                 n_workers=None,
                 init_ray = True,
                 rm_existing_ray = True,
                 rm_local_model=True,
                 redis_address = None):
        """

        :param model:
        :param n_workers:
        """
        if init_ray:
            if ray.is_initialized():
                if rm_existing_ray:
                    ray.shutdown()
                    pype.init_ray()
            else:
                pype.init_ray()

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
            print("Completed!")
        else:
            self.model.predict(data)
        if stop_ray:
            ray.shutdown()

    def close(self, stop_ray=False):
        self.data_server.close.remote()
        if stop_ray:
            ray.shutdown()

    def is_ready(self):
        return ray.get(self.data_server.all_ready.remote())


def main():
    from tensorflow.keras.applications import ResNet50

    n_data = 1000
    test_data = np.float16(np.random.uniform(-1, 1, (n_data, 224, 224, 3)))
    model = dKeras(ResNet50)
    # model = ResNet50()
    while True:
        if model.is_ready():
            break
        else:
            time.sleep(1e-3)
    print("Workers are ready")

    start_time = time.time()
    model.predict(test_data)
    elapsed = time.time()-start_time

    #model.close()
    #time.sleep(5)

    #output = ray.get(model.data_server.pull_results.remote())
    #print(np.asarray(output).shape)
    print("Time elapsed: {}\nFPS: {}".format(elapsed, n_data/elapsed))

if __name__ == "__main__":
    main()


"""
Serial
Time elapsed: 88.29588603973389
FPS: 11.325555978338466

3 workers
Time elapsed: 139.755868434906
FPS: 7.155334593092735

10 workers
Time elapsed: 173.76229000091553
FPS: 5.754988611134966

(10, 100, 1000)
Time elapsed: 92.77561902999878
FPS: 10.778693911777106

"""