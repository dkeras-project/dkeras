#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function, division
import numpy as np
import psutil
import time
import ray

from tensorflow.keras.applications import ResNet50


@ray.remote
class DataServer(object):

    def __init__(self, n_workers):
        self.n_workers = n_workers
        self.id_indexes = {}
        self.id = 0
        self.data = []
        self.indexes = []
        self.n_data = 0
        self.results = []

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def is_complete(self):
        return self.n_workers == len(self.results)

    def push_data(self, data):
        self.n_data = len(data)
        self.indexes = list(range(self.n_data))
        self.data = data

    def push(self, results, packet_id):
        self.results.append(results)

    def pull(self):
        if len(self.data) == 0:
            return None, []
        output = self.data[:self.batch_size]
        self.data = self.data[self.batch_size:]
        packet_id = str(self.id)
        self.id_indexes[packet_id] = self.indexes[:self.batch_size]
        self.indexes = self.indexes[self.batch_size:]
        self.id += 1
        return packet_id, output


class dKeras(object):

    def __init__(self, model, n_workers=None):
        self.model = model

        def make_model():
            return model()

        # self.n_workers = max(1, psutil.cpu_count() - 2)
        if n_workers is None:
            self.n_workers = 20
        else:
            self.n_workers = n_workers
        ds = DataServer.remote(self.n_workers)
        temp = make_model()
        weights = temp.get_weights()
        # weights = np.float16(weights)
        weights = ray.put(weights)
        del temp

        @ray.remote(num_cpus=1)
        def worker_task(weights, ds):
            worker_model = make_model()
            # weights = np.float32(weights)
            worker_model.set_weights(weights)
            while True:
                packet_id, data = ray.get(ds.pull.remote())
                if len(data) > 0:
                    data = np.asarray(data)
                    results = worker_model.predict(data)
                    # print(np.asarray(results).shape)
                    ds.push.remote(results, packet_id)
                else:
                    time.sleep(1e-3)

        for _ in range(self.n_workers):
            worker_task.remote(weights, ds)
        self.ds = ds

    def predict(self, data):
        n_data = len(data)
        start_time = time.time()
        self.ds.set_batch_size.remote(int(n_data / self.n_workers))
        self.ds.push_data.remote(data)
        while not ray.get(self.ds.is_complete.remote()):
            time.sleep(1e-4)
        elapsed = time.time() - start_time
        print("{}\nN Workers: {}\tN Data: {}".format('=' * 80, self.n_workers, n_data))
        print("Time elapsed: {}\nFPS: {}".format(elapsed, float(n_data / elapsed)))


def main():
    ray.init()

    n_data = 1000
    data = np.random.uniform(-1, 1, (n_data, 224, 224, 3))

    model = dKeras(ResNet50)
    time.sleep(20)

    model.predict(data)


if __name__ == "__main__":
    main()