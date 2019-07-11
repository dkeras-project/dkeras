#!/bin/env/python
#-*- encoding: utf-8 -*-
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

    def is_complete(self):
        #print(len(self.results))
        return self.n_workers == len(self.results)

    def push_data(self, data):
        self.n_data = len(data)
        self.indexes = list(range(self.n_data))
        self.data = data

    def push(self, results, packet_id):
        self.results.append(results)

    def pull(self, n_data):
        if len(self.data) == 0:
            return None, []
        print("Server got request of {} size data".format(n_data))
        output = self.data[:n_data]
        self.data = self.data[n_data:]
        packet_id = str(self.id)
        self.id_indexes[packet_id] = self.indexes[:n_data]
        self.indexes = self.indexes[n_data:]
        self.id += 1
        return packet_id, output


def make_model():
    return ResNet50()


@ray.remote(num_cpus=1)
def worker_task(weights, ds, batch_size):
    model = make_model()
    weights = np.float32(weights)
    model.set_weights(weights)
    while True:
        packet_id, data = ray.get(ds.pull.remote(batch_size))
        if len(data) > 0:
            data = np.asarray(data)
            results = model.predict(data)
            print(np.asarray(results).shape)
            ds.push.remote(results, packet_id)
        else:
            time.sleep(1e-3)


def local_test():
    n_data = 1000
    model = make_model()
    data = np.random.uniform(-1, 1, (n_data, 224, 224, 3))
    start_time = time.time()
    model.predict(data)
    elapsed_local = time.time() - start_time


def main():
    n_data = 1000
    data = np.random.uniform(-1, 1, (n_data, 224, 224, 3))

    ray.init()
    n_workers = max(1, psutil.cpu_count()-2)
    ds = DataServer.remote(n_workers)
    model = make_model()
    weights = model.get_weights()
    weights = np.float16(weights)
    weights = ray.put(weights)

    # n_workers = 5
    batch_size = int(n_data/n_workers)
    workers = [worker_task.remote(weights, ds, batch_size) for _ in range(n_workers)]

    time.sleep(10)

    start_time = time.time()
    ds.push_data.remote(data)
    while not ray.get(ds.is_complete.remote()):
        time.sleep(1e-4)
    elapsed = time.time()-start_time
    print("{}\nN Workers: {}\tN Data: {}".format('='*80, n_workers, n_data))
    print("Time elapsed: {}\nFPS: {}".format(elapsed, float(n_data/elapsed)))
    #start_time = time.time()
    #model.predict(data)
    #elapsed_local = time.time()-start_time
    #print("\n{}\nLocal\tN Data: {}".format('='*80, n_data))
    #print("Time elapsed: {}\nFPS: {}".format(elapsed_local, float(n_data/elapsed_local)))


if __name__ == "__main__":
    main()
