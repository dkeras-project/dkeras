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
        """

        :param n_workers:
        """
        self.n_workers = n_workers
        self.id_indexes = {}
        self.id = 0
        self.data = []
        self.indexes = []
        self.n_data = 0
        self.results = []

    def set_batch_size(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        self.batch_size = batch_size

    def is_complete(self):
        """

        :return:
        """
        return self.n_workers == len(self.results)

    def push_data(self, data):
        """

        :param data:
        :return:
        """
        self.n_data = len(data)
        self.indexes = list(range(self.n_data))
        self.data = data

    def push(self, results, packet_id):
        """

        :param results:
        :param packet_id:
        :return:
        """
        self.results.append(results)

    def pull(self):
        """

        :return:
        """
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

    def __init__(self, model, n_workers=None, cpus_per_worker=1, gpus_per_worker=0):
        """

        :param model:
        :param n_workers:
        :param cpus_per_worker:
        :param gpus_per_worker:
        """
        self.model = model

        def make_model():
            return model()

        if n_workers is None:
            # self.n_workers = max(1, psutil.cpu_count() - 2)
            self.n_workers = 20
        else:
            self.n_workers = n_workers
        ds = DataServer.remote(self.n_workers)
        temp = make_model()
        weights = temp.get_weights()
        weights = ray.put(weights)
        del temp

        @ray.remote(num_cpus=cpus_per_worker, num_gpus=gpus_per_worker)
        def worker_task(weights, ds):
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

        for _ in range(self.n_workers):
            worker_task.remote(weights, ds)
        self.ds = ds

    def compile(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        self.model.compile(args, kwargs)

    def evaluate(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        # TODO: Implement Distributed Version
        self.model.evaluate(kwargs)

    def fit(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        # TODO: Implement Distributed Version
        self.model.fit(kwargs)

    def train_on_batch(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        # TODO: Implement Distributed Version
        self.model.train_on_batch(args, kwargs)

    def test_on_batch(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        # TODO: Implement Distributed Version
        self.model.test_on_batch(args, kwargs)

    def fit_generator(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        # TODO: Implement Distributed Version
        self.model.fit_generator(args, kwargs)

    def evaluate_generator(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        # TODO: Implement Distributed Version
        self.model.evaluate_generator(args, kwargs)

    def predict_on_batch(self, data):
        """

        :param data:
        :return:
        """
        self.predict(data)

    def predict(self, data, distributed=True):
        """

        :param data:
        :param distributed:
        :return:
        """
        if distributed:
            n_data = len(data)
            self.ds.set_batch_size.remote(int(n_data / self.n_workers))
            self.ds.push_data.remote(data)
            while not ray.get(self.ds.is_complete.remote()):
                time.sleep(1e-4)
        else:
            self.model.predict(data)