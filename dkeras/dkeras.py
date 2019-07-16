#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function, division
import numpy as np
import psutil
import time
import ray

from dkeras.data_server import DataServer
from dkeras.worker import worker_task


class dKeras(object):

    def __init__(self, model, n_workers=None, cpus_per_worker=1, gpus_per_worker=0):
        """

        :param model:
        :param n_workers:
        :param cpus_per_worker:
        :param gpus_per_worker:
        """
        if not ray.is_initialized():
            ray.init()

        self.model = model
        for k in self.model.__dict__.keys():
            if not (k in self.__dict__):
                self.__dict__[k] = self.model.__dict__[k]

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

    def predict(self, data, distributed=True, stop_ray=False):
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
        if stop_ray:
            ray.shutdown()