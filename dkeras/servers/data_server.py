#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function, division
import ray


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
