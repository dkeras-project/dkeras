#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import division, print_function

import numpy as np
import ray


@ray.remote
class DataServer(object):

    def __init__(self, n_workers, worker_ids):
        """

        :param n_workers:
        :param worker_ids:
        """
        self.n_workers = n_workers
        self.id_indexes = {}
        self.id = 0
        self.data = []
        self.indexes = []
        self.n_data = 0
        self.results = [-1 for _ in range(self.n_workers)]
        self.closed = False
        self.worker_status = {}
        for n in worker_ids:
            self.worker_status[n] = False

    def pull_results(self):
        """

        :return:
        """
        output = []
        for n in self.results:
            for x in n:
                output.append(x)
        output = np.asarray(output)
        self.results = [-1 for _ in range(self.n_workers)]
        return output

    def close(self):
        """

        :return:
        """
        self.closed = True

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
        for i in range(len(self.results)):
            if isinstance(self.results[i], int):
                return False
        return self.n_workers == len(self.results)

    def push_data(self, data):
        """

        :param data:
        :return:
        """
        self.n_data = len(data)
        self.data = data

    def parse_packet_id(self, packet_id):
        """

        :param packet_id:
        :return:
        """
        return int(packet_id)

    def push(self, results, packet_id):
        """

        :param results:
        :param packet_id:
        :return:
        """
        index = self.parse_packet_id(packet_id)
        self.results[index] = results

    def pull(self):
        """

        :return:
        """
        if self.closed:
            return 'STOP', None
        if len(self.data) == 0:
            return None, []
        output = self.data[:self.batch_size]
        self.data = self.data[self.batch_size:]
        packet_id = str(self.id)
        self.id += 1
        if len(self.data) == 0:
            self.id = 0
        return packet_id, output

    def is_ready(self, worker_id):
        """

        :param worker_id:
        :return:
        """
        self.worker_status[worker_id] = True

    def all_ready(self):
        """

        :return:
        """
        for worker_id in self.worker_status.keys():
            if not self.worker_status[worker_id]:
                return False
        return True
