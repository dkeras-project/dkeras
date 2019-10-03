#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import division, print_function

import numpy as np
import ray


@ray.remote
class DataServer(object):
    """
    Data server for transferring data to workers

    Arguments:
        n_workers: Number of workers
        worker_ids: List of worker IDs
    """

    def __init__(self, n_workers, worker_ids, datatype='float', mode='infer'):
        self.n_workers = n_workers
        self.id_indexes = {}
        self.id = 0
        self.data = []
        self.indexes = []
        self.n_data = 0
        self.datatype = datatype
        self.mode = mode
        self.results = [-1 for _ in range(self.n_workers)]
        self.closed = False
        self.worker_status = {}
        self.job_config = [None]
        for n in worker_ids:
            self.worker_status[n] = False

    def set_datatype(self, datatype):
        self.datatype = datatype

    def pull_results(self):
        """
        Pull the prediction results

        :return: Numpy array with the output of the workers
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
        Set the close flag to True, which will close the workers
        on the next worker data pull request

        :return: None
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

    def push_data(self, data,
                  mode='infer',
                  infer_config=None):
        """

        :param data:
        :param mode:
        :param infer_config:
        :return:
        """
        if infer_config is None:
            # verbose, steps, callbacks, max_queue_size,
            # workers, use_multiprocessing
            infer_config = [0, None, None, 10, 1, False]
        self.n_data = len(data)
        self.data = data
        self.mode = mode
        self.job_config = infer_config

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
            return '-1_STOP_float', None
        if len(self.data) == 0:
            return '-1_Empty_float', []
        output = self.data[:self.batch_size]
        self.data = self.data[self.batch_size:]
        packet_id = '{}_{}_{}'.format(self.id, self.mode, self.datatype)
        self.id += 1
        if len(self.data) == 0:
            self.id = 0
        return packet_id, output, self.job_config

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
