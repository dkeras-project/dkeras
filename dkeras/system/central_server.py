#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function, division
import time
import ray
import cv2

from dkeras.system.video_server import VideoServer


class QueueNotFoundError(Exception):

    def __init__(self, message, errors):
        super().__init__(self, message, errors)
        self.errors = errors


class FIFOQueue(object):

    def __init__(self,
                 use_locking: bool = False,
                 use_semaphore: bool = True,
                 semaphore: int = 10):
        """

        :param use_locking:
        :param use_semaphore:
        :param semaphore:
        """
        if not isinstance(use_locking, bool):
            raise TypeError("use_locking arg must be boolean, not {}".format(
                type(use_locking).__name__))
        if not isinstance(use_semaphore, bool):
            raise TypeError("use_semaphore arg must be boolean, not {}".format(
                type(use_semaphore).__name__))
        if not isinstance(use_semaphore, int):
            raise TypeError("semaphore arg must be int, not {}".format(
                type(semaphore).__name__))
        if semaphore < 1:
            raise UserWarning("Semaphore must be > 0, not {}".format(
                semaphore))
        if use_semaphore and use_locking:
            raise UserWarning("Cannot have use_locking and use_semaphore")
        self._queue = []
        self.use_locking = use_locking
        self.use_semaphore = use_semaphore
        self.semaphore = semaphore
        if use_locking:
            self.push_locked = False

    def preprocess_val(self, val):
        return val

    def postprocess_val(self, val):
        return val

    def can_pull(self):
        return len(self._queue) > 0

    def can_push(self):
        if self.use_locking:
            return not self.push_locked
        elif self.use_semaphore:
            return self.semaphore > 0
        else:
            return True

    def push(self, data, index=0, expand=False, verify=False):
        if self.use_locking:
            self.push_locked = True
        if expand:
            if verify:
                if not isinstance(data, list):
                    raise TypeError(
                        "Input data should be type list with expand=True")
            data = list(map(self.preprocess_val, data))
            self._queue = self._queue[:index] + data + self._queue[index:]
        else:
            data = self.preprocess_val(data)
            self._queue.insert(index, data)

    def pull(self,
             remove: bool = True,
             batch_size: int = 1,
             wait_batch: bool = False,
             wait_batch_time: float = 1e-4,
             index: int = 0,
             wrap: bool = False,
             verify: bool = True):
        """

        :param remove:
        :param batch_size:
        :param wait_batch:
        :param wait_batch_time:
        :param index:
        :param wrap:
        :param verify:
        :return:
        """
        if verify:
            if not isinstance(batch_size, int):
                raise TypeError("batch_size arg should be int, not: {}".format(
                    type(batch_size).__name__))
            if (batch_size < -1) or (batch_size == 0):
                raise UserWarning(
                    "batch_size should be -1 or > 0, not {}".format(
                        batch_size))
            if not isinstance(index, int):
                raise TypeError("index arg should be int, not: {}".format(
                    type(index).__name__))
            if not isinstance(wrap, int):
                raise TypeError("wrap arg should be boolean, not: {}".format(
                    type(wrap).__name__))
        if self.use_locking:
            self.push_locked = False
        if self.use_semaphore:
            self.semaphore += 1
        if wait_batch:
            if batch_size != -1:
                while True:
                    if len(self._queue) >= batch_size:
                        break
                    else:
                        time.sleep(wait_batch_time)
        if batch_size == -1:
            output = self._queue
        else:
            output = self._queue[index:index + batch_size]
        if len(output) == 0:
            return []
        output = list(map(self.postprocess_val, output))
        if not wrap:
            if batch_size == 1:
                output = output[0]
        if remove is True:
            if batch_size == -1:
                self._queue = []
            else:
                self._queue = self._queue[index + batch_size:]
        return output


@ray.remote
class RayFIFOQueue(FIFOQueue):

    def __init__(self, use_locking=False, use_semaphore=True, semaphore=10):
        super().__init__(use_locking=False, use_semaphore=True, semaphore=10)


@ray.remote
class CentralServer(object):

    def __init__(self, verbose=True):
        self.queues = {}
        self.verbose = verbose

    def _create_queue(self, queue_name):
        if self.verbose:
            print("Creating new queue: {}".format(queue_name))
        self.queues[queue_name] = RayFIFOQueue.remote()

    def pull(self, queue, create=True):
        if not (queue in self.queues.keys()):
            if create:
                self._create_queue(queue)
            return []
        else:
            if ray.get(self.queues[queue].can_pull.remote()):
                return ray.get(self.queues[queue].pull.remote())
            else:
                return []

    def push(self, data, queues, create=True, verify=True, expand=False):
        if isinstance(queues, str):
            queues = [queues]
        if verify:
            if not isinstance(queues, (str, list, tuple)):
                raise TypeError(
                    "Queues must be type (str, list, tuple) not {}".format(
                        type(queues).__name__))
            queue_names = list(self.queues.keys())
            for q in queues:
                if queue_names.count(q) > 1:
                    raise Exception("Multiple queue counts of {}".format(q))
        for q in queues:
            if not (q in self.queues.keys()):
                if create:
                    self._create_queue(q)
                else:
                    raise QueueNotFoundError('{} not found'.format(q))
            self.queues[q].push.remote(data, expand=expand)

    def can_push(self, queue, create=True):
        if not (queue in self.queues.keys()):
            if create:
                self._create_queue(queue)
            else:
                return False
        return ray.get(self.queues[queue].can_push.remote())

    def can_pull(self, queue, create=False):
        if not (queue in self.queues.keys()):
            if create:
                self._create_queue(queue)
            else:
                return False
        return ray.get(self.queues[queue].can_pull.remote())


def main():
    ray.init()
    server = CentralServer.remote()
    video_server = VideoServer.remote(server, camera=0, output_queues=('frames'))

    while True:
        data = ray.get(server.pull.remote('frames'))
        if (len(data) > 0):
            frame = data['frame']
            cv2.imshow('frames', frame)
            cv2.waitKey(1)


if __name__ == '__main__':
    main()
