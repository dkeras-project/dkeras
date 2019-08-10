#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import print_function, division
import numpy as np
import time
import ray


class FIFOQueue(object):

    def __init__(self, use_locking=False, use_semaphore=True, semaphore=10):
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

    def can_push(self):
        if self.use_locking:
            return not self.push_locked
        if self.use_semaphore:
            return self.semaphore > 0
        else:
            return True

    def push(self, val):
        if self.use_locking:
            self.push_locked = True
        if self.use_semaphore:
            self.semaphore -= 1
        val = self.preprocess_val(val)
        self._queue.insert(0, val)

    def push_batch(self, batch):
        if self.use_locking:
            self.push_locked = True
        if self.use_semaphore:
            self.semaphore -= 1
        batch = list(map(self.preprocess_val, batch))
        self._queue = batch + self._queue

    def pull(self, remove=True, index=0):
        if self.use_locking:
            self.push_locked = False
        if self.use_semaphore:
            self.semaphore += 1
        if len(self._queue) == 0:
            return []
        if remove is True:
            return self.postprocess_val(self._queue.pop(index))
        else:
            return self.postprocess_val(self._queue[index])

    def pull_batch(self, full=True, batch_size=None, remove=True, index=0):
        if self.use_locking:
            self.push_locked = False
        if self.use_semaphore:
            self.semaphore += 1
        if len(self._queue) == 0:
            return []
        if full:
            output = self._queue.copy()
            if remove is True:
                self._queue = []
        else:
            output = self._queue[-batch_size:]
            if remove is True:
                self._queue = self._queue[-batch_size:]
        return list(map(self.postprocess_val, output))


@ray.remote
class RayFIFOQueue(FIFOQueue):

    def __init__(self, use_locking=False, use_semaphore=True, semaphore=10):
        super().__init__(use_locking=False, use_semaphore=True, semaphore=10)
