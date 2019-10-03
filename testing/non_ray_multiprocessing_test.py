#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import division, print_function

import time

import numpy as np
from tensorflow.keras.applications import ResNet50


def main():
    n_data = 1000
    test_data = np.random.uniform(-1, 1, (n_data, 224, 224, 3))
    model = ResNet50()

    start_time = time.time()
    preds = model.predict(test_data, use_multiprocessing=True, multiprocessing_workers=4)
    elapsed = time.time() - start_time
    print("{}\nMultiprocessing: Elapsed {}\tFPS: {}".format('='*80, elapsed, n_data/elapsed))

    start_time = time.time()
    preds = model.predict(test_data)
    elapsed = time.time() - start_time
    print("{}\nSerial: Elapsed {}\tFPS: {}".format('=' * 80, elapsed, n_data / elapsed))


if __name__ == "__main__":
    main()

"""
workers = 4
================================================================================
Multiprocessing: Elapsed 108.5789122581482	FPS: 9.209891490002066
================================================================================
Serial: Elapsed 103.73267221450806	FPS: 9.640164266973738


"""
