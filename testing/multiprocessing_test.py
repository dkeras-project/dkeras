#!/bin/env/python
# -*- encoding: utf-8 -*-
"""

"""
from __future__ import division, print_function

import time

import numpy as np
from tensorflow.keras.applications import ResNet50

from dkeras import dKeras
import ray


def main():

    n_data = 80
    test_data = np.random.uniform(-1, 1, (n_data, 224, 224, 3))
    model = dKeras(ResNet50, weights='imagenet', wait_for_workers=True, n_workers=4)

    start_time = time.time()
    preds = model.predict(test_data, use_multiprocessing=True, multiprocessing_workers=4)
    elapsed = time.time() - start_time

    model.close()


if __name__ == "__main__":
    main()
